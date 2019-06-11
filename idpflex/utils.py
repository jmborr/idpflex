import os
from contextlib import contextmanager
import tempfile
import functools
import multiprocessing
from collections import namedtuple, Mapping

import MDAnalysis as mda


def write_frame(atom_group, iframe, file_name):
    r"""Write a single trajectory frame to file.

    Format is guessed from the file's extension.

    Parameters
    ----------
    atom_group : :class:`~MDAnalysis.AtomGroup`
        Atoms from the universe describing the simulation
    iframe : int
        Trajectory frame index (indexes begin with zero)
    file_name : str
        Name of the file to create
    """
    atom_group.universe.trajectory[iframe]
    # Create directory if not existing
    dir_name = os.path.dirname(file_name)
    if dir_name and not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    with mda.Writer(file_name) as writer:
        writer.write(atom_group)


@contextmanager
def temporary_file(**kwargs):
    r"""Creates a temporary file

    Parameters
    ----------
    kwargs : dict
        optional arguments to tempfile.mkstemp
    Yields
    ------
    str
        Absolute path name to file
    """
    handle, name = tempfile.mkstemp(**kwargs)
    try:
        yield name
    finally:
        os.remove(name)


def namedtuplefy(func):
    r"""
    Decorator to transform the return dictionary of a function into
    a namedtuple

    Parameters
    ----------
    func: function
        Function to be decorated
    name: str
        Class name for the namedtuple. If None, the name of the function
        will be used
    Returns
    -------
    Function
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        if wrapper.nt is None:
            if isinstance(res, Mapping) is False:
                raise ValueError('Cannot namedtuplefy a non-dict')
            wrapper.nt = namedtuple(func.__name__ + '_nt', res.keys())
        return wrapper.nt(**res)
    wrapper.nt = None
    return wrapper


def generate_profile_for_frame(atom_group, iframe, profile_class):
    r"""
    Utility function to generate profile properties for a frame in a trajectory.

    Parameters
    ----------
    atom_group: :class:`MDAnalysis.AtomGroup`
        The atom group representing the structure to calculate the profile for.
    iframe: int
        The index of a trajectory for which to calculate profiles of the associated atom_group.
    profile_class:
        The profile class to use for the properties to be returned and calculated. Must implement a `from_pdb` method.

    Returns
    -------
    Profile for the selected frame
    """  # noqa: E501
    with temporary_file(suffix='.pdb') as fname:
        # Copy the atom group to a new universe to avoid
        # changing frames upon writing
        u = atom_group.universe
        u2 = mda.Universe(u.filename, u.trajectory.filename)
        atoms2 = u2.atoms[atom_group.atoms.indices]
        write_frame(atoms2, iframe, fname)
        return profile_class().from_pdb(fname)


def generate_trajectory_profiles(atom_group, iframes, profile_class):
    r"""
    Utility function to generate profile properties for each frame in a trajectory.

    Parameters
    ----------
    atom_group: :class:`MDAnalysis.AtomGroup`
        The atom group representing the structure to calculate the profile for.
    iframes: List[int]
        The indices of a trajectory for which to calculate profiles of the associated atom_group.
    profile_class:
        The profile class to use for the properties to be returned and calculated. Must implement a `from_pdb` method.

    Returns
    -------
    List of the profiles.
    """  # noqa: E501
    with multiprocessing.Pool() as p:
        return p.starmap(generate_profile_for_frame,
                         [(atom_group, i, profile_class) for i in iframes])
