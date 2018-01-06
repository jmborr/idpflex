import os
import sys

from collections import namedtuple
import MDAnalysis as mda


def returns_tuple(typename, field_names, doc=None):
    """A namedtuple with a docstring and a `keys()` method for easy access of
    fields. `keys()` returns a list instance.

    Parameters
    ----------
    typename : str
        name of the generated namedtuple class
    field_names: str
        arguments of the tuple
    doc: str
        docstring for the namedtuple class

    Returns
    -------
    namedtuple
    """
    nt = namedtuple(typename, field_names)
    if doc is not None and sys.version_info[0] >= 3:
        nt.__doc__ = doc + '\n' + nt.__doc__
    nt.keys = lambda self: self._fields
    return nt


def write_frame(a_universe, iframe, file_name):
    r"""Write a one trajectory frame to file.

    Format is guessed from the file's extension.

    Parameters
    ----------
    a_universe : :class:`~MDAnalysis.core.universe.Universe
        Universe describing the simulation
    iframe : int
        Trajectory frame index (indexes begin with zero)
    file_name : str
        Name of the file to create
    """
    a_universe.trajectory[iframe]
    # Create directory if not existing
    dir_name = os.path.dirname(file_name)
    if dir_name and not os.path.isdir(dir_name):
        os.makedirs(dir_name)

    with mda.Writer(file_name) as writer:
        writer.write(a_universe)
