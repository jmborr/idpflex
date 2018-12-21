import os
from contextlib import contextmanager
import tempfile
import functools
from collections import namedtuple, Mapping

import MDAnalysis as mda


def write_frame(a_universe, iframe, file_name):
    r"""Write a single trajectory frame to file.

    Format is guessed from the file's extension.

    Parameters
    ----------
    a_universe : :class:`~MDAnalysis.core.universe.Universe`
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
    func: Function
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
