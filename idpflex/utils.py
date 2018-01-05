import sys
from collections import namedtuple


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
