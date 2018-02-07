"""Integration tests"""

from __future__ import print_function, absolute_import

import os
import tempfile
from numpy.testing import assert_array_equal
from idpflex import cnextend


def test_save(sans_fit):
    r"""Save and load a tree containing SANS properties"""
    t = sans_fit['tree']
    handle, name = tempfile.mkstemp()
    t.save(name)
    r = cnextend.load_tree(name)
    os.remove(name)
    assert_array_equal(t[42]['sans'].y, r[42]['sans'].y)
