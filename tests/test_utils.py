from __future__ import print_function, absolute_import

import sys
import pytest
import tempfile

from idpflex import utils


def test_returns_tuple():
    tt = utils.returns_tuple('testtuple', 'a b c', doc='test tuple')
    rt = tt(1, 2, 3)
    assert set(rt.keys()) == set(['a', 'b', 'c'])
    if sys.version_info[0] >= 3:
        assert 'test tuple' in rt.__doc__


def test_write_frame(trajectory_benchmark):
    utils.write_frame(trajectory_benchmark, 0, 'test.pdb')
    with open('test.pdb') as handle:
        content = handle.read()
        assert 'ATOM    535  OC2 TYR S  37      47.180  41.030' in content


if __name__ == '__main__':
    pytest.main()
