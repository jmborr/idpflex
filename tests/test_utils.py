from __future__ import print_function, absolute_import

import os
import pytest

from idpflex import utils


def test_write_frame(trajectory_benchmark):
    utils.write_frame(trajectory_benchmark, 0, 'test.pdb')
    with open('test.pdb') as handle:
        content = handle.read()
        assert 'ATOM    535  OC2 TYR S  37      47.180  41.030' in content
    os.remove('test.pdb')


if __name__ == '__main__':
    pytest.main()
