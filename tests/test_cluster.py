from __future__ import print_function, absolute_import

import pytest
from numpy.testing import assert_almost_equal

from idpflex import cluster as idpc


def test_cluster_trajectory(trajectory_benchmark):
    results = idpc.cluster_trajectory(trajectory_benchmark,
                                      selection='name CA',
                                      segment_length=100,
                                      n_representatives=4)
    assert_almost_equal(results.rmsd[0:3], (5.4,  8.6,  9.0), decimal=0.1)
    leaf = results.tree.root.representative(results.rmsd)
    assert leaf['iframe'].y == 345


if __name__ == '__main__':
    pytest.main()
