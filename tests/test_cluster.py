from __future__ import print_function, absolute_import

import pytest
from numpy.testing import assert_almost_equal

from idpflex import cluster as idpc
from idpflex import properties as iprp


def test_cluster_trajectory(trajectory_benchmark):
    results = idpc.cluster_trajectory(trajectory_benchmark,
                                      selection='name CA',
                                      segment_length=100,
                                      n_representatives=4)
    assert_almost_equal(results.rmsd[0:3], (5.4,  8.6,  9.0), decimal=0.1)
    leaf = results.tree.root.representative(results.rmsd)
    assert leaf['iframe'].y == 345


def test_cluster_with_properties(trajectory_benchmark):
    pcls = [iprp.SaSa, iprp.RadiusOfGyration, iprp.EndToEnd, iprp.Asphericity]
    results = idpc.cluster_with_properties(trajectory_benchmark, pcls,
                                           selection='name CA',
                                           segment_length=100,
                                           n_representatives=9)
    default_names = set(['sasa', 'rg', 'end_to_end', 'asphericity'])
    assert default_names == set(results.tree.root._properties.keys())
    assert_almost_equal(results.rmsd[0:3], (3.4, 1.9, 3.0), decimal=0.1)
    leaf = results.tree.root.representative(results.rmsd)
    assert leaf['iframe'].y == 942


if __name__ == '__main__':
    pytest.main()
