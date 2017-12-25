from __future__ import print_function, absolute_import

import pytest
import numpy as np
from numpy.testing import assert_allclose

import idpflex.distances as idpd


def test_extract_coordinates(trajectory_benchmark):
    group = trajectory_benchmark.select_atoms('resnum 2 and name CA')
    indexes = (1, 3)  # second and fourth frame in trajectory
    xyz = idpd.extract_coordinates(sim_benchmark, group, indexes)
    reference = np.array([[[39.239, 56.403, 42.303]],
                          [[39.909, 55.820, 42.711]]])
    assert_allclose(xyz, reference, atol=0.001)


def test_rmsd_matrix(trajectory_benchmark):
    group = trajectory_benchmark.select_atoms('resnum 2 and name CA')
    xyz = idpd.extract_coordinates(sim_benchmark, group, range(3))
    rmsd = idpd.rmsd_matrix(xyz, condensed=True)
    reference = np.array([])
    assert_allclose(rmsd, reference, atol=0.001)

def test_distance_submatrix():
    dist_mat = np.arange(16).reshape(4, 4)
    x = dist_mat[[1, 2]][:, [1, 2]]
    submatrix = idpd.distance_submatrix(dist_mat, (1, 2))
    reference = np.array([[5, 6], [9, 10]])
    assert_allclose(submatrix, reference, atol=0.001)

if __name__ == '__main__':
    pytest.main()
