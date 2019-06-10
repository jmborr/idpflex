import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import zscore

import idpflex.distances as idpd


def test_extract_coordinates(trajectory_benchmark):
    group = trajectory_benchmark.select_atoms('resnum 2 and name CA')
    indexes = (0, -1)  # first and last frame in trajectory
    xyz = idpd.extract_coordinates(trajectory_benchmark, group, indexes)
    reference = np.array([[[53.8, 54.6, 38.8]], [[48.3, 46.6, 43.1]]])
    assert_allclose(xyz, reference, atol=0.1)


def test_rmsd_matrix(trajectory_benchmark):
    group = trajectory_benchmark.select_atoms('name CA')
    indexes = (0, 2, -1)  # first, third, and last frames
    xyz = idpd.extract_coordinates(trajectory_benchmark, group, indexes)
    rmsd = idpd.rmsd_matrix(xyz, condensed=True)
    reference = np.array([8.73, 8.92, 8.57])
    assert_allclose(rmsd, reference, atol=0.01)


def test_distance_submatrix():
    a = np.arange(4)
    dist_mat = np.square(a - a[:, np.newaxis])
    submatrix = idpd.distance_submatrix(dist_mat, [1, 3])
    reference = np.array([[0, 4], [4, 0]])
    assert_allclose(submatrix, reference, atol=0.001)


def test_generate_distance_matrix():
    atol = 1e-5
    y1 = np.array([0, 1, 2])
    y2 = y1.copy()
    y3 = -5*y1.copy() + 1
    # Check unweighted distances
    dm = idpd.generate_distance_matrix([y1, y2, y3])
    assert abs(dm[0]) <= atol
    scores = zscore([y1, y2, y3])
    assert abs(
        dm[1] - np.sqrt(np.sum(np.power(scores[0] - scores[2], 2)))) <= atol
    # Check weighted distances
    ws = [(1, 0, 0) for _ in (y1, y2, y3)]
    dm_weighted = idpd.generate_distance_matrix([y1, y2, y3], weights=ws)
    scores_weighted = zscore([y1, y2, y3]) * np.array(ws)
    assert abs(dm_weighted[1]
               - np.sqrt(np.sum(np.power(scores_weighted[0]
                                         - scores_weighted[2], 2)))) <= atol


if __name__ == '__main__':
    pytest.main()
