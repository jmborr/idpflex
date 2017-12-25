from __future__ import print_function, absolute_import

import numpy as np
from scipy.spatial.distance import squareform
from MDAnalysis.analysis.rms import rmsd as find_rmsd


def extract_coordinates(a_universe, group, indexes=None):
    r"""XYZ coordinates for an atom selection for a subset of frames

    Parameters
    ----------
    a_universe :  MDAnalysis.core.universe.Universe
        Topology and trajectory.
    group : MDAnalysis.core.groups.AtomGroup
        Atom selection.
    indexes: sequence of int
        sequence of frame indexes
    Returns
    -------
    numpy.ndarray
        XYZ coordinates shape=(M, N, 3) with M number of indexes and
        N number of atoms in group.
    """
    xyz = list()
    if indexes is None:
        indexes = range(a_universe.trajectory.nframe)
    for i_frame in indexes:
        a_universe.trajectory[i_frame]  # go to this frame
        xyz.append(group.positions.copy())  # extract frame coords
    return np.asarray(xyz)


def rmsd_matrix(xyz, condensed=False):
    r"""RMSD matrix between coordinate frames.

    Parameters
    ----------
    xyz : numpy.ndarray
        Bare coordinates shape=(N, M, 3) with N: number of frames,
        M: number of atoms
    condensed: bool
        Flag return matrix as square or condensed
    Returns
    -------
    numpy.ndarray
        Square NxN or 1d N*(N+1)/2 RMSD matrix
    """
    n = len(xyz)
    rmsd = np.zeros(n * n).reshape(n, n)
    # To-Do: make parallel
    for i in range(0, n-1):
        ri = xyz[i]
        for j in range(i+1, n):
            rmsd[i][j] = find_rmsd(ri, xyz[j])
            rmsd[j][i] = rmsd[i][j]
    if condensed is True:
        rmsd = squareform(rmsd)
    return rmsd


def distance_submatrix(dist_mat, indexes):
    r"""Extract matrix of distances for a subset of indexes

    Parameters
    ----------
    dist_mat: numpy.ndarray
        NxN distance matrix
    indexes: sequence of int
        sequence of indexes from which a submatrix is extracted.
    Returns
    -------
    numpy.ndarray
    """
    return dist_mat[indexes][:, indexes]
