from __future__ import print_function, absolute_import

import random
from functools import partial
import multiprocessing
import numpy as np
from scipy.spatial.distance import squareform
from MDAnalysis.analysis.rms import rmsd as find_rmsd


def extract_coordinates(a_universe, group, indexes=None):
    r"""Obtain XYZ coordinates for an atom group and for a subset of frames

    Parameters
    ----------
    a_universe :  :class:`~MDAnalysis.core.universe.Universe`
        Topology and trajectory.
    group : :class:`~MDAnalysis.core.groups.AtomGroup`
        Atom selection.
    indexes: :py:class:`list`
        sequence of frame indexes
    Returns
    -------
    :class:`~numpy:numpy.ndarray`
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


def rmsd_rows(i_chunk, coords):
    r"""RMDS values for a set of row indexes

    Parameters
    ----------
    i_chunk : :py:class:`list`
        list of row indexes
    coords : :class:`~numpy:numpy.ndarray`
        Atomic coordinates

    Returns
    -------
    i_chunk : :py:class:`list`
        Input list of row indexes
    rmsd_chunk : :py:class:`list`
        list of lists where each list item is a list of RMSD values
    """
    n = len(coords)
    rmsd_chunk = list()
    for i in i_chunk:
        ri = coords[i]
        rmsd_values = [find_rmsd(ri, coords[j], superposition=True)
                       for j in range(i+1, n)]
        rmsd_chunk.append(rmsd_values)
    return i_chunk, rmsd_chunk


def rmsd_matrix(xyz, condensed=False):
    r"""RMSD matrix between coordinate frames.

    Parameters
    ----------
    xyz : :class:`~numpy:numpy.ndarray`
        Bare coordinates shape=(N, M, 3) with N: number of frames,
        M: number of atoms
    condensed: bool
        Flag return matrix as square or condensed
    Returns
    -------
    :class:`~numpy:numpy.ndarray`
        Square NxN matrix, or condensed N*(N+1)/2 matrix
    """
    n = len(xyz)
    rmsd = np.zeros(n * n).reshape(n, n)
    # RMSD is a symmetric matrix with zeros in the diagonal. Thus, we only
    # calculate the upper diagonal.
    # Divide all rows of RMSD among available cores
    indexes = list(range(0, n-1))  # row indexes
    random.shuffle(indexes)  # to balance load among cores
    n_cores = multiprocessing.cpu_count()
    m = int(n / n_cores)  # number of rows per core
    # Each chunk of rows is assigned to one core
    i_chunks = [indexes[i*m: (i+1)*m] for i in range(n_cores)]
    # left over rows assigned to last chunk
    i_chunks[-1].extend(indexes[m * n_cores:])

    with multiprocessing.Pool(processes=n_cores) as pool:
        for i_chunk, dists in pool.imap_unordered(partial(rmsd_rows, coords=xyz), i_chunks):
            for k, i in enumerate(i_chunk):
                rmsd[i][i+1:] = dists[k]
    rmsd += rmsd.transpose()

    if condensed is True:
        rmsd = squareform(rmsd)
    return rmsd


def distance_submatrix(dist_mat, indexes):
    r"""Extract matrix of distances for a subset of indexes

    If matrix is in condensed format, then the submatrix is returned in
    condensed format too.

    Parameters
    ----------
    dist_mat: :class:`~numpy:numpy.ndarray`
        NxN distance matrix
    indexes: sequence of int
        sequence of indexes from which a submatrix is extracted.
    Returns
    -------
    :class:`~numpy:numpy.ndarray`
    """
    m = dist_mat
    if dist_mat.ndim == 1:
        m = squareform(dist_mat)
    subm = m[indexes][:, indexes]
    if dist_mat.ndim == 1:
        subm = squareform(subm)
    return subm
