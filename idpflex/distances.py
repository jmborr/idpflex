import random
from contextlib import closing
from functools import partial
import multiprocessing
import numpy as np
import scipy
from scipy.stats import zscore
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


def _rmsd_rows(i_chunk, coords):
    r"""RMDS values for a set of row indexes

    This function is only used inside rmsd_matrix() but must
    be defined at the module's namespace so that it can be
    pickled and passed to multiprocessing.pool.imap_unordered

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

    # rmsd is a symmetric matrix with zeros in the diagonal. Thus, we only
    # calculate the upper diagonal; Distribute rows of RMSD among cores
    indexes = list(range(0, n-1))  # row indexes
    random.shuffle(indexes)  # to balance load among cores
    n_cores = multiprocessing.cpu_count()
    m = int(n / n_cores)  # number of rows per core
    # One chunk of rows is assigned to one core
    i_chunks = [indexes[i*m: (i+1)*m] for i in range(n_cores)]
    # left over rows assigned to last chunk
    i_chunks[-1].extend(indexes[m * n_cores:])

    rr = partial(_rmsd_rows, coords=xyz)
    # multiprocessing.Pool is a context manager only for python >= 3.3
    with closing(multiprocessing.Pool(processes=n_cores)) as pool:
        for i_chunk, dists in pool.imap_unordered(rr, i_chunks):
            for k, i in enumerate(i_chunk):
                rmsd[i][i+1:] = dists[k]
        pool.terminate()
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


def generate_distance_matrix(feature_vectors, weights=None,
                             func1d=zscore, func1d_args=None,
                             func1d_kwargs=None):
    r"""
    Calculate a distance matrix between measurements, based on
    features of the measurements

    Each measurement is characterized by a set of features, here
    implemented as a vector. Thus, we sample each feature vector-item
    a number of times equal to the number of measurements.

    Distance `d` between two feature vectors `f` and `g` given weights `w`
        `d**2 = sum_i w_i * (f_i - g_i)**2

    Parameters
    ----------
    feature_vectors: list
        List of feature vectors, one vector for each measurement
    weights: list
        List of feature weight vectors, one for each measurement
    func1d: function
        Apply this function to the set of values obtained for each
        feature vector-item. The size of the set is number of
        measurements. The default function transforms the
        values in each feature vector-item set such that the set
        has zero mean and unit standard deviation.
    func1d_args: list
        Positional arguments to func1D
    func1d_kwargs: dict
        Optional arguments for func1D
    Returns
    -------
    numpy.ndarray
        Distance matrix in vector-form distance
    """
    if len(set([len(fv) for fv in feature_vectors])) > 1:
        raise RuntimeError('features vectors have different sizes')
    xyz = np.array(feature_vectors)  # shape=(#vectors, #features)

    # Apply func1D to each feature vector-item set (axis 0 of array xyz)
    fargs = [] if func1d_args is None else list(func1d_args)
    fkwargs = {} if func1d_kwargs is None else dict(func1d_kwargs)
    xyz = np.apply_along_axis(func1d, 0, xyz, *fargs, **fkwargs)
    # weight each feature vector
    if weights is not None:
        if len(weights) != len(feature_vectors):
            raise RuntimeError('Number of weight vectors different than '
                               'number of feature vectors')
        xyz *= np.array(weights)
    return squareform(scipy.spatial.distance_matrix(xyz, xyz))
