from __future__ import print_function, absolute_import

import sys

import numpy as np
import pickle

import scipy
from scipy.spatial.distance import squareform
from scipy.stats import zscore
from scipy.cluster import hierarchy
from tqdm import tqdm
from collections import namedtuple

from idpflex.distances import (rmsd_matrix, extract_coordinates)
from idpflex.cnextend import Tree
from idpflex.properties import ScalarProperty, propagator_size_weighted_sum

class ClusterTrove(namedtuple('ClusterTrove', 'idx rmsd tree')):
    r"""A namedtuple with a `keys()` method for easy access of
    fields, described below under header `Parameters`

    Parameters
    ----------
    idx : :class:`list`
        Frame indexes for the representative structures (indexes start at zero)
    rmsd : :class:`~numpy:numpy.ndarray`
        distance matrix between representative structures.
    tree : :class:`~idpflex.cnextend.Tree`
        Clustering of representative structures. Leaf nodes contain property
        `iframe` containing the frame index of the corresponding representative
        structure.
    """

    def keys(self):
        r"""Return the list of field names"""
        return self._fields

    def save(self, filename):
        r"""Serialize the cluster trove and save to file

        Parameters
        ----------
        filename: str
            File name
        """
        with open(filename, 'wb') as outfile:
            pickle.dump(self, outfile)


def trajectory_centroids(a_universe, selection='not name H*',
                         segment_length=1000, n_representatives=1000):
    r"""Cluster a trajectory into a set of representative structures using
    structural similarity (RMSD)

    The simulated trajectory is divided into segments, and hierarchical
    clustering is performed on each segment to yield a limited number of
    representative structures.

    Frame indexes from each segment are collected as cluster representatives.

    Parameters
    ----------
    a_universe : :class:`~MDAnalysis.core.universe.Universe`
        Topology and trajectory.
    selection : str
        atoms for which to calculate RMSD
    segment_length: int
        divide trajectory into chunks of this length
    n_representatives : int
        Target total number of representative structures. The final number
        may be close but not equal to the target number.

    Returns
    -------
    list
        Frame indexes of representative structures (centroids)
    """
    group = a_universe.select_atoms(selection)
    # Fragmentation of the trajectory
    n_frame = len(a_universe.trajectory)
    n_segments = int(n_frame / segment_length)
    nc = max(1, int(n_representatives / n_segments))  # clusters per segment
    rep_ifr = list()  # frame indexes of representative structures
    info = """Clustering the trajectory:
Creating {} representatives by partitioning {} frames into {} segments
and retrieving {} representatives from each segment.
    """.format(nc * n_segments, n_frame, n_segments, nc)
    sys.stdout.write(info)
    sys.stdout.flush()

    # Hierarchical clustering on each trajectory fragment
    for i_segment in tqdm(range(n_segments)):
        indexes = range(i_segment * segment_length,
                        (i_segment + 1) * segment_length)
        xyz = extract_coordinates(a_universe, group, indexes)
        rmsd = rmsd_matrix(xyz, condensed=True)
        z = hierarchy.linkage(rmsd, method='complete')
        for node in Tree(z=z).nodes_at_depth(nc-1):
            # Find the frame of each representative structure
            i_frame = i_segment * segment_length + node.representative(rmsd).id
            rep_ifr.append(i_frame)
    rep_ifr.sort()
    return rep_ifr


def cluster_with_properties(a_universe, pcls, p_names=None,
                            selection='not name H*', segment_length=1000,
                            n_representatives=1000):
    r"""Cluster a set of representative structures by structural similarity
    (RMSD) and by a set of properties

    The simulated trajectory is divided into segments, and hierarchical
    clustering is performed on each segment to yield a limited number of
    representative structures (the centroids). Properties are calculated
    for each centroid, thus each centroid is described by a property
    vector. The dimensionality of the vector is just the number of
    properties. Distances between centroids are calculated
    as the Euclidean distance between their respective vector properties.

    The calculated properties are stored in the leafs of the clustering tree
    and the propagated up to the tree's root.

    Parameters
    ----------
    a_universe : :class:`~MDAnalysis.core.universe.Universe`
        Topology and trajectory.
    pcls : list
        Property classes, such as :class:`~idpflex.properties.Asphericity`
        of :class:`~idpflex.properties.SaSa`
    p_names : list
        Names for each property. If None, then property default names are used
    selection : str
        atoms for which to calculate RMSD
    segment_length: int
        divide trajectory into chunks of this length
    n_representatives : int
        Target total number of representative structures. The final number
        may be close but not equal to the target number.

    Returns
    -------
    :class:`~idpflex.cluster.ClusterTrove`
        clustering results for the representatives
    """
    rep_ifr = trajectory_centroids(a_universe, selection=selection,
                                   segment_length=segment_length,
                                   n_representatives=n_representatives)
    n_centroids = len(rep_ifr)  # can be different than n_representatives

    # Create names if not passed
    if p_names is None:
        p_names = [Property.default_name for Property in pcls]

    # Calculate properties for each centroid
    l_prop = list()
    for p_name, Pcl in zip(p_names, pcls):
        l_prop.append([Pcl(name=p_name).from_universe(a_universe, index=i)
                       for i in rep_ifr])

    # Calculate distances between pair of centroids
    xyz = np.zeros((len(pcls), n_centroids))
    for i_prop, prop in enumerate(l_prop):
        xyz[i_prop] = [p.y for p in prop]
    # zero mean and unity variance for each property
    xyz = np.transpose(zscore(xyz,axis=1))
    distance_matrix = squareform(scipy.spatial.distance_matrix(xyz, xyz))

    # Cluster the representative structures
    tree = Tree(z=hierarchy.linkage(distance_matrix, method='complete'))
    for i_leaf, leaf in enumerate(tree.leafs):
        leaf.add_property(ScalarProperty(name='iframe', y=rep_ifr[i_leaf]))

    # Propagate the properties up the tree
    [propagator_size_weighted_sum(prop, tree) for prop in l_prop]

    return ClusterTrove(rep_ifr, distance_matrix, tree)


def cluster_trajectory(a_universe, selection='not name H*',
                       segment_length=1000, n_representatives=1000):
    r"""Cluster a set of representative structures by structural similarity
    (RMSD)

    The simulated trajectory is divided into segments, and hierarchical
    clustering is performed on each segment to yield a limited number of
    representative structures. These are then clustered into the final
    hiearchical tree.

    Frame indexes from each segment are collected as cluster representatives.

    Parameters
    ----------
    a_universe : :class:`~MDAnalysis.core.universe.Universe`
        Topology and trajectory.
    selection : str
        atoms for which to calculate RMSD
    segment_length: int
        divide trajectory into chunks of this length
    n_representatives : int
        Target total number of representative structures. The final number
        may be close but not equal to the target number.
    distance_matrix: :class:`~numpy:numpy.ndarray`

    Returns
    -------
    :class:`~idpflex.cluster.ClusterTrove`
        clustering results for the representatives
    """
    rep_ifr = trajectory_centroids(a_universe, selection=selection,
                                   segment_length=segment_length,
                                   n_representatives=n_representatives)

    group = a_universe.select_atoms(selection)
    xyz = extract_coordinates(a_universe, group, rep_ifr)
    distance_matrix = rmsd_matrix(xyz, condensed=True)

    # Cluster the representative structures
    tree = Tree(z=hierarchy.linkage(distance_matrix, method='complete'))
    for i_leaf, leaf in enumerate(tree.leafs):
        leaf.add_property(ScalarProperty(name='iframe', y=rep_ifr[i_leaf]))

    return ClusterTrove(rep_ifr, distance_matrix, tree)


def load_cluster_trove(filename):
    r"""Load a previously saved ClusterTrove instance

    Parameters
    ----------
    filename: str
        File name containing the serialized tree

    Returns
    -------
    :class:`~idpflex.cluster.ClusterTrove`
        Cluster trove instance stored in file
    """
    with open(filename, 'rb') as infile:
        t = pickle.load(infile)
    return t
