from __future__ import print_function, absolute_import

import sys

import pickle
from scipy.cluster import hierarchy
from tqdm import tqdm
from collections import namedtuple

from idpflex.distances import (rmsd_matrix, extract_coordinates)
from idpflex.cnextend import Tree
from idpflex.properties import ScalarProperty


class ClusterTrove(namedtuple('ClusterTrove', 'idx rmsd tree')):
    r"""A namedtuple with a `keys()` method for easy access of
    fields, described below under header `Parameters`

    Parameters
    ----------
    idx : :class:`list`
        Frame indexes for the representative structures (indexes start at zero)
    rmsd : :class:`~numpy:numpy.ndarray`
        Root mean square devaition (RMDS) matrix between representative
        structures.
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


def cluster_trajectory(a_universe, selection='not name H*',
                       segment_length=1000, n_representatives=1000):
    r"""Cluster a set of representative structures

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

    Returns
    -------
    :class:`~idpflex.cluster.ClusterTrove`
        clustering results for the representatives
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

    # Cluster the representative structures
    xyz = extract_coordinates(a_universe, group, rep_ifr)
    rmsd = rmsd_matrix(xyz, condensed=True)
    tree = Tree(z=hierarchy.linkage(rmsd, method='complete'))
    for ileaf, leaf in enumerate(tree.leafs):
        leaf.add_property(ScalarProperty(name='iframe', y=rep_ifr[ileaf]))

    return ClusterTrove(rep_ifr, rmsd, tree)


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
