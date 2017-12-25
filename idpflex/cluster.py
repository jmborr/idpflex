from scipy.cluster import hierarchy
import tqdm
from collections import namedtuple
from idpflex.distances import (rmsd_matrix, extract_coordinates)
from idpflex.cnextend import Tree

ClusterTrove_ = namedtuple('ClusterTrove', 'rmsd tree')


class ClusterTrove(ClusterTrove_):
    r"""collect results from clustering"""
    __slots__ = ()  # force inmutable object

    @property
    def fields(self):
        return self._fields


def cluster_trajectory(a_universe, selection='not name H*',
                       segment_length=1000, n_representatives=1000):
    r"""Find frames indexes corresponding to representative structures.

    Simulation trajectory is divided into segments, and hierarchical
    clustering is performed on each segment to yield a number of clusters.
    Frame indexes from each segment are collected as cluster representatives.

    Parameters
    ----------
    a_universe :  MDAnalysis.core.universe.Universe
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
        Frame indexes for cluster representative structures.
    """
    group = a_universe.select_atoms(selection)

    # Fragmentation of the trajectory
    n_frame = len(a_universe.trajectory)
    n_segments = int(n_frame / segment_length)
    nc = int(n_representatives / n_segments)  # number of clusters per segment
    representative_indexes = list()

    # Hierarchical clustering on each trajectory fragment
    for i_segment in tqdm(range(n_segments)):
        indexes = range(i_segment * segment_length,
                        (i_segment + 1) * segment_length)
        xyz = extract_coordinates(a_universe, group, indexes)
        rmsd = rmsd_matrix(xyz, condensed=True)
        Z = hierarchy.linkage(rmsd, method='complete')
        for node in Tree(Z=Z).nodes_at_depth(nc-1):
            # Find the leaf ID for each representative structure
            leaf_ids = list(r.id for r in node.representative(rmsd))
            i_frames = i_segment * segment_length + leaf_ids
            representative_indexes.extend(i_frames)

    # Cluster the representative structures
    xyz = extract_coordinates(a_universe, group, indexes)
    rmsd = rmsd_matrix(xyz, condensed=True)
    tree = Tree(Z=hierarchy.linkage(rmsd, method='complete'))
    return ClusterTrove(rmsd, tree)
