from scipy.cluster import hierarchy
from tqdm import tqdm
from idpflex.utils import returns_tuple
from idpflex.distances import (rmsd_matrix, extract_coordinates)
from idpflex.cnextend import Tree
from idpflex.properties import ScalarProperty

info = """Results from clustering
rmsd : :class:`~numpy:numpy.ndarray`
    RMDS matrix between representative structures
tree : :class:`~idpflex.cnextend.Tree`
    Clustering of representative structures. Leaf nodes contain property
    `iframe` containing the frame index of the corresponding representative
    structure.
"""
ClusterTrove = returns_tuple('ClusterTrove', 'rmsd tree',
                             doc=info)


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
    :class:`~idpflex.cluster.ClusterTrove`
        RMSD condensed matrix and hierarchical :class:`~idpflex.cnextend.Tree`
    """
    group = a_universe.select_atoms(selection)

    # Fragmentation of the trajectory
    n_frame = len(a_universe.trajectory)
    n_segments = int(n_frame / segment_length)
    nc = max(1, int(n_representatives / n_segments))  # number of clusters per segment
    rep_ifr = list()  # frame indexes of representative structures

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

    # Cluster the representative structures
    xyz = extract_coordinates(a_universe, group, rep_ifr)
    rmsd = rmsd_matrix(xyz, condensed=True)
    tree = Tree(z=hierarchy.linkage(rmsd, method='complete'))
    for ileaf, leaf in enumerate(tree.leafs):
        leaf.add_property(ScalarProperty(name='iframe', y=rep_ifr[ileaf]))

    return ClusterTrove(rmsd, tree)
