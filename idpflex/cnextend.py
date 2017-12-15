from __future__ import print_function, absolute_import

from scipy.cluster import hierarchy
import numpy as np
from past.builtins import xrange

class ClusterNodeX(hierarchy.ClusterNode):
    """
    Extension of hierarchy.ClusterNode to accomodate a parent reference and a dictionary of
    properties associated to a specific node, like SANS profile
    """
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.parent = None
        self._properties = dict()

    def __getitem__(self, name):
        """
        property names are treated as keys of self
        :param name: name of the attribute or of the property
        :return: property or attribute of self.
        """
        if name in self._properties:
            return self._properties[name]
        else:
            return None

    @property
    def leafs(self):
        """
        Find the leaf nodes under this cluster node
        :return: list of node leafs ordered by increasing ID 
        """
        return sorted(self.pre_order(lambda x: x), key=lambda x: x.id)

    def add_property(self, a_property):
        """
        Insert or update a property in the set of properties
        :param a_property: property instance
        """
        self._properties[a_property.name] = a_property


class Tree(object):
    """
    Hierarchical binary tree
    """

    def __init__(self, Z=None):
        self.root = None  # topmost node
        self.Z = Z  # linkage matrix from which to create the tree
        # list of nodes, position in the list is node ID. Last in the list is the root node.
        self._nodes = list()  # list of nodes, starting from the leaf nodes
        self.nleafs = 0  # a leaf is a node at the bottom of the tree
        if self.Z is not None:
            self.from_linkage_matrix(self.Z)

    def __iter__(self):
        """Navigate the tree in order of decreasing node ID, starting from root node"""
        return (node for node in self._nodes[::-1])

    def __getitem__(self, index):
        """Return items from _nodes attribute"""
        return self._nodes.__getitem__(index)

    def __len__(self):
        """
        :return: Number of nodes in the tree 
        """
        return len(self._nodes)

    @property
    def leafs(self):
        return self._nodes[:self.nleafs]

    def from_linkage_matrix(self, Z, node_class=ClusterNodeX):
        """
        Refactored scipy.cluster.hierarchy.to_tree
        Converts a hierarchical clustering encoded in the matrix ``Z`` (by
        linkage) into an easy-to-use tree object.

        The reference r to the root node_class object is returned.

        Each node_class object has a left, right, dist, id, and count
        attribute. The left and right attributes point to node_class objects
        that were combined to generate the cluster. If both are None then
        the node_class object is a leaf node, its count must be 1, and its
        distance is meaningless but set to 0.

        :param Z: (ndarray) The linkage matrix (scipy.cluster.hierarchy.linkage)
        :param node_class the type of nodes composing the tree. Now supports
        ClusterNodeX and parent class scipy.cluster.hierarchy.ClusterNode
        """
        Z = np.asarray(Z, order='c')
        hierarchy.is_valid_linkage(Z, throw=True, name='Z')
        # Number of original objects is equal to the number of rows minus 1.
        n = Z.shape[0] + 1
        # Create a list full of None's to store the node objects
        d = [None] * (n * 2 - 1)
        # Create the nodes corresponding to the n original objects.
        for i in xrange(0, n):
            d[i] = node_class(i)
        nd = None
        for i in xrange(0, n - 1):
            fi = int(Z[i, 0])
            fj = int(Z[i, 1])
            if fi > i + n:
                raise ValueError(('Corrupt matrix Z. Index to derivative cluster '
                                  'is used before it is formed. See row %d, '
                                  'column 0') % fi)
            if fj > i + n:
                raise ValueError(('Corrupt matrix Z. Index to derivative cluster '
                                  'is used before it is formed. See row %d, '
                                  'column 1') % fj)
            nd = node_class(i + n, left=d[fi], right=d[fj], dist=Z[i, 2])
            if hasattr(nd, 'parent'):
                # True for ClusterNodeX objects
                d[fi].parent = nd
                d[fj].parent = nd
            if Z[i, 3] != nd.count:
                raise ValueError(('Corrupt matrix Z. The count Z[%d,3] is '
                                  'incorrect.') % i)
            d[n + i] = nd
        self.nleafs = n
        self.root = nd
        self._nodes = d

    def clusters_above_depth(self, depth=0):
        r"""Clusters nodes at or above depth from the root node
    
        Parameters
        ----------
        depth : int
            Depth level starting from the root level (depth=0)

        Returns
        -------
        clusters : list
            List of nodes ordered by increasing ID. Last one is the root node
        """
        clusters = [self.root, ]
        for d in range(1, 1+depth):
            cmax = clusters[-d]
            clusters.extend((cmax.left, cmax.right))
            clusters.sort(key=lambda cluster: cluster.id)
        return clusters


    def clusters_at_depth(self, depth=0):
        r"""Cluster nodes at a given depth from the root node
    
        Parameters
        ----------
        depth : int
            Depth level starting from the root level (depth=0)

        Returns
        -------
        clusters : list
            List of nodes corresponding to that particular level
        """
        if depth == 0:
            clusters = [self.root]
        else:
            clusters = self.clusters_above_depth(depth)[:-depth]
        return clusters