from __future__ import print_function, absolute_import

import pickle
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import numpy as np
from past.builtins import xrange

from idpflex.distances import distance_submatrix


class ClusterNodeX(hierarchy.ClusterNode):
    r"""Extension of :py:class:`~scipy:scipy.cluster.hierarchy.ClusterNode`
    to accommodate a parent reference and a protected dictionary
    of properties.
    """
    def __init__(self, *args, **kwargs):
        # Using of *super* is unfeasible because *ClusterNode* does not
        # inherit from *object*.
        # super(ClusterNodeX, self).__init__(*args, **kwargs)
        hierarchy.ClusterNode.__init__(self, *args, **kwargs)
        self.parent = None  # parent node
        self._tree = None
        self._properties = dict()

    def __getitem__(self, name):
        r"""Fetch a property from protected `_properties` dictionary.

        Parameters
        ----------
        name : str
            name of the property

        Returns
        -------
        property object, or `None` if no property is found with *name*
        """
        if name in self._properties:
            return self._properties[name]
        else:
            return None

    @property
    def tree(self):
        r"""Tree object owning the node

        Returns
        -------
        :class:`~idpflex.cnextend.Tree`
        """
        return self._tree

    @property
    def leafs(self):
        r"""Find the leaf nodes under this cluster node.

        Returns
        -------
        :py:class:`list`
            node leafs ordered by increasing ID
        """
        return sorted(self.pre_order(lambda x: x), key=lambda x: x.id)

    @property
    def leaf_ids(self):
        r"""ID's of the leafs under the tree, ordered by increasing ID.

        Returns
        -------
        :class:`list`
        """
        return list(leaf.id for leaf in self.leafs)

    def add_property(self, a_property):
        r"""Insert or update a property in the set of properties

        Parameters
        ----------
        a_property : :class:`~idpflex.properties.ProfileProperty`
            a property instance
        """
        self._properties[a_property.name] = a_property
        a_property.node = self

    def distance_submatrix(self, dist_mat):
        r"""Extract matrix of distances between leafs under the node.

        Parameters
        ----------
        dist_mat: numpy.ndarray
            Distance matrix (square or in condensed form) among all N leaves
            of the tree to which the node belongs to. The row indexes of
            `dist_mat` must correspond to the node IDs of the leaves.

        Returns
        -------
        :class:`~numpy:numpy.ndarray`
            square distance matrix MxM between the M leafs under the node
        """
        return distance_submatrix(dist_mat, self.leaf_ids)

    def representative(self, dist_mat, similarity=np.mean):
        r"""Find leaf under node that is most similar to all other leaves
        under the node

        Find the leaf that minimizes the similarity between itself and all
        the other leaves under the node. For instance, the average of all
        distances between one leaf and all the other leaves results in
        a similarity scalar for the leaf.

        Parameters
        ----------
        dist_mat: :class:`~numpy:numpy.ndarray`
            condensed or square distance matrix MxM or NxN among all N leaves
            in the tree or among all M leaves under the node.
            If dealing with the distance matrix among all leaves in the
            tree, self.distance_submatrix is first applied.
        similarity: function object
            reduction operation on a the list of distances between one leaf
            and the other (M-1) leaves.

        Returns
        -------
        :class:`~idpflex.cnextend.ClusterNodeX`
            representative leaf node
        """
        if len(self.leafs) == 1:
            return self
        # Find out if matrix is condensed
        square_dist_mat = dist_mat
        if dist_mat.ndim == 1:
            square_dist_mat = squareform(dist_mat)
        # Find out if matrix for all leafs in the tree or all leafs in the node
        if len(square_dist_mat) == len(self.leafs):
            submatrix = square_dist_mat
        elif len(square_dist_mat) == self.tree.nleafs:
            submatrix = self.distance_submatrix(square_dist_mat)
        else:
            raise ValueError('Incorrect distance matrix size')
        return self.leafs[similarity(submatrix, axis=0).argmin()]


class Tree(object):
    r"""Hierarchical binary tree.

    Parameters
    ----------
    z : :class:`~numpy:numpy.ndarray`
        linkage matrix from which to create the tree. See
        :func:`~scipy:scipy.cluster.hierarchy.linkage`
    """

    def __init__(self, z=None):
        self.root = None  # topmost node
        self.z = z
        # list of nodes, position in the list is node ID. Last is the root node
        self._nodes = list()  # list of nodes, starting from the leaf nodes
        self.nleafs = 0  # a leaf is a node at the bottom of the tree
        if self.z is not None:
            self.from_linkage_matrix(self.z)

    def __iter__(self):
        r"""Navigate the tree in order of decreasing node ID, starting from
        root node
        """
        return (node for node in self._nodes[::-1])

    def __getitem__(self, index):
        r"""Fetch items from *_nodes* attribute

        Returns
        -------
        :class:`~idpflex.cnextend.ClusterNodeX`
            node instance
        """
        return self._nodes.__getitem__(index)

    def __len__(self):
        """
        Returns
        -------
        int
            Number of nodes in the tree
        """
        return len(self._nodes)

    @property
    def leafs(self):
        r"""

        Returns
        -------
        :py:class:`list`
            leaf nodes ordered by increasing ID
        """
        return self._nodes[:self.nleafs]

    def from_linkage_matrix(self, z, node_class=ClusterNodeX):
        """Refactored :func:`~scipy:scipy.cluster.hierarchy.to_tree`
        converts a hierarchical clustering encoded in matrix `z`
        (by linkage) into a convenient tree object.

        Each *node_class* instance has a *left*, *right*, *dist*, *id*,
        and *count* attribute. The *left* and *right* attributes point
        to *node_class* instances that were combined to generate the
        cluster. If both are *None* then *node_class* is a leaf node,
        its count must be 1, and its distance is meaningless but set to 0.

        Parameters
        ----------
        z : :class:`~numpy:numpy.ndarray`
            linkage matrix. See :func:`~scipy:scipy.cluster.hierarchy.linkage`
        node_class : :class:`~idpflex.cnextend.ClusterNodeX`
            the type of nodes composing the tree. Now supports
            :class:`~idpflex.cnextend.ClusterNodeX`
            and parent class
            :class:`~scipy:scipy.cluster.hierarchy.ClusterNode`
        """
        z = np.asarray(z, order='c')
        hierarchy.is_valid_linkage(z, throw=True, name='z')
        # Number of original objects is equal to the number of rows minus 1.
        n = z.shape[0] + 1
        # Create a list full of None's to store the node objects
        d = [None] * (n * 2 - 1)
        # Create the nodes corresponding to the n original objects.
        for i in xrange(0, n):
            d[i] = node_class(i)
        nd = None
        for i in xrange(0, n - 1):
            fi = int(z[i, 0])
            fj = int(z[i, 1])
            if fi > i + n:
                raise ValueError(('Corrupt matrix z. Index to derivative '
                                  'cluster is used before it is formed. See '
                                  'row %d, column 0') % fi)
            if fj > i + n:
                raise ValueError(('Corrupt matrix z. Index to derivative '
                                  'cluster is used before it is formed. See '
                                  'row %d, column 1') % fj)
            nd = node_class(i + n, left=d[fi], right=d[fj], dist=z[i, 2])
            if hasattr(nd, 'parent'):  # True for ClusterNodeX objects
                d[fi].parent = nd
                d[fj].parent = nd
            if hasattr(nd, '_tree'):  # True for ClusterNodeX objects
                nd._tree = self
            if z[i, 3] != nd.count:
                message = 'Corrupt matrix z. The count z[{}, 3] is incorrect'
                raise ValueError(message.format(i))
            d[n + i] = nd
        self.nleafs = n
        self.root = nd
        self._nodes = d

    def nodes_above_depth(self, depth=0):
        r"""Nodes at or above depth from the root node

        Parameters
        ----------
        depth : int
            Depth level starting from the root level (depth=0)

        Returns
        -------
        :py:class:`list`
            List of nodes ordered by increasing ID. Last one is the root node
        """
        nodes = [self.root, ]
        for d in range(1, 1+depth):
            cmax = nodes[-d]
            nodes.extend((cmax.left, cmax.right))
            nodes.sort(key=lambda cluster: cluster.id)
        return nodes

    def nodes_at_depth(self, depth=0):
        r"""Nodes at a given depth from the root node

        Parameters
        ----------
        depth : int
            Depth level starting from the root level (depth=0)

        Returns
        -------
        :py:class:`list`
            List of nodes corresponding to that particular level
        """
        if depth == 0:
            nodes = [self.root]
        else:
            nodes = self.nodes_above_depth(depth)[:-depth]
        return nodes

    def save(self, filename):
        r"""Serialize the tree and save to file

        Parameters
        ----------
        filename: str
            File name
        """
        with open(filename, 'wb') as outfile:
            pickle.dump(self, outfile)


def load_tree(filename):
    r"""Load a previously saved tree

    Parameters
    ----------
    filename: str
        File name containing the serialized tree

    Returns
    -------
    :class:`~idpflex.cnextend.Tree`
        Tree instance stored in file
    """
    with open(filename, 'rb') as infile:
        t = pickle.load(infile)
    return t
