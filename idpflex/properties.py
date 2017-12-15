from __future__ import print_function, absolute_import

from six.moves import zip
from pdb import set_trace as tr

import functools
import numpy as np
import numbers

def register_as_node_property(cls, nxye):
    r"""Endows a class with the node property protocol.

    The node property assumes the existence of these attributes:
        name: name of the property
        x: property domain
        y: property values
        e: errors of the property values
    This function will endow class cls with these attributes, implemented
    through the python property pattern.
    Names for the corresponding storage attributes must be supplied when
    registering the class.

    Parameters
    ----------
    cls : class type
        The class type
    nxye : tuple (len==4)
        nxye is a four element tuple. Its elements are of the form:
        (property name, 'stores the name of the property'),
        (domain_storage_attribute_name, description of the domain),
        (values_storage_attribute_name, description of the values),
        (errors_storage_attribute_name, description of the errors)
        Example:
            (('name', 'stores the name of the property'),
             ('qvalues', 'Momentum transfer values'),
             ('profile', 'Profile intensities'),
             ('errors', 'Intensity errors'))
    """
    def property_item(attribute_name, docstring):
        r"""Factory of the node property items 'name', 'x', 'y', and 'e'
        
        Parameters
        ----------
        attribute_name :  str
            name of the storage attribute holding the info for the 
            respective node property item.
        docstring : str
            description of the storage attribute
        Returns
        -------
        A node-property item 
        """
        def getter(instance):
            return instance.__dict__[attribute_name]

        def setter(instance, value):
            instance.__dict__[attribute_name] = value

        return property(getter, setter, docstring)

    # Endow the class with properties name, x, y, and e
    for (prop, storage) in zip(('name', 'x', 'y', 'e'), nxye):
        setattr(cls, prop, property_item(*storage))

    return cls


def decorate_as_node_property(xye):
    r"""Decorator that endows a class with the node property protocol """
    def decorate(cls):
        return register_as_node_property(cls, xye)
    return decorate


class ScalarProperty(object):
    r"""Implementation of a node property for a number plus an error

    Instance have name, x, y, and e attributes, so they follow the
    property node protocol.
    """

    def __init__(self, name=None, y=0.0, x=0.0, e=0.0):
        self.name = name
        self.x = x
        self.e = e
        self.y = y

    def setScalar(self, y):
        if not isinstance(y, numbers.Real):
            raise TypeError("y must be a non-complex number")
        self.y = y


@decorate_as_node_property((('name', 'name of the profile'),
                            ('qvalues', 'Momentum transfer values'),
                            ('profile', 'Profile intensities'),
                            ('errors', 'Intensity errors')))
class ProfileProperty(object):
    r"""Implementation of a node property valid for SANS or X-Ray data"""

    def __init__(self, name=None, qvalues=None, profile=None, errors=None):
        """
        :param name: Property name. We could have more than one sans profile
        :param profile: Intensity values
        :param qvalues: Momentun transfer domain
        """
        self.name = name
        self.qvalues = qvalues
        self.profile = profile
        self.errors = errors


class SansLoaderMixin(object):
    r"""Mixin class providing a set of methods to load SANS data into a
    profile property"""

    def from_sassena(self, handle, profile_key='fqt', index=0):
        """
        Load SANS profile from sassena output. It is assumed that Q-values are
        stored under item 'qvalues' and listed in the X component, like this:
          q1 0.0 0.0
          q2 0.0 0.0
          ...
        :param handle: h5py reading handle to HDF5 file
        profile_key: item key where profiles are stored in the HDF5 file
        :param index: profile index, if data contains more than one profile
        """
        q = handle['qvectors'][:, 0]  # q values listed in the X component
        i = handle[profile_key][:, index][:, 0]  # profile
        # q values may be unordered
        sorting_order = np.argsort(q)
        q = q[sorting_order]
        i = i[sorting_order]
        self.qvalues = np.array(q, dtype=np.float)
        self.profile = np.array(i, dtype=np.float)
        self.errors = np.zeros(len(q), dtype=np.float)


class SansProperty(ProfileProperty, SansLoaderMixin):
    r"""Implementation of a node property for SANS data"""
    def __init__(self, *args, **kwargs):
        ProfileProperty.__init__(self, *args, **kwargs)
        if self.name is None:
            self.name = 'sans'  # Default name


class SaxsLoaderMixin(object):
    r"""Mixin class providing a set of methods to load X-ray data into a
    profile property"""

    def from_crysol_int(self, file_name):
        r"""Load profile from a crysol *.int file output.
        Expected file format:
        1. First line is a comment
        2. Subsequent rows have five columns:
           col1: momentum transfer
           col2: profile of the protein in solution
           col3: profile of the protein in vacuum
           col4: profile of the solvent
           col5: profile from the hydration shell

        Parameters
        ----------
        file_name : str
            File path
        """
        contents = np.loadtxt(file_name, skiprows=1, usecols=(0, 1))
        self.qvalues = contents[:, 0]
        self.profile = contents[:, 1]
        self.errors = np.zeros(len(self.qvalues), dtype=float)

    def from_crysol_fit(self, file_name):
        r"""Load profile from a crysol *.fit file output.
        Expected file format:
        1. First line is a comment
        2. Subsequent rows have four columns:
           col1: momentum transfer
           col2: experimental data
           col3: residuals
           col4: fitted profile

        Parameters
        ----------
        file_name : str
            File path
        """
        contents = np.loadtxt(file_name, skiprows=1, usecols=(0, 3))
        self.qvalues = contents[:, 0]
        self.profile = contents[:, 1]
        self.errors = np.zeros(len(self.qvalues), dtype=float)

    def from_ascii(self, file_name):
        r"""Load profile from an ascii file.
        Expected file format:
        Rows have three items separated by a blank space:
           col1: momentum transfer
           col2: profile
           col3: errors of the profile

        Parameters
        ----------
        file_name : str
            File path
        """
        contents = np.loadtxt(file_name, skiprows=0, usecols=(0, 1, 2))
        self.qvalues = contents[:, 0]
        self.profile = contents[:, 1]
        self.errors = contents[:, 2]

    def to_ascii(self, file_name):
        r"""Save profile as a three-column ascii file
        Rows have three items separated by a blank space:
           col1: momentum transfer
           col2: profile
           col3: errors of the profile"""
        xye = np.array([list(self.x), list(self.y), list(self.e)])
        np.savetxt(file_name, xye.transpose(),
                   header='Momentum-transfer Profile Profile-errors')


class SaxsProperty(ProfileProperty, SaxsLoaderMixin):
    r"""Implementation of a node property for SANS data"""
    def __init__(self, *args, **kwargs):
        ProfileProperty.__init__(self, *args, **kwargs)
        if self.name is None:
            self.name = 'saxs'   # Default name


def propagator_weighted_sum(values, node_tree,
                            weights=lambda left_node, right_node: (1.0, 1.0)):
    r"""Calculate a property of each node as the sum of its siblings' property
    values. Propagation applies only to non-leaf nodes.

    Parameters
    ----------
    values: list 
        List of property values (of same type), one item for each leaf node.
    node_tree: cnextend.Tree.
        Tree of nodes, with nodes of type ClusterNodeX.
    weights: tuple
        Callable of two arguments (left-node and right-node) returning 
        a tuple of left and right weights. Default callable returns (1.0, 1.0)
        always.
    """
    # Insert a property for each leaf
    if len(values) != node_tree.nleafs:
        msg = "len(values)={} but there are {} leafs".format(len(values),
                                                             node_tree.nleafs)
        raise ValueError(msg)
    for i, leaf in enumerate(node_tree.leafs):
        leaf.add_property(values[i])
    PropertyClass = values[0].__class__  # type of the property
    name = values[0].name  # name of the property
    # Propagate up the tree nodes
    for node in node_tree._nodes[node_tree.nleafs:]:
        prop = PropertyClass()
        prop.name = name
        left_prop = node.left[name]
        right_prop = node.right[name]
        w = weights(node.left, node.right)
        prop.x = left_prop.x
        prop.y = w[0] * left_prop.y + w[1] * right_prop.y
        if left_prop.e is not None and right_prop.e is not None:
            prop.e = np.sqrt(w[0] * left_prop.e**2 + w[1] * right_prop.e**2)
        else:
            prop.e = None
        node.add_property(prop)


def weights_by_size(left_node, right_node):
    r"""Calculate the relative size of two nodes
    
    Parameters
    ----------
    left_node : ClusterNodeX
    right_node : ClusterNodeX

    Returns
    -------
    Tuple of weights representing the relative populate of two nodes

    """
    w = float(left_node.count) / (left_node.count + right_node.count)
    return w, 1-w

propagator_size_weighted_sum = functools.partial(propagator_weighted_sum,
                                                 weights=weights_by_size)
propagator_size_weighted_sum.__name__ = 'propagator_size_weighted_sum'
propagator_size_weighted_sum.__doc__ = r"""Calculate a property of the node
as the sum of its siblings' property values, weighted by the relative cluster
sizes of the siblings

Parameters
----------
values: list 
    List of property values (of same type), one item for each leaf node.
node_tree: cnextend.Tree.
    Tree of nodes, with nodes of type ClusterNodeX.
"""