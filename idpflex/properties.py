from __future__ import print_function, absolute_import

import os
from six.moves import zip
import subprocess
import tempfile
import fnmatch
import functools
import numpy as np
import numbers


def register_as_node_property(cls, nxye):
    r"""Endows a class with the node property protocol.

    | The node property assumes the existence of these attributes
    | - *name* name of the property
    | - *x* property domain
    | - *y* property values
    | - *e* errors of the property values

    This function will endow class *cls* with these attributes, implemented
    through the python property pattern.
    Names for the corresponding storage attributes must be supplied when
    registering the class.

    Parameters
    ----------
    cls : class type
        The class type
    nxye : tuple (len==4)
        nxye is a four element tuple. Its elements are in this order:

        (property name, 'stores the name of the property'),
        (domain_storage_attribute_name, description of the domain),
        (values_storage_attribute_name, description of the values),
        (errors_storage_attribute_name, description of the errors)

        Example:

        (('name', 'stores the name of the property'),
        ('qvalues', 'momentum transfer values'),
        ('profile', 'profile intensities'),
        ('errors', 'intensity errors'))
    """
    def property_item(attr_name, docstring):
        r"""Factory of the node property items *name*, *x*, *y*, and *e*

        Parameters
        ----------
        attr_name :  str
            name of the storage attribute holding the info for the
            respective node property item.
        docstring : str
            description of the storage attribute

        Returns
        -------
        :py:class:`property`
            A node-property item
        """
        def getter(instance):
            return instance.__dict__[attr_name]

        def setter(instance, value):
            instance.__dict__[attr_name] = value

        return property(fget=getter,
                        fset=setter,
                        doc='property *{}* : {}'.format(attr_name, docstring))

    # Endow the class with properties name, x, y, and e
    for (prop, storage) in zip(('name', 'x', 'y', 'e'), nxye):
        setattr(cls, prop, property_item(*storage))

    return cls


def decorate_as_node_property(nxye):
    r"""Decorator that endows a class with the node property protocol

    For details, see :func:`~idpflex.properties.register_as_node_property`

    Parameters
    ----------
    nxye : list
        list of (name, description) pairs denoting the property components
    """
    def decorate(cls):
        return register_as_node_property(cls, nxye)
    return decorate


class ScalarProperty(object):
    r"""Implementation of a node property for a number plus an error.

    Instances have *name*, *x*, *y*, and *e* attributes, so they will
    follow the property node protocol.
    """

    def __init__(self, name=None, x=0.0, y=0.0, e=0.0):
        r"""

        Parameters
        ----------
        name : str
            Name associated to this type of property
        x : float
            Domain of the property
        y : float
            value of the property
        e: float
            error of the property's value
        """
        self.name = name
        self.x = x
        self.e = e
        self.y = y

    def set_scalar(self, y):
        if not isinstance(y, numbers.Real):
            raise TypeError("y must be a non-complex number")
        self.y = y


@decorate_as_node_property((('name', '(str) name of the profile'),
                            ('qvalues', '(:class:`~numpy:numpy.ndarray`) momentum transfer values'),  # noqa: E501
                            ('profile', '(:class:`~numpy:numpy.ndarray`) profile intensities'),  # noqa: E501
                            ('errors', '(:class:`~numpy:numpy.ndarray`) intensity errors')))  # noqa: E501
class ProfileProperty(object):
    r"""Implementation of a node property valid for SANS or X-Ray data.

    Parameters
    ----------
    name : str
        Property name. We could have more than one sans profile
    qvalues : :class:`~numpy:numpy.ndarray`
        Momentun transfer domain
    profile : :class:`~numpy:numpy.ndarray`
        Intensity values
    errors : :class:`~numpy:numpy.ndarray`
        Errors in the intensity values
    """

    def __init__(self, name=None, qvalues=None, profile=None, errors=None):
        self.name = name
        self.qvalues = qvalues
        self.profile = profile
        self.errors = errors


class SansLoaderMixin(object):
    r"""Mixin class providing a set of methods to load SANS data into a
    profile property
    """

    def from_sassena(self, handle, profile_key='fqt', index=0):
        """Load SANS profile from sassena output.

        It is assumed that Q-values are stored under item *qvalues* and
        listed under the *X* column.

        Parameters
        ----------
        handle : h5py.File
            h5py reading handle to HDF5 file
        profile_key : str
            item key where profiles are stored in the HDF5 file
        param index : int
            profile index, if data contains more than one profile

        Returns
        -------
        self : :class:`~idpflex.properties.SansProperty`
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
        return self


class SansProperty(ProfileProperty, SansLoaderMixin):
    r"""Implementation of a node property for SANS data
    """
    def __init__(self, *args, **kwargs):
        ProfileProperty.__init__(self, *args, **kwargs)
        if self.name is None:
            self.name = 'sans'  # Default name


class SaxsLoaderMixin(object):
    r"""Mixin class providing a set of methods to load X-ray data into a
    profile property
    """

    def from_crysol_int(self, file_name):
        r"""Load profile from a `crysol \*.int <https://www.embl-hamburg.de/biosaxs/manuals/crysol.html#output>`_ file

        Parameters
        ----------
        file_name : str
            File path

        Returns
        -------
        self : :class:`~idpflex.properties.SaxsProperty`
        """  # noqa: E501
        contents = np.loadtxt(file_name, skiprows=1, usecols=(0, 1))
        self.qvalues = contents[:, 0]
        self.profile = contents[:, 1]
        self.errors = np.zeros(len(self.qvalues), dtype=float)
        return self

    def from_crysol_fit(self, file_name):
        r"""Load profile from a `crysol \*.fit <https://www.embl-hamburg.de/biosaxs/manuals/crysol.html#output>`_ file.

        Parameters
        ----------
        file_name : str
            File path

        Returns
        -------
        self : :class:`~idpflex.properties.SaxsProperty`
        """  # noqa: E501
        contents = np.loadtxt(file_name, skiprows=1, usecols=(0, 3))
        self.qvalues = contents[:, 0]
        self.profile = contents[:, 1]
        self.errors = np.zeros(len(self.qvalues), dtype=float)
        return self

    def from_crysol_pdb(self, file_name, command='crysol',
                        args='-lm 20 -sm 0.6 -ns 500 -un 1 -eh -dro 0.075',
                        silent=True):
        r"""Calculate profile with crysol from a PDB file

        Parameters
        ----------
        file_name : str
            Path to PDB file
        command : str
            Command to invoke crysol
        args : str
            Arguments to pass to crysol
        silent : bool
            Suppress crysol standard output and error
        Returns
        -------
        self : :class:`~idpflex.properties.SaxsProperty`
        """
        # Write crysol file within a temporary directory
        curr_dir = os.getcwd()
        temp_dir = tempfile.mkdtemp()
        os.chdir(temp_dir)
        call_stack = [command] + args.split() + [file_name]
        if silent:
            FNULL = open(os.devnull, 'w')  # silence crysol output
            subprocess.call(call_stack, stdout=FNULL, stderr=subprocess.STDOUT)
        else:
            subprocess.call(call_stack)
        # Load the crysol file
        ext_2_load = dict(int=self.from_crysol_int, fit=self.from_crysol_fit)
        stop_search = False
        for name in os.listdir(temp_dir):
            for ext in ext_2_load:
                if fnmatch.fnmatch(name, '*.{}'.format(ext)):
                    ext_2_load[ext](name)
                    stop_search = True
                    break
            if stop_search:
                break
        # Delete the temporary directory
        os.chdir(curr_dir)
        subprocess.call('/bin/rm -rf {}'.format(temp_dir).split())
        return self

    def from_ascii(self, file_name):
        r"""Load profile from an ascii file.

        | Expected file format:
        | Rows have three items separated by a blank space:
        | - *col1* momentum transfer
        | - *col2* profile
        | - *col3* errors of the profile

        Parameters
        ----------
        file_name : str
            File path

        Returns
        -------
        self : :class:`~idpflex.properties.SaxsProperty`
        """
        contents = np.loadtxt(file_name, skiprows=0, usecols=(0, 1, 2))
        self.qvalues = contents[:, 0]
        self.profile = contents[:, 1]
        self.errors = contents[:, 2]
        return self

    def to_ascii(self, file_name):
        r"""Save profile as a three-column ascii file.

        | Rows have three items separated by a blank space
        | - *col1* momentum transfer
        | - *col2* profile
        | - *col3* errors of the profile
        """
        dir_name = os.path.dirname(file_name)
        if dir_name and not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        xye = np.array([list(self.x), list(self.y), list(self.e)])
        np.savetxt(file_name, xye.transpose(),
                   header='Momentum-transfer Profile Profile-errors')


class SaxsProperty(ProfileProperty, SaxsLoaderMixin):
    r"""Implementation of a node property for SAXS data
    """
    def __init__(self, *args, **kwargs):
        ProfileProperty.__init__(self, *args, **kwargs)
        if self.name is None:
            self.name = 'saxs'   # Default name


def propagator_weighted_sum(values, tree,
                            weights=lambda left_node, right_node: (1.0, 1.0)):
    r"""Calculate the property of a node as the sum of its two siblings'
    property values. Propagation applies only to non-leaf nodes.

    Parameters
    ----------
    values: list
        List of property values (of same type), one item for each leaf node.
    tree: :class:`~idpflex.cnextend.Tree`
        Tree of :class:`~idpflex.cnextend.ClusterNodeX` nodes
    weights: tuple
        Callable of two arguments (left-node and right-node) returning
        a tuple of left and right weights. Default callable returns (1.0, 1.0)
        always.
    """
    # Insert a property for each leaf
    if len(values) != tree.nleafs:
        msg = "len(values)={} but there are {} leafs".format(len(values),
                                                             tree.nleafs)
        raise ValueError(msg)
    for i, leaf in enumerate(tree.leafs):
        leaf.add_property(values[i])
    property_class = values[0].__class__  # type of the property
    name = values[0].name  # name of the property
    # Propagate up the tree nodes
    for node in tree._nodes[tree.nleafs:]:
        prop = property_class()
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
    left_node : :class:`~idpflex.cnextend.ClusterNodeX`
        One of the two sibling nodes
    right_node : :class:`~idpflex.cnextend.ClusterNodeX`
        One of the two sibling nodes

    Returns
    -------
    tuple
        Weights representing the relative populations of two nodes

    """
    w = float(left_node.count) / (left_node.count + right_node.count)
    return w, 1-w


#: Calculate a property of the node as the sum of its siblings' property
#:  values, weighted by the relative cluster sizes of the siblings.
#:
#: Parameters
#: ----------
#: values : list
#:     List of property values (of same type), one item for each leaf node.
#: node_tree : :class:`~idpflex.cnextend.Tree`
#:     Tree of :class:`~idpflex.cnextend.ClusterNodeX` nodes
propagator_size_weighted_sum = functools.partial(propagator_weighted_sum,
                                                 weights=weights_by_size)
propagator_size_weighted_sum.__name__ = 'propagator_size_weighted_sum'
propagator_size_weighted_sum.__doc__ = r"""Calculate a property of the node
as the sum of its siblings' property values, weighted by the relative cluster
sizes of the siblings.

Parameters
----------
values : list
    List of property values (of same type), one item for each leaf node.
node_tree : :class:`~idpflex.cnextend.Tree`
    Tree of :class:`~idpflex.cnextend.ClusterNodeX` nodes
"""
