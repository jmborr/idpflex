from __future__ import print_function, absolute_import

import os
import sys
from copy import deepcopy

import pytest
import h5py
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
import MDAnalysis as mda

from idpflex import cnextend
from idpflex import properties as idprop

# Resolve the path to the "external data"
this_module_path = sys.modules[__name__].__file__
data_dir = os.path.join(os.path.dirname(this_module_path), 'data')


@idprop.decorate_as_node_property((('name',       'name of the property'),
                                  ('domain_bar', 'property domain'),
                                  ('bar',        'property_value'),
                                  ('error_bar',  'property error')))
class SimpleProperty(object):
    """
    An integer property, only for testing purposes
    """
    def __init__(self, value=0):
        """
        :param value: integer value
        """
        self.name = 'foo'  # name of the simple property
        self.domain_bar = 0.0
        self.bar = int(value)  # value of the property
        self.error_bar = 0.0


@pytest.fixture(scope='session')
def small_tree():
    n_leafs = 9
    a = np.arange(n_leafs)
    dist_mat = squareform(np.square(a - a[:, np.newaxis]))
    z = linkage(dist_mat, method='complete')
    return {'dist_mat': dist_mat,
            'z': z,
            'tree': cnextend.Tree(z),
            'simple_property': [SimpleProperty(i) for i in range(n_leafs)],
            }


@pytest.fixture(scope='session')
def benchmark():
    z = np.loadtxt(os.path.join(data_dir, 'linkage_matrix'))
    t = cnextend.Tree(z)
    n_leafs = 22379
    # Instantiate scalar properties for the leaf nodes, then propagate
    # up the tree
    sc = np.random.normal(loc=100.0, size=n_leafs)
    sc_p = [idprop.ScalarProperty(name='sc', y=s) for s in sc]
    idprop.propagator_size_weighted_sum(sc_p, t)
    return {'z': z,
            'tree': t,
            'nnodes': 44757,
            'nleafs': n_leafs,
            'simple_property': [SimpleProperty(i) for i in range(22379)],
            }


@pytest.fixture(scope='session')
def ss_benchmark():
    r"""DSSP output

    Returns
    -------
    dict
        'dssp_file': absolute path to file.
    """
    return dict(dssp_file=os.path.join(data_dir, 'simulation', 'hiAPP.dssp'),
                pdb_file=os.path.join(data_dir, 'simulation', 'hiAPP.pdb'))


@pytest.fixture(scope='session')
def trajectory_benchmark():
    r"""Load a trajectory into an MDAnalysis Universe instance

    Returns
    -------
    :class:`~MDAnalysis:MDAnalysis.core.universe.Universe`
    """
    sim_dir = os.path.join(data_dir, 'simulation')
    u = mda.Universe(os.path.join(sim_dir, 'hiAPP.pdb'))
    trajectory = os.path.join(sim_dir, 'hiAPP.xtc')
    u.load_new(trajectory)
    return u


@pytest.fixture(scope='session')
def saxs_benchmark():
    r"""Crysol output for one structure

    Returns
    ------
    dict
        'crysol_file': absolute path to file.
    """

    crysol_file = os.path.join(data_dir, 'saxs', 'crysol.dat')
    return dict(crysol_file=crysol_file)


@pytest.fixture(scope='session')
def sans_benchmark(request):
    r"""Sassena output containing 1000 I(Q) profiles for the hiAPP centroids.

    Yields
    ------
    dict
        'profiles' : HDF5 handle to the file containing the I(Q) profiles
        'property_list' : list of SansProperty instances, one for each leaf
        'tree_with_no_property' : cnextend.Tree with random distances among
            leafs and without included properties.
    """

    # setup or initialization
    handle = h5py.File(os.path.join(data_dir, 'sans', 'profiles.h5'), 'r')
    profiles = handle['fqt']
    n_leafs = len(profiles)

    # Create a node tree.
    # m is a 1D compressed matrix of distances between leafs
    m = np.random.random(int(n_leafs * (n_leafs - 1) / 2))
    z = linkage(m)
    tree = cnextend.Tree(z)

    # values is a list of SansProperty instances, one for each tree leaf
    values = list()
    for i in range(tree.nleafs):
        sans_property = idprop.SansProperty()
        sans_property.from_sassena(handle, index=i)
        values.append(sans_property)

    def teardown():
        handle.close()
    request.addfinalizer(teardown)
    return dict(profiles=handle, property_list=values,
                tree_with_no_property=tree)


@pytest.fixture(scope='session')
def sans_fit(sans_benchmark):
    r"""

    Parameters
    ----------
    sans_benchmark : :function:`~pytest.fixture`

    Returns
    -------
    dict
        A dictionary containing the following key, value pairs:
    tree: :class:`~idpflex.cnextend.Tree`
        A hiearchical tree with random distances among leafs, and endowed
        with a :class:`~idpflex.properties.SansProperty`.
    property_name: str
        Just the name of the property
    depth: int
        Tree depth resulting in the best fit to experiment_property
    coefficients: :py:`dict`
        weights of each node at Tree depth resulting in best fit. (key, val)
        pair is (node ID, weight).
    background : float
        Flat background added to the profile at depth for optimal fit
    experiment_property: :class:`~idpflex.properties.SansProperty`
        Experimental profile from a linear combination of the profiles
        at depth for optimal fit using `coefficients` and `background`.
    """
    tree = deepcopy(sans_benchmark['tree_with_no_property'])
    values = sans_benchmark['property_list']
    name = values[0].name  # property name
    idprop.propagator_size_weighted_sum(values, tree)
    # create a SANS profile as a linear combination of the clusters at a
    # particular depth
    depth = 4
    coeffs = (0.45, 0.00, 0.07, 0.25, 0.23)  # they must add to one
    coefficients = dict()
    nodes = tree.nodes_at_depth(depth)
    n_nodes = 1 + depth  # depth=0 corresponds to the root node (nclusters=1)
    q_values = (tree.root[name].x[:-1] + tree.root[name].x[1:]) / 2  # midpoint
    profile = np.zeros(len(q_values))
    for i in range(n_nodes):
        coefficients[nodes[i].id] = coeffs[i]
        p = nodes[i][name]
        profile += coeffs[i] * (p.y[:-1] + p.y[1:]) / 2
    background = 0.05 * max(profile)  # flat background
    profile += background
    experiment_property = idprop.SansProperty(name=name,
                                              qvalues=q_values,
                                              profile=profile,
                                              errors=0.1*profile)
    return {'tree': tree,
            'property_name': name,
            'depth': depth,
            'coefficients': coefficients,
            'background': background,
            'experiment_property': experiment_property}
