from __future__ import print_function, absolute_import

import numpy as np
import pytest

from idpflex import properties as ps
from idpflex.test.test_helper import benchmark, sans_benchmark, saxs_benchmark


class TestRegisterDecorateProperties(object):

    def test_register_as_node_property(self):
        class SomeProperty(object):
            def __init__(self):
                attrs = dict(id='foo', a='ax', b='by', c='ce')
                self.__dict__.update(attrs)
        associations = (('id', 'name of the property'),
                        ('a', 'this is x'),
                        ('b', 'this is y'),
                        ('c', 'this is e'))
        ps.register_as_node_property(SomeProperty, associations)
        # Test for class attributes
        assert isinstance(ps.ProfileProperty.name, property)
        assert isinstance(ps.ProfileProperty.x, property)
        assert isinstance(ps.ProfileProperty.y, property)
        assert isinstance(ps.ProfileProperty.e, property)
        # Test for managed attributes
        some_prop = SomeProperty()
        assert some_prop.name == 'foo'
        assert some_prop.x == 'ax'
        assert some_prop.y == 'by'
        assert some_prop.e == 'ce'

    def test_decorate_as_node_property(self):
        associations = (('id', 'name of the property'),
                        ('a', 'this is x'),
                        ('b', 'this is y'),
                        ('c', 'this is e'))

        @ps.decorate_as_node_property(associations)
        class SomeProperty(object):
            def __init__(self):
                attrs = dict(id='foo', a='ax', b='by', c='ce')
                self.__dict__.update(attrs)
        # Test for class attributes
        assert isinstance(ps.ProfileProperty.name, property)
        assert isinstance(ps.ProfileProperty.x, property)
        assert isinstance(ps.ProfileProperty.y, property)
        assert isinstance(ps.ProfileProperty.e, property)
        # Test for managed attributes
        some_prop = SomeProperty()
        assert some_prop.name == 'foo'
        assert some_prop.x == 'ax'
        assert some_prop.y == 'by'
        assert some_prop.e == 'ce'


class TestProfileProperty(object):

    def test_class_decorated_as_node_property(self):
        assert isinstance(ps.ProfileProperty.name, property)
        assert isinstance(ps.ProfileProperty.x, property)
        assert isinstance(ps.ProfileProperty.y, property)
        assert isinstance(ps.ProfileProperty.e, property)

    def test_instance_decorated_as_node_property(self):
        v = np.arange(9)
        profile_prop = ps.ProfileProperty(name='foo', qvalues=v, profile=10*v,
                                          errors=0.1*v)
        assert profile_prop.name == 'foo'
        assert np.array_equal(profile_prop.x, v)
        assert np.array_equal(profile_prop.y, 10*v)
        assert np.array_equal(profile_prop.e, 0.1*v)


class TestSansProperty(object):

    def test_registered_as_node_property(self):
        assert isinstance(ps.SansProperty.name, property)
        assert isinstance(ps.SansProperty.x, property)
        assert isinstance(ps.SansProperty.y, property)
        assert isinstance(ps.SansProperty.e, property)

    def test_default_name(self):
        sans_prop = ps.SansProperty()
        assert sans_prop.name == 'sans'

    def test_from_sassena(self, sans_benchmark):
        sans_prop = ps.SansProperty()
        sans_prop.from_sassena(sans_benchmark['profiles'], index=666)
        assert sans_prop.qvalues[13].item() - 0.0656565651298 < 0.000000001
        assert sans_prop.profile[13].item() - 741970.84461578 < 0.000001


class TestSaxsProperty(object):

    def test_registered_as_node_property(self):
        assert isinstance(ps.SaxsProperty.name, property)
        assert isinstance(ps.SaxsProperty.x, property)
        assert isinstance(ps.SaxsProperty.y, property)
        assert isinstance(ps.SaxsProperty.e, property)

    def test_default_name(self):
        saxs_prop = ps.SaxsProperty()
        assert saxs_prop.name == 'saxs'

    def test_from_crysol_int(self, saxs_benchmark):
        saxs_prop = ps.SaxsProperty()
        saxs_prop.from_crysol_int(saxs_benchmark['crysol_file'])
        assert saxs_prop.qvalues[8] == 0.008
        assert saxs_prop.profile[8] == 1740900.0
        assert saxs_prop.errors[8] == 0.0


class TestPropagators(object):

    def test_propagator_weighted_sum(self, benchmark):
        tree = benchmark['tree']
        ps.propagator_weighted_sum(benchmark['simple_property'], tree)
        l = benchmark['nleafs']
        assert tree.root['foo'].bar == int(l * (l-1) / 2)

    def test_propagator_size_weighted_sum(self, sans_benchmark):
        tree = sans_benchmark['tree_with_no_property']
        values = sans_benchmark['property_list']
        ps.propagator_size_weighted_sum(values, tree)
        # Test the propagation of the profiles for a node randomly picked
        node_id = np.random.randint(tree.nleafs, len(tree))  # exclude leafs
        node = tree[node_id]
        ln = node.left
        rn = node.right
        w = float(ln.count) / (ln.count + rn.count)
        lnp = ln['sans']  # profile of the "left" sibling node
        rnp = rn['sans']
        y = w * lnp.y + (1 - w) * rnp.y
        assert np.array_equal(y, node['sans'].y)


if __name__ == '__main__':
    pytest.main()
