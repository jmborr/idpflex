from __future__ import print_function, absolute_import

import random
import numpy as np
import pytest

from idpflex import properties as ps
from idpflex.properties import SecondaryStructureProperty as SSP


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


class TestScalarProperty(object):

    def test_histogram(self, benchmark):
        root_prop = benchmark['tree'].root['sc']
        edges, h, e = root_prop.histogram(bins=1, errors=True)
        assert h[0] == benchmark['nleafs']
        assert e[0] == np.sqrt(h[0])

    def test_plot_histogram(self, benchmark):
        root_prop = benchmark['tree'].root['sc']
        ax = root_prop.plot(kind='histogram', errors=True, bins=1)
        assert ax.patches[0]._height == benchmark['nleafs']


class TestAsphericity(object):

    def test_from_pdb(self, ss_benchmark):
        filename = ss_benchmark['pdb_file']
        prop = ps.Asphericity().from_pdb(filename)
        np.testing.assert_almost_equal(prop.asphericity, 0.71, decimal=2)


class TestEndToEnd(object):

    def test_from_pdb(self, ss_benchmark):
        filename = ss_benchmark['pdb_file']
        prop = ps.EndToEnd().from_pdb(filename)
        np.testing.assert_almost_equal(prop.end_to_end, 9.244, decimal=3)


class TestSaSa(object):

    def test_from_pdb(self, ss_benchmark):
        filename = ss_benchmark['pdb_file']
        prop = ps.SaSa().from_pdb(filename)
        np.testing.assert_allclose(prop.sasa, 2964, rtol=0.10)
        prop = ps.SaSa().from_pdb(filename, n_sphere_points=3)
        np.testing.assert_allclose(prop.sasa, 2989, rtol=0.10)
        prop = ps.SaSa().from_pdb(filename, selection='resid 0 to 10')
        np.testing.assert_allclose(prop.sasa, 1350, rtol=0.16)


class TestRadiusOfGyration(object):

    def test_from_pdb(self, ss_benchmark):
        filename = ss_benchmark['pdb_file']
        prop = ps.RadiusOfGyration().from_pdb(filename, 'name CA')
        np.testing.assert_almost_equal(prop.rg, 8.75, decimal=2)


class TestResidueContactMap(object):

    def test_from_universe(self, trajectory_benchmark):
        cm = ps.ResidueContactMap().from_universe(trajectory_benchmark,
                                                  8, 'name CA')
        assert np.sum(cm.y) == 363
        cm = ps.ResidueContactMap().from_universe(trajectory_benchmark, 4)
        assert np.sum(cm.y) == 313

    def test_from_pdb(self, ss_benchmark):
        filename = ss_benchmark['pdb_file']
        cm = ps.ResidueContactMap().from_pdb(filename, 8, 'name CA')
        assert np.sum(cm.y) == 351

    @pytest.mark.skip(reason="Plotting not enabled in the CI")
    def test_plot(self, trajectory_benchmark):
        cm = ps.ResidueContactMap().from_universe(trajectory_benchmark,
                                                  8, 'name CA')
        cm.plot()


class TestSecondaryStructureProperty(object):

    def test_class_decorated_as_node_property(self):
        assert isinstance(SSP.name, property)
        assert isinstance(SSP.x, property)
        assert isinstance(SSP.y, property)
        assert isinstance(SSP.e, property)

    def test_instance_decorated_as_node_property(self):
        ss = 'GTEL'
        v = np.random.rand(len(ss), SSP.n_codes)
        v /= np.sum(v, axis=1)[:, np.newaxis]  # normalize rows
        profile_prop = SSP(name='foo', aa=ss, profile=v, errors=0.1*v)
        assert profile_prop.name == 'foo'
        assert np.array_equal(profile_prop.x, ss)
        assert np.array_equal(profile_prop.y, v)
        assert np.array_equal(profile_prop.e, 0.1*v)

    def test_default_name(self):
        ss_prop = SSP()
        assert ss_prop.name == 'ss'

    def test_from_dssp_sequence(self):
        seq = ''.join(random.sample(SSP.dssp_codes, SSP.n_codes))
        ss_prop = SSP().from_dssp_sequence(seq)
        np.testing.assert_array_equal(ss_prop.y[-1], SSP.code2profile(seq[-1]))

    def test_from_dssp(self, ss_benchmark):
        name = ss_benchmark['dssp_file']
        ss_prop = SSP().from_dssp(name)
        np.testing.assert_array_equal(ss_prop.y[-1], SSP.code2profile(' '))

    @pytest.mark.skip(reason="DSSP may not be installed in the machine")
    def test_from_dssp_pdb(self, ss_benchmark):
        name = ss_benchmark['pdb_file']
        ss_prop = SSP().from_dssp_pdb(name)
        np.testing.assert_array_equal(ss_prop.y[-1], SSP.code2profile(' '))

    def test_propagator_size_weighted_sum(self, small_tree):
        r"""Create random secondary sequences by shufling all codes and
        assign to the leafs of the tree. Then, propagate the profiles up
        the tree hiearchy. Finally, compare the profile of the root with
        expected profile.
        """
        tree = small_tree['tree']
        ss_props = list()
        for i in range(tree.nleafs):
            seq = ''.join(random.sample(SSP.dssp_codes, SSP.n_codes))
            ss_props.append(SSP().from_dssp_sequence(seq))
        ps.propagator_size_weighted_sum(ss_props, tree)
        # Manually calculate the average profile for the last residue
        y = np.asarray([ss_props[i].y for i in range(tree.nleafs)])
        average_profile = np.mean(y, axis=0)
        np.testing.assert_array_almost_equal(average_profile,
                                             tree.root['ss'].y, decimal=12)

    def test_fractions(self):
        profile = np.random.rand(42, SSP.n_codes)  # not normalized
        prop = SSP(profile=profile)
        f = prop.fractions
        assert f['H'] == np.sum(profile, axis=0)[0] / 42

    def test_collapse(self):
        profile = np.random.rand(42, SSP.n_codes)  # not normalized
        prop = SSP(profile=profile)
        c = prop.collapsed
        assert c[0] == np.argmax(profile[0])

    def test_disparity(self):
        p = np.random.rand(42, SSP.n_codes)  # not normalized
        o = np.zeros((42, SSP.n_codes))
        pr = SSP(profile=p)
        assert pr.disparity(SSP(profile=-p)) == 4 * \
            pr.disparity(SSP(profile=o))

    @pytest.mark.skip(reason="Plotting not enabled in the CI")
    def test_plot_percents(self):
        profile = np.random.rand(42, SSP.n_codes)  # not normalized
        profile /= np.sum(profile, axis=1)[:, np.newaxis]  # normalized
        prop = SSP(profile=profile)
        prop.plot('percents')

    @pytest.mark.skip(reason="Plotting not enabled in the CI")
    def test_plot_node(self):
        profile = np.random.rand(42, SSP.n_codes)  # not normalized
        profile /= np.sum(profile, axis=1)[:, np.newaxis]  # normalized
        prop = SSP(profile=profile)
        prop.plot('node')

    @pytest.mark.skip(reason="Plotting not enabled in the CI")
    def test_plot_leafs(self, small_tree):
        tree = small_tree['tree']
        ss_props = list()
        for i in range(tree.nleafs):
            seq = ''.join(random.sample(1000*SSP.dssp_codes, 42))
            ss_props.append(SSP().from_dssp_sequence(seq))
        ps.propagator_size_weighted_sum(ss_props, tree)
        tree.root['ss'].plot('leafs')


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
        lfs = benchmark['nleafs']
        assert tree.root['foo'].bar == int(lfs * (lfs-1) / 2)

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
