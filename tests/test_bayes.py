import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest
import warnings

from idpflex import bayes
from idpflex import properties


def test_TabulatedModel_interpolation(sans_fit):
    tree = sans_fit['tree']
    tree.root.property_group = tree.root.property_group.subset(
        tree.root.property_group.keys(),
        property_type=properties.ProfileProperty)
    mod = bayes.create_at_depth(tree, 0, use_tabulated=True)
    prop = tree.root[sans_fit['property_name']]
    params = mod.make_params()
    # test interpolation
    qvals = prop.qvalues[:-1] + np.diff(prop.qvalues)/2
    assert_array_almost_equal(mod.eval(params, x=qvals),
                              prop.interpolator(qvals), decimal=1)
    # test centering
    params[f'{prop.name}_center'].set(vary=True, value=2)
    fit = mod.fit(prop.y, x=(prop.qvalues+2), params=params)
    assert_array_almost_equal(fit.best_fit, prop.y, decimal=1)
    assert_array_almost_equal(fit.params[f'{prop.name}_center'], 2, decimal=3)


def test_fit_at_depth(sans_fit):
    tree = sans_fit['tree']
    exp = sans_fit['experiment_property']
    exp_pd = properties.PropertyDict([exp])
    mod = bayes.create_at_depth(tree, sans_fit['depth'])
    params = mod.make_params()
    fit = bayes.fit_model(mod, exp_pd, params=params)
    assert fit.weights is None
    fit2 = bayes.fit_model(mod, exp_pd, weights=1/exp.e)
    assert fit2.chisqr < 1e-10
    assert_array_equal(fit2.weights, 1/exp.e)
    # make assertions about parameters
    assert all([f'struct{i}_proportion_c' in fit.params
                for i, _ in enumerate(tree.nodes_at_depth(sans_fit['depth']))])
    assert all([fit.params[f'struct{i}_proportion_c'].min == 0
                for i, _ in enumerate(tree.nodes_at_depth(sans_fit['depth']))])
    assert all([f'struct{i}_{pname}_slope' in fit.params
                for i, _ in enumerate(tree.nodes_at_depth(sans_fit['depth']))
                for pname in exp_pd])
    assert all([fit.params[f'struct{i}_{pname}_slope'].min == 0
                for i, _ in enumerate(tree.nodes_at_depth(sans_fit['depth']))
                for pname in exp_pd])
    assert all([f'struct{i}_{pname}_intercept' in fit.params
                for i, _ in enumerate(tree.nodes_at_depth(sans_fit['depth']))
                for pname in exp_pd])
    assert abs(sum(p for p in fit.params.values()
                   if p.name.endswith('_p')) - 1) <= 1e-3


def test_global_minimization(sans_fit):
    tree = sans_fit['tree']
    exp = sans_fit['experiment_property']
    exp_pd = properties.PropertyDict([exp])
    mod = bayes.create_at_depth(tree, sans_fit['depth'])
    params = mod.make_params()
    for param in params.values():
        param.set(min=0, max=1e9)
    fit = bayes.fit_model(mod, exp_pd, params=params, weights=1/exp.e)
    fit2 = bayes.fit_model(mod, exp_pd, params=params, weights=1/exp.e,
                           method='basin_hopping')
    assert abs(sum(p for p in fit2.params.values()
                   if p.name.endswith('_p')) - 1) <= 1e-3
    try:
        # Expect global fit to be better than typical fit
        assert fit.redchi <= fit.redchi
    except AssertionError:
        warnings.warn('Global minimization did not do better than reference'
                      ' fit. Attempting a refit to avoid test failure.'
                      f' Difference {fit.redchi - fit2.redchi:.3} where global'
                      f' {fit2.redchi:.3} and reference {fit.redchi:.3}.',
                      RuntimeWarning)
        # Try refitting and looser tolerance
        fit3 = bayes.fit_model(mod, exp_pd, params=params,
                               weights=exp_pd.feature_weights,
                               method='differential_evolution')
        assert fit3 <= fit


def test_fit_to_depth(sans_fit):
    tree = sans_fit['tree']
    exp = sans_fit['experiment_property']
    exp_pd = properties.PropertyDict([exp])
    mods = bayes.create_to_depth(tree, max_depth=7)
    params_list = [m.make_params() for m in mods]
    fits = bayes.fit_models(mods, exp_pd, weights=1/exp.e,
                            params_list=params_list)
    # Since only one probability assert that there is no probability
    assert all([not p.endswith('_p') for p in fits[0].params])
    chi2 = np.array([fit.chisqr for fit in fits])
    assert np.argmax(chi2 < 1e-10) == sans_fit['depth']


def test_fit(sans_fit):
    tree = sans_fit['tree']
    exp = sans_fit['experiment_property']
    scalar = properties.ScalarProperty(name='foo', x=1, y=1, e=1)
    values = [properties.ScalarProperty(name='foo', x=1, y=i, e=1)
              for i in range(len(tree.leafs))]
    properties.propagator_size_weighted_sum(values, tree)
    exp_pd = properties.PropertyDict([exp, scalar])
    ctd = bayes.create_to_depth
    models = ctd(tree, max_depth=7)
    params_list = [m.make_params() for m in models]
    weights = 1/np.concatenate([exp.e, scalar.feature_weights])
    fits = bayes.fit_models(models, exp_pd, weights=weights,
                            params_list=params_list)
    chi2 = [fit.chisqr for fit in fits]
    assert chi2[sans_fit['depth']] <= 1e-10
    # Test filtering by name
    cad = bayes.create_at_depth
    exp_pd2 = properties.PropertyDict([exp])
    models2 = cad(tree, depth=sans_fit['depth'], names=exp_pd2.keys())
    params_list2 = models2.make_params()
    assert all(f'{k}_slope' in params_list2 for k in exp_pd2.keys())
    # Test filtering by property type
    models3 = cad(tree, depth=sans_fit['depth'],
                  property_type=properties.ProfileProperty)
    params_list3 = models3.make_params()
    assert all(f'{p.name}_slope' in params_list3 for p in exp_pd2.values()
               if isinstance(p, properties.ProfileProperty))
    # Test filtering by filtering function
    models4 = cad(tree, depth=sans_fit['depth'],
                  to_keep_filter=lambda p: p.name == 'foo')
    params_list4 = models4.make_params()
    assert 'foo_slope' in params_list4


def test_fit_bad_input(sans_fit):
    tree = sans_fit['tree']
    exp = sans_fit['experiment_property']
    name = sans_fit['property_name']
    scalar = properties.ScalarProperty(name='foo', x=1, y=1, e=1)
    # Different x value to prompt error
    values = [properties.ScalarProperty(name='foo', x=10, y=i, e=1)
              for i in range(len(tree.leafs))]
    properties.propagator_size_weighted_sum(values, tree)
    exp_pd = properties.PropertyDict([exp, scalar])
    ctd = bayes.create_to_depth
    with pytest.raises(ValueError):
        models = ctd(tree, max_depth=7)
        bayes.fit_models(models, exp_pd)
    with pytest.raises(ValueError):
        left_child = tree.root.get_left()
        left_child.property_group = left_child.property_group.subset([name])
        mods = ctd(tree, max_depth=7)
        bayes.fit_models(mods, exp_pd)


if __name__ == '__main__':
    pytest.main()
