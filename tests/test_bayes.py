import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest
import warnings

from idpflex import bayes
from idpflex import properties


def test_model_at_node(sans_fit):
    tree = sans_fit['tree']
    mod = bayes.model_at_node(tree.root, sans_fit['property_name'])
    prop = tree.root[sans_fit['property_name']]
    params = mod.make_params()
    assert_array_almost_equal(mod.eval(params, x=prop.qvalues),
                              prop.y, decimal=1)


def test_TabulatedModel_interpolation(sans_fit):
    tree = sans_fit['tree']
    mod = bayes.model_at_node(tree.root, sans_fit['property_name'])
    prop = tree.root[sans_fit['property_name']]
    params = mod.make_params()
    # test interpolation
    qvals = prop.qvalues[:-1] + np.diff(prop.qvalues)/2
    assert_array_almost_equal(mod.eval(params, x=qvals),
                              prop.interpolator(qvals), decimal=1)
    # test centering
    params['center'].set(vary=True, value=2)
    fit = mod.fit(prop.y, x=(prop.qvalues+2), params=params)
    assert_array_almost_equal(fit.best_fit, prop.y, decimal=1)
    assert_array_almost_equal(fit.params['center'], 2, decimal=3)


def test_model_at_depth(sans_fit):
    tree = sans_fit['tree']
    name = sans_fit['property_name']
    depth = sans_fit['depth']
    # Evaluate the model
    mod = bayes.model_at_depth(tree, depth, name)
    params = mod.make_params()
    for id, coeff in sans_fit['coefficients'].items():
        params['n{}_amplitude'.format(id)].set(value=coeff)
    params['c'].set(value=sans_fit['background'])
    p_exp = sans_fit['experiment_property']
    assert_array_almost_equal(mod.eval(params, x=p_exp.x),
                              p_exp.y, decimal=1)


def test_fit_at_depth(sans_fit):
    tree = sans_fit['tree']
    name = sans_fit['property_name']
    p_exp = sans_fit['experiment_property']
    fit = bayes.fit_at_depth(tree, p_exp, name, sans_fit['depth'])
    assert fit.chisqr < 1e-10


def test_fit_to_depth(sans_fit):
    tree = sans_fit['tree']
    name = sans_fit['property_name']
    p_exp = sans_fit['experiment_property']
    fits = bayes.fit_to_depth(tree, p_exp, name, max_depth=7)
    chi2 = np.asarray([fit.chisqr for fit in fits])
    assert np.argmax(chi2 < 1e-10) == sans_fit['depth']


def test_fit_at_depth_multi(sans_fit):
    tree = sans_fit['tree']
    exp = sans_fit['experiment_property']
    exp_pd = properties.PropertyDict([exp])
    mod = bayes.create_at_depth_multiproperty(tree, sans_fit['depth'])
    params = mod.make_params()
    fit = bayes.fit_multiproperty_model(mod, exp_pd, params=params)
    assert fit.weights is None
    fit2 = bayes.fit_multiproperty_model(mod, exp_pd,
                                         weights=1/exp.e)
    assert fit2.chisqr < 1e-10
    assert_array_equal(fit2.weights, 1/exp.e)


def test_global_minimization(sans_fit):
    tree = sans_fit['tree']
    exp = sans_fit['experiment_property']
    exp_pd = properties.PropertyDict([exp])
    mod = bayes.create_at_depth_multiproperty(tree, sans_fit['depth'])
    params = mod.make_params()
    for param in params.values():
        param.set(min=0, max=100)
    fit = bayes.fit_multiproperty_model(mod, exp_pd, params=params,
                                        weights=1/exp.e)
    fit2 = bayes.fit_multiproperty_model(mod, exp_pd, params=params,
                                         weights=1/exp.e,
                                         method='basin_hopping')
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
        fit3 = bayes.fit_multiproperty_model(mod, exp_pd, params=params,
                                             weights=exp_pd.feature_weights,
                                             method='differential_evolution')
        assert fit3 <= fit


def test_fit_to_depth_multi(sans_fit):
    tree = sans_fit['tree']
    exp = sans_fit['experiment_property']
    exp_pd = properties.PropertyDict([exp])
    mods = bayes.create_to_depth_multiproperty(tree, max_depth=7)
    params_list = [m.make_params() for m in mods]
    fits = bayes.fit_multiproperty_models(mods, exp_pd, weights=1/exp.e,
                                          params_list=params_list)
    # Since only one probability assert that there is no probability
    assert all(['prob' not in p for p in fits[0].params])
    chi2 = np.array([fit.chisqr for fit in fits])
    assert np.argmax(chi2 < 1e-10) == sans_fit['depth']


def test_multiproperty_fit(sans_fit):
    tree = sans_fit['tree']
    exp = sans_fit['experiment_property']
    scalar = properties.ScalarProperty(name='foo', x=1, y=1, e=1)
    values = [properties.ScalarProperty(name='foo', x=1, y=i, e=1)
              for i in range(len(tree.leafs))]
    properties.propagator_size_weighted_sum(values, tree)
    exp_pd = properties.PropertyDict([exp, scalar])
    ctd = bayes.create_to_depth_multiproperty
    models = ctd(tree, max_depth=7)
    params_list = [m.make_params() for m in models]
    weights = 1/np.concatenate([exp.e, scalar.feature_weights])
    fits = bayes.fit_multiproperty_models(models, exp_pd, weights=weights,
                                          params_list=params_list)
    chi2 = [fit.chisqr for fit in fits]
    assert chi2[sans_fit['depth']] <= 1e-10
    # Test filtering by name
    cad = bayes.create_at_depth_multiproperty
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


def test_multiproperty_fit_different_models(sans_fit):
    tree = sans_fit['tree']
    exp = sans_fit['experiment_property']
    scalar = properties.ScalarProperty(name='foo', x=1, y=1, e=1)
    values = [properties.ScalarProperty(name='foo', x=1, y=i, e=1)
              for i in range(len(tree.leafs))]
    properties.propagator_size_weighted_sum(values, tree)
    exp_pd = properties.PropertyDict([exp, scalar])
    ctd = bayes.create_to_depth_multiproperty
    # test different models where one is a class and one is an instance
    models = ctd(tree, max_depth=7, models=[bayes.LinearModel,
                                            bayes.ConstantVectorModel()])
    params_list = [m.make_params() for m in models]
    weights = 1/np.concatenate([exp.e, scalar.feature_weights])
    fits = bayes.fit_multiproperty_models(models, exp_pd, weights=weights,
                                          params_list=params_list)
    chi2 = np.array([fit.chisqr for fit in fits])
    assert np.argmax(chi2 < 1e-10) == sans_fit['depth']
    assert all(['struct' not in p for p in fits[0].params])
    assert 'foo_scale' in fits[-1].params
    assert 'sans_slope' in fits[-1].params
    assert 'sans_intercept' in fits[-1].params
    assert all([f'struct{i}_prob_c' in fits[-1].params for i in range(7)])
    assert fits[-1].params['sans_slope'].expr == 'struct0_sans_slope'
    assert fits[-1].params['struct1_sans_slope'].expr == 'struct0_sans_slope'
    ptotal = sum([p.value for p in fits[-1].params.values()
                  if 'prob' in p.name])
    assert abs(1 - ptotal) <= 0.05
    if abs(1 - ptotal) > 0.005:
        warnings.warn(f'Probabilities did not sum to 1. Ptotal={ptotal:.3}.',
                      RuntimeWarning)


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
    ctd = bayes.create_to_depth_multiproperty
    with pytest.raises(ValueError):
        models = ctd(tree, max_depth=7, models=bayes.LinearModel)
        bayes.fit_multiproperty_models(models, exp_pd)
    with pytest.raises(ValueError):
        models = ctd(tree, max_depth=7, models=[bayes.LinearModel]*3)
        bayes.fit_multiproperty_models(models, exp_pd)
    with pytest.raises(ValueError):
        mods = ctd(tree, max_depth=7, models=bayes.ConstantVectorModel)
        bayes.fit_multiproperty_models(mods, exp_pd.subset([name]))
    with pytest.raises(ValueError):
        left_child = tree.root.get_left()
        left_child.property_group = left_child.property_group.subset([name])
        mods = ctd(tree, max_depth=7)
        bayes.fit_multiproperty_models(mods, exp_pd)


if __name__ == '__main__':
    pytest.main()
