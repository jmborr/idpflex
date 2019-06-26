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
        param.set(min=0, max=10)
    fit = bayes.fit_multiproperty_model(mod, exp_pd, params=params,
                                        weights=exp_pd.feature_weights)
    fit2 = bayes.fit_multiproperty_model(mod, exp_pd, params=params,
                                         weights=exp_pd.feature_weights,
                                         method='differential_evolution')
    try:
        # Expect less than 20% difference between these
        diff = abs(1 - fit.redchi/fit2.redchi)
        assert diff <= 0.20
    except AssertionError:
        warnings.warn('Global minimization did not get within 20% of reference'
                      f' fit. Relative difference {diff:.3}.',
                      RuntimeWarning)
        # Try refitting and looser tolerance
        fit3 = bayes.fit_multiproperty_model(mod, exp_pd, params=params,
                                             weights=exp_pd.feature_weights,
                                             method='differential_evolution')
        assert abs(1 - fit.redchi/fit2.redchi) <= 0.50\
            or abs(1 - fit3.redchi/fit2.redchi) <= 0.50


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
    models = bayes.create_to_depth_multiproperty(tree, max_depth=7)
    params_list = [m.make_params() for m in models]
    weights = 1/np.concatenate([exp.e, scalar.feature_weights])
    fits = bayes.fit_multiproperty_models(models, exp_pd, weights=weights,
                                          params_list=params_list)
    chi2 = np.array([fit.chisqr for fit in fits])
    assert np.argmax(chi2 < 1e-10) == sans_fit['depth']


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
    with pytest.raises(ValueError):
        models = bayes.create_to_depth_multiproperty(tree, max_depth=7)
        bayes.fit_multiproperty_models(models, exp_pd)
    with pytest.raises(ValueError):
        mods = bayes.create_to_depth_multiproperty(tree, max_depth=7)
        bayes.fit_multiproperty_models(mods, exp_pd.subset([name]))
    with pytest.raises(ValueError):
        left_child = tree.root.get_left()
        left_child.property_group = left_child.property_group.subset([name])
        mods = bayes.create_to_depth_multiproperty(tree, max_depth=7)
        bayes.fit_multiproperty_models(mods, exp_pd)


if __name__ == '__main__':
    pytest.main()
