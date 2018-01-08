from __future__ import print_function, absolute_import

import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from idpflex import bayes


def test_model_at_node(sans_fit):
    tree = sans_fit['tree']
    mod = bayes.model_at_node(tree.root, sans_fit['property_name'])
    prop = tree.root[sans_fit['property_name']]
    q_values = (prop.x[:-1] + prop.x[1:]) / 2  # midpoints
    params = mod.make_params()
    y = mod.eval(params, x=q_values)
    assert_array_almost_equal(y, (prop.y[:-1] + prop.y[1:]) / 2, decimal=1)


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


def test_fit_to_depth(sans_fit):
    tree = sans_fit['tree']
    name = sans_fit['property_name']
    p_exp = sans_fit['experiment_property']
    fits = bayes.fit_to_depth(tree, p_exp, name, max_depth=7)
    chi2 = np.asarray([fit.chisqr for fit in fits])
    assert np.argmax(chi2 < 1e-10) == sans_fit['depth']


if __name__ == '__main__':
    pytest.main()
