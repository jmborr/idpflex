from __future__ import print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mantid.simpleapi import CreateWorkspace, FlatBackground

from idpflex import bayes
from idpflex.test.test_helper import sans_benchmark, sans_fit


def test_model_at_depth(sans_fit):
    tree = sans_fit['tree']
    property_name = sans_fit['property_name']
    depth = 3   # four clusters, since depth=0 represents the root node
    f = bayes.model_at_depth(tree, depth, property_name)
    assert len(f) == 1 + depth


def test_do_fit_at_depth(sans_fit):
    tree = sans_fit['tree']
    property_name = sans_fit['property_name']
    depth = sans_fit['depth']
    experiment = sans_fit['experiment_property']
    model = bayes.model_at_depth(tree, depth, property_name)
    model += FlatBackground()
    fit_output = bayes.do_fit(model, experiment,
                              prefix='fit')
    print(fit_output._fields)
    print(type(fit_output))
    parameters_table = fit_output.OutputParameters
    coeff = sans_fit['coefficients']  # weight of each cluster
    coeff_index = 0
    for row in parameters_table:
        if '.Scaling' in row['Name']:
            assert abs(row['Value'] - coeff[coeff_index]) < 0.01
            coeff_index += 1
        if '.A0' in row['Name']:
            assert abs(row['Value'] - sans_fit['background']) < 0.01


def test_do_fit_tree(sans_fit):
    tree = sans_fit['tree']
    experiment_property = sans_fit['experiment_property']
    property_name = sans_fit['property_name']
    fits_output = bayes.do_fit_at_depth_tree(tree, experiment_property,
                                             property_name, max_depth=10,
                                             background=FlatBackground())
    chis_squared = [fit_output.OutputChi2overDoF for fit_output in fits_output]
    # Weights necessary because chis_s quared flattens above sans_fit['depth']
    weights = 10**np.arange(len(chis_squared))
    chis_squared_weighted = weights*chis_squared
    #plt.semilogy(chis_squared_weighted)
    #plt.show()
    assert np.argmin(chis_squared_weighted) == sans_fit['depth']

if __name__ == '__main__':
    pytest.main()