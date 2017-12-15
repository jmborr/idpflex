"""
Bayes fit of the levels of a clustering tree for containing a property
against an experimental profile.
"""

from __future__ import print_function, absolute_import

import numpy as np
from mantid.simpleapi import (Fit, CreateWorkspace, TabulatedFunction,
                              CompositeFunctionWrapper, FlatBackground)


def model_at_node(node, property_name):
    r"""Fit model as a Mantid Tabulated function with only Scaling
    as free parameter, plus a flat background

    Parameters
    ----------
    node : ClusterNodeX
        Any node of a cnextend.Tree
    property_name : str
        name of the property to create the model for.
    
    Returns
    -------
    c : TabulatedFunction + background
        Mantid fit function describing a model.
    """
    x = list(node[property_name].x)
    y = list(node[property_name].y)
    f = TabulatedFunction(X=x, Y=y)
    f.fix('Shift')
    f.fix('XScaling')
    c = CompositeFunctionWrapper()
    c += f
    c += FlatBackground()
    return c


def model_at_depth(tree, depth, property_name):
    r"""Fit model as a linear combination of functions for each node of the
    tree level.

    Parameters
    ----------
    tree : cnextend.Tree
        tree of nodes clustered by structural similarity and each node
        containing a simulated property.
    depth: int
        depth level starting from the root level (depth=0)
    property_name : str
        name of the property to create the model for.

    Returns
    -------
    c : CompositeFunction
        Mantid fit function describing a model for the depth of the tree.
    """

    c = CompositeFunctionWrapper()
    for node in tree.clusters_at_depth(depth):
        x = list(node[property_name].x)
        y = list(node[property_name].y)
        f = TabulatedFunction(X=x, Y=y)
        f['Scaling'] = 1.0/(1 + depth)
        c += f
    [c.fixAll(param) for param in ('Shift', 'XScaling')]
    c.constrainAll('0<Scaling')
    return c


def do_fit(a_function, experiment, prefix='fit', run_fabada=False):
    r"""Carries out fitting of a model against an experimental profile.
    
    An initial quick fit using Levenberg-Marquardt minimizer is followed
    by a longer minimization using the FABADA minimizer.

    Parameters
    ----------
    a_function : FunctionWrapper
        fit function model.
    experiment : properties.NodeProperty
        A property containing the experimental info.
    prefix : str
        prefix all output workspaces from the fit with this prefix string.

    Returns
    -------
    fit_output: namedtuple
        Output of Mantid's Fit algorithm
    """

    # Create a workspace for the experiment
    x = experiment.x
    y = experiment.y
    e = experiment.e
    ws = CreateWorkspace(x, y, e, NSpec=1, UnitX='MomentumTransfer')

    # Initialization of some variables
    degrees_of_freedom = sum([not int(a_function.isFixed(i))
                              for i in range(a_function.nParams())])
    # Initial quick minimization with Levenberg-Marquardt
    minimizer = 'Levenberg-Marquardt'
    fit_output = Fit(Function=a_function,
                     InputWorkspace=ws,
                     WorkspaceIndex=0,
                     CreateOutput=True,
                     Output=prefix,
                     Minimizer=minimizer,
                     MaxIterations=2000*degrees_of_freedom)
    if run_fabada:
        # Now run FABADA with previous minimization as initial guess
        chain_length = 2500 * (1 + degrees_of_freedom)
        minimizer = 'FABADA,Chains={}_chains,'.format(prefix) + \
                    'ConvergedChain={}_convchains,'.format(prefix) + \
                    'ChainLength={},'.format(chain_length) + \
                    'NumberBinsPDF=50,ConvergenceCriteria=0.1'
        other_function = fit_output.Function
        fit_output = Fit(Function=fit_output.Function,
                         InputWorkspace=ws,
                         WorkspaceIndex=0,
                         CreateOutput=True,
                         Output=prefix,
                         Minimizer=minimizer,
                         MaxIterations=2000*degrees_of_freedom)

    return fit_output


def do_fit_at_depth_tree(tree, experiment, property_name, max_depth=10,
                         background=FlatBackground()):
    r"""Mantid fit for each level of a tree up to a maximum depth.

    Parameters
    ----------
    tree : cnextend.Tree
        Tree of nodes clustered by structural similarity and each node
        containing a simulated property.
    experiment : properties.NodeProperty
        A property containing the experimental info.
    property_name: str
        The name of the simulated property to compare against experiment.
    max_depth : int
        Fit at each depth up to (but not including) max_depth.
    background : FunctionWrapper
        Mantid fit function describing the background

    Returns
    -------
    fits_output : list
        list containing fit outputs at each level of the tree up to max_depth
    """

    # Fit each level of the tree
    fits_output = list()
    fits_prob = list()
    for depth in range(max_depth):
        print('Fitting at depth = {}'.format(depth))
        model = model_at_depth(tree, depth, property_name)
        model += background
        fit_output = do_fit(model, experiment,
                            prefix='fit{}'.format(depth))
        fits_output.append(fit_output)

        """
        # Calculate probability of this level.
        # Assumptions:
        # 1 Model is a combination of tabulated functions plus flat background
        # 2 Background is the last entry

        # Extract the normalized variance-covariance matrix
        table = fit_output.OutputNormalisedCovarianceMatrix
        # The first column of the table contains parameter names.
        # The last column and the last row relate to the flat background
        # parameter.
        n_row, n_cols = table.rowCount() - 1, table.columnCount() - 2
        normalized_var_cov = np.zeros(n_row * n_cols).reshape((n_row, n_cols))
        for i_col in range(1, table.columnCount()-1):
            normalized_var_cov[:, i_col-1] = table.column(i_col)[:-1]
        # mantid outputs the normalized covariance matrix such that the
        # diagonal elements are 100.
        normalized_var_cov /= 100.0
        # Extract the variances
        errors = list()
        param_table = fit_output.OutputParameters
        for i_row in range(param_table.rowCount()):
            if '.Scaling' in param_table.row(i_row)['Name']:
                errors.append(param_table.row(i_row)['Error'])
        variances = np.array(errors) ** 2
        # Compute the variance-covariance matrix
        var_cov = normalized_var_cov * \
                  np.sqrt(variances * variances[np.newaxis].transpose())
        # Compute the likelihood
        likelihood = (4 * np.pi) ** depth *\
        np.exp(0.5 * fit_output.OutputChi2overDoF) *\
        np.sqrt(np.linalg.det(var_cov))
        # Compute the prior probability
        scaling = 0.0
        for i_row in range(param_table.rowCount()):
            if '.Scaling' in param_table.row(i_row)['Name']:
                scaling += param_table.row(i_row)['Value']
        prior_prob = np.math.factorial(depth) / scaling ** depth
        fits_prob.append(likelihood * prior_prob)
        """
    #return fits_output, fits_prob
    return fits_output
