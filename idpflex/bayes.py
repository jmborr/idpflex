# from qef.models import TabulatedFunctionModel
from lmfit.models import (Model, ConstantModel)
from lmfit import Parameter
from scipy.interpolate import interp1d
import numpy as np
from idpflex.properties import ScalarProperty


class TabulatedFunctionModel(Model):
    r"""A fit model that uses a table of (x, y) values to interpolate.

    Uses :class:`~scipy.interpolate.interp1d`

    Fitting parameters:
        - integrated intensity ``amplitude`` :math:`A`
        - position of the peak ``center`` :math:`E_0`
        - nominal relaxation time ``tau`` :math:`\tau`
        - stretching exponent ``beta`` :math:`\beta`

    Parameters
    ----------
    xdata : :class:`~numpy:numpy.ndarray`
        X-values to construct the interpolator
    ydata : :class:`~numpy:numpy.ndarray`
        Y-values to construct the interpolator
    interpolator_kind : str
        Interpolator that :class:`~scipy.interpolate.interp1d` should use
    """

    def __init__(self, xdata, ydata, interpolator_kind='linear',
                 prefix='', missing=None, name=None,
                 **kwargs):
        kwargs.update({'prefix': prefix, 'missing': missing})
        self._interpolator = interp1d(xdata, ydata, kind=interpolator_kind)
        self._ydata = ydata
        self._xdata = xdata

        def tabulate(x, amplitude, center):
            if not np.allclose(x, self._xdata):
                raise ValueError("Attempting to fit with experimental xdata\
                                   that does not match the xdata of the model.\
                                   Interpolate before or after.")
            return amplitude * self._ydata
        #     return amplitude * self._interpolator(x - center)
        # def tabulate(x, amplitude):

        super(TabulatedFunctionModel, self).__init__(tabulate, **kwargs)
        self.set_param_hint('amplitude', min=0, value=1)


class MultiPropertyModel(Model):
    r"""A fit model that uses potentially multiple PropertyDicts.

    Parameters
    ----------
    property_groups : list of property groups used to make a model
        Properties used to create a feature vector and a model.
    """

    def __init__(self, property_groups, **kwargs):
        if len({len(pg) for pg in property_groups}) > 1:
            raise ValueError("Property groups must be same length")

        def func(x, **params):
            if not all([np.allclose(x, p.feature_domain)
                        for p in property_groups]):
                raise ValueError("Attempting to fit with experimental xdata\
                                   that does not match the xdata of the model.\
                                   Interpolate before or after.")
            names = [p.name for p in property_groups[0].values()]
            ps = [params[f'p_{i}'] for i in range(len(property_groups))]
            ms = [params[f'scale_{name}'] for name in names]
            cs = [params[f'const_{name}'] for name in names
                  if not isinstance(property_groups[0][name], ScalarProperty)]
            # start with the right proportion of each structure
            mod = sum([ps[i] * pg.feature_vector
                       for i, pg in enumerate(property_groups)])
            # scale each property of the model by the appropriate factor
            scaling = np.concatenate([ms[i]*np.ones(len(p.feature_vector))
                                      for i, p in
                                      enumerate(property_groups[0].values())])
            mod *= scaling
            # finally, add a constant for each property of the model
            mod += np.concatenate([cs[i]*np.ones(len(p.feature_vector))
                                   if not isinstance(p, ScalarProperty)
                                   else np.ones(len(p.feature_vector))
                                   for i, p in
                                   enumerate(property_groups[0].values())
                                   ])
            return mod

        super(MultiPropertyModel, self).__init__(func, **kwargs)
        self.params = self.make_params()
        for i in range(1, len(property_groups)):
            self.params.add(f'p_{i}', vary=True, min=0, max=1,
                            value=1.0/len(property_groups))
        eq = '1-('+'+'.join([f'p_{j}'
                             for j in range(1, len(property_groups))])+')'
        if len(property_groups) == 1:
            self.params.add('p_0', value=1, min=0, max=1)
        else:
            self.params.add('p_0', value=1, min=0, max=1, expr=eq)
        for prop in property_groups[0].values():
            self.params[f'scale_{prop.name}'] = Parameter(value=1)
            if not isinstance(prop, ScalarProperty):
                self.params[f'const_{prop.name}'] = Parameter(value=0)


def model_at_node(node, property_name):
    r"""Generate fit model as a tabulated function with a scaling parameter, plus a flat background.

    Parameters
    ----------
    node : :class:`~idpflex.cnextend.ClusterNodeX`
        One node of the hierarchical :class:`~idpflex.cnextend.Tree`
    property_name : str
        Name of the property to create the model for

    Returns
    -------
    :class:`~lmfit.model.CompositeModel`
        A model composed of a :class:`~idpflex.bayes.TabulatedFunctionModel`
        and a :class:`~lmfit.models.ConstantModel`
    """  # noqa: E501
    p = node[property_name]
    mod = TabulatedFunctionModel(p.x, p.y) + ConstantModel()
    mod.set_param_hint('center', vary=False)
    return mod


def model_at_depth(tree, depth, property_name):
    r"""Generate a fit model at a particular tree depth.

    Parameters
    ----------
    tree : :class:`~idpflex.cnextend.Tree`
        Hierarchical tree
    depth: int
        depth level, starting from the tree's root (depth=0)
    property_name : str
        Name of the property to create the model for

    Returns
    -------
    :class:`~lmfit.model.CompositeModel`
        A model composed of a :class:`~idpflex.bayes.TabulatedFunctionModel`
        for each node plus a :class:`~lmfit.models.ConstantModel` accounting
        for a flat background
    """  # noqa: E501
    mod = ConstantModel()
    for node in tree.nodes_at_depth(depth):
        p = node[property_name]
        m = TabulatedFunctionModel(p.x, p.y, prefix='n{}_'.format(node.id))
        m.set_param_hint('center', vary=False)
        m.set_param_hint('amplitude', value=1.0 / (1 + depth))
        mod += m
    return mod


def fit_at_depth(tree, experiment, property_name, depth):
    r"""Fit at a particular tree depth from the root node.

    Fit experiment against the property stored in the nodes. The fit model
    is generated by :func:`~idpflex.bayes.model_at_depth`

    Parameters
    ----------
    tree : :class:`~idpflex.cnextend.Tree`
        Hierarchical tree
    experiment : :class:`~idpflex.properties.ProfileProperty`
        A property containing the experimental info.
    property_name: str
        The name of the simulated property to compare against experiment
    depth : int
        Fit at this depth

    Returns
    -------
    :class:`~lmfit.model.ModelResult`
        Results of the fit
    """
    mod = model_at_depth(tree, depth, property_name)
    params = mod.make_params()
    return mod.fit(experiment.y,
                   x=experiment.x,
                   weights=1.0 / experiment.e,
                   params=params)


def fit_to_depth(tree, experiment, property_name, max_depth=5):
    r"""Fit at each tree depth from the root node up to a maximum depth.

    Fit experiment against the property stored in the nodes. The fit model
    is generated by :func:`~idpflex.bayes.model_at_depth`

    Parameters
    ----------
    tree : :class:`~idpflex.cnextend.Tree`
        Hierarchical tree
    experiment : :class:`~idpflex.properties.ProfileProperty`
        A property containing the experimental info.
    property_name: str
        The name of the simulated property to compare against experiment
    max_depth : int
        Fit at each depth up to (and including) max_depth

    Returns
    -------
    :py:class:`list`
        A list of :class:`~lmfit.model.ModelResult` items containing the
        fit at each level of the tree up to and including `max_depth`
    """
    # Fit each level of the tree
    return [fit_at_depth(tree, experiment, property_name, depth) for
            depth in range(max_depth + 1)]


def create_at_depth_multiproperty(tree, depth, experiment=None):
    r"""Create a model at a particular tree depth from the root node.

    Parameters
    ----------
    tree : :class:`~idpflex.cnextend.Tree`
        Hierarchical tree
    depth : int
        Fit at this depth
    experiment : PropertyDict, optional
        A PropertyDict containing the experimental data.
        If provided, will use only the keys in the experiment.

    Returns
    -------
    :class:`~lmfit.model.ModelResult`
        Model for the depth
    """
    property_names = experiment.keys() if experiment is not None else None
    pgs = [node.property_group.subset(property_names)
           for node in tree.nodes_at_depth(depth)]
    return MultiPropertyModel(pgs, experiment_property_group=experiment)


def create_to_depth_multiproperty(tree, max_depth, experiment=None):
    r"""Create models to a particular tree depth from the root node.

    Parameters
    ----------
    tree : :class:`~idpflex.cnextend.Tree`
        Hierarchical tree
    max_depth : int
        Fit at each depth up to (and including) max_depth
    experiment : PropertyDict, optional
        A PropertyDict containing the experimental data.

    Returns
    -------
    list of :class:`~lmfit.model.ModelResult`
        Models for each depth
    """
    return [create_at_depth_multiproperty(tree, i, experiment)
            for i in range(max_depth + 1)]


def fit_multiproperty_model(model, experiment, weights=None):
    """Apply a fit to a particular model.

    Parameters
    ----------
    model: :class:`~lmfit.model.ModelResult`
        Model to be fit
    experiment: :class:`~idpflex.properties.PropertyDict`
        Set of experimental properties to be fit.

    Returns
    -------
    :class:`~lmfit.model.ModelResult`
        The fit of the model
    """
    return model.fit(experiment.feature_vector, weights=weights,
                     x=experiment.feature_domain, params=model.params)


def fit_multiproperty_models(models, experiment, weights=None):
    """Apply fitting to a list of models."""
    return [fit_multiproperty_model(model, experiment, weights=weights)
            for model in models]
