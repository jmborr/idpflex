# from qef.models import TabulatedFunctionModel
from lmfit.models import (Model, ConstantModel)
from lmfit import CompositeModel
from scipy.interpolate import interp1d
import numpy as np
import copy
from itertools import cycle
from functools import reduce
import operator


class TabulatedFunctionModel(Model):
    r"""A fit model that uses a table of (x, y) values to interpolate.

    Uses :class:`~scipy.interpolate.interp1d`

    Fitting parameters:
        - integrated intensity ``amplitude`` :math:`A`
        - position of the peak ``center`` :math:`E_0`

    Parameters
    ----------
    prop : :class:`~idpflex.properties.ScalarProperty` or :class:`~idpflex.properties.ProfileProperty`
        Property used to create interpolator and model
    interpolator_kind : str
        Interpolator that :class:`~scipy.interpolate.interp1d` should use
    """  # noqa: E501

    def __init__(self, prop, interpolator_kind='linear',
                 fill_value='extrapolate', prefix='', missing=None, name=None,
                 **kwargs):
        kwargs.update({'prefix': prefix, 'missing': missing, 'name': name})
        self._interpolator = interp1d(prop.x, prop.y, kind=interpolator_kind,
                                      fill_value=fill_value)
        self.prop = prop

        def tabulate(x, amplitude, center, prop=None):
            return amplitude * prop.interpolator(x - center)

        super(TabulatedFunctionModel, self).__init__(tabulate, prop=prop,
                                                     **kwargs)
        self.set_param_hint('amplitude', min=0, value=1)
        self.set_param_hint('center', value=0)


class ConstantVectorModel(Model):
    r"""A fit model that fits :math:`scale*prop.y = exp`.

    Fitting parameters:
        - scaling factor ``scale``

    Parameters
    ----------
    prop : :class:`~idpflex.properties.ScalarProperty` or :class:`~idpflex.properties.ProfileProperty`
        Property used to create interpolator and model
    """  # noqa: E501

    def __init__(self, prop=None, **kwargs):
        def constant_vector(x, scale, prop=None):
            if not set(x).issuperset(prop.feature_domain):
                raise ValueError('The domain of the experiment does not align '
                                 'with the domain of the profile being fitted.'
                                 ' Interpolate before creating the model.')
            return scale*prop.y
        super(ConstantVectorModel, self).__init__(constant_vector, prop=prop,
                                                  **kwargs)
        self.set_param_hint('scale', value=1, min=0)


class LinearModel(Model):
    r"""A fit model that fits :math:`m*prop.y + b = exp`.

    Fitting parameters:
        - slope ``slope``
        - intercept ``intercept``

    Parameters
    ----------
    prop : :class:`~idpflex.properties.ScalarProperty` or :class:`~idpflex.properties.ProfileProperty`
        Property used to create interpolator and model
    """  # noqa: E501

    def __init__(self, prop=None, **kwargs):
        def line(x, slope, intercept, prop=None):
            if not set(x).issuperset(prop.feature_domain):
                raise ValueError('The domain of the experiment does not align '
                                 'with the domain of the profile being fitted.'
                                 ' Interpolate before creating the model.')
            return slope*prop.y + intercept
        super(LinearModel, self).__init__(line, prop=prop,
                                          **kwargs)
        self.set_param_hint('slope', value=1, min=0)
        self.set_param_hint('intercept', value=0, min=0)


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
    mod = TabulatedFunctionModel(prop=p) + ConstantModel()
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
        m = TabulatedFunctionModel(p, prefix='n{}_'.format(node.id))
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


def create_at_depth_multiproperty(tree, depth, models=LinearModel,
                                  experiment=None):
    r"""Create a model at a particular tree depth from the root node.

    Parameters
    ----------
    tree : :class:`~idpflex.cnextend.Tree`
        Hierarchical tree
    depth : int
        Fit at this depth
    models: (list of) class/subclasses/instances of lmfit.Model
        The models to apply to each property. If only one model, apply it to
        all properties. The model's function must have an independent
        parameter x and a keyword argument prop=None.
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
    return create_model_from_property_groups(pgs, models)


def create_to_depth_multiproperty(tree, max_depth, models=LinearModel,
                                  experiment=None):
    r"""Create models to a particular tree depth from the root node.

    Parameters
    ----------
    tree : :class:`~idpflex.cnextend.Tree`
        Hierarchical tree
    max_depth : int
        Fit at each depth up to (and including) max_depth
    models: (list of) class/subclasses/instances of lmfit.Model
        The models to apply to each property. If only one model, apply it to
        all properties. The model's function must have an independent
        parameter x and a keyword argument prop=None.
    experiment : PropertyDict, optional
        A PropertyDict containing the experimental data.

    Returns
    -------
    list of :class:`~lmfit.model.ModelResult`
        Models for each depth
    """
    return [create_at_depth_multiproperty(tree, i, models, experiment)
            for i in range(max_depth + 1)]


def fit_multiproperty_model(model, experiment, params=None, weights=None,
                            method='leastsq'):
    """Apply a fit to a particular model.

    Parameters
    ----------
    model: :class:`~lmfit.model.ModelResult`
        Model to be fit
    experiment: :class:`~idpflex.properties.PropertyDict`
        Set of experimental properties to be fit.
    params: Parameters
        Parameters of the model to be used. Can default to model.make_params()
    weights: numpy.ndarray, optional
        Array of weights to be used for fitting
    method: str, optional
        Choice of which fitting method to use with lmfit. Defaults to 'leastsq'
        but can choose methods such as 'differential_evolution' to find global
        minimizations for the parameters.

    Returns
    -------
    :class:`~lmfit.model.ModelResult`
        The fit of the model
    """
    if params is None:
        params = model.make_params()
    return model.fit(experiment.feature_vector, weights=weights,
                     x=experiment.feature_domain, params=params,
                     method=method)


def fit_multiproperty_models(models, experiment, params_list=None,
                             weights=None, method='leastsq'):
    """Apply fitting to a list of models."""
    if params_list is None:
        params_list = [m.make_params() for m in models]
    return [fit_multiproperty_model(model, experiment, params=params,
                                    weights=weights, method=method)
            for model, params in zip(models, params_list)]


def _create_model_from_property_group(property_group, models):
    """Create a composite model from a PropertyDict and a set of models.

    Parameters
    ----------
    property_group: :class:`~idpflex.properties.PropertyDict`
        The set of properties used to create a composite model.
    models: (list of) class/subclasses/instances of lmfit.Model
        The models to apply to each property. If only one model, apply it to
        all properties. The model's function must have an independent
        parameter x and a keyword argument prop=None.


    Returns
    -------
    :class:`~lmfit.CompositeModel`
        The composite model created by applying the model to the corresponging
        property and concatenating the results.
    """  # noqa: E501
    if not isinstance(models, list):
        models = [models]
    elif len(models) != 1 and len(models) != len(property_group):
        raise ValueError(f'Number of Properties {len(property_group)} '
                         f'and number of models {len(models)} do not match '
                         'and more than one model was provided.')
    # Create new model instances or copy model instances
    model_objs = [m(prop=p) if isinstance(m, type) else copy.deepcopy(m)
                  for m, p in zip(cycle(models), property_group.values())]
    # Prefix all models with the associated property name
    # Set the model's function prop arg to use the property
    for i, p in enumerate(property_group.values()):
        model_objs[i].opts['prop'] = p
        model_objs[i].prefix = p.name + '_'
    # Reduce models to a single composite model
    return reduce(lambda joined_model, m:
                  CompositeModel(joined_model, m, lambda l, r:
                                 np.concatenate([np.atleast_1d(l),
                                                 np.atleast_1d(r)])),
                  model_objs)


def create_model_from_property_groups(property_groups, models):
    """Create a composite model from a list of PropertyDict and a set of models.

    Parameters
    ----------
    property_groups: list of :class:`~idpflex.properties.PropertyDict`
        The set of properties used to create a composite model.
    models: (list of) class/subclasses/instances of lmfit.Model
        The models to apply to each property. If only one model, apply it to
        all properties. The model's function must have an independent
        parameter x and a keyword argument prop=None.

    Returns
    -------
    :class:`~lmfit.CompositeModel`
        The composite model created by applying the model to the corresponging
        property and concatenating the results.
    """
    if not isinstance(property_groups, list):
        property_groups = [property_groups]

    if len(property_groups) == 1:
        return _create_model_from_property_group(property_groups[0], models)

    def create_submodel(i, property_group):
        submodel = _create_model_from_property_group(property_group, models)
        submodel = ConstantModel(prefix='prob_')*submodel
        for component in submodel.components:
            component.prefix = f'struct{i}_' + component.prefix
        return submodel

    model = reduce(operator.add, (create_submodel(i, pg)
                                  for i, pg in enumerate(property_groups)))
    # For each property, create single parameter without struct prefix
    for param in (param for param in model.param_names
                  for prop in property_groups[0].values()
                  if prop.name in param and not param.startswith('struct0_')):
        pbase = param.partition('_')[-1]  # param name without struct prefix
        model.set_param_hint(pbase, expr='struct0_'+pbase)
        model.set_param_hint(param, expr='struct0_'+pbase)
    # Bound probabilites and force sum to 1
    prob_names = [p for p in model.param_names if p.endswith('prob_c')]
    eq = '1-('+'+'.join(prob_names[1:])+')'
    model.set_param_hint('struct0_prob_c', min=0, max=1, expr=eq)
    for p in prob_names:
        model.set_param_hint(p, min=0, max=1)
    return model
