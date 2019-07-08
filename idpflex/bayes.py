# from qef.models import TabulatedFunctionModel
import warnings
import operator
import numpy as np
from lmfit.models import (Model, ConstantModel)
from lmfit import CompositeModel
from functools import reduce


class TabulatedFunctionModel(Model):
    r"""A fit model that uses a table of (x, y) values to interpolate.

    Uses an individual property's `interpolator` for the interpolation.
    Control of the interpolator can be set using the property's
    `create_interpolator` method.

    Fitting parameters:
        - integrated intensity ``amplitude`` :math:`A`
        - position of the peak ``center`` :math:`E_0`

    Parameters
    ----------
    prop : :class:`~idpflex.properties.ScalarProperty` or :class:`~idpflex.properties.ProfileProperty`
        Property used to create interpolator and model
    """  # noqa: E501

    def __init__(self, prop, prefix='', missing=None, name=None, **kwargs):
        kwargs.update({'prefix': prefix, 'missing': missing, 'name': name})

        def tabulate(x, amplitude, center, intercept, prop=None):
            return amplitude * prop.interpolator(x - center) + intercept

        super().__init__(tabulate, prop=prop, **kwargs)
        self.set_param_hint('amplitude', min=0, value=1)
        self.set_param_hint('center', value=0)
        self.set_param_hint('intercept', value=0)


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
        super().__init__(line, prop=prop, **kwargs)
        self.set_param_hint('slope', value=1, min=0)
        self.set_param_hint('intercept', value=0, min=0)


def create_at_depth(tree, depth, names=None, use_tabulated=False,
                    **subset_kws):
    r"""Create a model at a particular tree depth from the root node.

    Parameters
    ----------
    tree : :class:`~idpflex.cnextend.Tree`
        Hierarchical tree
    depth : int
        Fit at this depth
    names : list or str, optional
        kwarg to pass on when creating subset of property dict
    use_tabulated: bool
        Decide to use an tabulated (interpolated) model or a linear model with
        out interpolation. Useful to "center" data.
    subset_kws : additional args for subset filtering, optional
        kwargs to pass on when creating subset of property dict example
        includes `property_type`

    Returns
    -------
    :class:`~lmfit.model.ModelResult`
        Model for the depth
    """
    pgs = [node.property_group.subset(names or node.property_group.keys(),
                                      **subset_kws)
           for node in tree.nodes_at_depth(depth)]
    return create_models(pgs, use_tabulated)


def create_to_depth(tree, max_depth, names=None,
                    use_tabulated=False, **subset_kws):
    r"""Create models to a particular tree depth from the root node.

    Parameters
    ----------
    tree : :class:`~idpflex.cnextend.Tree`
        Hierarchical tree
    max_depth : int
        Fit at each depth up to (and including) max_depth
    names : list or str, optional
        kwarg to pass on when creating subset of property dict
    use_tabulated: bool
        Decide to use an tabulated (interpolated) model or a linear model with
        out interpolation. Useful to "center" data.
    subset_kws : additional args for subset filtering, optional
        kwargs to pass on when creating subset of property dict example
        includes `property_type`

    Returns
    -------
    list of :class:`~lmfit.model.ModelResult`
        Models for each depth
    """
    return [create_at_depth(tree, i, names=names, **subset_kws)
            for i in range(max_depth + 1)]


def fit_model(model, experiment, params=None, weights=None, method='leastsq'):
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
    result = model.fit(experiment.feature_vector, weights=weights,
                       x=experiment.feature_domain, params=params,
                       method=method)
    # If there are structure probabilites, ensure they sum close to 1
    if any(pname.endswith('prob_c') for pname in result.params):
        ptotal = sum(p.value for p in result.params.values()
                     if p.name.endswith('prob_c'))
        if abs(1 - ptotal) > .05:
            warnings.warn('Fit produced probabilites that did not sum to 1.'
                          f' The probabilies summed to {ptotal}.'
                          ' Recommended action is to refit with different'
                          ' starting parameter values.')
    return result


def fit_models(models, experiment, params_list=None, weights=None,
               method='leastsq'):
    """Apply fitting to a list of models."""
    if params_list is None:
        params_list = [m.make_params() for m in models]
    return [fit_model(model, experiment, params=params,
                      weights=weights, method=method)
            for model, params in zip(models, params_list)]


def create_model(property_group, use_tabulated=False):
    """Create a composite model from a PropertyDict and a set of models.

    Parameters
    ----------
    property_group: :class:`~idpflex.properties.PropertyDict`
        The set of properties used to create a composite model.
    use_tabulated: bool
        Decide to use an tabulated (interpolated) model or a linear model with
        out interpolation. Useful to "center" data.

    Returns
    -------
    :class:`~lmfit.CompositeModel`
        The composite model created by applying the model to the corresponging
        property and concatenating the results.
    """  # noqa: E501
    # Create new model instances or copy model instances
    if use_tabulated:
        model_objs = [TabulatedFunctionModel(prop=p)
                      for p in property_group.values()]
    else:
        model_objs = [LinearModel(prop=p) for p in property_group.values()]
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


def create_models(property_groups, use_tabulated=False):
    """Create a composite model from a list of PropertyDict and a set of models.

    Parameters
    ----------
    property_groups: list of :class:`~idpflex.properties.PropertyDict`
        The set of properties used to create a composite model.
    use_tabulated: bool
        Decide to use an tabulated (interpolated) model or a linear model with
        out interpolation. Useful to "center" data.

    Returns
    -------
    :class:`~lmfit.CompositeModel`
        The composite model created by applying the model to the corresponging
        property and concatenating the results.
    """
    try:
        property_groups = list(property_groups)
    except TypeError:
        property_groups = [property_groups]

    if len(property_groups) == 1:
        return create_model(property_groups[0], use_tabulated)

    def create_submodel(i, property_group):
        submodel = create_model(property_group, use_tabulated)
        submodel = ConstantModel(prefix='proportion_')*submodel
        for component in submodel.components:
            component.prefix = f'struct{i}_' + component.prefix
        return submodel

    model = reduce(operator.add, (create_submodel(i, pg)
                                  for i, pg in enumerate(property_groups)))
    # for each structure calculate a probability using propotions
    proportion_names = [p for p in model.param_names
                        if p.endswith('proportion_c')]
    total_eq = '(' + '+'.join(proportion_names) + ')'
    model.set_param_hint('total', expr=total_eq)
    for p in proportion_names:
        struct = p.partition('_')[0]  # param name struct prefix
        model.set_param_hint(p, min=0, value=1)  # start with equal proportions
        model.set_param_hint(f'{struct}_p', expr=f'{p}/total')

    # For each property, create single parameter without struct prefix that
    # is appropriately scaled and equate the internal parameters
    for param in (param for param in model.param_names
                  for prop in property_groups[0].values()
                  if prop.name in param and not param.startswith('struct0_')):
        pbase = param.partition('_')[-1]  # param name without struct prefix
        model.set_param_hint(pbase, expr=f'struct0_{pbase}/total')
        model.set_param_hint(param, expr=f'struct0_{pbase}')

    return model
