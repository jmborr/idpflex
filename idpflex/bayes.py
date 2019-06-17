# from qef.models import TabulatedFunctionModel
from lmfit.models import (Model, ConstantModel, index_of)
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

        def tabulate(x, amplitude, center):
            return amplitude * self._ydata
        #     return amplitude * self._interpolator(x - center)
        # def tabulate(x, amplitude):

        super(TabulatedFunctionModel, self).__init__(tabulate, **kwargs)
        self.set_param_hint('amplitude', min=0, value=1)
        # self.set_param_hint('center', value=0)

    def guess(self, y, x=None, **kwargs):
        r"""Estimate fitting parameters from input data.

        Parameters
        ----------
        y : :class:`~numpy:numpy.ndarray`
            Values to fit to, e.g., SANS or SAXS intensity values
        x : :class:`~numpy:numpy.ndarray`
            independent variable, e.g., momentum transfer

        Returns
        -------
        :class:`~lmfit.parameter.Parameters`
            Parameters with estimated initial values.
        """
        amplitude = 1.0
        center = 0.0
        if x is not None:
            center = x[index_of(y, max(y))]  # assumed peak within domain x
            amplitude = max(y)
        return self.make_params(amplitude=amplitude, center=center)


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

        number_of_constants = len(property_groups[0])
        number_of_structures = len(property_groups)

        def func(x, **params):
            ms = list(params.values())[:number_of_structures]
            cs = list(params.values())[-number_of_constants:]
            mod = sum([ms[i] * pg.feature_vector  # *pg.feature_weights
                       for i, pg in enumerate(property_groups)])
            mod += np.concatenate([cs[i]*np.ones(len(p))  # *p.feature_weights
                                   for i, p in
                                   enumerate(property_groups[0].values())])
            return mod

        super(MultiPropertyModel, self).__init__(func, **kwargs)
        for i in range(len(property_groups)):
            self.set_param_hint(f'group_{i}', min=0, value=1)
        for prop in property_groups[0].values():
            if isinstance(prop, ScalarProperty):
                self.set_param_hint(f'const_{prop.name}', value=0, vary=False)
            else:
                self.set_param_hint(f'const_{prop.name}', value=1)


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
                   # weights=1.0 / experiment.e,
                   weights=experiment.feature_weights,
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


def fit_at_depth_multiproperty(tree, experiment, depth):
    r"""Fit at a particular tree depth from the root node.

    Fit experiment against the properties stored in the nodes.

    Parameters
    ----------
    tree : :class:`~idpflex.cnextend.Tree`
        Hierarchical tree
    experiment : PropertyDict
        A PropertyDict containing the experimental data.
    depth : int
        Fit at this depth

    Returns
    -------
    :class:`~lmfit.model.ModelResult`
        Results of the fit
    """
    property_names = experiment.keys()
    pgs = [node.property_group.subset(property_names)
           for node in tree.nodes_at_depth(depth)]
    m = MultiPropertyModel(pgs)
    params = m.make_params()
    return m.fit(experiment.feature_vector,
                 weights=experiment.feature_weights,
                 x=experiment.feature_vector, params=params)


def fit_to_depth_multiproperty(tree, experiment, max_depth):
    r"""Fit to a particular tree depth from the root node.

    Parameters
    ----------
    tree : :class:`~idpflex.cnextend.Tree`
        Hierarchical tree
    experiment : PropertyDict
        A PropertyDict containing the experimental data.
    max_depth : int
        Fit at each depth up to (and including) max_depth

    Returns
    -------
    :class:`~lmfit.model.ModelResult`
        Results of the fit
    """
    return [fit_at_depth_multiproperty(tree, experiment, i)
            for i in range(max_depth + 1)]
