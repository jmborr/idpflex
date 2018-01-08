"""
Bayes fit of the levels of a clustering tree for containing a property
against an experimental profile.
"""

from __future__ import print_function, absolute_import

# from qef.models import TabulatedFunctionModel
from lmfit.models import (Model, ConstantModel, index_of)
from scipy.interpolate import interp1d


class TabulatedFunctionModel(Model):

    def __init__(self, xdata, ydata, interpolator_kind='linear',
                 prefix='', missing=None, name=None,
                 **kwargs):
        kwargs.update({'prefix': prefix, 'missing': missing})
        self._interpolator = interp1d(xdata, ydata, kind=interpolator_kind)

        def tabulate(x, amplitude, center):
            return amplitude * self._interpolator(x - center)

        super(TabulatedFunctionModel, self).__init__(tabulate, **kwargs)
        self.set_param_hint('amplitude', value=1)
        self.set_param_hint('center', value=0)

    def guess(self, y, x=None, **kwargs):
        amplitude = 1.0
        center = 0.0
        if x is not None:
            center = x[index_of(y, max(y))]  # assumed peak within domain x
            amplitude = max(y)
        return self.make_params(amplitude=amplitude, center=center)


def model_at_node(node, property_name):
    r"""Fit model as a tabulated function with a scaling parameter, plus a
    flat background.

    Parameters
    ----------
    node : :class:`~idpflex.cnextend.ClusterNodeX`
        Any node of the hierarchical :class:`~idpflex.cnextend.Tree`
    property_name : str
        Name of the property to create the model for.

    Returns
    -------
    :class:`~lmfit.model.CompositeModel`
        A model composed of a :class:`~qef.models.tabulatedfunction.TabulatedFunctionModel`
        and a :class:`~lmfit.models.ConstantModel`
    """  # noqa: E501
    p = node[property_name]
    mod = TabulatedFunctionModel(p.x, p.y) + ConstantModel()
    mod.set_param_hint('center', vary=False)
    return mod


def model_at_depth(tree, depth, property_name):
    r"""Fit model for the nodes of the tree at a particular depth

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
    c : :class:`~lmfit.model.CompositeModel`
        A model composed of a :class:`~qef.models.tabulatedfunction.TabulatedFunctionModel`
        for each node plus a :class:`~lmfit.models.ConstantModel`
    """  # noqa: E501
    mod = ConstantModel()
    for node in tree.nodes_at_depth(depth):
        p = node[property_name]
        m = TabulatedFunctionModel(p.x, p.y, prefix='n{}_'.format(node.id))
        m.set_param_hint('center', vary=False)
        m.set_param_hint('amplitude', value=1.0 / (1 + depth))
        mod += m
    return mod


def fit_to_depth(tree, experiment, property_name, max_depth=5):
        r"""Fit at each tree depth from the root node up to a maximum depth.

        Fit experiment against the property stored in the nodes. The model
        is generated by :function:`~idpflex.bayes.model_at_depth`.

        Parameters
        ----------
        tree : :class:`~idpflex.cnextend.Tree`
            Hierarchical tree
        experiment : :class:`~idpflex.properties.ProfileProperty`
            A property containing the experimental info.
        property_name: str
            The name of the simulated property to compare against experiment.
        max_depth : int
            Fit at each depth up to (and including) max_depth.

        Returns
        -------
        fits_output : :py:class:`list`
            A list of :class:`~lmfit.model.ModelResult` items containing the
            fit at each level of the tree up to and including `max_depth`.
        """

        # Fit each level of the tree
        fits_output = list()
        # fits_prob = list()
        for depth in range(max_depth + 1):
            print('Fitting at depth = {}'.format(depth))
            mod = model_at_depth(tree, depth, property_name)
            params = mod.make_params()
            fits_output.append(mod.fit(experiment.y,
                                       x=experiment.x,
                                       weights=1.0 / experiment.e,
                                       params=params))
        return fits_output
