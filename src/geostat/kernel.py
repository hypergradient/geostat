from dataclasses import dataclass
from typing import Callable, List, Union
import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag

from .op import Op
from .metric import Euclidean, PerAxisDist2, ed
from .param import ppp, upp, bpp, ppp_list
from .mean import get_trend_coefs

__all__ = ['Kernel']

class Kernel(Op):
    """
    Kernel class representing a covariance function for Gaussian Processes (GPs).

    The `Kernel` class defines the structure of a GP's covariance function. It supports operations
    such as addition and multiplication with other kernels, enabling the construction of more
    complex kernels through combinations. The class also provides methods for computing the
    covariance matrix between sets of locations.

    Parameters:
        fa (dict or callable):
            A dictionary or callable representing the functional attributes of the kernel.
        autoinputs (dict):
            A dictionary specifying the automatic input mappings for the kernel. If 'offset',
            'locs1', or 'locs2' keys are not present, they are added with default values.

    Examples
    --------
    Creating and using a `Kernel` object:

    ```
    from geostat.kernel import Kernel
    kernel = Kernel(fa={'alpha': 1.0}, autoinputs={})
    locs1 = np.array([[0.0, 0.0], [1.0, 1.0]])
    locs2 = np.array([[2.0, 2.0], [3.0, 3.0]])
    covariance_matrix = kernel({'locs1': locs1, 'locs2': locs2})
    ```

    Combining two kernels using addition and multiplication:

    ```
    kernel1 = Kernel(fa={'alpha': 1.0}, autoinputs={})
    kernel2 = Kernel(fa={'range': 0.5}, autoinputs={})
    combined_kernel = kernel1 + kernel2  # Adding kernels
    product_kernel = kernel1 * kernel2   # Multiplying kernels
    ```

    Notes
    -----
    - The `__call__` method computes the covariance matrix between two sets of locations 
        (`locs1` and `locs2`) and ensures the result is correctly broadcasted to the appropriate shape.
    - The `report` method provides a summary of the kernel's parameters and their values.
    - This class serves as a base class for more specialized kernel functions in GP modeling.
    """

    def __init__(self, fa, autoinputs):
        if 'offset' not in autoinputs: autoinputs['offset'] = 'offset'
        if 'locs1' not in autoinputs: autoinputs['locs1'] = 'locs1'
        if 'locs2' not in autoinputs: autoinputs['locs2'] = 'locs2'
        super().__init__(fa, autoinputs)

    def __add__(self, other):
        if other is None:
            return self
        else:
            return Stack([self]) + other

    def __mul__(self, other):
        return Product([self]) * other

    def call(self, e):
        """
        Returns tuple `(mean, covariance)` for locations.
        Return values may be unbroadcasted.
        """
        pass

    def __call__(self, e):
        """
        Returns tuple `(mean, covariance)` for locations.
        Return values have correct shapes.
        """
        C = self.call(e)
        if C is None:
            C = 0.0
        n1 = e['locs1'].shape[0]
        n2 = e['locs2'].shape[0]
        C = jnp.broadcast_to(C, (n1, n2))
        return C

    # def __call__(self, e):
    #     """
    #     Returns tuple `(mean, covariance)` for locations.
    #     Return values have correct shapes.
    #     """
    #     C = self.call(e)
    #     if C is None: C = 0.
    #     n1 = tf.shape(e['locs1'])[0]
    #     n2 = tf.shape(e['locs2'])[0]
    #     C = tf.broadcast_to(C, [n1, n2])
    #     return C

    def report(self):
        string = ', '.join('%s %4.2f' % (v.name, p[v.name]) for v in self.vars())
        return '[' + string + ']'

class TrendPrior(Kernel):
    """
    TrendPrior class representing a kernel with a linear trend prior for Gaussian Processes (GPs).

    The `TrendPrior` class defines a kernel that incorporates a linear trend in the covariance structure
    using a provided featurizer function. This kernel is particularly useful when the underlying process
    is expected to exhibit a trend that can be captured by the specified features.

    Parameters:
        featurizer (Callable):
            A function that takes input locations and returns a feature matrix. This function defines
            the features used in the trend prior.
        alpha (float or tf.Variable):
            The scaling factor (weight) applied to the trend prior.

    Examples
    --------
    Defining a TrendPrior kernel with a custom featurizer:

    ```
    import tensorflow as tf
    from geostat.kernel import TrendPrior

    # Define a simple featurizer function
    def simple_featurizer(x):
        return x, x**2

    alpha = 0.5
    trend_prior_kernel = TrendPrior(featurizer=simple_featurizer, alpha=alpha)
    
    locs1 = tf.constant([[1.0], [2.0], [3.0]])
    locs2 = tf.constant([[1.5], [2.5], [3.5]])
    covariance_matrix = trend_prior_kernel({'locs1': locs1, 'locs2': locs2, 'alpha': alpha})
    ```

    Notes
    -----
    - The `call` method computes the covariance matrix using the features generated by the featurizer
        function and scales it by the `alpha` parameter.
    - The `vars` method returns the parameter dictionary for `alpha` using the `ppp` function.
    - The `TrendPrior` kernel is typically used when the GP model needs to account for linear or 
        polynomial trends in the data.
    """

    def __init__(self, featurizer, alpha):
        fa = dict(alpha=alpha)
        self.featurizer = featurizer
        super().__init__(fa, dict(locs1='locs1', locs2='locs2'))

    def vars(self):
        return ppp(self.fa['alpha'])

    def call(self, e):
        F1 = jnp.array(self.featurizer(e['locs1']), jnp.float32)
        F2 = jnp.array(self.featurizer(e['locs2']), jnp.float32)
        return e['alpha'] * jnp.einsum('ba,ca->bc', F1, F2)

def scale_to_metric(scale, metric):
    assert scale is None or metric is None
    if metric is None:
        if scale is None:
            metric = 'euclidean'
        else:
            metric = Euclidean(scale)
    return metric

def scale_to_metric_2(e, metric):
    scale = [e['xscale'], e['yscale'], e['zscale']]
    assert scale is None or metric is None
    if metric is None:
        if scale is None:
            metric = 'euclidean'
        else:
            metric = Euclidean(scale)
    return metric

class Constant(Kernel):
    """
    Constant kernel class for Gaussian Processes (GPs).

    The `Constant` class defines a simple kernel that produces a constant covariance value across
    all pairs of input locations. This kernel is typically used to represent a baseline level of
    variance (sill) in the GP model.

    Parameters:
        sill (float or tf.Variable):
            The constant value representing the sill (baseline variance) of the kernel.

    Examples
    --------
    Creating and using a `Constant` kernel:

    ```
    from geostat.kernel import Constant

    # Create a Constant kernel with a sill value of 2.0
    constant_kernel = Constant(sill=2.0)
    
    locs1 = np.array([[0.0], [1.0], [2.0]])
    locs2 = np.array([[0.0], [1.0], [2.0]])
    covariance_matrix = constant_kernel({'locs1': locs1, 'locs2': locs2, 'sill': 2.0})
    ```

    Notes
    -----
    - The `call` method returns the constant value specified by `sill` for all pairs of input locations.
    - The `vars` method returns the parameter dictionary for `sill` using the `ppp` function.
    - The `Constant` kernel is useful when you want to add a fixed variance component to your GP model.
    """

    def __init__(self, sill):
        fa = dict(sill=sill)
        super().__init__(fa, dict())

    def vars(self):
        return ppp(self.fa['sill'])

    def call(self, e):
        return e['sill']

class SquaredExponential(Kernel):
    """
    SquaredExponential kernel class for Gaussian Processes (GPs).

    The `SquaredExponential` class defines a widely used kernel that models smooth and continuous 
    covariance structures. It is parameterized by a sill (variance) and a range (length scale) 
    and can optionally use a metric for scaling.

    Parameters:
        sill (float or tf.Variable):
            The variance (sill) of the kernel, representing the maximum covariance value.
        range (float or tf.Variable):
            The length scale parameter that controls how quickly the covariance decreases 
            with distance.
        scale (optional):
            An optional scale parameter that can be used to modify the metric. Default is None.
        metric (optional):
            An optional metric used for distance calculation. Default is None.

    Examples
    --------
    Creating and using a `SquaredExponential` kernel:

    ```
    from geostat.kernel import SquaredExponential

    # Create a SquaredExponential kernel with a sill of 1.0 and a range of 2.0
    sq_exp_kernel = SquaredExponential(sill=1.0, range=2.0)
    
    locs1 = np.array([[0.0], [1.0], [2.0]])
    locs2 = np.array([[0.0], [1.0], [2.0]])
    covariance_matrix = sq_exp_kernel({'locs1': locs1, 'locs2': locs2, 'sill': 1.0, 'range': 2.0})
    ```

    Notes
    -----
    - The `call` method computes the covariance matrix using the squared exponential formula:
        \\( C(x, x') = \\text{sill} \cdot \exp\left(-0.5 \\frac{d^2}{\\text{range}^2}\\right) \\),
        where \\(d^2\\) is the squared distance between `locs1` and `locs2`.
    - The `vars` method returns the parameter dictionary for both `sill` and `range` using the `ppp` function.
    - This kernel is appropriate for modeling smooth, continuous processes with no abrupt changes.
    """

    def __init__(self, sill, range, scale=None, metric=None):
        fa = dict(sill=sill, range=range)
        autoinputs = scale_to_metric(scale, metric)
        super().__init__(fa, dict(d2=autoinputs))

    def vars(self):
        return ppp(self.fa['sill']) | ppp(self.fa['range'])

    def call(self, e):
        return e['sill'] * jnp.exp(-0.5 * e['d2'] / jnp.square(e['range']))

class GammaExponential(Kernel):
    """
    GammaExponential kernel class for Gaussian Processes (GPs).

    The `GammaExponential` class defines a kernel that generalizes the Squared Exponential kernel by introducing
    a gamma parameter, allowing for greater flexibility in modeling covariance structures. It can capture processes
    with varying degrees of smoothness, depending on the value of `gamma`.

    Parameters:
        range (float or tf.Variable):
            The length scale parameter that controls how quickly the covariance decreases with distance.
        sill (float or tf.Variable):
            The variance (sill) of the kernel, representing the maximum covariance value.
        gamma (float or tf.Variable):
            The smoothness parameter. A value of 1 results in the standard exponential kernel, while a value of 2 
            recovers the Squared Exponential kernel. Values between 0 and 2 adjust the smoothness of the kernel.
        scale (optional):
            An optional scale parameter that can be used to modify the metric. Default is None.
        metric (optional):
            An optional metric used for distance calculation. Default is None.

    Examples
    --------
    Creating and using a `GammaExponential` kernel:

    ```
    from geostat.kernel import GammaExponential

    # Create a GammaExponential kernel with sill=1.0, range=2.0, and gamma=1.5
    gamma_exp_kernel = GammaExponential(range=2.0, sill=1.0, gamma=1.5)
    
    locs1 = np.array([[0.0], [1.0], [2.0]])
    locs2 = np.array([[0.0], [1.0], [2.0]])
    covariance_matrix = gamma_exp_kernel({'locs1': locs1, 'locs2': locs2, 'sill': 1.0, 'range': 2.0, 'gamma': 1.5})
    ```

    Notes
    -----
    - The `call` method computes the covariance matrix using the gamma-exponential formula:
        \\( C(x, x') = \\text{sill} \cdot \exp\left(-\left(\\frac{d^2}{\\text{range}^2}\\right)^{\\text{gamma} / 2}\\right) \\),
        where \\(d^2\\) is the squared distance between `locs1` and `locs2`.
    - The `vars` method returns the parameter dictionary for `sill`, `range`, and `gamma` using the `ppp` and `bpp` functions.
    - The `GammaExponential` kernel provides a more flexible covariance structure than the Squared Exponential kernel,
        allowing for varying degrees of smoothness.
    """

    def __init__(self, range, sill, gamma, scale=None, metric=None):
        fa = dict(sill=sill, range=range, gamma=gamma, scale=scale)
        autoinputs = scale_to_metric(scale, metric)
        super().__init__(fa, dict(d2=autoinputs))

    def vars(self):
        return ppp(self.fa['sill']) | ppp(self.fa['range']) | bpp(self.fa['gamma'], 0., 2.)

    def call(self, e):
        return e['sill'] * gamma_exp(e['d2'] / jnp.square(e['range']), e['gamma'])

# @tf.custom_gradient
def ramp(x):
    ax = tf.abs(x)
    def grad(upstream):
        return upstream * tf.where(ax < 1., -tf.sign(x), 0.)
    return tf.maximum(0., 1. - ax), grad

class Ramp(Kernel):
    """
    Ramp kernel class for Gaussian Processes (GPs).

    The `Ramp` class defines a kernel that produces a covariance structure resembling a "ramp" function.
    It is characterized by a sill (variance) and a range (length scale) and can optionally use a metric for scaling.

    Parameters:
        range (float or tf.Variable):
            The length scale parameter that controls how quickly the covariance decreases with distance.
        sill (float or tf.Variable):
            The variance (sill) of the kernel, representing the maximum covariance value.
        scale (optional):
            An optional scale parameter that can be used to modify the metric. Default is None.
        metric (optional):
            An optional metric used for distance calculation. Default is None.

    Examples
    --------
    Creating and using a `Ramp` kernel:

    ```
    from geostat.kernel import Ramp

    # Create a Ramp kernel with sill=1.0 and range=2.0
    ramp_kernel = Ramp(range=2.0, sill=1.0)
    
    locs1 = np.array([[0.0], [1.0], [2.0]])
    locs2 = np.array([[0.0], [1.0], [2.0]])
    covariance_matrix = ramp_kernel({'locs1': locs1, 'locs2': locs2, 'sill': 1.0, 'range': 2.0})
    ```

    Notes
    -----
    - The `call` method computes the covariance matrix using the ramp function:
        \\( C(x, x') = \\text{sill} \cdot \\text{ramp}\left(\\frac{\sqrt{d^2}}{\\text{range}}\\right) \\),
        where \\(d^2\\) is the squared distance between `locs1` and `locs2`.
    - The `vars` method returns the parameter dictionary for both `sill` and `range` using the `ppp` function.
    - The `Ramp` kernel can be used in cases where the covariance structure exhibits a linear decay with increasing distance.
    """

    def __init__(self, range, sill, scale=None, metric=None):
        fa = dict(sill=sill, range=range, scale=scale)
        autoinputs = scale_to_metric(scale, metric)
        super().__init__(fa, dict(d2=autoinputs))

    def vars(self):
        return ppp(self.fa['sill']) | ppp(self.fa['range'])

    def call(self, e):
        return e['sill'] * ramp(tf.sqrt(e['d2']) / e['range'])

# @tf.custom_gradient
# def rampstack(x, sills, ranges):
#     """
#     `x` has arbitrary shape [...].
#     `sills` and `ranges` both have shape [K].
#     """
#     ax = ed(tf.abs(x)) # [..., 1]
#     y = sills * tf.maximum(0., 1. - ax / ranges) # [..., K]
#     def grad(upstream):
#         ax = ed(tf.abs(x)) # [..., 1]
#         y = sills * tf.maximum(0., 1. - ax / ranges) # [..., K]
#         K = tf.shape(sills)[0]
#         grad_x = upstream * tf.reduce_sum(tf.where(ax < ranges, -tf.sign(ed(x)), 0.), -1) # [...]
#         grad_sills = tf.reduce_sum(tf.reshape(ed(upstream) * y, [-1, K]), 0) # [K}
#         grad_ranges = tf.where(ax < ranges, sills * ax / tf.square(ranges), 0.) # [..., K}
#         grad_ranges = tf.reduce_sum(tf.reshape(ed(upstream) * grad_ranges, [-1, K]), 0) # [K]
#         return grad_x, grad_sills, grad_ranges
#     return tf.reduce_sum(y, -1), grad

# @tf.custom_gradient
def rampstack(x, sills, ranges):
    """
    `x` has arbitrary shape [...], but must be non-negative.
    `sills` and `ranges` both have shape [K].
    """
    ax = ed(tf.abs(x)) # [..., 1]
    y = sills * tf.maximum(0., 1. - ax / ranges) # [..., K]
    def grad(upstream):
        ax = ed(tf.abs(x)) # [..., 1]
        y = sills * tf.maximum(0., 1. - ax / ranges) # [..., K]
        K = tf.shape(sills)[0]
        small = ax < ranges
        grad_x = upstream * tf.reduce_sum(tf.where(small, -tf.sign(ed(x)) * (sills / ranges), 0.), -1) # [...]
        grad_sills = tf.einsum('ak,a->k', tf.reshape(y, [-1, K]), tf.reshape(upstream, [-1]))
        grad_ranges = tf.where(small, ax * (sills / tf.square(ranges)), 0.) # [..., K}
        grad_ranges = tf.einsum('ak,a->k', tf.reshape(grad_ranges, [-1, K]), tf.reshape(upstream, [-1]))
        return grad_x, grad_sills, grad_ranges
    return tf.reduce_sum(y, -1), grad

class RampStack(Kernel):
    """
    RampStack kernel class for Gaussian Processes (GPs).

    The `RampStack` class defines a kernel that extends the standard `Ramp` kernel by allowing for multiple 
    sills and ranges, effectively creating a "stacked" ramp function. This kernel can capture more complex 
    covariance structures with multiple levels of decay.

    Parameters:
        range (list or tf.Variable):
            A list or TensorFlow variable representing the length scale parameters that control how quickly 
            the covariance decreases with distance.
        sill (list or tf.Variable):
            A list or TensorFlow variable representing the variance (sill) values for each ramp component.
        scale (optional):
            An optional scale parameter that can be used to modify the metric. Default is None.
        metric (optional):
            An optional metric used for distance calculation. Default is None.

    Examples
    --------
    Creating and using a `RampStack` kernel:

    ```
    from geostat.kernel import RampStack

    # Create a RampStack kernel with multiple sills and ranges
    ramp_stack_kernel = RampStack(range=[2.0, 3.0], sill=[1.0, 0.5])
    
    locs1 = np.array([[0.0], [1.0], [2.0]])
    locs2 = np.array([[0.0], [1.0], [2.0]])
    covariance_matrix = ramp_stack_kernel({'locs1': locs1, 'locs2': locs2, 'sill': [1.0, 0.5], 'range': [2.0, 3.0]})
    ```

    Notes
    -----
    - The `call` method computes the covariance matrix using the `rampstack` function, which applies 
        multiple ramp functions based on the provided `sill` and `range` values for each component.
    - The `vars` method returns the parameter dictionary for both `sill` and `range` using the `ppp_list` function.
    - The `RampStack` kernel is useful for modeling complex processes that exhibit multiple levels of variability 
        or changes in smoothness at different scales.
    """

    def __init__(self, range, sill, scale=None, metric=None):
        fa = dict(sill=sill, range=range, scale=scale)
        autoinputs = scale_to_metric(scale, metric)
        super().__init__(fa, dict(d2=autoinputs))

    def vars(self):
        return ppp_list(self.fa['sill']) | ppp_list(self.fa['range'])

    def call(self, e):
        if isinstance(e['sill'], (tuple, list)):
            e['sill'] = tf.stack(e['sill'])
        if isinstance(e['range'], (tuple, list)):
            e['range'] = tf.stack(e['range'])

        return rampstack(tf.sqrt(e['d2']), e['sill'], e['range'])

# @tf.recompute_grad
def smooth_convex(x, sills, ranges):
    """
    `x` has arbitrary shape [...], but must be non-negative.
    `sills` and `ranges` both have shape [K].
    """
    r2 = ranges
    r1 = tf.pad(ranges[:-1], [[1, 0]])
    ex = ed(x)
    ax = tf.abs(ex)
    rx = ax / r2 - 1.

    c1 = 2. / (r1 + r2)
    c2 = 1. / (1. - tf.square(r1/r2))

    # i1 = tf.cast(ax <= r1, tf.float32) # Indicates x <= r1.
    # i2 = tf.cast(ax <= r2, tf.float32) * (1. - i1) # Indicates r1 < x <= r2.
    # v = i1 * (1. - c1 * ax) + i2 * c2 * tf.square(rx)

    v = tf.where(ax <= r1, 1. - c1 * ax, c2 * tf.square(rx))
    v = tf.where(ax <= r2, v, 0.)

    y = tf.einsum('...k,k->...', v, sills)
    return y

# @tf.custom_gradient
def smooth_convex_grad(x, sills, ranges):
    """
    `x` has arbitrary shape [...], but must be non-negative.
    `sills` and `ranges` both have shape [K].
    """
    r2 = ranges
    r1 = tf.pad(ranges[:-1], [[1, 0]])
    ex = ed(x)
    ax = tf.abs(ex)
    rx = ax / r2 - 1.

    c1 = 2. / (r1 + r2)
    c2 = 1. / (1. - tf.square(r1/r2))

    v = tf.where(ax <= r1, 1. - c1 * ax, c2 * tf.square(rx))
    v = tf.where(ax <= r2, v, 0.)

    y = tf.einsum('...k,k->...', v, sills)

    def grad(upstream):
        r2 = ranges
        r1 = tf.pad(ranges[:-1], [[1, 0]])
        ex = ed(x)
        ax = tf.abs(ex)
        rx = ax / r2 - 1.
        i1 = tf.cast(ax <= r1, tf.float32) # Indicates x <= r1.
        i2 = tf.cast(ax <= r2, tf.float32) * (1. - i1) # Indicates r1 < x <= r2.

        c1 = 2. / (r1 + r2)
        c2 = 1. / (1. - tf.square(r1 / r2))
        c3 = 1. / (r2 - tf.square(r1) / r2)

        v = i1 * (1. - c1 * ax) + i2 * c2 * tf.square(rx)

        sx = tf.sign(ex)

        K = tf.shape(sills)[0]
        gx = sx * sills * (i1 * -c1 + i2 * rx * (2 * c3))
        grad_x = upstream * tf.reduce_sum(gx, -1) # [...]

        grad_sills = tf.einsum('ak,a->k', tf.reshape(v, [-1, K]), tf.reshape(upstream, [-1]))

        u = 2 / tf.square(r1 + r2) * ax * i1
        yr1 = u + i2 * tf.square(rx * c3) * 2 * r1
        yr2 = u - 2 * i2 * (rx * c3 + tf.square(rx * c2) / r2)
        yr1 = sills * tf.reshape(yr1, [-1, K])
        yr2 = sills * tf.reshape(yr2, [-1, K])
        yr = tf.pad(yr1[:, 1:], [[0, 0], [0, 1]]) + yr2
        grad_ranges = tf.einsum('ak,a->k', yr, tf.reshape(upstream, [-1]))

        return grad_x, grad_sills, grad_ranges

    return y, grad

class SmoothConvex(Kernel):
    """
    SmoothConvex kernel class for Gaussian Processes (GPs).

    The `SmoothConvex` class defines a kernel that produces a smooth and convex covariance structure. 
    It allows for multiple sills and ranges, enabling a more complex representation of covariance that 
    smoothly transitions across different scales.

    Parameters:
        range (list or tf.Variable):
            A list or TensorFlow variable representing the length scale parameters that control how 
            quickly the covariance decreases with distance.
        sill (list or tf.Variable):
            A list or TensorFlow variable representing the variance (sill) values for each smooth convex component.
        scale (optional):
            An optional scale parameter that can be used to modify the metric. Default is None.
        metric (optional):
            An optional metric used for distance calculation. Default is None.

    Examples
    --------
    Creating and using a `SmoothConvex` kernel:

    ```
    from geostat.kernel import SmoothConvex

    # Create a SmoothConvex kernel with multiple sills and ranges
    smooth_convex_kernel = SmoothConvex(range=[2.0, 3.0], sill=[1.0, 0.5])
    
    locs1 = np.array([[0.0], [1.0], [2.0]])
    locs2 = np.array([[0.0], [1.0], [2.0]])
    covariance_matrix = smooth_convex_kernel({'locs1': locs1, 'locs2': locs2, 'sill': [1.0, 0.5], 'range': [2.0, 3.0]})
    ```

    Notes
    -----
    - The `call` method computes the covariance matrix using the `smooth_convex` function, which applies 
        multiple convex functions based on the provided `sill` and `range` values for each component.
    - The `vars` method returns the parameter dictionary for both `sill` and `range` using the `ppp_list` function.
    - The `SmoothConvex` kernel is useful for modeling processes that require smooth transitions and convexity 
        in their covariance structure across different scales.
    """

    def __init__(self, range, sill, scale=None, metric=None):
        fa = dict(sill=sill, range=range, scale=scale)
        autoinputs = scale_to_metric(scale, metric)
        super().__init__(fa, dict(d2=autoinputs))

    def vars(self):
        return ppp_list(self.fa['sill']) | ppp_list(self.fa['range'])

    def call(self, e):
        if isinstance(e['sill'], (tuple, list)):
            e['sill'] = tf.stack(e['sill'])
        if isinstance(e['range'], (tuple, list)):
            e['range'] = tf.stack(e['range'])

        return smooth_convex(tf.sqrt(e['d2']), e['sill'], e['range'])

# @tf.recompute_grad
def quadstack(x, sills, ranges):
    """
    `x` has arbitrary shape [...], but must be non-negative.
    `sills` and `ranges` both have shape [K].
    """
    ex = ed(x)
    ax = tf.maximum(0., 1. - tf.abs(ex) / ranges) # [..., 1]
    y = sills * tf.square(ax) # [..., K]
    return tf.reduce_sum(y, -1)

class QuadStack(Kernel):
    """
    QuadStack kernel class for Gaussian Processes (GPs).

    The `QuadStack` class defines a kernel that combines multiple quadratic components to model complex 
    covariance structures. It allows for multiple sills and ranges, providing flexibility in capturing 
    covariance that varies across different scales.

    Parameters:
        range (list or tf.Variable):
            A list or TensorFlow variable representing the length scale parameters that control how 
            quickly the covariance decreases with distance.
        sill (list or tf.Variable):
            A list or TensorFlow variable representing the variance (sill) values for each quadratic component.
        scale (optional):
            An optional scale parameter that can be used to modify the metric. Default is None.
        metric (optional):
            An optional metric used for distance calculation. Default is None.

    Examples
    --------
    Creating and using a `QuadStack` kernel:

    ```
    from geostat.kernel import QuadStack

    # Create a QuadStack kernel with multiple sills and ranges
    quad_stack_kernel = QuadStack(range=[2.0, 3.0], sill=[1.0, 0.5])
    
    locs1 = np.array([[0.0], [1.0], [2.0]])
    locs2 = np.array([[0.0], [1.0], [2.0]])
    covariance_matrix = quad_stack_kernel({'locs1': locs1, 'locs2': locs2, 'sill': [1.0, 0.5], 'range': [2.0, 3.0]})
    ```

    Notes
    -----
    - The `call` method computes the covariance matrix using the `quadstack` function, which applies 
        multiple quadratic functions based on the provided `sill` and `range` values for each component.
    - The `vars` method returns the parameter dictionary for both `sill` and `range` using the `ppp_list` function.
    - The `QuadStack` kernel is useful for modeling processes that exhibit multiple levels of variability 
        or changes in smoothness at different scales with a quadratic structure.
    """

    def __init__(self, range, sill, scale=None, metric=None):
        fa = dict(sill=sill, range=range, scale=scale)
        autoinputs = scale_to_metric(scale, metric)
        super().__init__(fa, dict(d2=autoinputs))

    def vars(self):
        return ppp_list(self.fa['sill']) | ppp_list(self.fa['range'])

    def call(self, e):
        if isinstance(e['sill'], (tuple, list)):
            e['sill'] = tf.stack(e['sill'])
        if isinstance(e['range'], (tuple, list)):
            e['range'] = tf.stack(e['range'])

        return quadstack(tf.sqrt(e['d2']), e['sill'], e['range'])

class Wiener(Kernel):
    """
    Wiener kernel class for Gaussian Processes (GPs).

    The `Wiener` class defines a kernel that represents a Wiener process (or Brownian motion) in one dimension.
    It models the covariance based on the minimum distance along a specified axis of integration, starting 
    from a given point.

    Parameters:
        axis (int):
            The axis along which the Wiener process operates (e.g., 0 for x-axis, 1 for y-axis).
        start (float):
            The starting point of the Wiener process along the specified axis.

    Examples
    --------
    Creating and using a `Wiener` kernel:

    ```
    from geostat.kernel import Wiener

    # Create a Wiener kernel operating along the x-axis starting from 0.0
    wiener_kernel = Wiener(axis=0, start=0.0)
    
    locs1 = np.array([[0.0], [1.0], [2.0]])
    locs2 = np.array([[0.0], [1.0], [2.0]])
    covariance_matrix = wiener_kernel({'locs1': locs1, 'locs2': locs2})
    ```

    Notes
    -----
    - The `call` method computes the covariance matrix using the Wiener process formula, which is based 
        on the minimum distance along the specified `axis` from the `start` point.
    - The `vars` method returns an empty dictionary since the Wiener kernel does not have tunable parameters.
    - The `Wiener` kernel is suitable for modeling processes that evolve with time or any other 
        ordered dimension, representing a type of random walk or Brownian motion.
    """

    def __init__(self, axis, start):

        self.axis = axis
        self.start = start

        # Include the element of scale corresponding to the axis of
        # integration as an explicit formal argument.
        fa = dict()

        super().__init__(fa, dict(locs1='locs1', locs2='locs2'))

    def vars(self):
        return {}

    def call(self, e):
        x1 = e['locs1'][..., self.axis]
        x2 = e['locs2'][..., self.axis]
        k = tf.maximum(0., tf.minimum(ed(x1, 1), ed(x2, 0)) - self.start)
        return k

class IntSquaredExponential(Kernel):
    """
    Integrated Squared Exponential (IntSquaredExponential) kernel class for Gaussian Processes (GPs).

    The `IntSquaredExponential` class defines a kernel that integrates the Squared Exponential kernel
    along a specified axis. This kernel is useful for modeling processes with smooth variations along 
    one dimension, starting from a given point.

    Parameters:
        axis (int):
            The axis along which the integration is performed (e.g., 0 for x-axis, 1 for y-axis).
        start (float):
            The starting point of the integration along the specified axis.
        range (float or tf.Variable):
            The length scale parameter that controls how quickly the covariance decreases with distance.

    Examples
    --------
    Creating and using an `IntSquaredExponential` kernel:

    ```
    from geostat.kernel import IntSquaredExponential

    # Create an IntSquaredExponential kernel integrating along the x-axis starting from 0.0 with a range of 2.0
    int_sq_exp_kernel = IntSquaredExponential(axis=0, start=0.0, range=2.0)
    
    locs1 = np.array([[0.0], [1.0], [2.0]])
    locs2 = np.array([[0.0], [1.0], [2.0]])
    covariance_matrix = int_sq_exp_kernel({'locs1': locs1, 'locs2': locs2, 'range': 2.0})
    ```

    Notes
    -----
    - The `call` method computes the integrated squared exponential covariance matrix based on the 
        specified axis, starting point, and range.
    - The `vars` method returns the parameter dictionary for `range` using the `ppp` function.
    - The `IntSquaredExponential` kernel is suitable for modeling smooth processes with integrated 
        covariance structures along one dimension.
    """

    def __init__(self, axis, start, range):

        self.axis = axis
        self.start = start

        # Include the element of scale corresponding to the axis of
        # integration as an explicit formal argument.
        fa = dict(range=range)

        super().__init__(fa, dict(locs1='locs1', locs2='locs2'))

    def vars(self):
        return ppp(self.fa['range'])

    def call(self, e):
        x1 = tf.pad(e['locs1'][..., self.axis] - self.start, [[1, 0]])
        x2 = tf.pad(e['locs2'][..., self.axis] - self.start, [[1, 0]])

        r = e['range']
        sdiff = (ed(x1, 1) - ed(x2, 0)) / (r * np.sqrt(2.))
        k = -tf.square(r) * (np.sqrt(np.pi) * sdiff * tf.math.erf(sdiff) + tf.exp(-tf.square(sdiff)))
        k -= k[0:1, :]
        k -= k[:, 0:1]
        k = k[1:, 1:]
        k = tf.maximum(0., k)

        return k

class IntExponential(Kernel):
    """
    Integrated Exponential (IntExponential) kernel class for Gaussian Processes (GPs).

    The `IntExponential` class defines a kernel that integrates the Exponential kernel
    along a specified axis. This kernel is useful for modeling processes with exponential 
    decay along one dimension, starting from a given point.

    Parameters:
        axis (int):
            The axis along which the integration is performed (e.g., 0 for x-axis, 1 for y-axis).
        start (float):
            The starting point of the integration along the specified axis.
        range (float or tf.Variable):
            The length scale parameter that controls how quickly the covariance decreases with distance.

    Examples
    --------
    Creating and using an `IntExponential` kernel:

    ```
    from geostat.kernel import IntExponential

    # Create an IntExponential kernel integrating along the x-axis starting from 0.0 with a range of 2.0
    int_exp_kernel = IntExponential(axis=0, start=0.0, range=2.0)
    
    locs1 = np.array([[0.0], [1.0], [2.0]])
    locs2 = np.array([[0.0], [1.0], [2.0]])
    covariance_matrix = int_exp_kernel({'locs1': locs1, 'locs2': locs2, 'range': 2.0})
    ```

    Notes
    -----
    - The `call` method computes the integrated exponential covariance matrix based on the 
        specified axis, starting point, and range.
    - The `vars` method returns the parameter dictionary for `range` using the `ppp` function.
    - The `IntExponential` kernel is suitable for modeling processes with exponential decay 
        along one dimension with integrated covariance structures.
    """

    def __init__(self, axis, start, range):

        self.axis = axis
        self.start = start

        # Include the element of scale corresponding to the axis of
        # integration as an explicit formal argument.
        fa = dict(range=range)

        super().__init__(fa, dict(locs1='locs1', locs2='locs2'))

    def vars(self):
        return ppp(self.fa['range'])

    def call(self, e):
        x1 = tf.pad(e['locs1'][..., self.axis] - self.start, [[1, 0]])
        x2 = tf.pad(e['locs2'][..., self.axis] - self.start, [[1, 0]])

        r = e['range']
        sdiff = tf.abs(ed(x1, 1) - ed(x2, 0)) / r
        k = -tf.square(r) * (sdiff + tf.exp(-sdiff))
        k -= k[0:1, :]
        k -= k[:, 0:1]
        k = k[1:, 1:]
        k = tf.maximum(0., k)

        return k

class Noise(Kernel):
    """
    Noise kernel class for Gaussian Processes (GPs).

    The `Noise` class defines a kernel that models the nugget effect, which represents uncorrelated noise
    in the data. It produces a diagonal covariance matrix with the specified `nugget` value, indicating 
    the presence of noise at each location.

    Parameters:
        nugget (float or tf.Variable):
            The variance (nugget) representing the noise level. This value is added to the diagonal 
            of the covariance matrix.

    Examples
    --------
    Creating and using a `Noise` kernel:

    ```
    from geostat.kernel import Noise

    # Create a Noise kernel with a nugget value of 0.1
    noise_kernel = Noise(nugget=0.1)
    
    locs1 = np.array([[0.0], [1.0], [2.0]])
    locs2 = np.array([[0.0], [1.0], [2.0]])
    covariance_matrix = noise_kernel({'locs1': locs1, 'locs2': locs2, 'nugget': 0.1})
    ```

    Notes
    -----
    - The `call` method computes a diagonal covariance matrix where the diagonal elements are equal 
        to `nugget`, representing noise at each location. Off-diagonal elements are set to 0.
    - The `vars` method returns the parameter dictionary for `nugget` using the `ppp` function.
    - The `Noise` kernel is useful for modeling independent noise in the data, especially when the 
        observations contain measurement error or variability that cannot be explained by the model.
    """

    def __init__(self, nugget):
        fa = dict(nugget=nugget)
        super().__init__(fa, dict(locs1='locs1', locs2='locs2'))

    def vars(self):
        return ppp(self.fa['nugget'])    

    def call(self, e):
        # Convert TensorFlow EagerTensors to JAX arrays
        offset = jnp.asarray(e['offset'])
        nugget = jnp.asarray(e['nugget'])

        indices1 = jnp.arange(e['locs1'].shape[0])
        indices2 = jnp.arange(e['locs2'].shape[0]) + offset
        C = jnp.where(jnp.expand_dims(indices1, -1) == indices2, nugget, 0.0)
        return C

    # def call(self, e):

    #     indices1 = tf.range(tf.shape(e['locs1'])[0])
    #     indices2 = tf.range(tf.shape(e['locs2'])[0]) + e['offset']
    #     C = tf.where(tf.equal(tf.expand_dims(indices1, -1), indices2), e['nugget'], 0.)
    #     return C

class Delta(Kernel):
    """
    Delta kernel class for Gaussian Processes (GPs).

    The `Delta` class defines a kernel that models a Dirac delta function effect, where covariance
    is non-zero only when the inputs are identical. This kernel is useful for capturing exact matches
    between input points, weighted by the specified `sill` parameter.

    Parameters:
        sill (float or tf.Variable):
            The variance (sill) representing the weight of the delta function. This value is applied 
            when input locations match exactly.
        axes (list or None, optional):
            A list of axes over which to apply the delta function. If not specified, the delta function 
            is applied across all axes.

    Examples
    --------
    Creating and using a `Delta` kernel:

    ```
    from geostat.kernel import Delta

    # Create a Delta kernel with a sill of 1.0, applied across all axes
    delta_kernel = Delta(sill=1.0)
    
    locs1 = np.array([[0.0], [1.0], [2.0]])
    locs2 = np.array([[0.0], [1.0], [2.0]])
    covariance_matrix = delta_kernel({'locs1': locs1, 'locs2': locs2, 'sill': 1.0})
    ```

    Using the `Delta` kernel with specified axes:

    ```
    delta_kernel_axes = Delta(sill=1.0, axes=[0])
    ```

    Notes
    -----
    - The `call` method computes a covariance matrix using a delta function, returning `sill` when the 
        squared distances are zero along the specified axes, and 0 otherwise.
    - The `vars` method returns the parameter dictionary for `sill` using the `ppp` function.
    - The `Delta` kernel is useful for modeling processes that exhibit exact matches or sharp changes 
        in covariance when inputs coincide, making it ideal for capturing discrete effects.
    """

    def __init__(self, sill, axes=None):
        fa = dict(sill=sill)
        self.axes = axes
        super().__init__(fa, dict(pa_d2='per_axis_dist2'))

    def vars(self):
        return ppp(self.fa['sill'])

    def call(self, e):

        if self.axes is not None:
            n = e['pa_d2'].shape[-1]
            mask = jnp.zeros(n)
            for axis in self.axes:
                mask += mask.at[axis].set(1.0)  # Create a mask with 1.0 on specified axis
            #mask = tf.math.bincount(self.axes, minlength=n, maxlength=n, dtype=tf.float32)
            d2 = jnp.einsum('abc,c->ab', e['pa_d2'], mask)
        else:
            d2 = jnp.sum(e['pa_d2'], axis=-1)

        return e['sill'] * jnp.where(d2 == 0., 1.0, 0.0)

class Mix(Kernel):
    """
    Mix kernel class for combining multiple Gaussian Process (GP) kernels.

    The `Mix` class defines a kernel that allows combining multiple input kernels, either using 
    specified weights or by directly mixing the component kernels. This provides a flexible way to 
    create complex covariance structures by blending the properties of different kernels.

    Parameters:
        inputs (list of Kernel objects):
            A list of kernel objects to be combined.
        weights (matrix, optional):
            A matrix specifying how the input kernels should be combined. If not provided, 
            the kernels are combined without weighting.

    Examples
    --------
    Combining multiple kernels with specified weights:

    ```
    from geostat.kernel import Mix, SquaredExponential, Noise

    # Create individual kernels
    kernel1 = SquaredExponential(sill=1.0, range=2.0)
    kernel2 = Noise(nugget=0.1)

    # Combine kernels using the Mix class
    mixed_kernel = Mix(inputs=[kernel1, kernel2], weights=[[0.6, 0.4], [0.4, 0.6]])
    
    locs1 = np.array([[0.0], [1.0], [2.0]])
    locs2 = np.array([[0.0], [1.0], [2.0]])
    covariance_matrix = mixed_kernel({'locs1': locs1, 'locs2': locs2, 'weights': [[0.6, 0.4], [0.4, 0.6]]})
    ```

    Using the `Mix` kernel without weights:

    ```
    mixed_kernel_no_weights = Mix(inputs=[kernel1, kernel2])
    ```

    Notes
    -----
    - The `call` method computes the covariance matrix by either using the specified weights 
        to combine the input kernels or directly combining them when weights are not provided.
    - The `vars` method gathers the parameters from all input kernels, allowing for easy 
        access and manipulation of their coefficients.
    - The `Mix` kernel is useful for creating complex, multi-faceted covariance structures 
        by blending different types of kernels, providing enhanced modeling flexibility.
    """

    def __init__(self, inputs, weights=None):
        self.inputs = inputs
        fa = {}
        ai = dict(cats1='cats1', cats2='cats2')

        # Special case if weights is not given.
        if weights is not None:
            fa['weights'] = weights
            ai['inputs'] = inputs

        super().__init__(fa, ai)

    def gather_vars(self, cache=None):
        """Make a special version of gather_vars because
           we want to gather variables from `inputs`
           even when it's not in autoinputs"""
        vv = super().gather_vars(cache)
        for iput in self.inputs:
            cache[id(self)] |= iput.gather_vars(cache)
        return cache[id(self)]

    def vars(self):
        if 'weights' in self.fa:
            return {k: p for row in self.fa['weights']
                      for k, p in get_trend_coefs(row).items()}
        else:
            return {}

    def call(self, e):
        if 'weights' in e:
            weights = []
            for row in e['weights']:
                if isinstance(row, (tuple, list)):
                    row = jnp.stack(row)
                    weights.append(row)
            weights = jnp.stack(weights)
            C = jnp.stack(e['inputs'], axis=-1) # [locs, locs, numinputs].
            Aaug1 = jnp.take(weights, e['cats1'], axis=0) # [locs, numinputs].
            Aaug2 = jnp.take(weights, e['cats2'], axis=0) # [locs, numinputs].
            outer = jnp.einsum('ac,bc->abc', Aaug1, Aaug2) # [locs, locs, numinputs].
            C = jnp.einsum('abc,abc->ab', C, outer) # [locs, locs].
            return C
        else:
            # When weights is not given, exploit the fact that we don't have
            # to compute every element in component covariance matrices.
            N = len(self.inputs)
            catcounts1 = jnp.bincount(e['cats1'], minlength=N, length=N)
            catcounts2 = jnp.bincount(e['cats2'], minlength=N, length=N)
            catindices1 = jnp.cumulative_sum(catcounts1, include_initial=True)
            catindices2 = jnp.cumulative_sum(catcounts2, include_initial=True)
            catdiffs = jnp.unstack(catindices2 - catindices1)[:-1]
            locsegs1 = jnp.split(e['locs1'], jnp.cumsum(catcounts1, axis=0)[:-1])
            locsegs2 = jnp.split(e['locs2'], jnp.cumsum(catcounts2, axis=0)[:-1])

            # TODO: Check that the below is still correct.
            CC = [] # Observation noise submatrices.
            for sublocs1, sublocs2, catdiff, iput in zip(locsegs1, locsegs2, catdiffs, self.inputs):
                cache = dict(
                    offset = e['offset'] + catdiff,
                    locs1 = sublocs1,
                    locs2 = sublocs2)
                cache['per_axis_dist2'] = PerAxisDist2().run(cache)
                cache['euclidean'] = Euclidean().run(cache)
                Csub = iput.run(cache)
                CC.append(Csub)

            return block_diag(*CC)

class Stack(Kernel):
    """
    Stack kernel class for combining multiple Gaussian Process (GP) kernels additively.

    The `Stack` class defines a kernel that combines multiple input kernels by stacking them together.
    This additive combination allows for capturing a more complex covariance structure by summing the
    contributions from each individual kernel.

    Parameters:
        parts (List[Kernel]):
            A list of kernel objects to be combined additively.

    Examples
    --------
    Creating and using a `Stack` kernel:

    ```
    from geostat.kernel import Stack, SquaredExponential, Noise

    # Create individual kernels
    kernel1 = SquaredExponential(sill=1.0, range=2.0)
    kernel2 = Noise(nugget=0.1)

    # Combine kernels using the Stack class
    stacked_kernel = Stack(parts=[kernel1, kernel2])
    
    locs1 = np.array([[0.0], [1.0], [2.0]])
    locs2 = np.array([[0.0], [1.0], [2.0]])
    covariance_matrix = stacked_kernel({'locs1': locs1, 'locs2': locs2})
    ```

    Adding another kernel to an existing `Stack`:

    ```
    kernel3 = SquaredExponential(sill=0.5, range=1.0)
    combined_stack = stacked_kernel + kernel3
    ```

    Notes
    -----
    - The `call` method computes the sum of all covariance matrices generated by the stacked kernels.
    - The `vars` method gathers parameters from all input kernels, making them accessible for optimization.
    - The `Stack` kernel is useful for building complex models where multiple covariance structures need to be 
        combined additively, enabling richer and more flexible GP models.
    """

    def __init__(self, parts: List[Kernel]):
        self.parts = parts
        super().__init__({}, dict(locs1='locs1', locs2='locs2', parts=parts))

    def vars(self):
        return {k: p for part in self.parts for k, p in part.vars().items()}

    def __add__(self, other):
        if isinstance(other, Kernel):
            return Stack(self.parts + [other])
    
    def call(self, e):
        return jnp.sum(jnp.array(e['parts']), axis=0)

    def report(self):
        return ' '.join(part.report(p) for part in self.parts)

class Product(Kernel):
    """
    Product kernel class for combining multiple Gaussian Process (GP) kernels multiplicatively.

    The `Product` class defines a kernel that combines multiple input kernels by multiplying them together.
    This multiplicative combination allows for capturing interactions between the individual kernels, resulting
    in a more complex and flexible covariance structure.

    Parameters:
        parts (List[Kernel]):
            A list of kernel objects to be combined multiplicatively.

    Examples
    --------
    Creating and using a `Product` kernel:

    ```
    from geostat.kernel import Product, SquaredExponential, Noise

    # Create individual kernels
    kernel1 = SquaredExponential(sill=1.0, range=2.0)
    kernel2 = Noise(nugget=0.1)

    # Combine kernels using the Product class
    product_kernel = Product(parts=[kernel1, kernel2])
    
    locs1 = np.array([[0.0], [1.0], [2.0]])
    locs2 = np.array([[0.0], [1.0], [2.0]])
    covariance_matrix = product_kernel({'locs1': locs1, 'locs2': locs2})
    ```

    Multiplying another kernel with an existing `Product` kernel:

    ```
    kernel3 = SquaredExponential(sill=0.5, range=1.0)
    combined_product = product_kernel * kernel3
    ```

    Notes
    -----
    - The `call` method computes the product of all covariance matrices generated by the multiplied kernels.
    - The `vars` method gathers parameters from all input kernels, making them accessible for optimization.
    - The `Product` kernel is useful for building models where the covariance structure results from 
        multiplicative interactions between different kernels, allowing for more complex GP models.
    """

    def __init__(self, parts: List[Kernel]):
        self.parts = parts
        super().__init__({}, dict(locs1='locs1', locs2='locs2', parts=parts))

    def vars(self):
        return {k: p for part in self.parts for k, p in part.vars().items()}

    def __mul__(self, other):
        if isinstance(other, Kernel):
            return Product(self.parts + [other])
    
    def call(self, e):
        return np.prod(e['parts'], axis=0)

    def report(self):
        return ' '.join(part.report(p) for part in self.parts)

@jax.custom_vjp
def safepow(x, a):
    return jnp.power(x, a)

# Forward function for the custom VJP
def safepow_fwd(x, a):
    y = jnp.power(x, a)
    return y, (x, a, y)

# Backward function for the custom VJP
def safepow_bwd(res, dy):
    x, a, y = res
    
    # Compute the gradient with respect to x
    dx = jnp.where(x <= 0.0, jnp.zeros_like(x), dy * jnp.power(x, a - 1))
    
    # Compute the gradient with respect to a
    da = jnp.where(x <= 0.0, jnp.zeros_like(a), dy * y * jnp.log(jnp.maximum(x, 1e-10)))
    
    # Use the unbroadcast function to ensure the shapes of dx and da match their respective inputs
    dx = unbroadcast(dx, x.shape)
    da = unbroadcast(da, a.shape)
    
    return dx, da

# Register the forward and backward functions with the custom VJP
safepow.defvjp(safepow_fwd, safepow_bwd)

def unbroadcast(x, target_shape):
    """
    Adjusts the shape of x to match target_shape by summing across broadcasted dimensions.
    
    Parameters:
        x (jnp.ndarray): The array to unbroadcast.
        target_shape (tuple): The desired shape to match.
    
    Returns:
        jnp.ndarray: The array with the shape adjusted to target_shape.
    """
    while len(x.shape) > len(target_shape):
        x = x.sum(axis=0)  # Reduce excess dimensions

    for axis, size in enumerate(target_shape):
        if size == 1 and x.shape[axis] != 1:
            x = x.sum(axis=axis, keepdims=True)

    return x

def gamma_exp(d2, gamma):
    return jnp.exp(-safepow(jnp.maximum(d2, 0.0), 0.5 * gamma))

class Observation(Op):

    def __init__(self,
        coefs: List,
        noise: Kernel
    ):
        self.coefs = coefs
        self.noise = noise
        super().__init__({}, self.noise)

    def vars(self):
        vv = {k: p for c in self.coefs for k, p in upp(c)}
        vv |= self.noise.vars()
        return vv

    def __call__(self, e):        
        return 0.
