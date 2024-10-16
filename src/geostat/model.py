import time
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Union
import numpy as np
from scipy.special import expit, logit
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Tensorflow is extraordinarily noisy. Catch warnings during import.
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions
    from tensorflow.core.function.trace_type import default_types

from . import mean as mn
from . import kernel as krn
from .metric import Euclidean, PerAxisDist2
from .op import Op
from .op import SingletonTraceType
from .param import get_parameter_values, ppp, upp, bpp
from .param import Parameter

MVN = tfp.distributions.MultivariateNormalTriL

__all__ = ['featurizer', 'GP', 'Mix', 'Model', 'Featurizer', 'NormalizingFeaturizer', 'StratigraphicWarp']

@dataclass
class GP:    
    """
    Gaussian Process (GP) model class with a mean function and a kernel.

    This class represents a Gaussian Process with specified mean
    and kernel functions.  If no mean is provided, a zero mean is
    used by default. The kernel must always be specified.  The class
    supports addition to combine two GP models, and it allows
    gathering variables from the mean and kernel.

    Parameters:    
        mean (mean.Mean, optional):
            The mean function of the Gaussian Process. If not provided or set to 0, 
            a ZeroTrend is used as the default mean.
        kernel (kernel.Kernel):
            The kernel function of the Gaussian Process. This parameter is required.

    Examples: Details:
        This is how to specify a GP with a squared exponential kernel 
        and superimposed uncorrelated noise:

        ```python
        import geostat.kernel as krn
        from geostat import GP, Model, Parameters

        p = Parameters(range=1., sill=1., nugget=1.)
        kernel = krn.SquaredExponential(range=p.range, sill=p.sill) + krn.Noise(nugget=p.nugget)
        gp = GP(0, kernel)
        ```

        To use the GP, it must be wrapped in a model:

        ```python
        model = Model(GP)
        ```

        This model object can then be used to generate synthetic data,
        fit its parameters to provided data, or make predictions, see
        [`Model`](#src.geostat.model.Model).

        In Geostat, GPs can be defined on locations in Euclidean space
        of any dimension \(\mathbb{R}^D\), with the number of dimensions
        specified implicitly by the shape of the location matrix given
        to `fit()` or `generate()` in the `locs` argument.
        
        GPs can also be defined on locations in \(\mathbb{R}^D \times
        \mathbb{Z}\) using the [`Mix`](#src.geostat.model.Mix) operator.
        This construction is for modeling multiple spatial quantities,
        with each quantity occupying a different 'plane' of the space.
        When multiple spatial quantities are involved, these are specified
        in the `cats` argument of `fit()`, `generate()` or `predict()`.

        GPs can be superimposed:
        ```python
        gp = gp1 + gp2
        ```

    Examples:    
        A linear regression is a special case of GP regression that
        can be modeled by Geostat. Suppose
        $$
        u_i = \beta_1 + \beta_2 x_i + \beta_3 y_i + \beta_4 x_i^2 \
            + \beta_5 x_i y_i + \beta_6 y_i^2 + \epsilon_i
        $$
        where \(u_i\) is an observation, \(x_i\) and \(y_i\) are model
        inputs, \(\epsilon_i \sim \mathcal{N}(0, \sigma^2)\)
        describes observation noise, and \(\beta_1, \ldots, \beta_6\)
        are regression coefficients.  Geostat can be used to fit this
        regression (though not in the most efficient way):

        ```python
        @featurizer
        def trend_terms(x, y):
            return 1, x, y, x*x, x*y, y*y
        p = Parameters(beta=np.zeros([6]), sigma2=1.)
        mean = mn.Trend(trend_terms, beta=p.beta)
        kernel = krn.Noise(nugget=p.sigma2)
        gp = GP(mean, kernel)
        Model(gp).fit(locs, vals) # locs.shape = [N, 2], vals.shape = [N]
        ```
    """

    mean: mn.Trend = None
    kernel: krn.Kernel = None

    def __post_init__(self):
        if self.mean is None or self.mean == 0:
            self.mean = mn.ZeroTrend()
        assert self.kernel is not None

    def __add__(self, other):
        return GP(self.mean + other.mean, self.kernel + other.kernel)

    def __tf_tracing_type__(self, context):
        return SingletonTraceType(self)

    def gather_vars(self):
        return self.mean.gather_vars() | self.kernel.gather_vars()

def Mix(inputs, weights=None):
    """
    Linearly combines multiple Gaussian Processes (GPs) into a single GP using specified weights.

    Parameters:
        inputs (list of GPs):  
            A list of GP objects to be combined.
        weights (matrix, optional):  
            A matrix specifying how the inputs are to be combined. If not provided, 
            an identity matrix is assumed, meaning the GPs are combined without weighting.

    Returns:
        GP (GP):
            A new GP object representing the linear combination of the input GPs with the specified weights.



    Examples:
        Combining two GPs into a new multi-output GP:

        Suppose you have two GPs: 
        \\(f_1(x) \sim \mathrm{GP}(\mu_1, K_1)\\) and \\(f_2(x) \sim \mathrm{GP}(\mu_2, K_2)\\), 
        and you want to create a new multi-output GP \\(\mathbf{g}(x)\\) defined as:

        $$
        \mathbf{g}(x) = \\begin{pmatrix}
        g_1(x) \\\\
        g_2(x) \\\\
        g_3(x)
        \end{pmatrix} = A \\begin{pmatrix}
        f_1(x) \\\\
        f_2(x)
        \end{pmatrix},
        $$

        where \\(A\\) is the weights matrix. This can be implemented as:

        ```python
        g = Mix([f1, f2], [[a11, a12], [a21, a22], [a31, a32]])
        ```

        The resulting GP \\(\mathbf{g}(x)\\) can then be used for fitting, generating, or predicting 
        with methods such as `g.fit()`, `g.generate()`, or `g.predict()` and its components are specified using the `cats` parameter.
    
    Examples: Notes:
        - The `weights` parameter defines how the input GPs are linearly combined. If omitted, 
        each GP is assumed to be independent, and the identity matrix is used.
        - The resulting GP supports all standard operations (e.g., `fit`, `generate`, `predict`).
    """

    return GP(
        mn.Mix([i.mean for i in inputs], weights), 
        krn.Mix([i.kernel for i in inputs], weights))

class Warp:
    def __call__(self, locs, prep):
        """
        `locs` is numpy.

        Returns a WarpLocations.
        """
        pass

    def gather_vars(self):
        pass

class WarpLocations(Op):
    def __init__(self, warped_locs):
        self.warped_locs = warped_locs
        super().__init__({}, {})

    def __call__(self, e):
        return tf.cast(self.warped_locs, dtype=tf.float32)

    def __tf_tracing_type__(self, context):
        return SingletonTraceType(self)

    def gather_vars(self):
        return {}

class NoWarp:
    def __call__(self, locs):
        return WarpLocations(locs)

    def gather_vars(self):
        return {}

class CrazyWarp(Warp):
    def __call__(self, locs):
        if locs.shape[-1] == 4:
            locs += e(np.sin(locs[:, 1])) * np.array([0., 0., 1., 0.])
        return WarpLocations(locs)

    def gather_vars(self):
        return {}

@tf.function
def interpolate_1d_tf(src, tgt, x):
    """
    `src`: (batch, breaks)
    `tgt`: (batch, breaks)
    `x`  : (batch)
    """
    x_shape = tf.shape(x)
    x = tf.reshape(x, [-1, 1]) # (batch, 1)
    bucket = tf.searchsorted(src, x)
    bucket = tf.clip_by_value(bucket - 1, 0, tf.shape(tgt)[0] - 2)
    src0 = tf.gather(src, bucket, batch_dims=1)
    src1 = tf.gather(src, bucket + 1, batch_dims=1)
    tgt0 = tf.gather(tgt, bucket, batch_dims=1)
    tgt1 = tf.gather(tgt, bucket + 1, batch_dims=1)
    xout = ((x - src0) * tgt1 + (src1 - x) * tgt0) / (src1 - src0)
    return tf.reshape(xout, x_shape)

@tf.function
def relax(s, t, distort):
    xi = distort / (1 - distort)
    ds = s[:, 1:] - s[:, :-1]
    x = s

    for i in range(5):
        # Compute objective.
        dx = x[:, 1:] - x[:, :-1]
        dxds = dx / ds
        a = tf.math.log(dxds)
        # obj = tf.reduce_sum(tf.math.square(a), axis=-1) \
        #     + tf.reduce_sum(xi * tf.math.square(x - t), axis=-1)

        g = a / dx
        zg = tf.pad(g, [[0, 0], [1, 0]])
        gz = tf.pad(g, [[0, 0], [0, 1]])
        halfgrad = zg - gz + xi * (x - t)

        # Compute hessian as tridiagonal matrix.
        h = (1 - a) / tf.square(dx)
        zh = tf.pad(h, [[0, 0], [1, 0]])
        hz = tf.pad(h, [[0, 0], [0, 1]])
        halfhess = tf.stack([-hz, hz + zh + xi, -zh], axis=-2)
        
        # Newton's method.
        x -= tf.linalg.tridiagonal_solve(halfhess, halfgrad)

    return x

class TweeningStratigraphicWarp(Warp):
    def __init__(self, surface_functions, surface_targets, tween):
        self.surface_functions = surface_functions
        self.surface_targets = surface_targets
        self.tween = tween

    def __call__(self, locs):
        sources = [f(locs[..., :2]) for f in self.surface_functions]
        sources = tf.sort(tf.cast(tf.stack(sources, axis=-1), tf.float32))
        targets = tf.constant(self.surface_targets, dtype=tf.float32)

        locs = tf.cast(locs, tf.float32)
        zout = interpolate_1d_tf(sources, targets, locs[..., 2])
        out = tf.stack([locs[..., 0], locs[..., 1], zout, locs[..., 3]], axis=-1)
        return TweeningStratigraphicWarpLocations(locs, out.numpy(), self.tween)

    def gather_vars(self):
        return bpp(self.tween, 0., 1.)

class TweeningStratigraphicWarpLocations(Op):
    def __init__(self, locs, warped_locs, tween):
        self.locs = locs
        self.warped_locs = warped_locs

        fa = dict(tween=tween)
        super().__init__(fa, {})

    def __call__(self, e):
        tween = e['tween']
        locs0 = tf.cast(self.locs, dtype=tf.float32)
        locs1 = tf.cast(self.warped_locs, dtype=tf.float32)
        return locs0 * (1. - tween) + locs1 * tween

    def __tf_tracing_type__(self, context):
        return SingletonTraceType(self)

    def gather_vars(self):
        return bpp(self.fa['tween'], 0., 1.)

class StratigraphicWarp(Warp):
    def __init__(self, surface_functions, surface_targets, distort):
        self.surface_functions = surface_functions
        self.surface_targets = surface_targets
        self.distort = distort

    def __call__(self, locs):
        sources = [f(locs[..., :2]) for f in self.surface_functions]
        sources = tf.sort(tf.cast(tf.stack(sources, axis=-1), tf.float32))
        targets = tf.constant(self.surface_targets, dtype=tf.float32)
        locs = tf.cast(locs, tf.float32)
        return StratigraphicWarpLocations(locs, sources, targets, self.distort)

    def gather_vars(self):
        return bpp(self.distort, 0., 1.)

class StratigraphicWarpLocations(Op):
    def __init__(self, locs, sources, targets, distort):
        self.locs = locs
        self.sources = sources
        self.targets = targets

        fa = dict(distort=distort)
        super().__init__(fa, {})

    def __call__(self, e):
        distort = e['distort']

        locs0 = self.locs
        s = self.sources
        t = self.targets
        r = relax(s, t, distort)
        z = interpolate_1d_tf(s, r, locs0[:, 2])
        locs1 = tf.stack([locs0[:, 0], locs0[:, 1], z, locs0[:, 3]], axis=-1)

        return locs1

    def __tf_tracing_type__(self, context):
        return SingletonTraceType(self)

    def gather_vars(self):
        return bpp(self.fa['tween'], 0., 1.)

class NormalizingFeaturizer:
    """
    NormalizingFeaturizer class for producing normalized feature matrices (F matrix) with an intercept.

    The `NormalizingFeaturizer` takes raw location data and applies a specified featurization function.
    It normalizes the resulting features and remembers normalization parameters using the mean and standard deviation calculated from the 
    original data and adds an intercept feature (a column of ones) to the matrix.

    Parameters:
        featurization (Callable):
            A function or callable that defines how the input location data should be featurized.
        locs (array-like or Tensor):
            The input location data used for calculating normalization parameters (mean and standard 
            deviation) and featurizing new data.

    Examples:
        Creating a `NormalizingFeaturizer` using a custom featurization function and location data:

        ```python
        import tensorflow as tf
        from geostat.model import NormalizingFeaturizer

        # Define a simple featurization function
        def custom_featurizer(x, y):
            return x, y, x * y

        # Sample location data
        locs = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        # Create the NormalizingFeaturizer
        norm_featurizer = NormalizingFeaturizer(custom_featurizer, locs)
        ```

        Using the `NormalizingFeaturizer` to featurize new location data:

        ```python
        new_locs = tf.constant([[7.0, 8.0], [9.0, 10.0]])
        F_matrix = norm_featurizer(new_locs)
        print(F_matrix) # F_matrix will contain normalized features with an additional intercept column
        # tf.Tensor(
        # [[1.        2.4494898 2.4494898 3.5676992]
        #  [1.        3.6742349 3.6742349 6.50242  ]], shape=(2, 4), dtype=float32)
        ```

    Examples: Notes:
        - The normalization parameters (`unnorm_mean` and `unnorm_std`) are calculated based on the 
        initial `locs` data provided during initialization.
        - The `__call__` method applies the normalization and adds an intercept feature when used 
        to featurize new location data.
    """

    def __init__(self, featurization, locs):
        self.unnorm_featurizer = Featurizer(featurization)
        F_unnorm = self.unnorm_featurizer(locs)
        self.unnorm_mean = tf.reduce_mean(F_unnorm, axis=0)
        self.unnorm_std = tf.math.reduce_std(F_unnorm, axis=0)

    def __call__(self, locs):
        ones = tf.ones([tf.shape(locs)[0], 1], dtype=tf.float32)
        F_unnorm = self.unnorm_featurizer(locs)
        F_norm = (F_unnorm - self.unnorm_mean) / self.unnorm_std
        return tf.concat([ones, F_norm], axis=1)

class Featurizer:
    """
    Featurizer class for producing feature matrices (F matrix) from location data.

    The `Featurizer` applies a specified featurization function to the input location data 
    and generates the corresponding feature matrix. If no featurization function is provided, 
    it produces a matrix with appropriate dimensions containing only ones.

    Parameters:
        featurization (Callable or None):
            A function that takes in the individual components of location data and returns the features.
            If set to `None`, the featurizer will produce an empty feature matrix (i.e., only ones).

    Examples:
        Creating a `Featurizer` using a custom featurization function:

        ```python
        import tensorflow as tf
        from geostat.model import Featurizer

        # Define a custom featurization function
        def simple_featurizer(x, y):
            return x, y, x * y

        # Initialize the Featurizer
        featurizer = Featurizer(simple_featurizer)
        ```

        Using the `Featurizer` to transform location data:

        ```python
        locs = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        F_matrix = featurizer(locs)
        print(F_matrix) # F_matrix will contain the features: (x, y, x*y) for each location
        # tf.Tensor(
        # [[ 1.  2.  2.]
        #  [ 3.  4. 12.]
        #  [ 5.  6. 30.]], shape=(3, 3), dtype=float32)
        ```

        Handling the case where no featurization is provided:

        ```python
        featurizer_no_feat = Featurizer(None)
        F_matrix = featurizer_no_feat(locs)
        print(F_matrix) # Since no featurization function is provided, F_matrix will have shape (3, 0)
        # tf.Tensor([], shape=(3, 0), dtype=float32)
        ```

    Examples: Notes:
        - The `__call__` method is used to apply the featurization to input location data.
        - If `featurization` returns a tuple, it is assumed to represent multiple features, 
        which will be stacked to form the feature matrix.
    """

    def __init__(self, featurization):
        self.featurization = featurization

    def __call__(self, locs):
        locs = tf.cast(locs, tf.float32)
        if self.featurization is None: # No features.
            return tf.ones([tf.shape(locs)[0], 0], dtype=tf.float32)

        feats = self.featurization(*tf.unstack(locs, axis=1))
        if isinstance(feats, tuple): # One or many features.
            if len(feats) == 0:
                return tf.ones([tf.shape(locs)[0], 0], dtype=tf.float32)
            else:
                feats = self.featurization(*tf.unstack(locs, axis=1))
                feats = [tf.broadcast_to(tf.cast(f, tf.float32), [tf.shape(locs)[0]]) for f in feats]
                return tf.stack(feats, axis=1)
        else: # One feature.
            return e(feats)

def featurizer(normalize=False):
    def helper(f):
        if not normalize:
            return Featurizer(f)
        else:
            return NormalizingFeaturizer(f, normalize)
    return helper

def e(x, a=-1):
    return tf.expand_dims(x, a)

@tf.function
def gp_covariance(gp, locs, cats):
    return gp_covariance2(gp, locs, cats, locs, cats, 0)

@tf.function
def gp_covariance2(gp, locs1, cats1, locs2, cats2, offset):
    """
    `offset` is i2-i1, where i1 and i2 are the starting indices of locs1
    and locs2.  It is used to create the diagonal non-zero elements
    of a Noise covariance function.  An non-zero offset results in a
    covariance matrix with non-zero entries along an off-center diagonal.
    """

    # assert np.all(cats1 == np.sort(cats1)), '`cats1` must be in non-descending order'
    # assert np.all(cats2 == np.sort(cats2)), '`cats2` must be in non-descending order'

    cache = {}
    cache['offset'] = offset
    cache['locs1'] = locs1
    cache['locs2'] = locs2
    cache['cats1'] = cats1
    cache['cats2'] = cats2
    cache['per_axis_dist2'] = PerAxisDist2().run(cache)
    cache['euclidean'] = Euclidean().run(cache)

    M = gp.mean.run(cache)
    C = gp.kernel.run(cache)
    M = tf.cast(M, tf.float64)
    C = tf.cast(C, tf.float64)
    return M, C

def mvn_log_pdf(u, m, cov):
    """Log PDF of a multivariate gaussian."""
    u_adj = u - m
    logdet = tf.linalg.logdet(2 * np.pi * cov)
    quad = tf.matmul(e(u_adj, 0), tf.linalg.solve(cov, e(u_adj, -1)))[0, 0]
    return tf.cast(-0.5 * (logdet + quad), tf.float32)

@tf.function
def gp_log_likelihood(data, gp):
    m, S = gp_covariance(gp, data['warplocs'].run({}), data['cats'])
    u = tf.cast(data['vals'], tf.float64)
    return mvn_log_pdf(u, m, S)

def gp_train_step(
    optimizer,
    data,
    parameters: Dict[str, Parameter],
    gp,
    reg=None
):
    with tf.GradientTape() as tape:
        ll = gp_log_likelihood(data, gp)

        if reg:
            # TODO: Put in cache later.
            reg_penalty = reg.run({})
        else:
            reg_penalty = 0.

        loss = -ll + reg_penalty

    up = [p.underlying for p in parameters.values()]
    gradients = tape.gradient(loss, up)
    optimizer.apply_gradients(zip(gradients, up))
    return ll, reg_penalty

@dataclass
class Model():
    """
    Model class for performing Gaussian Process (GP) training and prediction with optional warping.

    The `Model` class integrates a GP model with optional data warping, and supports data generation on given location,
    training on given location and observation data, and prediction on given location.

    Parameters:
        gp (GP):
            The Gaussian Process model to be used for training and prediction.
        warp (Warp, optional):
            An optional warping transformation applied to the data. If not specified, `NoWarp` 
            is used by default.
        parameter_sample_size (int, optional):
            The number of parameter samples to draw. Default is None.
        locs (np.ndarray, optional):
            A NumPy array containing location data.
        vals (np.ndarray, optional):
            A NumPy array containing observed values corresponding to `locs`.
        cats (np.ndarray, optional):
            A NumPy array containing categorical data.
        report (Callable, optional):
            A custom reporting function to display model parameters. If not provided, a default 
            reporting function is used.
        verbose (bool, optional):
            Whether to print model parameters and status updates. Default is True.

    Examples: Details:
        To generate synthetic data at \(n\) locations in \(k\)-dimensional
        space, pass the locations into `generate()`:
        ```python
        vals = model.generate(locs) # locs has shape (n, k).
        ```

        To fit to data at \(n\) locations, pass locations and values into
        `fit()`:

    Examples:    
        Initializing a `Model` with a Gaussian Process:

        ```python
        from geostat import GP, Model, Parameters
        from geostat.kernel import Noise
        import numpy as np

        # Create parameters.
        p = Parameters(nugget=1.)

        # Define the Gaussian Process and the model
        gp = GP(kernel=Noise(nugget=p.nugget))
        locs = np.array([[0.0, 1.0], [1.0, 2.0]])
        vals = np.array([1.0, 2.0])
        model = Model(gp=gp, locs=locs, vals=vals)
        ```

    Examples: Notes:
        - The `__post_init__` method sets up default values, initializes the warping if not provided, 
        and sets up reporting and data preprocessing.
    """

    gp: GP
    warp: Warp = None
    parameter_sample_size: Optional[int] = None
    locs: np.ndarray = None
    vals: np.ndarray = None
    cats: np.ndarray = None
    report: Callable = None
    verbose: bool = True

    def __post_init__(self):
        # '''
        # Parameters:
        #         x : Pandas DataFrame with columns for locations.

        #         u : A Pandas Series containing observations.

        #         featurization : function, optional
        #             Should be a function that takes x1 (n-dim array of input data)
        #             and returns the coordinates, i.e., x, y, x**2, y**2.
        #             Example: def featurization(x1):
        #                         return x1[:, 0], x1[:, 1], x1[:, 0]**2, x1[:, 1]**2.
        #             Default is None.

        #         latent : List[GP]
        #              Name of the covariance function to use in the GP.
        #              Should be 'squared-exp' or 'gamma-exp'.
        #              Default is 'squared-exp'.

        #         verbose : boolean, optional
        #             Whether or not to print parameters.
        #             Default is True.

        # Performs Gaussian process training and prediction.
        # '''

        if self.warp is None: self.warp = NoWarp()

        # Default reporting function.
        def default_report(p, prefix=None):
            if prefix: print(prefix, end=' ')

            def fmt(x):
                if isinstance(x, tf.Tensor):
                    x = x.numpy()

                if isinstance(x, (int, np.int32, np.int64)):
                    return '{:5d}'.format(x)
                if isinstance(x, (float, np.float32, np.float64)):
                    return '{:5.2f}'.format(x)
                else:
                    with np.printoptions(precision=2, formatter={'floatkind': '{:5.2f}'.format}):
                        return str(x)

            print('[%s]' % (' '.join('%s %s' % (k, fmt(v)) for k, v in p.items())))

        if self.report == None: self.report = default_report

        if self.locs is not None: self.locs = np.array(self.locs)
        if self.vals is not None: self.vals = np.array(self.vals)
        if self.cats is not None: self.cats = np.array(self.cats)

        # Collect parameters and create TF parameters.
        for p in self.gather_vars().values():
            p.create_tf_variable()

    def gather_vars(self):
        return self.gp.gather_vars() | self.warp.gather_vars()

    def set(self, **values):
        """
        Sets the values of the model's parameters based on the provided keyword arguments.
        Each parameter specified must exist in the model; otherwise, a `ValueError` is raised.

        Parameters:
            values (keyword arguments):
                A dictionary of parameter names and their corresponding values that should be 
                set in the model. Each key corresponds to a parameter name, and the value is 
                the value to be assigned to that parameter.

        Returns:
            self (Model):
                The model instance with updated parameter values, allowing for method chaining.

        Raises:
            ValueError:
                If a provided parameter name does not exist in the model's parameters.

        Examples:
            Update parameter value using `set`:

            ```python
            from geostat import GP, Model, Parameters
            from geostat.kernel import Noise

            # Create parameters.
            p = Parameters(nugget=1.)

            # Create model
            kernel = Noise(nugget=p.nugget)
            model = Model(GP(0, kernel))

            # Update parameters
            model.set(nugget=0.5)
            ```

        Examples: Notes:
            - The `set` method retrieves the current parameters using `gather_vars` and updates 
            their values. The associated TensorFlow variables are also recreated.
            - This method is useful for dynamically updating the model's parameters after initialization.
        """

        parameters = self.gather_vars()
        for name, v in values.items():
            if name in parameters:
                parameters[name].value = v
                parameters[name].create_tf_variable()
            else:
                raise ValueError(f"{name} is not a parameter")
        return self

    def fit(self, locs, vals, cats=None, step_size=0.01, iters=100, reg=None):
        """
        Trains the model using the provided location and value data by optimizing the parameters of the Gaussian Process (GP)
        using the Adam optimizer. Optionally performs regularization and can handle categorical data.

        Parameters:        
            locs (np.ndarray):
                A NumPy array containing the input locations for training.
            vals (np.ndarray):
                A NumPy array containing observed values corresponding to the `locs`.
            cats (np.ndarray, optional):
                A NumPy array containing categorical data for each observation in `locs`. If provided,
                the data is sorted according to `cats` to enable stratified training. Defaults to None.
            step_size (float, optional):
                The learning rate for the Adam optimizer.
            iters (int, optional):
                The total number of iterations to run for training.
            reg (float or None, optional):
                Regularization penalty parameter. If None, no regularization is applied.

        Returns:
            self (Model):
                The model instance with updated parameters, allowing for method chaining.

        Examples:
            Fitting a model using training data:

            ```python
            from geostat import GP, Model, Parameters
            from geostat.kernel import Noise
            import numpy as np

            # Create parameters.
            p = Parameters(nugget=1.)

            # Create model
            kernel = Noise(nugget=p.nugget)
            model = Model(GP(0, kernel))

            # Fit model
            locs = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
            vals = np.array([10.0, 15.0, 20.0])
            model.fit(locs, vals, step_size=0.05, iters=500)
            # [iter    50 ll -63.71 time  2.72 reg  0.00 nugget  6.37]
            # [iter   100 ll -32.94 time  0.25 reg  0.00 nugget 13.97]
            # [iter   150 ll -23.56 time  0.25 reg  0.00 nugget 22.65]
            # [iter   200 ll -19.26 time  0.25 reg  0.00 nugget 32.27]
            # [iter   250 ll -16.92 time  0.25 reg  0.00 nugget 42.63]
            # [iter   300 ll -15.52 time  0.24 reg  0.00 nugget 53.50]
            # [iter   350 ll -14.63 time  0.24 reg  0.00 nugget 64.71]
            # [iter   400 ll -14.03 time  0.24 reg  0.00 nugget 76.10]
            # [iter   450 ll -13.61 time  0.25 reg  0.00 nugget 87.52]
            # [iter   500 ll -13.32 time  0.24 reg  0.00 nugget 98.85]
            ```

            Using categorical data for training:

            ```python
            cats = np.array([1, 1, 2])
            model.fit(locs, vals, cats=cats, step_size=0.01, iters=300)
            # [iter    30 ll -12.84 time  0.25 reg  0.00 nugget 131.53]
            # [iter    60 ll -12.62 time  0.15 reg  0.00 nugget 164.41]
            # [iter    90 ll -12.53 time  0.16 reg  0.00 nugget 191.70]
            # [iter   120 ll -12.50 time  0.16 reg  0.00 nugget 211.74]
            # [iter   150 ll -12.49 time  0.15 reg  0.00 nugget 225.07]
            # [iter   180 ll -12.49 time  0.16 reg  0.00 nugget 233.15]
            # [iter   210 ll -12.49 time  0.15 reg  0.00 nugget 237.64]
            # [iter   240 ll -12.49 time  0.15 reg  0.00 nugget 239.92]
            # [iter   270 ll -12.49 time  0.15 reg  0.00 nugget 240.98]
            # [iter   300 ll -12.49 time  0.15 reg  0.00 nugget 241.42]
            ```

        Examples: Notes:
            - The `fit` method uses the Adam optimizer to minimize the negative log-likelihood (`ll`) and any regularization 
            penalties specified by `reg`.
            - During training, if `cats` are provided, data points are sorted according to `cats` to ensure grouped training.
            - The `verbose` flag determines whether training progress is printed after each iteration.
            - After training, parameter values are saved and can be accessed or updated using the model's attributes.
        """

        # Collect parameters and create TF parameters.
        parameters = self.gather_vars()

        # Permute datapoints if cats is given.
        if cats is not None:
            cats = np.array(cats)
            perm = np.argsort(cats)
            locs, vals, cats = locs[perm], vals[perm], cats[perm]
        else:
            cats = np.zeros(locs.shape[:1], np.int32)
            perm = None

        # Data dict.
        self.data = {
            'warplocs': self.warp(locs),
            'vals': tf.constant(vals, dtype=tf.float32),
            'cats': tf.constant(cats, dtype=tf.int32)}

        optimizer = tf.keras.optimizers.Adam(learning_rate=step_size)

        j = 0 # Iteration count.
        for i in range(10):
            t0 = time.time()
            while j < (i + 1) * iters / 10:
                ll, reg_penalty = gp_train_step(
                    optimizer, self.data, parameters, self.gp, reg)
                j += 1

            time_elapsed = time.time() - t0
            if self.verbose == True:
                self.report(
                  dict(iter=j, ll=ll, time=time_elapsed, reg=reg_penalty) |
                  {p.name: p.surface() for p in parameters.values()})

        # Save parameter values.
        for p in parameters.values():
            p.update_value()

        # Restore order if things were permuted.
        if perm is not None:
            revperm = np.argsort(perm)
            locs, vals, cats = locs[revperm], vals[revperm], cats[revperm]

        self.locs = locs
        self.vals = vals
        self.cats = cats

        return self

    def mcmc(self, locs, vals, cats=None,
            chains=4, step_size=0.1, move_prob=0.5,
            samples=1000, burnin=500, report_interval=100):

        assert samples % report_interval == 0, '`samples` must be a multiple of `report_interval`'
        assert burnin % report_interval == 0, '`burnin` must be a multiple of `report_interval`'

        # Permute datapoints if cats is given.
        if cats is not None:
            cats = np.array(cats)
            perm = np.argsort(cats)
            locs, vals, cats = locs[perm], vals[perm], cats[perm]

        # Data dict.
        self.data = {
            'locs': tf.constant(locs, dtype=tf.float32),
            'vals': tf.constant(vals, dtype=tf.float32),
            'cats': None if cats is None else tf.constant(cats, dtype=tf.int32)}

        # Initial MCMC state.
        initial_up = self.parameter_space.get_underlying(self.parameters)

        # Unnormalized log posterior distribution.
        def g(up):
            sp = self.parameter_space.get_surface(up)
            return gp_log_likelihood(self.data, sp, self.gp)

        def f(*up_flat):
            up = tf.nest.pack_sequence_as(initial_up, up_flat)
            ll = tf.map_fn(g, up, fn_output_signature=tf.float32)
            # log_prior = -tf.reduce_sum(tf.math.log(1. + tf.square(up_flat)), axis=0)
            return ll # + log_prior

        # Run the chain for a burst.
        @tf.function
        def run_chain(current_state, final_results, kernel, iters):
            samples, results, final_results = tfp.mcmc.sample_chain(
                num_results=iters,
                current_state=current_state,
                kernel=kernel,
                return_final_kernel_results=True,
                trace_fn=lambda _, results: results)

            return samples, results, final_results
        
        def new_state_fn(scale, dtype):
          direction_dist = tfd.Normal(loc=dtype(0), scale=dtype(1))
          scale_dist = tfd.Exponential(rate=dtype(1/scale))
          pick_dist = tfd.Bernoulli(probs=move_prob)

          def _fn(state_parts, seed):
            next_state_parts = []
            part_seeds = tfp.random.split_seed(
                seed, n=len(state_parts), salt='rwmcauchy')
            for sp, ps in zip(state_parts, part_seeds):
                pick = tf.cast(pick_dist.sample(sample_shape=sp.shape, seed=ps), tf.float32)
                direction = direction_dist.sample(sample_shape=sp.shape, seed=ps)
                scale_val = scale_dist.sample(seed=ps)
                next_state_parts.append(sp + tf.einsum('a...,a->a...', pick * direction, scale_val))
            return next_state_parts
          return _fn

        inv_temps = 0.5**np.arange(chains, dtype=np.float32)

        def make_kernel_fn(target_log_prob_fn):
            return tfp.mcmc.RandomWalkMetropolis(
                target_log_prob_fn=target_log_prob_fn,
                new_state_fn=new_state_fn(scale=step_size / np.sqrt(inv_temps), dtype=np.float32))

        kernel = tfp.mcmc.ReplicaExchangeMC(
            target_log_prob_fn=f,
            inverse_temperatures=inv_temps,
            make_kernel_fn=make_kernel_fn)

        # Do bursts.
        current_state = tf.nest.flatten(initial_up)
        final_results = None
        acc_states = []
        num_bursts = (samples + burnin) // report_interval
        burnin_bursts = burnin // report_interval
        for i in range(num_bursts):
            is_burnin = i < burnin_bursts

            if self.verbose and (i == 0 or i == burnin_bursts):
                print('BURNIN\n' if is_burnin else '\nSAMPLING')
            
            t0 = time.time()
            states, results, final_results = run_chain(current_state, final_results, kernel, report_interval)

            if self.verbose == True:
                if not is_burnin: print()
                accept_rates = results.post_swap_replica_results.is_accepted.numpy().mean(axis=0)
                print('[iter {:4d}] [time {:.1f}] [accept rates {}]'.format(
                    ((i if is_burnin else i - burnin_bursts) + 1) * report_interval,
                    time.time() - t0,
                    ' '.join([f'{x:.2f}' for x in accept_rates.tolist()])))

            if not is_burnin:
                acc_states.append(tf.nest.map_structure(lambda x: x.numpy(), states))
                all_states = [np.concatenate(x, 0) for x in zip(*acc_states)]
                up = tf.nest.pack_sequence_as(initial_up, all_states)
                sp = self.parameter_space.get_surface(up, numpy=True) 

                # Reporting
                if self.verbose == True:
                    for p in [5, 50, 95]:
                        x = tf.nest.map_structure(lambda x: np.percentile(x, p, axis=0), sp)
                        self.report(x, prefix=f'{p:02d}%ile')

            current_state = [s[-1] for s in states]

        posterior = self.parameter_space.get_surface(up, numpy=True)

        # Restore order if things were permuted.
        if cats is not None:
            revperm = np.argsort(perm)
            locs, vals, cats = locs[revperm], vals[revperm], cats[revperm]

        return replace(self, 
            parameters=posterior,
            parameter_sample_size=samples,
            locs=locs, vals=vals, cats=cats)

    def generate(self, locs, cats=None):
        """
        Generates synthetic data values from the Gaussian Process (GP) model based on the provided location data.
        This method simulates values based on the GP's covariance structure, allowing for random sample generation.

        Parameters:
            locs (np.ndarray):
                A NumPy array containing the input locations for which to generate synthetic values.
            cats (np.ndarray, optional):
                A NumPy array containing categorical data corresponding to `locs`. If provided, data points 
                are permuted according to `cats` for stratified generation. Defaults to None.

        Returns:
            self (Model):
                The model instance with generated values stored in `self.vals` and corresponding locations stored 
                in `self.locs`. This enables method chaining.

        Examples:
            Generating synthetic values for a set of locations:

            ```python
            from geostat import GP, Model, Parameters
            from geostat.kernel import Noise
            import numpy as np

            # Create parameters.
            p = Parameters(nugget=1.)

            # Create model
            kernel = Noise(nugget=p.nugget)
            model = Model(GP(0, kernel))

            # Generate values based on locs
            locs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            model.generate(locs)
            generated_vals = model.vals  # Access the generated values
            print(generated_vals)
            # [0.45151083 1.23276189 0.3822659 ] (Values are non-deterministic)
            ```

        Examples: Notes:
            - Conditional generation is currently not supported, and this method will raise an assertion error if 
            `self.locs` and `self.vals` are already defined.
            - Generation from a distribution is not yet supported, and an assertion error will be raised if 
            `self.parameter_sample_size` is not `None`.
            - If `cats` are provided, the data is permuted according to `cats` for stratified generation, and 
            the original order is restored before returning.
        """

        assert self.locs is None and self.vals is None, 'Conditional generation not yet supported'
        assert self.parameter_sample_size is None, 'Generation from a distribution not yet supported'

        locs = np.array(locs)

        # Permute datapoints if cats is given.
        if cats is not None:
            cats = np.array(cats)
            perm = np.argsort(cats)
            locs, cats = locs[perm], cats[perm]
        else:
            cats = np.zeros(locs.shape[:1], np.int32)
            perm = None

        m, S = gp_covariance(
            self.gp,
            self.warp(locs).run({}),
            None if cats is None else tf.constant(cats, dtype=tf.int32))

        vals = MVN(m, tf.linalg.cholesky(S)).sample().numpy()

        # Restore order if things were permuted.
        if perm is not None:
            revperm = np.argsort(perm)
            locs, vals, cats = locs[revperm], vals[revperm], cats[revperm]

        self.locs = locs
        self.vals = vals
        self.cats = cats

        return self

    def predict(self, locs2, cats2=None, *, subsample=None, reduce=None, tracker=None, pair=False):
        """
        Performs Gaussian Process (GP) predictions of the mean and variance for the given location data.
        Supports batch predictions for large datasets and can handle categorical data.

        Parameters:
            locs2 (np.ndarray):
                A NumPy array containing the input locations for which predictions are to be made.
            cats2 (np.ndarray, optional):
                A NumPy array containing categorical data for the prediction locations (`locs2`). If provided,
                the data points will be permuted according to `cats2`. Default is None.
            subsample (int, optional):
                Specifies the number of parameter samples to be used for prediction when `parameter_sample_size` is set.
                Only valid if parameters are sampled. Default is None.
            reduce (str, optional):
                Specifies the reduction method ('mean' or 'median') to aggregate predictions from multiple parameter samples.
                Only valid if parameters are sampled. Default is None.
            tracker (Callable, optional):
                A tracking function for monitoring progress when making predictions across multiple samples. Default is None.
            pair (bool, optional):
                If True, performs pairwise predictions of mean and variance for each pair of input points in `locs2`.

        Returns:
            m (np.ndarray):
                The predicted mean values for the input locations.
            v (np.ndarray):
                The predicted variances for the input locations.

        Examples:
            Making predictions for a set of locations:

            ```python
            from geostat import GP, Model, Parameters
            from geostat.kernel import SquaredExponential
            import numpy as np

            # Create parameters.
            p = Parameters(sill=1.0, range=2.0)

            # Create model
            kernel = SquaredExponential(sill=p.sill, range=p.range)
            model = Model(GP(0, kernel))

            # Fit model
            locs = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
            vals = np.array([10.0, 15.0, 20.0])
            model.fit(locs, vals, step_size=0.05, iters=500)
            # [iter    50 ll -40.27 time  2.29 reg  0.00 sill  6.35 range  1.96]
            # [iter   100 ll -21.79 time  0.40 reg  0.00 sill 13.84 range  2.18]
            # [iter   150 ll -16.17 time  0.39 reg  0.00 sill 22.31 range  2.44]
            # [iter   200 ll -13.55 time  0.39 reg  0.00 sill 31.75 range  2.76]
            # [iter   250 ll -12.08 time  0.38 reg  0.00 sill 42.08 range  3.12]
            # [iter   300 ll -11.14 time  0.38 reg  0.00 sill 53.29 range  3.48]
            # [iter   350 ll -10.50 time  0.38 reg  0.00 sill 65.36 range  3.85]
            # [iter   400 ll -10.05 time  0.39 reg  0.00 sill 78.29 range  4.22]
            # [iter   450 ll -9.70 time  0.39 reg  0.00 sill 92.07 range  4.59]
            # [iter   500 ll -9.43 time  0.39 reg  0.00 sill 106.70 range  4.95]

            # Run predictions
            locs2 = np.array([[1.5, 1.5], [2.5, 4.0]])
            mean, variance = model.predict(locs2)
            print(mean)
            # [ 9.89839798 18.77077269]
            print(variance)
            # [2.1572128  0.54444738]
            ```

        Examples: Notes:
            - If `subsample` is specified, it should be used only when `parameter_sample_size` is defined.
            - The `reduce` parameter allows aggregation of predictions, but it's valid only with sampled parameters.
            - The method supports pairwise predictions by setting `pair=True`, which is useful for predicting 
            the covariance between two sets of locations.
            - The internal `interpolate_batch` and `interpolate_pair_batch` functions handle the prediction computations
            in a batched manner to support large datasets efficiently.
        """

        assert subsample is None or self.parameter_sample_size is not None, \
            '`subsample` is only valid with sampled parameters'

        assert reduce is None or self.parameter_sample_size is not None, \
            '`reduce` is only valid with sampled parameters'

        assert subsample is None or reduce is None, \
            '`subsample` and `reduce` cannot both be given'

        if tracker is None: tracker = lambda x: x

        assert self.locs.shape[-1] == locs2.shape[-1], 'Mismatch in location dimensions'
        if cats2 is not None:
            assert cats2.shape == locs2.shape[:1], 'Mismatched shapes in cats and locs'
        else:
            cats2 = np.zeros(locs2.shape[:1], np.int32)

        def interpolate_batch(A11i, locs1, vals1diff, cats1, locs2, cats2):
            """
            Inputs:
              locs1.shape = [N1, K]
              vals1diff.shape = [N1]
              cats1.shape = [N1]
              locs2.shape = [N2, K]
              cats2.shape = [N2]

            Outputs:
              u2_mean.shape = [N2]
              u2_var.shape = [N2]
            """

            N1 = len(locs1) # Number of measurements.

            # Permute datapoints if cats is given.
            if cats2 is not None:
                perm = np.argsort(cats2)
                locs2, cats2 = locs2[perm], cats2[perm]
                locs2 = self.warp(locs2).run({})

            _, A12 = gp_covariance2(
                self.gp,
                tf.constant(locs1, dtype=tf.float32),
                tf.constant(cats1, dtype=tf.int32),
                tf.constant(locs2, dtype=tf.float32),
                tf.constant(cats2, dtype=tf.int32),
                N1)

            m2, A22 = gp_covariance(
                self.gp,
                tf.constant(locs2, dtype=tf.float32),
                tf.constant(cats2, dtype=tf.int32))

            # Restore order if things were permuted.
            if cats2 is not None:
                revperm = np.argsort(perm)
                m2 = tf.gather(m2, revperm)
                A12 = tf.gather(A12, revperm, axis=-1)
                A22 = tf.gather(tf.gather(A22, revperm), revperm, axis=-1)

            u2_mean = m2 + tf.einsum('ab,a->b', A12, tf.einsum('ab,b->a', A11i, vals1diff))
            u2_var = tf.linalg.diag_part(A22) -  tf.einsum('ab,ab->b', A12, tf.matmul(A11i, A12))

            return u2_mean, u2_var

        def interpolate_pair_batch(A11i, locs1, vals1diff, cats1, locs2, cats2):
            """
            Inputs:
              locs1.shape = [N1, K]
              vals1diff.shape = [N1]
              cats1.shape = [N1]
              locs2.shape = [N2, 2, K]
              cats2.shape = [N2]

            Outputs:
              u2_mean.shape = [N2, 2]
              u2_var.shape = [N2, 2, 2]
            """

            N1 = len(locs1) # Number of measurements.
            N2 = len(locs2) # Number of prediction pairs.

            # Permute datapoints if cats is given.
            perm = np.argsort(cats2)
            locs2, cats2 = locs2[perm], cats2[perm]

            # Warp locs2.
            locs2_shape = locs2.shape
            locs2 = locs2.reshape([-1, locs2_shape[-1]])  # Shape into matrix.
            locs2 = self.warp(locs2).run({})
            locs2 = tf.reshape(locs2, locs2_shape)  # Revert shape.

            _, A12 = gp_covariance2(
                self.gp,
                tf.constant(locs1, dtype=tf.float32),
                tf.constant(cats1, dtype=tf.int32),
                tf.constant(locs2[:, 0, :], dtype=tf.float32),
                tf.constant(cats2, dtype=tf.int32),
                N1)

            _, A13 = gp_covariance2(
                self.gp,
                tf.constant(locs1, dtype=tf.float32),
                tf.constant(cats1, dtype=tf.int32),
                tf.constant(locs2[:, 1, :], dtype=tf.float32),
                tf.constant(cats2, dtype=tf.int32),
                N1)

            m2, A22 = gp_covariance(
                self.gp,
                tf.constant(locs2[:, 0, :], dtype=tf.float32),
                tf.constant(cats2, dtype=tf.int32))

            m3, A33 = gp_covariance(
                self.gp,
                tf.constant(locs2[:, 1, :], dtype=tf.float32),
                tf.constant(cats2, dtype=tf.int32))

            _, A23 = gp_covariance2(
                self.gp,
                tf.constant(locs2[:, 0, :], dtype=tf.float32),
                tf.constant(cats2, dtype=tf.int32),
                tf.constant(locs2[:, 1, :], dtype=tf.float32),
                tf.constant(cats2, dtype=tf.int32),
                N2)

            # Reassemble into more useful shapes.

            A12 = tf.stack([A12, A13], axis=-1) # [N1, N2, 2]

            m2 = tf.stack([m2, m3], axis=-1) # [N2, 2]

            A22 = tf.linalg.diag_part(A22)
            A33 = tf.linalg.diag_part(A33)
            A23 = tf.linalg.diag_part(A23)
            A22 = tf.stack([tf.stack([A22, A23], axis=-1), tf.stack([A23, A33], axis=-1)], axis=-2) # [N2, 2, 2]

            # Restore order if things were permuted.
            revperm = np.argsort(perm)
            m2 = tf.gather(m2, revperm)
            A12 = tf.gather(A12, revperm, axis=1)
            A22 = tf.gather(A22, revperm)

            u2_mean = m2 + tf.einsum('abc,a->bc', A12, tf.einsum('ab,b->a', A11i, vals1diff))
            u2_var = A22 - tf.einsum('abc,abd->bcd', A12, tf.einsum('ae,ebd->abd', A11i, A12))

            return u2_mean, u2_var

        def interpolate(locs1, vals1, cats1, locs2, cats2, pair=False):
            # Interpolate in batches.
            batch_size = locs1.shape[0] // 2

            for_gp = []

            for start in np.arange(0, len(locs2), batch_size):
                stop = start + batch_size
                subset = locs2[start:stop], cats2[start:stop]
                for_gp.append(subset)

            # Permute datapoints if cats is given.
            if cats1 is not None:
                perm = np.argsort(cats1)
                locs1, vals1, cats1 = locs1[perm], vals1[perm], cats1[perm]

            locs1 = self.warp(locs1).run({})

            m1, A11 = gp_covariance(
                self.gp,
                tf.constant(locs1, dtype=tf.float32),
                tf.constant(cats1, dtype=tf.int32))

            A11i = tf.linalg.inv(A11)

            u2_mean_s = []
            u2_var_s = []

            f = interpolate_pair_batch if pair else interpolate_batch

            for locs_subset, cats_subset in for_gp:
                u2_mean, u2_var = f(A11i, locs1, vals1 - m1, cats1, locs_subset, cats_subset)
                u2_mean = u2_mean.numpy()
                u2_var = u2_var.numpy()
                u2_mean_s.append(u2_mean)
                u2_var_s.append(u2_var)

            u2_mean = np.concatenate(u2_mean_s)
            u2_var = np.concatenate(u2_var_s)

            return u2_mean, u2_var

        if self.parameter_sample_size is None:
            m, v = interpolate(self.locs, self.vals, self.cats, locs2, cats2, pair)
        elif reduce == 'median':
            raise NotImplementedError
            p = tf.nest.map_structure(lambda x: np.quantile(x, 0.5, axis=0).astype(np.float32), self.parameters)
            m, v = interpolate(self.locs, self.vals, self.cats, locs2, cats2, p, pair)
        elif reduce == 'mean':
            raise NotImplementedError
            p = tf.nest.map_structure(lambda x: x.mean(axis=0).astype(np.float32), self.parameters)
            m, v = interpolate(self.locs, self.vals, self.cats, locs2, cats2, p, pair)
        else:
            raise NotImplementedError
            samples = self.parameter_sample_size

            if subsample is not None:
                assert subsample <= samples, '`subsample` may not exceed sample size'
            else:
                subsample = samples

            # Thin by picking roughly equally-spaced samples.
            a = np.arange(samples) * subsample / samples % 1
            pick = np.concatenate([[True], a[1:] >= a[:-1]])
            parameters = tf.nest.map_structure(lambda x: x[pick], self.parameters)

            # Make a prediction for each sample.
            results = []
            for i in tracker(range(subsample)):
                p = tf.nest.map_structure(lambda x: x[i], parameters)
                results.append(interpolate(self.locs, self.vals, self.cats, locs2, cats2, p, pair))

            mm, vv = [np.stack(x) for x in zip(*results)]
            m = mm.mean(axis=0)
            v = (np.square(mm) + vv).mean(axis=0) - np.square(m)

        return m, v
