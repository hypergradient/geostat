from dataclasses import dataclass, replace
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import expit
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Tensorflow is extraordinarily noisy. Catch warnings during import.
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow as tf
    import tensorflow_probability as tfp

from .spatialinterpolator import SpatialInterpolator

MVN = tfp.distributions.MultivariateNormalTriL

__all__ = ['GP', 'NormalizingFeaturizer']

# Produces featurized locations (F matrix) and remembers normalization parameters.
class NormalizingFeaturizer:
    def __init__(self, featurization, locs):
        self.featurization = featurization
        F_unnorm = self.get_unnorm_features(locs)
        self.unnorm_mean = np.mean(F_unnorm, axis=0)
        self.unnorm_std = np.std(F_unnorm, axis=0)

    def get_unnorm_features(self, locs):
        if self.featurization is None: # No features.
            return np.ones([locs.shape[0], 0])

        feats = self.featurization(*np.transpose(locs))
        if isinstance(feats, tuple): # One or many features.
            if len(feats) == 0:
                return np.ones([locs.shape[0], 0])
            else:
                return np.stack(self.featurization(*np.transpose(locs)), axis=1)
        else: # One feature.
            return feats[:, np.newaxis]

    def __call__(self, locs):
        ones = np.ones([locs.shape[0], 1])
        F_unnorm = self.get_unnorm_features(locs)
        F_norm = (F_unnorm - self.unnorm_mean) / self.unnorm_std
        return np.concatenate([ones, F_norm], axis=1)

def e(x, a=-1):
    return tf.expand_dims(x, a)

def one_axes(x):
    return tf.where(tf.equal(x.shape, 1))[:, 0]

# Squared exponential covariance function.
def gp_covariance_sq_exp(D, F, range_, sill, nugget, alpha):
    C = sill * tf.exp(-tf.square(D / range_)) \
            + (nugget + 1e-6) * tf.eye(D.shape[0], dtype=tf.float64) \
            + alpha * tf.einsum('ba,ca->bc', F, F)
    return C

# Gamma exponential covariance function.
@tf.custom_gradient
def safepow(x, a):
    y = tf.pow(x, a)
    def grad(dy):
        dx = tf.where(x <= 0.0, tf.zeros_like(x), dy * tf.pow(x, a-1))
        dx = tf.reduce_sum(dx, axis=one_axes(x), keepdims=True)
        da = tf.where(x <= 0.0, tf.zeros_like(a), dy * y * tf.math.log(x))
        da = tf.reduce_sum(da, axis=one_axes(a), keepdims=True)
        return dx, da
    return y, grad

def gamma_exp(d2, halfgamma):
    return tf.exp(-safepow(tf.maximum(d2, 0.0), halfgamma))

def gp_covariance_gamma_exp(D, F, range_, sill, nugget, halfgamma, alpha):
    range_ = e(e(range_))
    sill = e(e(sill))
    halfgamma = e(e(halfgamma))
    C = sill * gamma_exp(tf.square(D / range_), halfgamma) + (nugget + 1e-6) * tf.eye(tf.shape(D)[0], dtype=tf.float64)
    C += tf.einsum('ba,ca->bc', F, F) * alpha
    return C

# Transform parameters.
@tf.function
def gp_xform_parameters(parameters, covariance_func):
    """
    Transform parameters from the underlying representation
    (which has the whole real number line as its range) to
    a surface representation (which is bounded).
    """
    range_ = tf.exp(parameters['log_range'])
    sill = tf.exp(parameters['log_sill'])
    nugget = tf.exp(parameters['log_nugget'])

    if covariance_func == 'squared-exp':
        param_dict = {'range': range_, 'sill': sill, 'nugget': nugget}
        return param_dict

    elif covariance_func == 'gamma-exp':
        halfgamma = tf.sigmoid(parameters['logit_halfgamma'])
        param_dict = {'range': range_, 'sill': sill, 'nugget': nugget, 'halfgamma': halfgamma}
        return param_dict

# Log likelihood.
@tf.function
def gp_log_likelihood(u, m, cov):
    """Log likelihood of is the PDF of a multivariate gaussian."""
    u_adj = u - m
    logdet = tf.linalg.logdet(2 * np.pi * cov)
    quad = tf.matmul(e(u_adj, 0), tf.linalg.solve(cov, e(u_adj, -1)))[0, 0]
    return -0.5 * (logdet + quad)


# GP training.
def gpm_train_step(optimizer, data, parameters, hyperparameters, covariance_func):
    with tf.GradientTape() as tape:
        p = gp_xform_parameters(parameters, covariance_func)
        beta_prior = hyperparameters['alpha']

        if covariance_func == 'squared-exp':
            A = gp_covariance_sq_exp(data['D'], data['F'], p['range'], p['sill'], p['nugget'], beta_prior)
        elif covariance_func == 'gamma-exp':
            A = gp_covariance_gamma_exp(data['D'], data['F'], p['range'], p['sill'], p['nugget'], p['halfgamma'], beta_prior)

        ll = gp_log_likelihood(data['u'], 0., A)

        if hyperparameters['reg'] != None:
            reg = hyperparameters['reg'] * tf.reduce_sum(tf.square(parameters['log_range']))
            loss = -ll + reg
        else:
            loss = -ll

    gradients = tape.gradient(loss, parameters.values())
    optimizer.apply_gradients(zip(gradients, parameters.values()))
    return p, ll

@dataclass
class GP(SpatialInterpolator):

    featurizer: NormalizingFeaturizer
    covariance_func: str = 'squared-exp'
    parameters: object = None
    hyperparameters: object = None
    verbose: bool = True

    def __post_init__(self):

        '''
        Parameters:
                x : Pandas DataFrame with columns for locations.

                u : A Pandas Series containing observations.

                featurization : function, optional
                    Should be a function that takes x1 (n-dim array of input data)
                    and returns the coordinates, i.e., x, y, x**2, y**2.
                    Example: def featurization(x1):
                                return x1[:, 0], x1[:, 1], x1[:, 0]**2, x1[:, 1]**2.
                    Default is None.

                covariance_func : str
                     Name of the covariance function to use in the GP.
                     Should be 'squared-exp' or 'gamma-exp'.
                     Default is 'squared-exp'.

                parameters : dict, optional
                    The starting point for the parameters.
                    Example: parameters=dict(range=2.0, sill=5.0, nugget=1.0).
                    Default is None.

                hyperparameters : dict
                    Dictionary of the hyperparameters.
                      - alpha: the prior distribution for the trend. Default 10.
                      - reg: how much regularization to use. Default None (no regularization).
                      - train_iters: number of training iterations. Default 300.

                verbose : boolean, optional
                    Whether or not to print parameters.
                    Default is True.

        Performs Gaussian process training and prediction.
        '''

        super().__init__()

        # Supply defaults.
        default_hyperparameters = dict(alpha=10, reg=None, train_iters=300)
        self.hyperparameters = dict(default_hyperparameters, **self.hyperparameters)

        # Default parameters.
        if self.parameters == None:
            if self.covariance_func == 'squared-exp':
                self.parameters = dict(range=1.0, sill=1.0, nugget=1.0)
            elif self.covariance_func == 'gamma-exp':
                self.parameters = dict(range=1.0, sill=1.0, nugget=1.0, gamma=1.0)

        # Other needed functions.
        def logodds(p):
            return -np.log(1/p-1)

        def logodds_half(x):
            return logodds(x/2)

        # def sigmoid(o): return 1 / (1 + np.exp(-o))
        # def doub_sigmoid(o): return 2 * sigmoid(o)

        # List the needed graph functions.

        # Set the user desired covariance function.
        if self.covariance_func not in ['squared-exp', 'gamma-exp']:
            raise ValueError("Only 'squared-exp' and 'gamma-exp' are currently supported.")

        # Define other inputs.
        self.train_iters = self.hyperparameters['train_iters']

        if self.hyperparameters['reg']:
            self.hp = {'alpha': tf.constant(self.hyperparameters['alpha'], dtype=tf.float64),
                                    'reg': tf.constant(self.hyperparameters['reg'], dtype=tf.float64)}
        if not self.hyperparameters['reg']:
            self.hp = {'alpha': tf.constant(self.hyperparameters['alpha'], dtype=tf.float64),
                                    'reg': None}

    def fit(self, x, u):
        # Distance matrix.
        D = cdist(x, x)

        # Feature matrix.
        F = self.featurizer(x)

        # Data dict.
        self.data = {'D': tf.constant(D), 'F': tf.constant(F), 'u': tf.constant(u)}

        # Train the GP.
        def gpm_fit(data, parameters, hyperparameters):
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

            j = 0 # Iteration count.
            for i in range(10):
                while j < (i + 1) * self.train_iters / 10:
                    p, ll = gpm_train_step(optimizer, data, parameters, hyperparameters, self.covariance_func)
                    j += 1

                if self.verbose == True:
                    if self.covariance_func == 'squared-exp':
                        print('[iter %d] [ll %7.2f] [range %4.2f, sill %4.2f, nugget %4.2f]' %
                            (j, ll, p['range'], p['sill'], p['nugget']))

                    elif self.covariance_func == 'gamma-exp':
                        print('[iter %d] [ll %7.2f] [range %4.2f, sill %4.2f, nugget %4.2f, gamma %4.2f]' %
                            (j, ll, p['range'], p['sill'], p['nugget'], p['halfgamma'] * 2))

        up = self.get_underlying_parameters()

        gpm_fit(self.data, up, self.hyperparameters)

        return replace(self, parameters = self.get_surface_parameters(up))

    def get_underlying_parameters(self):
        if self.covariance_func == 'squared-exp':
            up = dict(
                log_range = tf.Variable(np.log(self.parameters['range']), dtype=tf.float64),
                log_sill = tf.Variable(np.log(self.parameters['sill']), dtype=tf.float64),
                log_nugget = tf.Variable(np.log(self.parameters['nugget']), dtype=tf.float64))
        elif self.covariance_func == 'gamma-exp':
            up = dict(
                log_range = tf.Variable(np.log(self.parameters['range']), dtype=tf.float64),
                log_sill = tf.Variable(np.log(self.parameters['sill']), dtype=tf.float64),
                log_nugget = tf.Variable(np.log(self.parameters['nugget']), dtype=tf.float64),
                logit_halfgamma = tf.Variable(logodds_half(self.parameters['gamma']), dtype=tf.float64))
        return up

    def get_surface_parameters(self, up):
        sp = {}
        sp['range'] = np.exp(up['log_range'].numpy())
        sp['sill'] = np.exp(up['log_sill'].numpy())
        sp['nugget'] = np.exp(up['log_nugget'].numpy())
        if 'logit_halfgamma' in up:
            sp['gamma'] = 2.0 * expit(parameters['logit_halfgamma'])
        return sp

    def generate(self, x):
        xr = x.reshape([-1, x.shape[-1]])

        D = cdist(xr, xr)

        F = self.featurizer(xr)

        up = self.get_underlying_parameters()

        p = gp_xform_parameters(up, self.covariance_func)

        if self.covariance_func == 'squared-exp':
            A = gp_covariance_sq_exp(D, F, p['range'], p['sill'], p['nugget'], self.hyperparameters['alpha'])
        elif self.covariance_func == 'gamma-exp':
            A = gp_covariance_gamma_exp(D, F, p['range'], p['sill'], p['nugget'], p['halfgamma'], self.hyperparameters['alpha'])

        z = tf.zeros_like(A[0, :])

        return MVN(z, tf.linalg.cholesky(A)).sample().numpy().reshape(x.shape[:-1])

    def predict(self, x1, u1, x2):

        '''
        Parameters:
                x2 : n-dim array
                    Locations to make predictions.

        Returns:
                u2_mean : array
                    GP mean.

                u2_var : array
                    GP variance.


        Performs GP predictions of the mean and variance.
        Has support for batch predictions for large data sets.

        '''

        # Define inputs.
        self.batch_size = x1.shape[0] // 2


        # Needed functions.
        def e(x, a=-1):
            return tf.expand_dims(x, a)

        def one_axes(x):
            return tf.where(tf.equal(x.shape, 1))[:, 0]

        ##############################################
        def interpolate_gp(X1, u1, X2, parameters, hyperparameters):

            N1 = len(X1) # Number of measurements.
            N2 = len(X2) # Number of predictions.

            X = np.concatenate([X1, X2], axis=0)
            D = cdist(X, X)

            # Use the user given featurization.
            F = self.featurizer(X)

            p = gp_xform_parameters(parameters, self.covariance_func)

            if self.covariance_func == 'squared-exp':
                A = gp_covariance_sq_exp(D, F, p['range'], p['sill'], p['nugget'], hyperparameters['alpha'])
            elif self.covariance_func == 'gamma-exp':
                A = gp_covariance_gamma_exp(D, F, p['range'], p['sill'], p['nugget'], p['halfgamma'], hyperparameters['alpha'])

            A11 = A[:N1, :N1]
            A12 = A[:N1, N1:]
            A21 = A[N1:, :N1]
            A22 = A[N1:, N1:]

            u2_mean = tf.matmul(A21, tf.linalg.solve(A11, e(u1, -1)))[:, 0]
            u2_var = tf.linalg.diag_part(A22) -  tf.reduce_sum(A12 * tf.linalg.solve(A11, A12), axis=0)

            return u2_mean, u2_var


        # Interpolate in batches.
        for_gp = []
        x2r = x2.reshape([-1, x2.shape[-1]])

        for start in np.arange(0, len(x2r), self.batch_size):
            stop = start + self.batch_size
            subset = x2r[start:stop]
            for_gp.append(subset)

        up = self.get_underlying_parameters()

        u2_mean_s = []
        u2_var_s = []

        for subset in for_gp:
            u2_mean, u2_var = interpolate_gp(x1, u1, subset, up, self.hyperparameters)
            u2_mean = u2_mean.numpy()
            u2_var = u2_var.numpy()
            u2_mean_s.append(u2_mean)
            u2_var_s.append(u2_var)

        u2_mean = np.concatenate(u2_mean_s).reshape(x2.shape[:-1])
        u2_var = np.concatenate(u2_var_s).reshape(x2.shape[:-1])

        return u2_mean, u2_var
