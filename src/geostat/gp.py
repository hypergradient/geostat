from dataclasses import dataclass, replace
import numpy as np
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
from .covfunc import CovarianceFunction

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

# Other needed functions.
def logodds(p):
    return -np.log(1/p-1)

def logodds_half(x):
    return logodds(x/2)

def e(x, a=-1):
    return tf.expand_dims(x, a)

def gp_covariance(covariance, X, F, p, alpha):
    X = tf.cast(X, tf.float32)
    F = tf.cast(F, tf.float32)
    C = covariance.matrix(X, p)
    C += 1e-6 * tf.eye(X.shape[0])
    C += alpha * tf.einsum('ba,ca->bc', F, F)
    return tf.cast(C, tf.float64)

# Transform parameters.
@tf.function
def gp_xform_parameters(up):
    """
    Transform parameters from the underlying representation
    (which has the whole real number line as its range) to
    a surface representation (which is bounded).
    """
    sp = {}
    for v in up.keys():
        if v == 'gamma':
            sp[v] = 2.0 * tf.sigmoid(up[v])
        else:
            sp[v] = tf.exp(up[v])
    return sp

# Log likelihood.
@tf.function
def gp_log_likelihood(u, m, cov):
    """Log likelihood of is the PDF of a multivariate gaussian."""
    u_adj = u - m
    logdet = tf.linalg.logdet(2 * np.pi * cov)
    quad = tf.matmul(e(u_adj, 0), tf.linalg.solve(cov, e(u_adj, -1)))[0, 0]
    return tf.cast(-0.5 * (logdet + quad), tf.float32)


# GP training.
def gpm_train_step(optimizer, data, parameters, hyperparameters, covariance):
    with tf.GradientTape() as tape:
        p = gp_xform_parameters(parameters)
        beta_prior = hyperparameters['alpha']

        A = gp_covariance(covariance, data['X'], data['F'], p, hyperparameters['alpha'])

        ll = gp_log_likelihood(data['u'], 0., A)

        if hyperparameters['reg'] != None:
            reg = hyperparameters['reg'] * covariance.reg(p)
        else:
            reg = 0.

        loss = -ll + reg

    gradients = tape.gradient(loss, parameters.values())
    optimizer.apply_gradients(zip(gradients, parameters.values()))
    return p, ll

@dataclass
class GP(SpatialInterpolator):

    featurizer: NormalizingFeaturizer
    covariance: CovarianceFunction
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

                covariance : CovarianceFunction
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

        # def sigmoid(o): return 1 / (1 + np.exp(-o))
        # def doub_sigmoid(o): return 2 * sigmoid(o)

        # List the needed graph functions.

        # Define other inputs.
        self.train_iters = self.hyperparameters['train_iters']

        if self.hyperparameters['reg']:
            self.hp = {'alpha': tf.constant(self.hyperparameters['alpha'], dtype=tf.float32),
                       'reg': tf.constant(self.hyperparameters['reg'], dtype=tf.float32)}
        if not self.hyperparameters['reg']:
            self.hp = {'alpha': tf.constant(self.hyperparameters['alpha'], dtype=tf.float32),
                       'reg': None}

    def fit(self, x, u):
        # Feature matrix.
        F = self.featurizer(x)

        # Data dict.
        self.data = {'X': tf.constant(x), 'F': tf.constant(F), 'u': tf.constant(u)}

        # Train the GP.
        def gpm_fit(data, parameters, hyperparameters):
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

            j = 0 # Iteration count.
            for i in range(10):
                while j < (i + 1) * self.train_iters / 10:
                    p, ll = gpm_train_step(optimizer, data, parameters, hyperparameters, self.covariance)
                    j += 1

                if self.verbose == True:
                    s = '[iter %4d, ll %7.2f]' % (j, ll)
                    s += self.covariance.report(p)
                    print(s)

        up = self.get_underlying_parameters()

        gpm_fit(self.data, up, self.hyperparameters)

        return replace(self, parameters = self.get_surface_parameters(up))

    def get_underlying_parameters(self):
        up = {}
        for v in self.parameters.keys():
            if v == 'gamma':
                up[v] = tf.Variable(logodds_half(self.parameters[v]), dtype=tf.float32)
            else:
                up[v] = tf.Variable(np.log(self.parameters[v]), dtype=tf.float32)
        return up

    def get_surface_parameters(self, up):
        sp = {}
        for v in up.keys():
            if v == 'gamma':
                sp[v] = 2.0 * expit(up[v])
            else:
                sp[v] = np.exp(up[v])
        return sp

    def generate(self, x):
        X = x.reshape([-1, x.shape[-1]])

        F = self.featurizer(X)

        up = self.get_underlying_parameters()

        p = gp_xform_parameters(up)

        A = gp_covariance(self.covariance, X, F, p, self.hyperparameters['alpha'])

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

        def interpolate_gp(X1, u1, X2, parameters, hyperparameters):

            N1 = len(X1) # Number of measurements.
            N2 = len(X2) # Number of predictions.

            X = np.concatenate([X1, X2], axis=0)

            F = self.featurizer(X)

            p = gp_xform_parameters(parameters)

            A = gp_covariance(self.covariance, X, F, p, hyperparameters['alpha'])

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
