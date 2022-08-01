import numpy as np
from scipy.spatial.distance import cdist
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Tensorflow is extraordinarily noisy. Catch warnings during import.
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow as tf

from .spatialinterpolator import SpatialInterpolator

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

class GP(SpatialInterpolator):
    
    def __init__(self, 
                 x=None,
                 u=None,
                 featurizer=None,
                 covariance_func='squared-exp',
                 parameters=None,
                 hyperparameters=None,
                 verbose=True,
                 ):
        
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
        hyperparameters = dict(default_hyperparameters, **hyperparameters)

        # This provides a filter to create the tf.Variables() only if the call_flag is None.
        # This is needed to avoid "ValueError: tf.function-decorated function 
        # tried to create variables on non-first call."
        self.call_flag = None
        
        # Other needed functions.
        def logodds(p): 
            return -np.log(1/p-1)
        
        def logodds_half(x): 
            return logodds(x/2)
        
        # def sigmoid(o): return 1 / (1 + np.exp(-o))
        # def doub_sigmoid(o): return 2 * sigmoid(o)
    
        # List the needed graph functions.
        
        # Transform parameters.
        @tf.function
        def gp_xform_parameters(parameters):
            """
            Transform parameters from the underlying representation
            (which has the whole real number line as its range) to
            a surface representation (which is bounded).
            """
            range_ = tf.exp(parameters['log_range'])
            sill = tf.exp(parameters['log_sill'])
            nugget = tf.exp(parameters['log_nugget'])
            
            if self.covariance_func == 'squared-exp':
                param_dict = {'range': range_, 'sill': sill, 'nugget': nugget}
                return param_dict
                
            elif self.covariance_func == 'gamma-exp':
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
        @tf.function
        def gpm_train_step(optimizer, data, parameters, hyperparameters):
            with tf.GradientTape() as tape:
                p = gp_xform_parameters(parameters)
                beta_prior = hyperparameters['alpha']
                
                if self.covariance_func == 'squared-exp':
                    A = gp_covariance_sq_exp(data['D'], data['F'], p['range'], p['sill'], p['nugget'], beta_prior)
                elif self.covariance_func == 'gamma-exp':
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

        # Set the user desired covariance function.
        if covariance_func not in ['squared-exp', 'gamma-exp']:
            raise ValueError("Only 'squared-exp' and 'gamma-exp' are currently supported.")
        self.covariance_func = covariance_func
        
        # Define other inputs.
        self.verbose = verbose
        self.train_iters = hyperparameters['train_iters']
        self.gp_xform_parameters = gp_xform_parameters  # Need for predict.
        
        if hyperparameters['reg']:
            self.hyperparameters = {'alpha': tf.constant(hyperparameters['alpha'], dtype=tf.float64),
                                    'reg': tf.constant(hyperparameters['reg'], dtype=tf.float64)}
        if not hyperparameters['reg']:
            self.hyperparameters = {'alpha': tf.constant(hyperparameters['alpha'], dtype=tf.float64),
                                    'reg': None}
        
        
        # Build the tf.Variable() dict.
        if parameters != None:
            if self.call_flag is None:
                if self.covariance_func == 'squared-exp':
                    # Log the starting point parameters that where provided.
                    for key in parameters:    
                        parameters[key] = np.log(parameters[key])
                        
                    self.parameters = {
                          'log_range': tf.Variable(parameters['range'], dtype=tf.float64),
                          'log_sill': tf.Variable(parameters['sill'], dtype=tf.float64),
                          'log_nugget': tf.Variable(parameters['nugget'], dtype=tf.float64)}
                    # print(self.parameters.items())
                
                elif self.covariance_func == 'gamma-exp':
                    for key in parameters:
                        if key == 'gamma':
                            parameters[key] = logodds_half(parameters[key])
                        else:
                            parameters[key] = np.log(parameters[key])
                            
                    self.parameters = {
                              'log_range': tf.Variable(parameters['range'], dtype=tf.float64),
                              'log_sill': tf.Variable(parameters['sill'], dtype=tf.float64),
                              'log_nugget': tf.Variable(parameters['nugget'], dtype=tf.float64),
                              'logit_halfgamma': tf.Variable(parameters['gamma'], dtype=tf.float64)}
                    # print(self.parameters.items())
                    

        elif parameters == None:
            if self.call_flag is None:
                if self.covariance_func == 'squared-exp':
                    self.parameters = {
                          'log_range': tf.Variable(0.0, dtype=tf.float64),
                          'log_sill': tf.Variable(0.0, dtype=tf.float64),
                          'log_nugget': tf.Variable(0.0, dtype=tf.float64)}
                    # print(self.parameters.items())
                elif self.covariance_func == 'gamma-exp':
                    self.parameters = {
                          'log_range': tf.Variable(0.0, dtype=tf.float64),
                          'log_sill': tf.Variable(0.0, dtype=tf.float64),
                          'log_nugget': tf.Variable(0.0, dtype=tf.float64),
                          'logit_halfgamma': tf.Variable(0.0, dtype=tf.float64)}
                    # print(self.parameters.items())


        # # Build parameters dict. (This may work when GP can take n params).
        # self.parameter_names = parameter_names
        # tf_var = [tf.Variable(0.0, dtype=tf.float64)] * len(self.parameter_names)
        # self.parameters = dict(zip(self.parameter_names, tf_var))

        self.featurizer = featurizer

        if u is not None and x is not None:
            self.u1 = u
        
            # Projection.
            self.x1 = x

            # Distance matrix.
            self.D = cdist(self.x1, self.x1)       

            # Feature matrix.
            self.F = self.featurizer(self.x1)
        
            # Data dict.
            self.data = {'D': tf.constant(self.D), 'F': tf.constant(self.F), 'u': tf.constant(self.u1)}        
        
            # Train the GP.
            def gpm_fit(data, parameters, hyperparameters):
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

                j = 0 # Iteration count.
                for i in range(10):
                    while j < (i + 1) * self.train_iters / 10:
                        p, ll = gpm_train_step(optimizer, data, parameters, hyperparameters)
                        j += 1
                    
                    if self.verbose == True:
                        if self.covariance_func == 'squared-exp':
                            print('[iter %d] [ll %7.2f] [range %4.2f, sill %4.2f, nugget %4.2f]' % 
                                (j, ll, p['range'], p['sill'], p['nugget']))

                        elif self.covariance_func == 'gamma-exp':
                            print('[iter %d] [ll %7.2f] [range %4.2f, sill %4.2f, nugget %4.2f, gamma %4.2f]' % 
                                (j, ll, p['range'], p['sill'], p['nugget'], p['halfgamma'] * 2))


            gpm_fit(self.data, self.parameters, self.hyperparameters)
        
        
############################################################################
############################################################################

    def generate(self, x2_pred):
        X = x2_pred
        D = cdist(X, X)

        # Feature matrix.
        F = self.featurizer(X)

        p = self.gp_xform_parameters(self.parameters)
        
        if self.covariance_func == 'squared-exp':
            A = gp_covariance_sq_exp(D, F, p['range'], p['sill'], p['nugget'], self.hyperparameters['alpha'])
        elif self.covariance_func == 'gamma-exp':
            A = gp_covariance_gamma_exp(D, F, p['range'], p['sill'], p['nugget'], p['halfgamma'], self.hyperparameters['alpha'])

        A = A.numpy()
        return np.random.multivariate_normal(np.zeros([A.shape[0]]), A)

    def predict(self, x2_pred):
        
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
        self.batch_size = self.x1.shape[0] // 2
        
        
        # Needed functions.
        def e(x, a=-1):
            return tf.expand_dims(x, a)

        def one_axes(x):
            return tf.where(tf.equal(x.shape, 1))[:, 0]

        # Project.
        self.x2 = x2_pred

        ##############################################
        def interpolate_gp(X1, u1, X2, parameters, hyperparameters):

            N1 = len(X1) # Number of measurements.
            N2 = len(X2) # Number of predictions.

            X = np.concatenate([X1, X2], axis=0)
            D = cdist(X, X)
            
            # Use the user given featurization.
            F = self.featurizer(X)

            p = self.gp_xform_parameters(parameters)
            
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

        for start in np.arange(0, len(self.x2), self.batch_size):
            stop = start + self.batch_size    
            subset = self.x2[start:stop]
            for_gp.append(subset)

        u2_mean_s = []
        u2_var_s = []

        for subset in for_gp:
            u2_mean, u2_var = interpolate_gp(self.x1, self.u1, subset, self.parameters, self.hyperparameters)
            u2_mean = u2_mean.numpy()
            u2_var = u2_var.numpy()
            u2_mean_s.append(u2_mean)
            u2_var_s.append(u2_var)
            
        u2_mean = np.concatenate(u2_mean_s)
        u2_var = np.concatenate(u2_var_s)

        return u2_mean, u2_var
