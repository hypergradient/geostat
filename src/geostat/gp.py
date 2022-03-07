import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
import geopandas as gpd
import tensorflow as tf
from shapely.geometry import Point

from .spatialinterpolator import SpatialInterpolator

__all__ = ['GP']

class GP(SpatialInterpolator):
    
    def __init__(self, 
                 x1,
                 u1, 
                 covariance_func='squared-exp',
                 featurization=None,
                 parameter0=None,
                 train_epochs=300, 
                 hyperparameters=dict(alpha=10, reg=None),
                 projection=None,
                 verbose=True,
                 ):
        
        '''
        Parameters:
                x1 : n-dim array
                    Locations of input data.
                    
                u1 : 1-d array
                    Values to be modeled.
                
                covariance_func : str
                     Name of the covariance function to use in the GP. 
                     Should be 'squared-exp' or 'gamma-exp'.
                     Default is 'squared-exp'.
                    
                featurization : function, optional
                    Should be a function that takes x1 (n-dim array of input data) 
                    and returns the coordinates, i.e., x, y, x**2, y**2.
                    Example: def featurization(x1):    
                                return x1[:, 0], x1[:, 1], x1[:, 0]**2, x1[:, 1]**2.
                    Default is None.
                
                parameter0 : dict, optional
                    The starting point for the parameters. Use "vrange" for the range.
                    Example: parameter0=dict(vrange=2.0, sill=5.0, nugget=1.0).
                    Default is None.
        
                train_epochs : int, optional
                    The number of training epochs.
                    Default is 300.
                
                hyperparameters : dict
                    Dictionary of the hyperparameters. Should contain "alpha", the prior distribution for the trend,
                    and "reg", the value for regularization. If no regularization is wanted use reg=None.
                    Default is alpha=10 and reg=None.
                    
                project : function, opt
                    A function that takes multiple vectors, and returns
                    a tuple of projected vectors.
                
                verbose : boolean, optional
                    Whether or not to print parameters.
                    Default is True.


        Performs Gaussian process training and prediction.
  
        '''
        
        super().__init__(projection=projection)
        
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
    
        def e(x, a=-1):
            return tf.expand_dims(x, a)

        def one_axes(x):
            return tf.where(tf.equal(x.shape, 1))[:, 0]

        def Featurizer(x1, featurization):
            F_list = []
            for i in list(featurization(x1)):
                F_list.append(i.T)
            F = np.concatenate([np.ones([1, len(x1)]), np.vstack(F_list)])
            return F



        # List the needed graph functions.
        
        # Transform parameters.
        @tf.function
        def gp_xform_parameters(parameters):
            """
            Transform parameters from the underlying representation
            (which has the whole real number line as its range) to
            a surface representation (which is bounded).
            """
            vrange = tf.exp(parameters['log_vrange'])
            sill = tf.exp(parameters['log_sill'])
            nugget = tf.exp(parameters['log_nugget'])
            
            if self.covariance_func == 'squared-exp':
                param_dict = {'vrange': vrange, 'sill': sill, 'nugget': nugget}
                return param_dict
                
            elif self.covariance_func == 'gamma-exp':
                halfgamma = tf.sigmoid(parameters['logit_halfgamma'])
                param_dict = {'vrange': vrange, 'sill': sill, 'nugget': nugget, 'halfgamma': halfgamma}
                return param_dict
                

        # Squared exponential covariance function.
        @tf.function
        def gp_covariance_sq_exp(D, F, vrange, sill, nugget, alpha):
            C = sill * tf.exp(-tf.square(D / vrange)) \
                    + (nugget + 1e-6) * tf.eye(D.shape[0], dtype=tf.float64) \
                    + alpha * tf.einsum('ab,ac->bc', F, F)
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
        
        @tf.function
        def gamma_exp(d2, halfgamma):
            return tf.exp(-safepow(tf.maximum(d2, 0.0), halfgamma))

        @tf.function
        def gp_covariance_gamma_exp(D, F, vrange, sill, nugget, halfgamma, alpha):
            vrange = e(e(vrange))
            sill = e(e(sill))
            halfgamma = e(e(halfgamma))
            C = sill * gamma_exp(tf.square(D / vrange), halfgamma) + (nugget + 1e-6) * tf.eye(tf.shape(D)[0], dtype=tf.float64)
            C += tf.einsum('ab,ac->bc', F, F) * alpha   
            return C
        

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
                    A = self.gp_covariance(data['D'], data['F'], p['vrange'], p['sill'], p['nugget'], beta_prior)
                
                elif self.covariance_func == 'gamma-exp':
                    A = self.gp_covariance(data['D'], data['F'], p['vrange'], p['sill'], p['nugget'], p['halfgamma'], beta_prior)
                
                ll = gp_log_likelihood(data['u'], 0., A)
                
                if hyperparameters['reg'] != None:
                    reg = hyperparameters['reg'] * tf.reduce_sum(tf.square(parameters['log_vrange']))
                    loss = -ll + reg
                else:
                    loss = -ll
                
            gradients = tape.gradient(loss, parameters.values())
            optimizer.apply_gradients(zip(gradients, parameters.values()))
            return p, ll


        
        # Set the user desired covariance function.
        if covariance_func == 'squared-exp':
            self.covariance_func = 'squared-exp'
            self.gp_covariance = gp_covariance_sq_exp
        
        elif covariance_func == 'gamma-exp':
            self.covariance_func = 'gamma-exp'
            self.gp_covariance = gp_covariance_gamma_exp

        else:
            raise ValueError("Only 'squared-exp' and 'gamma-exp' are currently supported.")
            
        
        
        # Define other inputs.
        self.verbose = verbose
        self.train_epochs = train_epochs
        self.featurization = featurization
        self.gp_xform_parameters = gp_xform_parameters  # Need for predict.
        
        if hyperparameters['reg']:
            self.hyperparameters = {'alpha': tf.constant(hyperparameters['alpha'], dtype=tf.float64),
                                    'reg': tf.constant(hyperparameters['reg'], dtype=tf.float64)}
        if not hyperparameters['reg']:
            self.hyperparameters = {'alpha': tf.constant(hyperparameters['alpha'], dtype=tf.float64),
                                    'reg': None}
        
        
        # Build the tf.Variable() dict.
        if parameter0 != None:
            if self.call_flag is None:
                if self.covariance_func == 'squared-exp':
                    # Log the starting point parameters that where provided.
                    for key in parameter0:    
                        parameter0[key] = np.log(parameter0[key])
                        
                    self.parameters = {
                          'log_vrange': tf.Variable(parameter0['vrange'], dtype=tf.float64),
                          'log_sill': tf.Variable(parameter0['sill'], dtype=tf.float64),
                          'log_nugget': tf.Variable(parameter0['nugget'], dtype=tf.float64)}
                    # print(self.parameters.items())
                
                elif self.covariance_func == 'gamma-exp':
                    for key in parameter0:
                        if key == 'gamma':
                            parameter0[key] = logodds_half(parameter0[key])
                        else:
                            parameter0[key] = np.log(parameter0[key])
                            
                    self.parameters = {
                              'log_vrange': tf.Variable(parameter0['vrange'], dtype=tf.float64),
                              'log_sill': tf.Variable(parameter0['sill'], dtype=tf.float64),
                              'log_nugget': tf.Variable(parameter0['nugget'], dtype=tf.float64),
                              'logit_halfgamma': tf.Variable(parameter0['gamma'], dtype=tf.float64)}
                    # print(self.parameters.items())
                    

        elif parameter0 == None:
            if self.call_flag is None:
                if self.covariance_func == 'squared-exp':
                    self.parameters = {
                          'log_vrange': tf.Variable(0.0, dtype=tf.float64),
                          'log_sill': tf.Variable(0.0, dtype=tf.float64),
                          'log_nugget': tf.Variable(0.0, dtype=tf.float64)}
                    # print(self.parameters.items())
                elif self.covariance_func == 'gamma-exp':
                    self.parameters = {
                          'log_vrange': tf.Variable(0.0, dtype=tf.float64),
                          'log_sill': tf.Variable(0.0, dtype=tf.float64),
                          'log_nugget': tf.Variable(0.0, dtype=tf.float64),
                          'logit_halfgamma': tf.Variable(0.0, dtype=tf.float64)}
                    # print(self.parameters.items())


        # # Build parameters dict. (This may work when GP can take n params).
        # self.parameter_names = parameter_names
        # tf_var = [tf.Variable(0.0, dtype=tf.float64)] * len(self.parameter_names)
        # self.parameters = dict(zip(self.parameter_names, tf_var))

    
        # Check and change the shape of u1 if needed.
        if u1.ndim == 1:
            self.u1 = u1
        elif u1.shape[1] == 1:
            self.u1 = u1.reshape(-1)
        else:
            raise ValueError("Check dimensions of 'u1'.")
        
        # Projection.
        self.x1 = self.project(x1)

        # Distance matrix.
        self.D = cdist(self.x1, self.x1)       
        
        # Feature matrix.
        self.F = Featurizer(self.x1, self.featurization)
        
        # Data dict.
        self.data = {'D': tf.constant(self.D), 'F': tf.constant(self.F), 'u': tf.constant(self.u1)}        
        
        
        
        #####################################################

        # Train the GP.
        def gpm_fit(data, parameters, hyperparameters):
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

            for i in range(10):
                for j in range(self.train_epochs):
                    
                    p, ll = gpm_train_step(optimizer, data, parameters, hyperparameters)
                
                if self.verbose == True:
                    if self.covariance_func == 'squared-exp':
                        print('[ll %7.2f] [range %4.2f, sill %4.2f, nugget %4.2f]' % 
                            (ll, p['vrange'], p['sill'], p['nugget']))

                    elif self.covariance_func == 'gamma-exp':
                        print('[ll %7.2f] [range %4.2f, sill %4.2f, nugget %4.2f, gamma %4.2f]' % 
                            (ll, p['vrange'], p['sill'], p['nugget'], p['halfgamma'] * 2))


        gpm_fit(self.data, self.parameters, self.hyperparameters)
        
        
############################################################################
############################################################################

    def predict(self, x2_pred, batch_size=None):
        
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
        self.batch_size = batch_size
        
        
        # Needed functions.
        def e(x, a=-1):
            return tf.expand_dims(x, a)

        def one_axes(x):
            return tf.where(tf.equal(x.shape, 1))[:, 0]

        def Featurizer(x1, featurization):
            F_list = []
            for i in list(featurization(x1)):
                F_list.append(i.T)
            F = np.concatenate([np.ones([1, len(x1)]), np.vstack(F_list)])
            return F
        
        # Project.
        self.x2 = self.project(x2_pred)

        ##############################################
        def interpolate_gp(X1, u1, X2, parameters, hyperparameters):

            N1 = len(X1) # Number of measurements.
            N2 = len(X2) # Number of predictions.

            X = np.concatenate([X1, X2])
            D = cdist(X, X)
            
            # Use the user given featurization.
            F = Featurizer(X, self.featurization)

            p = self.gp_xform_parameters(parameters)
            
            if self.covariance_func == 'squared-exp':
                A = self.gp_covariance(D, F, p['vrange'], p['sill'], p['nugget'], hyperparameters['alpha'])
                
            if self.covariance_func == 'gamma-exp':
                A = self.gp_covariance(D, F, p['vrange'], p['sill'], p['nugget'], p['halfgamma'], hyperparameters['alpha'])

            A11 = A[:N1, :N1]
            A12 = A[:N1, N1:]
            A21 = A[N1:, :N1]
            A22 = A[N1:, N1:]

            u2_mean = tf.matmul(A21, tf.linalg.solve(A11, e(u1, -1)))[:, 0]
            u2_var = tf.linalg.diag_part(A22) -  tf.reduce_sum(A12 * tf.linalg.solve(A11, A12), axis=0)

            return u2_mean, u2_var
        
        
        # Interpolate in batches.
        if self.batch_size == None:
            u2_mean, u2_var = interpolate_gp(self.x1, self.u1, self.x2, self.parameters, self.hyperparameters)
            return u2_mean.numpy(), u2_var.numpy()
        
        elif self.batch_size != None:
            
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

    # Access the projected coords of the input data.
    def get_projected(self):
        return self.x1[:, 0], self.x1[:, 1]

        

