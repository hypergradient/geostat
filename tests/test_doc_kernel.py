# from geostat.kernel import Kernel
# import numpy as np

# # Construct kernel and call it on locations
# locs1 = np.array([[0.0, 0.0], [1.0, 1.0]])
# locs2 = np.array([[2.0, 2.0], [3.0, 3.0]])
# kernel = Kernel(fa={'alpha': 1.0}, autoinputs={})
# covariance_matrix = kernel({'locs1': locs1, 'locs2': locs2})
# print(covariance_matrix) # Covariance matrix only has zero entries as no kernel function was given
# # tf.Tensor(
# # [[0. 0.]
# #  [0. 0.]], shape=(2, 2), dtype=float32)

# kernel1 = Kernel(fa={'alpha': 1.0}, autoinputs={})
# kernel2 = Kernel(fa={'range': 0.5}, autoinputs={})
# combined_kernel = kernel1 + kernel2  # Adding kernels
# product_kernel = kernel1 * kernel2   # Multiplying kernels

#--------------------------ERROR-------------------------------#

import tensorflow as tf
from geostat import Parameters
from geostat.kernel import TrendPrior

# Define a simple featurizer function
def simple_featurizer(x):
    return x, 2*x, x**2

# Create parameters.
p = Parameters(alpha=0.5)

# Construct kernel and call it
locs1 = tf.constant([[1.0], [2.0], [3.0]])
locs2 = tf.constant([[1.5], [2.5], [3.5], [4.5]])
trend_prior_kernel = TrendPrior(featurizer=simple_featurizer, alpha=p.alpha)
covariance_matrix = trend_prior_kernel({'locs1': locs1, 'locs2': locs2, 'alpha': p.alpha.value})
print(covariance_matrix)

#-----------------------------------------------------------------#

# from geostat import Parameters
# from geostat.kernel import Constant
# import numpy as np

# # Create parameters.
# p = Parameters(sill=2.0)

# # Create a Constant kernel with a sill value of 2.0 and call it
# locs1 = np.array([[0.0], [1.0], [2.0]])
# locs2 = np.array([[0.0], [1.0], [2.0]])
# constant_kernel = Constant(sill=p.sill)
# covariance_matrix = constant_kernel({'locs1': locs1, 'locs2': locs2, 'sill': 2.0})
# print(covariance_matrix)
# # tf.Tensor(
# # [[2. 2. 2.]
# #  [2. 2. 2.]
# #  [2. 2. 2.]], shape=(3, 3), dtype=float32)

#-----------------------------------------------------------------#

# from geostat import Parameters
# from geostat.metric import PerAxisDist2
# from geostat.kernel import SquaredExponential
# import numpy as np

# # Create parameters.
# p = Parameters(sill=1.0, range=2.0)

# # Create a SquaredExponential kernel with a sill of 1.0 and a range of 2.0 and call it
# locs1 = np.array([[0.0], [1.0], [2.0]])
# locs2 = np.array([[0.0], [1.0], [2.0]])
# d2 = PerAxisDist2({'locs1': locs1, 'locs2': locs2})
# sq_exp_kernel = SquaredExponential(sill=p.sill, range=p.range, metric='euclidean')
# covariance_matrix = sq_exp_kernel({'locs1': locs1, 'locs2': locs2, 'sill': 1.0, 'range': 2.0})
# print(covariance_matrix)