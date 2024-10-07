import tensorflow as tf
from geostat import Parameters
from geostat.kernel import TrendPrior

def test_trendprior():

    # Define a simple featurizer function
    def simple_featurizer(x):
        return x, x**2

    # Create parameters.
    p = Parameters(alpha=0.5)

    # Construct kernel and call it
    locs1 = tf.constant([[1.0], [2.0], [3.0]])
    locs2 = tf.constant([[1.5], [2.5], [3.5]])
    #locs1 = tf.constant([[1.0], [2.0]])
    #locs2 = tf.constant([[1.5], [2.5]])
    trend_prior_kernel = TrendPrior(featurizer=simple_featurizer, alpha=p.alpha)
    covariance_matrix = trend_prior_kernel({'locs1': locs1, 'locs2': locs2, 'alpha': p.alpha})
    print(covariance_matrix)