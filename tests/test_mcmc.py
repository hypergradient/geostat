import numpy as np
from geostat import GP, NormalizingFeaturizer
import geostat.covfunc as cf


def test_mcmc():
    
    # Create 100 random locations in a square centered on the origin.
    locs1 = np.random.uniform(-1., 1., [1000, 2])

    # Initialize featurizer of location for trends.
    def trend_terms(x, y): return x, y, x*y
    featurizer = NormalizingFeaturizer(trend_terms, locs1)
    covariance = cf.Trend(featurizer) + cf.SquaredExponential(sill=1.) + cf.Noise()

    # Generating GP.
    gp1 = GP(
        covariance = covariance,
        parameters = dict(alpha=1., range=0.5, nugget=1.),
        verbose=True)

    # Generate data.
    vals1 = gp1.generate(locs1).vals

    # Fit GP.
    gp2 = GP(
        covariance = covariance,
        parameters = dict(alpha=2., range=1., nugget=0.5),
        hyperparameters = dict(reg=0, train_iters=500),
        verbose=True).mcmc(locs1, vals1)
