import numpy as np
from geostat import GP, TrendFeaturizer
import pyproj

def test_gp():
    # Create N by N grid and project.
    locs = np.random.uniform(-1., 1., [30, 2])

    # Initialize featurizer of location for trends.
    def featurization(locs):
        x, y = np.transpose(locs)
        return x, y, x*y
    featurizer = TrendFeaturizer(featurization, locs)

    # Generating GP.
    gp = GP(featurizer = featurizer,
            covariance_func = 'squared-exp',
            parameter0 = dict(vrange=0.5, sill=1., nugget=1.),
            hyperparameters = dict(alpha=1.),
            verbose=True)

    # Generate data.
    vals = gp.generate(locs)

    # Fit GP.
    gp = GP(locs, vals,
            featurizer = featurizer,
            covariance_func = 'squared-exp',
            parameter0 = dict(vrange=1.0, sill=0.5, nugget=0.5),
            hyperparameters = dict(alpha=vals.ptp()**2, reg=0, train_iters=200),
            verbose=True)

    # Interpolate using GP.
    N = 20
    lat = np.repeat(np.linspace(-1., 1., N), N)
    lon = np.tile(np.linspace(-1., 1., N), N)
    locs = np.stack([lat, lon], axis=1)

    mean, var = gp.predict(locs)
