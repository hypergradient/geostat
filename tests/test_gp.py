import numpy as np
from geostat import GP, NormalizingFeaturizer

def test_gp():
    # Create 100 random locations in a square centered on the origin.
    locs1 = np.random.uniform(-1., 1., [100, 2])

    # Initialize featurizer of location for trends.
    def trend_terms(x, y): return x, y, x*y
    featurizer = NormalizingFeaturizer(trend_terms, locs1)

    # Generating GP.
    gp1 = GP(featurizer = featurizer,
            covariance_func = 'squared-exp',
            parameters = dict(range=0.5, sill=1., nugget=1.),
            hyperparameters = dict(alpha=1.),
            verbose=True)

    # Generate data.
    vals1 = gp1.generate(locs1)

    # Fit GP.
    gp2 = GP(featurizer = featurizer,
            covariance_func = 'squared-exp',
            parameters = dict(range=1.0, sill=0.5, nugget=0.5),
            hyperparameters = dict(alpha=vals1.ptp()**2, reg=0, train_iters=200),
            verbose=True).fit(locs1, vals1)

    # Interpolate using GP.
    N = 20
    lat = np.repeat(np.linspace(-1., 1., N), N)
    lon = np.tile(np.linspace(-1., 1., N), N)
    locs2 = np.stack([lat, lon], axis=1)

    mean, var = gp2.predict(locs1, vals1, locs2)
