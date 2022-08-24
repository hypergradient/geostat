import numpy as np
from geostat import GP, NormalizingFeaturizer
import geostat.covfunc as cf


def test_gp2d():
    # Create 100 random locations in a square centered on the origin.
    locs1 = np.random.uniform(-1., 1., [500, 2])

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
    vals1 = gp1.generate(locs1)

    # Fit GP.
    gp2 = GP(
        covariance = covariance,
        parameters = dict(alpha=2., range=1., nugget=0.5),
        hyperparameters = dict(reg=0, train_iters=200),
        verbose=True).fit(locs1, vals1)

    # Interpolate using GP.
    N = 20
    xx, yy = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    locs2 = np.stack([xx, yy], axis=-1) # Okay if locs2 rank is greater than 2.

    mean, var = gp2.predict(locs1, vals1, locs2)
    mean2, var2 = gp2.predict(locs1, vals1, locs2)

    assert np.all(mean == mean2)
    assert np.all(var == var2)


def test_gp3d():
    # Create 100 random locations in a square centered on the origin.
    locs1 = np.random.uniform(-1., 1., [500, 3])

    # Initialize featurizer of location for trends.
    def trend_terms(x, y, z): return z, z*z
    featurizer = NormalizingFeaturizer(trend_terms, locs1)
    covariance = \
        cf.Trend(featurizer) + \
        cf.GammaExponential(scale=[1., 1., 'zscale']) + \
        cf.Delta(axes=[0, 1]) + \
        cf.Noise()

    # Generating GP.
    gp1 = GP(
        covariance = covariance,
        parameters = dict(alpha=1., zscale=5., range=0.5, sill=1., gamma=1., dsill=1., nugget=1.),
        verbose=True)

    # Generate data.
    vals1 = gp1.generate(locs1)

    # Fit GP.
    gp2 = GP(
        covariance = covariance,
        parameters = dict(alpha=2., zscale=1., range=1.0, sill=0.5, gamma=0.5, dsill=0.5, nugget=0.5),
        hyperparameters = dict(reg=0, train_iters=500),
        verbose=True).fit(locs1, vals1)

    # Interpolate using GP.
    N = 10
    xx, yy, zz = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    locs2 = np.stack([xx, yy, zz], axis=-1) # Okay if locs2 rank is greater than 2.

    mean, var = gp2.predict(locs1, vals1, locs2)
    mean2, var2 = gp2.predict(locs1, vals1, locs2)

    assert np.all(mean == mean2)
    assert np.all(var == var2)


def test_gp3d_stacked():
    # Create random locations centered on the origin.
    locs1 = np.random.normal(size=[1000, 3])

    # Initialize featurizer of location for trends.
    def trend_terms(x, y, z): return z, z*z
    featurizer = NormalizingFeaturizer(trend_terms, locs1)

    # Covariance structure
    covariance = \
        cf.Trend(featurizer) + \
        cf.SquaredExponential(range='r1', sill='s1', scale=[1., 1., 'zscale']) + \
        cf.SquaredExponential(range='r2', sill='s2', scale=[1., 1., 0.]) + \
        cf.Noise()

    # Generating GP.
    gp1 = GP(
        covariance = covariance,
        parameters = dict(alpha=1., zscale=5., r1=0.25, s1=1., r2=1.0, s2=0.25, nugget=1.),
        verbose=True)

    # Generate data.
    vals1 = gp1.generate(locs1)

    # Fit GP.
    gp2 = GP(
        covariance = covariance,
        parameters = dict(alpha=2., zscale=2.5, r1=0.125, s1=0.5, r2=0.5, s2=0.125, nugget=0.5),
        hyperparameters = dict(reg=0, train_iters=500),
        verbose=True).fit(locs1, vals1)

    # Interpolate using GP.
    N = 10
    xx, yy, zz = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    locs2 = np.stack([xx, yy, zz], axis=-1) # Okay if locs2 rank is greater than 2.

    mean, var = gp2.predict(locs1, vals1, locs2)
    mean2, var2 = gp2.predict(locs1, vals1, locs2)

    assert np.all(mean == mean2)
    assert np.all(var == var2)
