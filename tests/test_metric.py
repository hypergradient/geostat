import numpy as np
from geostat import GP, NormalizingFeaturizer
import geostat.covfunc as cf
import geostat.metric as gm

def test_euclidean():
    # Create 100 random locations in a square centered on the origin.
    locs = np.random.uniform(-1., 1., [1000, 3])

    metric = gm.Euclidean(scale=[1., 1., 'zscale'])
    covariance = \
        cf.GammaExponential(metric=metric) + \
        cf.SquaredExponential(range=2., sill='alpha') + \
        cf.Noise()

    # Generating GP.
    gp1 = GP(
        covariance = covariance,
        parameters = dict(alpha=1., zscale=5., range=0.5, sill=1., gamma=1., nugget=1.),
        verbose=True)

    # Generate data.
    vals = gp1.generate(locs).vals

    # Fit GP.
    gp2 = GP(
        covariance = covariance,
        parameters = dict(alpha=2., zscale=1., range=1.0, sill=0.5, gamma=0.5, nugget=0.5),
        hyperparameters = dict(reg=0, train_iters=500),
        verbose=True).fit(locs, vals)

    # Interpolate using GP.
    N = 10
    xx, yy, zz = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    locs2 = np.stack([xx, yy, zz], axis=-1).reshape([-1, 3])

    mean, var = gp2.predict(locs2)
    mean2, var2 = gp2.predict(locs2)

    assert np.all(mean == mean2)
    assert np.all(var == var2)

def test_poincare():
    np.random.seed(1235)

    # Create random locations in a square centered on the origin.
    locs = np.random.uniform(-2., 2., [1000, 3])
    locs[:, 2] = np.exp(locs[:, 2]) # Constrain z to be positive.

    metric = gm.Poincare(axis=2, scale=[1., 1., 'zscale'], zoff='zoff')
    covariance = \
        cf.GammaExponential(metric=metric) + \
        cf.SquaredExponential(metric=metric, range=2., sill='alpha') + \
        cf.Noise()

    # Generating GP.
    gp1 = GP(
        covariance = covariance,
        parameters = dict(zoff=2.0, alpha=1., zscale=5., range=0.5, sill=1., gamma=1., nugget=1.),
        verbose=True)

    # Generate data.
    vals = gp1.generate(locs).vals

    # Fit GP.
    gp2 = GP(
        covariance = covariance,
        parameters = dict(zoff=2.0, alpha=2., zscale=1., range=1.0, sill=0.5, gamma=0.5, nugget=0.5),
        hyperparameters = dict(reg=0, train_iters=500),
        verbose=True).fit(locs, vals)

    # Interpolate using GP.
    N = 10
    xx, yy, zz = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    locs2 = np.stack([xx, yy, zz], axis=-1).reshape([-1, 3])

    mean, var = gp2.predict(locs2)
    mean2, var2 = gp2.predict(locs2)

    assert np.all(mean == mean2)
    assert np.all(var == var2)

