import numpy as np
from geostat import gp, Model, NormalizingFeaturizer
import geostat.metric as gm

def test_euclidean():
    # Create 100 random locations in a square centered on the origin.
    locs = np.random.uniform(-1., 1., [1000, 3])

    metric = gm.Euclidean(scale=[1., 1., 'zscale'])
    covariance = \
        gp.GammaExponential(metric=metric) + \
        gp.SquaredExponential(range=2., sill='alpha') + \
        gp.Noise()

    # Generating GP.
    model1 = Model(
        latent = covariance,
        parameters = dict(alpha=1., zscale=5., range=0.5, sill=1., gamma=1., nugget=1.),
        verbose=True)

    # Generate data.
    vals = model1.generate(locs).vals

    # Fit GP.
    model2 = Model(
        latent = covariance,
        parameters = dict(alpha=2., zscale=1., range=1.0, sill=0.5, gamma=0.5, nugget=0.5),
        verbose=True).fit(locs, vals, iters=500)

    # Interpolate using GP.
    N = 10
    xx, yy, zz = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    locs2 = np.stack([xx, yy, zz], axis=-1).reshape([-1, 3])

    mean, var = model2.predict(locs2)
    mean2, var2 = model2.predict(locs2)

    assert np.all(mean == mean2)
    assert np.all(var == var2)

def test_poincare():
    np.random.seed(1235)

    # Create random locations in a square centered on the origin.
    locs = np.random.uniform(-2., 2., [1000, 3])
    locs[:, 2] = 3. - np.exp(locs[:, 2]) # Make z to be 3 or lower.

    # Transform z to be positive, and make it the first axis.
    zmax = locs[:, 2].max()
    def xform(x, y, z): return zmax-z, x, y

    metric = gm.Poincare(xform=xform, scale=['zscale', 1., 1.], zoff='zoff')
    covariance = \
        gp.GammaExponential(metric=metric) + \
        gp.SquaredExponential(metric=metric, range=2., sill='alpha') + \
        gp.Noise()

    # Generating GP.
    model1 = Model(
        latent = covariance,
        parameters = dict(zoff=2.0, alpha=1., zscale=5., range=0.5, sill=1., gamma=1., nugget=1.),
        verbose=True)

    # Generate data.
    vals = model1.generate(locs).vals

    # Fit GP.
    model2 = Model(
        latent = covariance,
        parameters = dict(zoff=2.0, alpha=2., zscale=1., range=1.0, sill=0.5, gamma=0.5, nugget=0.5),
        verbose=True).fit(locs, vals, iters=500)

    # Interpolate using GP.
    N = 10
    xx, yy, zz = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    locs2 = np.stack([xx, yy, zz], axis=-1).reshape([-1, 3])

    mean, var = model2.predict(locs2)
    mean2, var2 = model2.predict(locs2)

    assert np.all(mean == mean2)
    assert np.all(var == var2)

