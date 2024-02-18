import numpy as np
from geostat import GP, Model, NormalizingFeaturizer, Parameters
import geostat.kernel as krn
import geostat.metric as gm

def test_euclidean():
    # Create 100 random locations in a square centered on the origin.
    locs = np.random.uniform(-1., 1., [1000, 3])

    p = Parameters(zscale=5., range=0.5, sill=1., gamma=1., nugget=1.)

    metric = gm.Euclidean(scale=[1., 1., p.zscale])
    gp = GP(0,
        krn.GammaExponential(range=p.range, sill=p.sill, gamma=p.gamma, metric=metric) +
        krn.SquaredExponential(range=2., sill=p.sill) +
        krn.Noise(nugget=p.nugget))

    # Generating GP.
    model = Model(gp)

    # Generate data.
    vals = model.generate(locs).vals

    # Fit GP.
    model.set(zscale=1., range=1.0, sill=0.5, gamma=0.5, nugget=0.5)
    model.fit(locs, vals, iters=500)

    # Interpolate using GP.
    N = 10
    xx, yy, zz = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    locs2 = np.stack([xx, yy, zz], axis=-1).reshape([-1, 3])

    mean, var = model.predict(locs2)
    mean2, var2 = model.predict(locs2)

    assert np.all(mean == mean2)
    assert np.all(var == var2)

def test_poincare():
    np.random.seed(1235)

    # Create random locations in a square centered on the origin.
    locs = np.random.uniform(-2., 2., [1000, 3])
    locs[:, 2] = 3. - np.exp(locs[:, 2]) # Make z to be 3 or lower.

    p = Parameters(zoff=2.0, alpha=1., zscale=5., range=0.5, sill=1., gamma=1., nugget=1.)

    # Transform z to be positive, and make it the first axis.
    zmax = locs[:, 2].max()
    def xform(x, y, z): return zmax-z, x, y

    metric = gm.Poincare(xform=xform, scale=[p.zscale, 1., 1.], zoff=p.zoff)
    gp = GP(0,
        krn.GammaExponential(range=p.range, sill=p.sill, gamma=p.gamma, metric=metric) +
        krn.SquaredExponential(metric=metric, range=2., sill=p.sill) +
        krn.Noise(nugget=p.nugget))

    # Generating GP.
    model = Model(gp)

    # Generate data.
    vals = model.generate(locs).vals

    # Fit GP.
    model.set(zoff=2.0, zscale=1., range=1.0, sill=0.5, gamma=0.5, nugget=0.5)
    model.fit(locs, vals, iters=500)

    # Interpolate using GP.
    N = 10
    xx, yy, zz = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    locs2 = np.stack([xx, yy, zz], axis=-1).reshape([-1, 3])

    mean, var = model.predict(locs2)
    mean2, var2 = model.predict(locs2)

    assert np.all(mean == mean2)
    assert np.all(var == var2)

