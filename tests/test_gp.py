import numpy as np
import tensorflow as tf
from geostat import Featurizer, GP, Model, NormalizingFeaturizer, Trend
import geostat.kernel as krn

def test_gp_with_trend():
    np.random.seed(2)
    tf.random.set_seed(2)

    # Create random locations in a square centered on the origin.
    locs1 = np.random.normal(size=[1000, 2])

    # Initialize featurizer of location for trends.
    def trend_terms(x, y): return 1., x, y, x*y
    featurizer = Featurizer(trend_terms)
    trend = Trend(featurizer, beta='beta')
    kernel = krn.SquaredExponential(sill=1.) + krn.Noise()

    # Generate data.
    vals1 = Model(
        GP(trend, kernel),
        parameters = dict(range=0.33, nugget=1., beta=[4., 3., 2., 1.]),
        verbose=True).generate(locs1).vals

    # Fit GP.
    model = Model(
        GP(trend, kernel),
        parameters = dict(range=1., nugget=0.5, beta=[0., 0., 0., 0.]),
        verbose=True).fit(locs1, vals1, iters=100, step_size=1e-1)

    assert np.allclose(model.parameters['beta'], [4., 3., 2., 1.], rtol=0.3)

    assert np.allclose(
        [model.parameters[p] for p in ['range', 'nugget']],
        [0.33, 1.],
        rtol=0.3)

    # Interpolate using GP.
    N = 20
    xx, yy = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    locs2 = np.stack([xx, yy], axis=-1).reshape([-1, 2])

    mean, var = model.predict(locs2)
    mean2, var2 = model.predict(locs2)

    assert np.all(mean == mean2)
    assert np.all(var == var2)

def test_gp2d():
    np.random.seed(2)
    tf.random.set_seed(2)

    # Create random locations in a square centered on the origin.
    locs1 = np.random.normal(size=[1000, 2])

    # Initialize featurizer of location for trends.
    def trend_terms(x, y): return x, y, x*y
    featurizer = NormalizingFeaturizer(trend_terms, locs1)
    kernel = krn.TrendPrior(featurizer) + krn.SquaredExponential(sill=1.) + krn.Noise()

    # Generate data.
    vals1 = Model(
        GP(0, kernel),
        parameters = dict(alpha=1., range=0.33, nugget=1.),
        verbose=True).generate(locs1).vals

    # Fit GP.
    model = Model(
        GP(0, kernel),
        parameters = dict(alpha=2., range=1., nugget=0.5),
        verbose=True).fit(locs1, vals1, iters=100, step_size=1e-1)

    assert np.allclose(
        [model.parameters[p] for p in ['range', 'nugget']],
        [0.33, 1.],
        rtol=0.3)

    # Interpolate using GP.
    N = 20
    xx, yy = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    locs2 = np.stack([xx, yy], axis=-1).reshape([-1, 2])

    mean, var = model.predict(locs2)
    mean2, var2 = model.predict(locs2)

    assert np.all(mean == mean2)
    assert np.all(var == var2)

def test_gp3d():
    np.random.seed(2)
    tf.random.set_seed(2)

    # Create random locations in a square centered on the origin.
    locs1 = np.random.normal(size=[600, 3])
    locs1 = np.concatenate([locs1, locs1 * [1., 1., 0.8], locs1 * [1., 1., 1.1]])

    # Initialize featurizer of location for trends.
    def trend_terms(x, y, z): return z, z*z
    featurizer = NormalizingFeaturizer(trend_terms, locs1)
    kernel = \
        krn.TrendPrior(featurizer) + \
        krn.GammaExponential(scale=[1., 1., 'zscale']) + \
        krn.Delta(axes=[0, 1]) + \
        krn.Noise()

    # Generate data.
    vals1 = Model(
        GP(0, kernel),
        parameters = dict(alpha=1., zscale=5., range=0.5, sill=1., gamma=1., dsill=0.1, nugget=0.1),
        verbose=True).generate(locs1).vals

    # Fit GP.
    model = Model(
        GP(0, kernel),
        parameters = dict(alpha=2., zscale=1., range=1., sill=0.5, gamma=0.5, dsill=0.5, nugget=0.5),
        verbose=True).fit(locs1, vals1, iters=200, step_size=1e-1)

    assert np.allclose(
        [model.parameters[p] for p in ['zscale', 'range', 'sill', 'gamma', 'dsill', 'nugget']],
        [5., 0.5, 1., 1., 0.1, 0.1],
        rtol=0.5)

    # Interpolate using GP.
    N = 10
    xx, yy, zz = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    locs2 = np.stack([xx, yy, zz], axis=-1).reshape([-1, 3])

    mean, var = model.predict(locs2)
    mean2, var2 = model.predict(locs2)

    assert np.all(mean == mean2)
    assert np.all(var == var2)


def test_gp3d_stacked():
    np.random.seed(2)
    tf.random.set_seed(2)

    # Create random locations centered on the origin.
    locs1 = np.random.normal(size=[2500, 3])

    # Initialize featurizer of location for trends.
    def trend_terms(x, y, z): return z, z*z
    featurizer = NormalizingFeaturizer(trend_terms, locs1)

    # Covariance structure
    kernel = \
        krn.TrendPrior(featurizer) + \
        krn.SquaredExponential(range='r1', sill='s1', scale=[1., 1., 'zscale']) + \
        krn.SquaredExponential(range='r2', sill='s2', scale=[1., 1., 0.]) + \
        krn.Noise()

    # Generate data.
    vals1 = Model(
        GP(0, kernel),
        parameters = dict(alpha=1., zscale=5., r1=0.25, s1=1., r2=1.0, s2=0.25, nugget=1.),
        verbose=True).generate(locs1).vals

    # Fit GP.
    model = Model(
        GP(0, kernel),
        parameters = dict(alpha=2., zscale=2.5, r1=0.125, s1=0.5, r2=0.5, s2=0.125, nugget=0.5),
        verbose=True).fit(locs1, vals1, iters=100, step_size=1e-1)

    assert np.allclose(
        [model.parameters[p] for p in ['zscale', 'r1', 's1', 'r2', 's2', 'nugget']],
        [5., 0.25, 1., 1., 0.25, 1.],
        rtol=0.3)

    # Interpolate using GP.
    N = 10
    xx, yy, zz = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    locs2 = np.stack([xx, yy, zz], axis=-1).reshape([-1, 3])

    mean, var = model.predict(locs2)
    mean2, var2 = model.predict(locs2)

    assert np.all(mean == mean2)
    assert np.all(var == var2)
