import numpy as np
from geostat import GP, Model, Featurizer, NormalizingFeaturizer, Parameters, Trend
import geostat.kernel as krn

def test_coefs1():
    # Create 100 random locations in a square centered on the origin.
    locs1 = np.random.uniform(-1., 1., [100, 2])

    p = Parameters(b=[1.0, 0.5], nugget=0.1)

    # Initialize featurizer of location for trends.
    def trend_terms(x, y): return 1, y
    featurizer = Featurizer(trend_terms)
    gp = GP(Trend(featurizer, beta=p.b),
            krn.SquaredExponential(range=0.5, sill=0.01) + krn.Noise(nugget=p.nugget))

    # Generating GP.
    model = Model(gp)

    # Generate data.
    vals1 = model.generate(locs1).vals

    # Fit GP.
    model.set(b=[1.0, 0.5], nugget=0.1)
    model.fit(locs1, vals1, iters=200)

    # Interpolate using GP.
    N = 2
    xx, yy = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    locs2 = np.stack([xx, yy], axis=-1).reshape([-1, 2])

    mean, var = model.predict(locs2)

    target_mean = [1., 1., 1.5, 1.5]

    assert np.square(mean - target_mean).mean() < 0.1

def test_coefs2():
    # Create 500 random locations in a square centered on the origin.
    locs1 = np.random.uniform(-1., 1., [500, 2])

    p = Parameters(b1=0.5, b2=0.5, b3=0.25, range=0.5, nugget=1.)

    # Initialize featurizer of location for trends.
    def trend_terms(x, y): return x, y, x*y
    featurizer = NormalizingFeaturizer(trend_terms, locs1)
    gp = GP(Trend(featurizer, beta=[1., p.b1, p.b2, p.b3]),
            krn.SquaredExponential(range=p.range, sill=1.) + krn.Noise(nugget=p.nugget))

    # Generating GP.
    model = Model(gp)

    # Generate data.
    vals1 = model.generate(locs1).vals

    # Fit GP.
    model.set(b1=0., b2=0., b3=0., range=1., nugget=0.5)
    model.fit(locs1, vals1, iters=200)

    # # MCMC.
    # model = Model(
    #     gp,
    #     parameters = dict(b1=0., b2=0., b3=0., range=1., nugget=0.5),
    #     verbose = True).mcmc(locs1, vals1, burnin=500, samples=500)

    # Interpolate using GP.
    N = 20
    xx, yy = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    locs2 = np.stack([xx, yy], axis=-1).reshape([-1, 2])

    mean, var = model.predict(locs2)
    mean2, var2 = model.predict(locs2)

    assert np.all(mean == mean2)
    assert np.all(var == var2)

def test_explicit_trend():
    # Create 500 random locations in a square centered on the origin.
    locs1 = np.random.uniform(-1., 1., [500, 2])

    p = Parameters(beta=[1., 0.5, 0.5, 0.25], range=0.5, nugget=1.)

    # Initialize featurizer of location for trends.
    def trend_terms(x, y): return x, y, x*y
    featurizer = NormalizingFeaturizer(trend_terms, locs1)
    gp = GP(Trend(featurizer, beta=p.beta),
            krn.SquaredExponential(range=p.range, sill=1.) + krn.Noise(nugget=p.nugget))

    # Generating GP.
    model = Model(gp)

    # Generate data.
    vals1 = model.generate(locs1).vals

    # Fit GP.
    model.set(beta=[0., 0., 0., 0.], range=1., nugget=0.5)
    model.fit(locs1, vals1, iters=200)

    # # MCMC.
    # model = Model(
    #     gp,
    #     parameters = dict(beta=[0., 0., 0., 0.], range=1., nugget=0.5),
    #     verbose = True).mcmc(locs1, vals1, burnin=500, samples=500)

    # Interpolate using GP.
    N = 20
    xx, yy = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    locs2 = np.stack([xx, yy], axis=-1).reshape([-1, 2])

    mean, var = model.predict(locs2)
    mean2, var2 = model.predict(locs2)

    assert np.all(mean == mean2)
    assert np.all(var == var2)
