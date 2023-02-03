import numpy as np
from geostat import gp, Model, NormalizingFeaturizer

def test_explicit_coefs():
    # Create 100 random locations in a square centered on the origin.
    locs1 = np.random.uniform(-1., 1., [500, 2])

    # Initialize featurizer of location for trends.
    def trend_terms(x, y): return x, y, x*y
    featurizer = NormalizingFeaturizer(trend_terms, locs1)
    covariance = gp.Trend(featurizer, beta=[1., 'b1', 'b2', 'b3']) \
        + gp.SquaredExponential(sill=1.) + gp.Noise()

    # Generating GP.
    model1 = Model(
        latent = covariance,
        parameters = dict(b1=0.5, b2=0.5, b3=0.25, range=0.5, nugget=1.),
        verbose = True)

    # Generate data.
    vals1 = model1.generate(locs1).vals

    # Fit GP.
    model2 = Model(
        latent = covariance,
        parameters = dict(b1=0., b2=0., b3=0., range=1., nugget=0.5),
        verbose = True).fit(locs1, vals1, iters=200)
    print()

    # MCMC.
    model3 = Model(
        latent = covariance,
        parameters = dict(b1=0., b2=0., b3=0., range=1., nugget=0.5),
        verbose = True).mcmc(locs1, vals1, burnin=500, samples=500)

    # Interpolate using GP.
    N = 20
    xx, yy = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    locs2 = np.stack([xx, yy], axis=-1).reshape([-1, 2])

    mean, var = model2.predict(locs2)
    mean2, var2 = model2.predict(locs2)

    assert np.all(mean == mean2)
    assert np.all(var == var2)

def test_explicit_trend():
    # Create 100 random locations in a square centered on the origin.
    locs1 = np.random.uniform(-1., 1., [500, 2])

    # Initialize featurizer of location for trends.
    def trend_terms(x, y): return x, y, x*y
    featurizer = NormalizingFeaturizer(trend_terms, locs1)
    covariance = gp.Trend(featurizer, beta='beta') \
        + gp.SquaredExponential(sill=1.) + gp.Noise()

    # Generating GP.
    model1 = Model(
        latent = covariance,
        parameters = dict(beta=[1., 0.5, 0.5, 0.25], range=0.5, nugget=1.),
        verbose = True)

    # Generate data.
    vals1 = model1.generate(locs1).vals

    # Fit GP.
    model2 = Model(
        latent = covariance,
        parameters = dict(beta=[0., 0., 0., 0.], range=1., nugget=0.5),
        verbose = True).fit(locs1, vals1, iters=200)
    print()

    # MCMC.
    model3 = Model(
        latent = covariance,
        parameters = dict(beta=[0., 0., 0., 0.], range=1., nugget=0.5),
        verbose = True).mcmc(locs1, vals1, burnin=500, samples=500)

    # Interpolate using GP.
    N = 20
    xx, yy = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    locs2 = np.stack([xx, yy], axis=-1).reshape([-1, 2])

    mean, var = model2.predict(locs2)
    mean2, var2 = model2.predict(locs2)

    assert np.all(mean == mean2)
    assert np.all(var == var2)
