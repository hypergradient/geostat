import numpy as np
from geostat import GP, NormalizingFeaturizer, Trend
import geostat.covfunc as cf

def test_explicit_trend():
    # Create 100 random locations in a square centered on the origin.
    locs1 = np.random.uniform(-1., 1., [500, 2])

    # Initialize featurizer of location for trends.
    def trend_terms(x, y): return x, y, x*y
    featurizer = NormalizingFeaturizer(trend_terms, locs1)
    covariance = cf.SquaredExponential(sill=1.) + cf.Noise()

    # Generating GP.
    gp1 = GP(
        trend = Trend(featurizer, beta=['b0', 'b1', 'b2', 'b3']),
        covariance = covariance,
        parameters = dict(b0=1., b1=0.5, b2=0.5, b3=0.25, range=0.5, nugget=1.),
        verbose = True)

    # Generate data.
    vals1 = gp1.generate(locs1).vals

    # Fit GP.
    gp2 = GP(
        trend = Trend(featurizer, beta=['b0', 'b1', 'b2', 'b3']),
        covariance = covariance,
        parameters = dict(b0=0., b1=0., b2=0., b3=0., range=1., nugget=0.5),
        verbose = True).fit(locs1, vals1, iters=200)
    print()

    # MCMC.
    gp3 = GP(
        trend = Trend(featurizer, beta=['b0', 'b1', 'b2', 'b3']),
        covariance = covariance,
        parameters = dict(b0=0., b1=0., b2=0., b3=0., range=1., nugget=0.5),
        verbose = True).mcmc(locs1, vals1, burnin=500, samples=500)

    # Interpolate using GP.
    N = 20
    xx, yy = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    locs2 = np.stack([xx, yy], axis=-1).reshape([-1, 2])

    mean, var = gp2.predict(locs2)
    mean2, var2 = gp2.predict(locs2)

    assert np.all(mean == mean2)
    assert np.all(var == var2)

def foo_test_explicit_trend():
    # Create 100 random locations in a square centered on the origin.
    locs1 = np.random.uniform(-1., 1., [500, 2])

    # Initialize featurizer of location for trends.
    def trend_terms(x, y): return x, y, x*y
    featurizer = NormalizingFeaturizer(trend_terms, locs1)
    covariance = cf.SquaredExponential(sill=1.) + cf.Noise()

    # Generating GP.
    gp1 = GP(
        trend = Trend(featurizer, beta='beta'),
        covariance = covariance,
        parameters = dict(beta=[1., 0.5, 0.5, 0.25], range=0.5, nugget=1.),
        verbose = True)

    # Generate data.
    vals1 = gp1.generate(locs1).vals

    # Fit GP.
    gp2 = GP(
        trend = Trend(featurizer, beta='beta'),
        covariance = covariance,
        parameters = dict(beta=[0., 0., 0., 0.], range=1., nugget=0.5),
        verbose = True).fit(locs1, vals1, iters=200)
    print()

    # MCMC.
    gp3 = GP(
        trend = Trend(featurizer, beta='beta'),
        covariance = covariance,
        parameters = dict(beta=[0., 0., 0., 0.], range=1., nugget=0.5),
        verbose = True).mcmc(locs1, vals1, burnin=500, samples=500)

    # Interpolate using GP.
    N = 20
    xx, yy = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    locs2 = np.stack([xx, yy], axis=-1).reshape([-1, 2])

    mean, var = gp2.predict(locs2)
    mean2, var2 = gp2.predict(locs2)

    assert np.all(mean == mean2)
    assert np.all(var == var2)
