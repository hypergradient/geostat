import numpy as np
from geostat import gp, Model, Featurizer, NormalizingFeaturizer
from types import SimpleNamespace

def test_mcmc():
    
    # Create 100 random locations in a square centered on the origin.
    locs1 = np.random.uniform(-1., 1., [1000, 2])

    # Initialize featurizer of location for trends.
    def trend_terms(x, y): return x, y, x*y
    featurizer = NormalizingFeaturizer(trend_terms, locs1)
    covariance = gp.TrendPrior(featurizer) + gp.SquaredExponential(sill=1.) + gp.Noise()

    # Generating GP.
    model1 = Model(
        latent = covariance,
        parameters = dict(alpha=1., range=0.5, nugget=1.),
        verbose=True)

    # Generate data.
    vals1 = model1.generate(locs1).vals

    # Fit GP.
    model2 = Model(
        latent = covariance,
        parameters = dict(alpha=2., range=1., nugget=0.5),
        verbose=True).mcmc(locs1, vals1,
            step_size=0.05, samples=100, burnin=100, report_interval=50)

    # Interpolate using GP.
    N = 20
    xx, yy = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    locs2 = np.stack([xx, yy], axis=-1).reshape([-1, 2])

    mean, var = model2.predict(locs2, subsample=20)

def test_mcmc_multigp():

    # Create random locations in a square centered on the origin.
    N = 200
    locs1 = np.random.uniform(-1., 1., [3*N, 2])
    # Triple data with offsets.
    N *= 3
    locs1 = np.concatenate([locs1 - [0.1, 0], locs1, locs1 + [0.1, 0]], axis=-1).reshape([-1, 2])

    # Initialize featurizer of location for trends.
    def trend_terms(x, y): return x, y, x*y
    featurizer = NormalizingFeaturizer(trend_terms, locs1)
    cov1 = gp.TrendPrior(featurizer, alpha='a1') + gp.SquaredExponential(sill='s1', range='r1')
    cov2 = gp.TrendPrior(featurizer, alpha='a2') + gp.SquaredExponential(sill='s2', range='r2')

    f2 = Featurizer(lambda x, y: (1, x + y*y))
    obs1 = gp.Observation([1., 0.], gp.Trend(f2, beta=[0., 0.]) + gp.Noise(nugget='n1'))
    obs2 = gp.Observation([0., 1.], gp.Trend(f2, beta=[1., 0.]) + gp.Noise(nugget='n2'))
    obs3 = gp.Observation(['k1', 'k2'], gp.Trend(f2, beta=[0., 1.]) + gp.Noise(nugget='n3') + gp.Delta(dsill='d', axes=[1]))

    # Generating GP.
    model1 = Model(
        latent = [cov1, cov2],
        observed = [obs1, obs2, obs3],
        parameters = dict(
            a1=1., s1=1., r1=0.5, k1=2.,
            a2=1., s2=1., r2=0.5, k2=3.,
            n1=0.1, n2=0.2, n3=0.3, d=0.1),
        verbose=True)

    # Generate data.
    cats1 = [0] * N + [1] * N + [2] * N
    vals1 = model1.generate(locs1, cats1).vals

    # Reporting function.
    def report(p, prefix=''):
        x = SimpleNamespace(**p)
        print('{} {:5.3s} {:5.3s} {:5.3s} {:5.3s} {:5.3s} {:5.3s} {:5.3s}'.format(prefix, '', 'a', 's', 'r', 'k', 'n', 'd'))
        print('{} {:5.3s} {:5.3f} {:5.3f} {:5.3f} {:5.3f} {:5.3f} {:5.3s}'.format(prefix, '1', x.a1, x.s1, x.r1, x.k1, x.n1, ''))
        print('{} {:5.3s} {:5.3f} {:5.3f} {:5.3f} {:5.3f} {:5.3f} {:5.3s}'.format(prefix, '2', x.a2, x.s2, x.r2, x.k2, x.n2, ''))
        print('{} {:5.3s} {:5.3s} {:5.3s} {:5.3s} {:5.3s} {:5.3f} {:5.3f}'.format(prefix, '3', '', '', '', '', x.n3, x.d))

    # Fit GP.
    model2 = Model(
        latent = [cov1, cov2],
        observed = [obs1, obs2, obs3],
        parameters = dict(
            a1=1., s1=1., r1=1., k1=0.,
            a2=1., s2=1., r2=1., k2=0.,
            n1=0.1, n2=0.1, n3=0.1, d=0.1),
        report=report,
        verbose=True
    ).mcmc(locs1, vals1, cats1,
        step_size=0.02, samples=2000, burnin=500, report_interval=100)
