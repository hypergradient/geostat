import numpy as np
from geostat import GP, NormalizingFeaturizer
import geostat.covfunc as cf
from types import SimpleNamespace

def foo_test_mcmc():
    
    # Create 100 random locations in a square centered on the origin.
    locs1 = np.random.uniform(-1., 1., [1000, 2])

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
    vals1 = gp1.generate(locs1).vals

    # Fit GP.
    gp2 = GP(
        covariance = covariance,
        parameters = dict(alpha=2., range=1., nugget=0.5),
        hyperparameters = dict(reg=0, train_iters=500),
        verbose=True).mcmc(locs1, vals1)

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
    cov1 = cf.Trend(featurizer, alpha='a1') + cf.SquaredExponential(sill='s1', range='r1')
    cov2 = cf.Trend(featurizer, alpha='a2') + cf.SquaredExponential(sill='s2', range='r2')

    obs1 = cf.Observation([1., 0.], 0., cf.Noise(nugget='n1'))
    obs2 = cf.Observation([0., 1.], 1., cf.Noise(nugget='n2'))
    def off3(x, y): return x + y*y
    obs3 = cf.Observation(['k1', 'k2'], off3, cf.Noise(nugget='n3') + cf.Delta(dsill='d', axes=[1]))

    # Generating GP.
    gp1 = GP(
        covariance = [cov1, cov2],
        observation = [obs1, obs2, obs3],
        parameters = dict(
            a1=1., s1=1., r1=0.5, k1=2.,
            a2=1., s2=1., r2=0.5, k2=3.,
            n1=0.1, n2=0.2, n3=0.3, d=0.1),
        verbose=True)

    # Generate data.
    cats1 = [0] * N + [1] * N + [2] * N
    vals1 = gp1.generate(locs1, cats1).vals

    # Reporting function.
    def report(p):
        x = SimpleNamespace(**p)
        print('{:5.3s} {:5.3s} {:5.3s} {:5.3s} {:5.3s} {:5.3s} {:5.3s}'.format('', 'a', 's', 'r', 'k', 'n', 'd'))
        print('{:5.3s} {:5.3f} {:5.3f} {:5.3f} {:5.3f} {:5.3f} {:5.3s}'.format('1', x.a1, x.s1, x.r1, x.k1, x.n1, ''))
        print('{:5.3s} {:5.3f} {:5.3f} {:5.3f} {:5.3f} {:5.3f} {:5.3s}'.format('2', x.a2, x.s2, x.r2, x.k2, x.n2, ''))
        print('{:5.3s} {:5.3s} {:5.3s} {:5.3s} {:5.3s} {:5.3f} {:5.3f}'.format('3', '', '', '', '', x.n3, x.d))

    # Fit GP.
    gp2 = GP(
        covariance = [cov1, cov2],
        observation = [obs1, obs2, obs3],
        parameters = dict(
            a1=1., s1=1., r1=1., k1=0.,
            a2=1., s2=1., r2=1., k2=0.,
            n1=0.1, n2=0.1, n3=0.1, d=0.1),
        report=report,
        verbose=True
    ).mcmc(locs1, vals1, cats1,
        step_size=0.02, samples=2000, burnin=500, report_interval=100)
