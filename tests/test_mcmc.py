import numpy as np
from geostat import GP, Model, Featurizer, NormalizingFeaturizer, Mix, Trend
import geostat.kernel as krn
from types import SimpleNamespace

def test_mcmc():
    
    # Create 100 random locations in a square centered on the origin.
    locs1 = np.random.uniform(-1., 1., [1000, 2])

    # Initialize featurizer of location for trends.
    def trend_terms(x, y): return x, y, x*y
    featurizer = NormalizingFeaturizer(trend_terms, locs1)
    kernel = krn.TrendPrior(featurizer) + krn.SquaredExponential(sill=1.) + krn.Noise()

    # Generating GP.
    model1 = Model(
        GP(0, kernel),
        parameters = dict(alpha=1., range=0.5, nugget=1.),
        verbose=True)

    # Generate data.
    vals1 = model1.generate(locs1).vals

    # Fit GP.
    model2 = Model(
        GP(0, kernel),
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
    in1 = GP(0, krn.TrendPrior(featurizer, alpha='a1') + krn.SquaredExponential(sill='s1', range='r1'))
    in2 = GP(0, krn.TrendPrior(featurizer, alpha='a2') + krn.SquaredExponential(sill='s2', range='r2'))

    f2 = Featurizer(lambda x, y: (1, x + y*y))
    out1 = GP(Trend(f2, beta=[0., 0.]), krn.Noise(nugget='n1'))
    out2 = GP(Trend(f2, beta=[1., 0.]), krn.Noise(nugget='n2'))
    out3 = GP(Trend(f2, beta=[0., 1.]), krn.Noise(nugget='n3') + krn.Delta(dsill='d', axes=[1]))

    gp = Mix([in1, in2], [[1., 0.], [0., 1.], ['k1', 'k2']]) + Mix([out1, out2, out3])

    # Generating GP.
    model1 = Model(
        gp,
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
        gp,
        parameters = dict(
            a1=1., s1=1., r1=1., k1=0.,
            a2=1., s2=1., r2=1., k2=0.,
            n1=0.1, n2=0.1, n3=0.1, d=0.1),
        report=report,
        verbose=True
    ).mcmc(locs1, vals1, cats1,
        step_size=0.02, samples=2000, burnin=500, report_interval=100)
