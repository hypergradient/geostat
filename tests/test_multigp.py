import numpy as np
from geostat import GP, NormalizingFeaturizer
import geostat.covfunc as cf


def test_multigp():
    N = 500

    # Create random locations in a square centered on the origin.
    locs1 = np.random.uniform(-1., 1., [3*N, 2])

    # Initialize featurizer of location for trends.
    def trend_terms(x, y): return x, y, x*y
    featurizer = NormalizingFeaturizer(trend_terms, locs1)
    cov1 = cf.Trend(featurizer, alpha='a1') + cf.SquaredExponential(sill='s1', range='r1')
    cov2 = cf.Trend(featurizer, alpha='a2') + cf.SquaredExponential(sill='s2', range='r2')

    obs1 = cf.Observation([1., 0.], 0., cf.Noise(nugget='n1'))
    obs2 = cf.Observation([0., 1.], 1., cf.Noise(nugget='n2'))
    def off3(x, y): return x + y*y
    obs3 = cf.Observation(['k1', 'k2'], off3, cf.Noise(nugget='n3'))

    # Generating GP.
    gp1 = GP(
        covariance = [cov1, cov2],
        observation = [obs1, obs2, obs3],
        parameters = dict(
            a1=1., s1=1., r1=0.5, k1=2.,
            a2=1., s2=1., r2=0.5, k2=3.,
            n1=0.1, n2=0.2, n3=0.3),
        verbose=True)

    # Generate data.
    cats1 = [0] * N + [1] * N + [2] * N
    vals1 = gp1.generate(locs1, cats1).vals

    # Fit GP.
    gp2 = GP(
        covariance = [cov1, cov2],
        observation = [obs1, obs2, obs3],

        parameters = dict(
            a1=1., s1=1., r1=1., k1=0.,
            a2=1., s2=1., r2=1., k2=0.,
            n1=0.1, n2=0.1, n3=0.1),
        hyperparameters = dict(reg=0, train_iters=5000),
        verbose=True).fit(locs1, vals1, cats1)

    # Interpolate using GP.
    N = 20
    xx, yy = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    locs2 = np.stack([xx, yy], axis=-1)
    for cats in range(3):
        mean, var = gp2.predict(locs2, cats * np.ones_like(locs2[..., 0], dtype=np.int32))
