import numpy as np
from geostat import gp, Model, Featurizer, NormalizingFeaturizer

def test_multigp():

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
    obs2 = gp.Observation([0., 1.], gp.Trend(f2, beta=['c2', 0.]) + gp.Noise(nugget='n2'))
    obs3 = gp.Observation(['k1', 'k2'], gp.Trend(f2, beta=[0., 1.]) + gp.Noise(nugget='n3') + gp.Delta(dsill='d', axes=[1]))

    # Generating GP.
    model1 = Model(
        latent = [cov1, cov2],
        observed = [obs1, obs2, obs3],
        parameters = dict(
            a1=1., s1=1., r1=0.5, k1=2.,
            a2=1., s2=1., r2=0.5, k2=3., c2=1.,
            n1=0.1, n2=0.2, n3=0.3, d=0.1),
        verbose=True)

    # Generate data.
    cats1 = [0] * N + [1] * N + [2] * N
    vals1 = model1.generate(locs1, cats1).vals

    # Fit GP.
    model2 = Model(
        latent = [cov1, cov2],
        observed = [obs1, obs2, obs3],

        parameters = dict(
            a1=1., s1=1., r1=1., k1=0.,
            a2=1., s2=1., r2=1., k2=0., c2=0.,
            n1=0.1, n2=0.1, n3=0.1, d=0.1),
        verbose=True).fit(locs1, vals1, cats1, iters=2000)

    # Interpolate using GP.
    N = 20
    xx, yy = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    locs2 = np.stack([xx, yy], axis=-1)
    for cats in range(3):
        mean, var = model2.predict(locs2, cats * np.ones_like(locs2[..., 0], dtype=np.int32))
