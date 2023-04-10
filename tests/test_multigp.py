from argparse import Namespace
import numpy as np
import tensorflow as tf
from geostat import Featurizer, GP, Model, NormalizingFeaturizer, Mix, Trend
import geostat.kernel as krn

def test_multigp():

    np.random.seed(12)
    tf.random.set_seed(12)

    # Create random locations in a square centered on the origin.
    N = 600
    locs1 = np.random.normal(size=[3*N, 2])
    cats1 = [0] * N + [1] * N + [2] * N

    # Initialize featurizer of location for trends.
    def trend_terms(x, y): return x, y, x*y
    featurizer = NormalizingFeaturizer(trend_terms, locs1)
    in1 = GP(0, krn.TrendPrior(featurizer, alpha='a1') + krn.SquaredExponential(sill='s1', range='r1'))
    in2 = GP(0, krn.TrendPrior(featurizer, alpha='a2') + krn.SquaredExponential(sill='s2', range='r2'))

    f2 = Featurizer(lambda x, y: (1, x + y*y))
    out1 = GP(Trend(f2, beta=[0., 0.]), krn.Noise(nugget='n1'))
    out2 = GP(Trend(f2, beta=['c2', 0.]), krn.Noise(nugget='n2'))
    out3 = GP(Trend(f2, beta=[0., 1.]), krn.Noise(nugget='n3'))

    gp = Mix([in1, in2], [[1., 0.], [0., 1.], ['k1', 'k2']]) + Mix([out1, out2, out3])

    def report(p, prefix=''):
        p = {k: (v.numpy() if hasattr(v, 'numpy') else v) for k, v in p.items()}
        p = Namespace(**p)

        if 'iter' in p: print(f'[iter {p.iter:5d}] [ll {p.ll:.3f}] [reg {p.reg:.3f}] [time {p.time:.1f}]')

        print(f"""
            a1={p.a1:.2f}, s1={p.s1:.2f}, r1={p.r1:.2f}, k1={p.k1:.2f},
            a2={p.a2:.2f}, s2={p.s2:.2f}, r2={p.r2:.2f}, k2={p.k2:.2f}, c2={p.c2:.2f},
            n1={p.n1:.2f}, n2={p.n2:.2f}, n3={p.n3:.2f}"""[1:])

    # Generating GP.
    vals1 = Model(
        gp,
        parameters = dict(
            a1=1., s1=1., r1=0.5, k1=2.,
            a2=1., s2=1., r2=0.5, k2=3., c2=1.,
            n1=0.1, n2=0.2, n3=0.3),
        verbose=True).generate(locs1, cats1).vals

    # Fit GP.
    model = Model(
        gp,
        parameters = dict(
            a1=1., s1=1., r1=1., k1=0.,
            a2=1., s2=1., r2=1., k2=0., c2=0.,
            n1=0.1, n2=0.1, n3=0.1),
        report=report,
        verbose=True).fit(locs1, vals1, cats1, iters=5000)

    # Interpolate using GP.
    N = 20
    xx, yy = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    locs2 = np.stack([xx, yy], axis=-1)
    for cats in range(3):
        mean, var = model.predict(locs2, cats * np.ones_like(locs2[..., 0], dtype=np.int32))

    assert np.allclose(
        [model.parameters[p] for p in 's1 r1 k1 s2 r2 k2 c2 n1 n2 n3'.split()],
        [1., 0.5, 2., 1., 0.5, 3., 1., 0.1, 0.2, 0.3],
        rtol=0.5)
