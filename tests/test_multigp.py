from argparse import Namespace
import numpy as np
import tensorflow as tf
from geostat import GP, Model, Mix, Parameters, Trend
import geostat
import geostat.kernel as krn

def test_multigp():

    np.random.seed(12)
    tf.random.set_seed(12)

    # Create random locations in a square centered on the origin.
    N = 600
    locs1 = np.random.normal(size=[3*N, 2])
    cats1 = [0] * N + [1] * N + [2] * N

    p = Parameters(
        a1=1., s1=1., r1=0.5, k1=2.,
        a2=1., s2=1., r2=0.5, k2=3., c2=1.,
        n1=0.1, n2=0.2, n3=0.3)

    # Initialize featurizer of location for trends.
    @geostat.featurizer(normalize=locs1)
    def trend_featurizer(x, y): return x, y, x*y

    in1 = GP(0, krn.TrendPrior(trend_featurizer, alpha=p.a1)
                + krn.SquaredExponential(sill=p.s1, range=p.r1))
    in2 = GP(0, krn.TrendPrior(trend_featurizer, alpha=p.a2)
                + krn.SquaredExponential(sill=p.s2, range=p.r2))

    @geostat.featurizer()
    def f2(x, y): return 1, x + y*y

    out1 = GP(Trend(f2, beta=[0., 0.]), krn.Noise(nugget=p.n1))
    out2 = GP(Trend(f2, beta=[p.c2, 0.]), krn.Noise(nugget=p.n2))
    out3 = GP(Trend(f2, beta=[0., 1.]), krn.Noise(nugget=p.n3))

    gp = Mix([in1, in2], [[1., 0.], [0., 1.], [p.k1, p.k2]]) + Mix([out1, out2, out3])

    def report(p, prefix=''):
        p = {k: (v.numpy() if hasattr(v, 'numpy') else v) for k, v in p.items()}
        p = Namespace(**p)

        if 'iter' in p: print(f'[iter {p.iter:5d}] [ll {p.ll:.3f}] [time {p.time:.1f}]')

        print(f"""
            a1={p.a1:.2f}, s1={p.s1:.2f}, r1={p.r1:.2f}, k1={p.k1:.2f},
            a2={p.a2:.2f}, s2={p.s2:.2f}, r2={p.r2:.2f}, k2={p.k2:.2f}, c2={p.c2:.2f},
            n1={p.n1:.2f}, n2={p.n2:.2f}, n3={p.n3:.2f}"""[1:])

    # Generating GP.
    model = Model(gp)
    vals1 = model.generate(locs1, cats1).vals

    # Fit GP.
    model.set(
        a1=1., s1=1., r1=1., k1=0.,
        a2=1., s2=1., r2=1., k2=0., c2=0.,
        n1=0.1, n2=0.1, n3=0.1)

    model.fit(locs1, vals1, cats1, iters=5000)

    assert np.allclose(
        [vars(p)[name].value for name in 's1 r1 k1 s2 r2 k2 c2 n1 n2 n3'.split()],
        [1., 0.5, 2., 1., 0.5, 3., 1., 0.1, 0.2, 0.3],
        rtol=0.5)

    # Interpolate using GP.
    N = 20
    xx, yy = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    locs2 = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    for cats in range(3):
        mean, var = model.predict(locs2, cats * np.ones(locs2.shape[:1], dtype=np.int32))
