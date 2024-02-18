import numpy as np
from geostat import GP, Model, Mix, Parameters, Trend
import geostat
import geostat.kernel as krn
import numpy as np
import tensorflow as tf
from argparse import Namespace


def test_delta():
    # Create 100 random locations in a square centered on the origin.
    np.random.seed(123)
    tf.random.set_seed(123)

    counts = np.array([100, 150, 200, 200]) * 3
    locs1 = np.random.uniform(-1., 1., [sum(counts) // 3, 4]) * [10., 10., 1., 1.]
    locs1 = locs1[:, np.newaxis, :] + np.pad(np.linspace(0.01, 0.1, 3)[:, np.newaxis], ((0, 0), (2, 1)))
    locs1 = locs1.reshape([-1, 4])

    cats1 = [x for i, count in enumerate(counts) for x in [i] * count]

    p_init = {
      'au': 1.,
      'su1': 0.01,
      'ru1': 0.5,
      'cu': -0.5,
      'wu': 0.01,
      'nu': 0.02,
      'ap': 1.,
      'sp1': 0.04,
      'rp1': 0.4,
      'cp': -0.3,
      'wp': 0.002,
      'np': 0.001,
      'at': 0.1,
      'st1': 0.02,
      'rt1': 1.,
      'wt': 0.01,
      'nt': 0.01,
      'ar': 1.,
      'wr': 0.03,
      'nr': 0.001,
      'zu': 10.,
      'zp': 15.,
      'zt': 20.,
      'tu': 10.,
      'tt': 10.}

    p = Parameters(**p_init)

    # Featurizer of location for trends.
    @geostat.featurizer(normalize=locs1)
    def trend_featurizer(x, y, z, t): return z, z*z, z*z*z

    i_u = GP(0, krn.TrendPrior(trend_featurizer, alpha=p.au)
                 + krn.SquaredExponential(sill=p.su1, range=p.ru1, scale=[1., 1., p.zu, p.tu]))
    i_p = GP(0, krn.TrendPrior(trend_featurizer, alpha=p.ap)
                 + krn.SquaredExponential(sill=p.sp1, range=p.rp1, scale=[1., 1., p.zp, 0.  ]))
    i_t = GP(0, krn.TrendPrior(trend_featurizer, alpha=p.at)
                 + krn.SquaredExponential(sill=p.st1, range=p.rt1, scale=[1., 1., p.zt, p.tt]))

    @geostat.featurizer()
    def unit_featurizer(x, y, z, t): return (1,)

    @geostat.featurizer()
    def neg_transformed_natural_gradient(x, y, z, t):
        degF = 62.2 + 57.1 * (0.15 - z)
        return -(tf.math.log(degF + 6.77) - tf.math.log(75 + 6.77))

    o_u = GP(0, krn.Noise(nugget=p.nu) + krn.Delta(sill=p.wu, axes=[0, 1]))
    o_p = GP(0, krn.Noise(nugget=p.np) + krn.Delta(sill=p.wp, axes=[0, 1]))
    o_t = GP(0, krn.Noise(nugget=p.nt) + krn.Delta(sill=p.wt, axes=[0, 1]))
    o_r = GP(Trend(neg_transformed_natural_gradient, beta=[1.]),
             krn.Noise(nugget=p.nr) + krn.Delta(sill=p.wr, axes=[0, 1])
             + krn.TrendPrior(unit_featurizer, alpha=p.ar))

    gp = Mix([i_u, i_p, i_t], [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [p.cu, p.cp, 0.]]) \
       + Mix([o_u, o_p, o_t, o_r])
        
    def report(p):
        p = {k: (v.numpy() if hasattr(v, 'numpy') else v) for k, v in p.items()}
        p = Namespace(**p)
        print(f'[iter {p.iter:5d}] [ll {p.ll:.3f}] [reg {p.reg:.3f}] [time {p.time:.1f}]')
        print(f'   {"alp":>6s} {" zs":>6s} {" ts":>6s} {" sil":>6s} {" rng":>6s} {"  c":>6s} {" wi":>6s} {"nug":>6s}')
        print(f'u: {p.au:6.3f} {p.zu:6.3f} {p.tu:6.3f} {p.su1:6.3f} {p.ru1:6.3f} {p.cu:6.3f} {p.wu:6.3f} {p.nu:6.3f}')
        print(f'p: {p.ap:6.3f} {p.zp:6.3f} {"   ":>6s} {p.sp1:6.3f} {p.rp1:6.3f} {p.cp:6.3f} {p.wp:6.3f} {p.np:6.3f}')
        print(f't: {p.at:6.3f} {p.zt:6.3f} {p.tt:6.3f} {p.st1:6.3f} {p.rt1:6.3f} {-1. :6.3f} {p.wt:6.3f} {p.nt:6.3f}')
        print(f'r: {p.ar:6.3f} {"    ":6s} {"    ":6s} {"     ":6s} {"     ":6s} {"    ":6s} {p.wr:6.3f} {p.nr:6.3f}')

    model = Model(gp, report = report)

    # Generate data.
    vals1 = model.generate(locs1, cats1).vals

    @geostat.function
    def reg(ru, rp, rt):
        return ru + rp + rt

    # Perturb model parameters and fit.
    model.set(**{k: 2*v if 'g' not in k else v for k, v in p_init.items()})
    model.fit(locs1, vals1, cats1, reg=reg(p.ru1, p.rp1, p.rt1), iters=500)
