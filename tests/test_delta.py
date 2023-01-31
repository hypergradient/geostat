import numpy as np
from geostat import GP, NormalizingFeaturizer, Observation
import geostat.covfunc as cf
import numpy as np
import tensorflow as tf
from argparse import Namespace


def test_multigp():
    # Create 100 random locations in a square centered on the origin.
    np.random.seed(123)
    tf.random.set_seed(123)

    counts = np.array([100, 150, 200, 200]) * 3
    locs1 = np.random.uniform(-1., 1., [sum(counts) // 3, 4]) * [10., 10., 1., 1.]
    locs1 = locs1[:, np.newaxis, :] + np.pad(np.linspace(0.01, 0.1, 3)[:, np.newaxis], ((0, 0), (2, 1)))
    locs1 = locs1.reshape([-1, 4])

    cats1 = [x for i, count in enumerate(counts) for x in [i] * count]
    # np.random.shuffle(cats1)
    
    # print(locs1.shape)
    # for i, (row, cat) in enumerate(zip(locs1, cats1)):
    #     print(i, ':', cat, ':', locs1[i])

    # Initialize featurizer of location for trends.
    def trend_terms(x, y, z, t): return z, z*z, z*z*z
    featurizer = NormalizingFeaturizer(trend_terms, locs1)

    covu = cf.TrendPrior(featurizer, alpha='au') \
         + cf.SquaredExponential(sill='su1', range='ru1', scale=[1., 1., 'zu', 'tu'])
    covp = cf.TrendPrior(featurizer, alpha='ap') \
         + cf.SquaredExponential(sill='sp1', range='rp1', scale=[1., 1., 'zp', 0.  ])
    covt = cf.TrendPrior(featurizer, alpha='at') \
         + cf.SquaredExponential(sill='st1', range='rt1', scale=[1., 1., 'zt', 'tt'])

    feat_r = NormalizingFeaturizer(lambda x, y, z, t: (), locs1)

    def neg_transformed_natural_gradient(x, y, z, t):
        degF = 62.2 + 57.1 * (0.15 - z)
        return -(tf.math.log(degF + 6.77) - tf.math.log(75 + 6.77))

    obsu = Observation([1., 0., 0.], 0., cf.Noise(nugget='nu') + cf.Delta(dsill='wu', axes=[0, 1]))
    obsp = Observation([0., 1., 0.], 0., cf.Noise(nugget='np') + cf.Delta(dsill='wp', axes=[0, 1]))
    obst = Observation([0., 0., 1.], 0., cf.Noise(nugget='nt') + cf.Delta(dsill='wt', axes=[0, 1]))
    obsR = Observation(['cu', 'cp', 0.], neg_transformed_natural_gradient,
        cf.Noise(nugget='nr') + cf.Delta(dsill='wr', axes=[0, 1]) + cf.TrendPrior(feat_r, alpha='ar'))

    p_init = {'au': 1.,
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

    def report(p):
        p = {k: (v.numpy() if hasattr(v, 'numpy') else v) for k, v in p.items()}
        p = Namespace(**p)
        print(f'[iter {p.iter:5d}] [ll {p.ll:.3f}] [reg {p.reg:.3f}] [time {p.time:.1f}]')
        print(f'   {"alp":>6s} {" zs":>6s} {" ts":>6s} {" sil":>6s} {" rng":>6s} {"  c":>6s} {" wi":>6s} {"nug":>6s}')
        print(f'u: {p.au:6.3f} {p.zu:6.3f} {p.tu:6.3f} {p.su1:6.3f} {p.ru1:6.3f} {p.cu:6.3f} {p.wu:6.3f} {p.nu:6.3f}')
        print(f'p: {p.ap:6.3f} {p.zp:6.3f} {"   ":>6s} {p.sp1:6.3f} {p.rp1:6.3f} {p.cp:6.3f} {p.wp:6.3f} {p.np:6.3f}')
        print(f't: {p.at:6.3f} {p.zt:6.3f} {p.tt:6.3f} {p.st1:6.3f} {p.rt1:6.3f} {-1. :6.3f} {p.wt:6.3f} {p.nt:6.3f}')
        print(f'r: {p.ar:6.3f} {"    ":6s} {"    ":6s} {"     ":6s} {"     ":6s} {"    ":6s} {p.wr:6.3f} {p.nr:6.3f}')

    gp1 = GP(
        covariance = [covu, covp, covt],
        observation = [obsu, obsp, obst, obsR],
        parameters = p_init,
        report = report,
        verbose = True)

    # Generate data.
    vals1 = gp1.generate(locs1, cats1).vals

    gp2 = GP(
        covariance = [covu, covp, covt],
        observation = [obsu, obsp, obst, obsR],
        parameters = {k: 2*v if 'g' not in k else v for k, v in p_init.items()},
        report = report,
        verbose=True).fit(locs1, vals1, cats1, reg=1., iters=500)

    # # Interpolate using GP.
    # N = 20
    # xx, yy = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    # locs2 = np.stack([xx, yy], axis=-1)
    # for cats in range(3):
    #     mean, var = gp2.predict(locs2, cats * np.ones_like(locs2[..., 0], dtype=np.int32))
