from dataclasses import dataclass
from typing import Callable, List, Union

# Tensorflow is extraordinarily noisy. Catch warnings during import.
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow as tf

from .op import Op
from .metric import Euclidean
from .param import get_parameter_values, ppp, upp, bpp

@dataclass
class GP(Op):
    def __init__(self, fa, autoinputs):
        if 'offset' not in autoinputs: autoinputs['offset'] = 'offset'
        if 'locs1' not in autoinputs: autoinputs['locs1'] = 'locs1'
        if 'locs2' not in autoinputs: autoinputs['locs2'] = 'locs2'
        super().__init__(fa, autoinputs)

    def __add__(self, other):
        return Stack([self]) + other

    def call(self, p, e):
        """
        Returns tuple `(mean, covariance)` for locations.
        Return values may be unbroadcasted.
        """
        pass

    def __call__(self, p, e):
        """
        Returns tuple `(mean, covariance)` for locations.
        Return values have correct shapes.
        """
        M, C = self.call(p, e)
        if M is None: M = 0.
        if C is None: C = 0.
        n1 = tf.shape(e['locs1'])[0]
        n2 = tf.shape(e['locs2'])[0]
        M = tf.broadcast_to(M, [n1])
        C = tf.broadcast_to(C, [n1, n2])
        return M, C

    def report(self, p):
        string = ', '.join('%s %4.2f' % (v.name, p[v.name]) for v in self.vars())
        return '[' + string + ']'

    def reg(self, p):
        pass

def get_trend_coefs(beta):
    if isinstance(beta, (list, tuple)):
        return [p for s in beta for p in upp(s)]
    elif isinstance(beta, str):
        return upp(beta)
    else:
        return []

class Trend(GP):
    def __init__(self, featurizer, beta='beta'):
        fa = dict(beta=beta)
        self.featurizer = featurizer
        super().__init__(fa, dict(locs1='locs1'))

    def vars(self):
        return get_trend_coefs(self.fa['beta'])

    def call(self, p, e):
        v = get_parameter_values(self.fa, p)
        x = tf.cast(self.featurizer(e['locs1']), tf.float32)
        if isinstance(v['beta'], (tuple, list)):
            v['beta'] = tf.stack(v['beta'])
        return tf.einsum('ab,b->a', x, v['beta']), None # [locs1]

    def reg(self, p):
        return 0.

class TrendPrior(GP):
    def __init__(self, featurizer, alpha='alpha'):
        fa = dict(alpha=alpha)
        self.featurizer = featurizer
        super().__init__(fa, dict(locs1='locs1', locs2='locs2'))

    def vars(self):
        return ppp(self.fa['alpha'])

    def call(self, p, e):
        v = get_parameter_values(self.fa, p)
        F1 = tf.cast(self.featurizer(e['locs1']), tf.float32)
        F2 = tf.cast(self.featurizer(e['locs2']), tf.float32)
        return None, v['alpha'] * tf.einsum('ba,ca->bc', F1, F2)

    def reg(self, p):
        return 0.

def scale_to_metric(scale, metric):
    assert scale is None or metric is None
    if metric is None:
        if scale is None:
            metric = 'euclidean'
        else:
            metric = Euclidean(scale)
    return metric

class SquaredExponential(GP):
    def __init__(self, sill='sill', range='range', scale=None, metric=None):
        fa = dict(sill=sill, range=range)
        autoinputs = scale_to_metric(scale, metric)
        super().__init__(fa, dict(d2=autoinputs))

    def vars(self):
        return ppp(self.fa['sill']) + ppp(self.fa['range'])

    def call(self, p, e):
        v = get_parameter_values(self.fa, p)
        return None, v['sill'] * tf.exp(-e['d2'] / tf.square(v['range']))

    def reg(self, p):
        v = get_parameter_values(self.fa, p)
        return v['range']

class GammaExponential(GP):
    def __init__(self, range='range', sill='sill', gamma='gamma', scale=None, metric=None):
        fa = dict(sill=sill, range=range, gamma=gamma, scale=scale)
        autoinputs = scale_to_metric(scale, metric)
        super().__init__(fa, dict(d2=autoinputs))

    def vars(self):
        return ppp(self.fa['sill']) + ppp(self.fa['range']) + bpp(self.fa['gamma'], 0., 2.)

    def call(self, p, e):
        v = get_parameter_values(self.fa, p)
        return None, v['sill'] * gamma_exp(e['d2'] / tf.square(v['range']), v['gamma'])

    def reg(self, p):
        v = get_parameter_values(self.fa, p)
        return v['range']

class Noise(GP):
    def __init__(self, nugget='nugget'):
        fa = dict(nugget=nugget)
        super().__init__(fa, dict(locs1='locs1', locs2='locs2'))

    def vars(self):
        return ppp(self.fa['nugget'])

    def call(self, p, e):
        v = get_parameter_values(self.fa, p)

        indices1 = tf.range(tf.shape(e['locs1'])[0])
        indices2 = tf.range(tf.shape(e['locs2'])[0]) + e['offset']
        C = tf.where(tf.equal(tf.expand_dims(indices1, -1), indices2), v['nugget'], 0.)
        return None, C

    def reg(self, p):
        return 0.

class Delta(GP):
    def __init__(self, dsill='dsill', axes=None):
        fa = dict(dsill=dsill)
        self.axes = axes
        super().__init__(fa, dict(pa_d2='per_axis_dist2'))

    def vars(self):
        return ppp(self.fa['dsill'])

    def call(self, p, e):
        v = get_parameter_values(self.fa, p)

        if self.axes is not None:
            n = tf.shape(e['pa_d2'])[-1]
            mask = tf.math.bincount(self.axes, minlength=n, maxlength=n, dtype=tf.float32)
            d2 = tf.einsum('abc,c->ab', e['pa_d2'], mask)
        else:
            d2 = tf.reduce_sum(e['pa_d2'], axis=-1)

        return None, v['dsill'] * tf.cast(tf.equal(d2, 0.), tf.float32)

    def reg(self, p):
        return 0.

class Stack(GP):
    def __init__(self, parts: List[GP]):
        self.parts = parts
        super().__init__({}, dict(locs1='locs1', locs2='locs2', parts=parts))

    def vars(self):
        return [p for part in self.parts for p in part.vars()]

    def __add__(self, other):
        if isinstance(other, GP):
            return Stack(self.parts + [other])
    
    def call(self, p, e):
        MM, CC = zip(*e['parts'])
        MM = [M for M in MM if M is not None]
        CC = [C for C in CC if C is not None]
        M = tf.reduce_sum(MM, axis=0)
        C = tf.reduce_sum(CC, axis=0)
        return M, C

    def report(self, p):
        return ' '.join(part.report(p) for part in self.parts)

    def reg(self, p):
        return tf.reduce_sum([part.reg(p) for part in self.parts], axis=0)

# Gamma exponential covariance function.
@tf.custom_gradient
def safepow(x, a):
    y = tf.pow(x, a)
    def grad(dy):
        dx = tf.where(x <= 0.0, tf.zeros_like(x), dy * tf.pow(x, a-1))
        dx = unbroadcast(dx, x.shape)
        da = tf.where(x <= 0.0, tf.zeros_like(a), dy * y * tf.math.log(x))
        da = unbroadcast(da, a.shape)
        return dx, da
    return y, grad

def unbroadcast(x, shape):
    excess_rank = tf.maximum(0, len(x.shape) - len(shape))
    x = tf.reduce_sum(x, axis=tf.range(excess_rank))
    axes_that_are_one = tf.where(tf.equal(shape, 1))[:, 0]
    x = tf.reduce_sum(x, axis=axes_that_are_one, keepdims=True)
    return x

def gamma_exp(d2, gamma):
    return tf.exp(-safepow(tf.maximum(d2, 0.0), 0.5 * gamma))

class Observation(Op):

    def __init__(self,
        coefs: List,
        noise: GP
    ):
        self.coefs = coefs
        self.noise = noise
        super().__init__({}, self.noise)

    def vars(self):
        vv = [p for c in self.coefs for p in upp(c)]
        vv += self.noise.vars()
        return vv

    def __call__(self, p, e):
        """
        Dummy.
        """
        return 0.
