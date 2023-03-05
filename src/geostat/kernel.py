from dataclasses import dataclass
from typing import Callable, List, Union
import numpy as np

# Tensorflow is extraordinarily noisy. Catch warnings during import.
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow as tf

from .op import Op
from .metric import Euclidean, ed
from .param import get_parameter_values, ppp, upp, bpp

__all__ = ['Kernel', 'Trend']

@dataclass
class Kernel(Op):
    def __init__(self, fa, autoinputs):
        if 'offset' not in autoinputs: autoinputs['offset'] = 'offset'
        if 'locs1' not in autoinputs: autoinputs['locs1'] = 'locs1'
        if 'locs2' not in autoinputs: autoinputs['locs2'] = 'locs2'
        super().__init__(fa, autoinputs)

    def __add__(self, other):
        return Stack([self]) + other

    def __mul__(self, other):
        return Product([self]) * other

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
        C = self.call(p, e)
        if C is None: C = 0.
        n1 = tf.shape(e['locs1'])[0]
        n2 = tf.shape(e['locs2'])[0]
        C = tf.broadcast_to(C, [n1, n2])
        return C

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

class Trend(Op):
    def __init__(self, featurizer, beta='beta'):
        self.featurizer = featurizer
        super().__init__(
            dict(beta=beta),
            dict(locs1='locs1'))

    def vars(self):
        return get_trend_coefs(self.fa['beta'])

    def __call__(self, p, e):
        v = get_parameter_values(self.fa, p)
        x = tf.cast(self.featurizer(e['locs1']), tf.float32)
        if isinstance(v['beta'], (tuple, list)):
            v['beta'] = tf.stack(v['beta'])
        return tf.einsum('ab,b->a', x, v['beta']) # [locs1]

class ZeroTrend(Op):
    def __init__(self):
        super().__init__({}, dict(locs1='locs1'))

    def vars(self):
        return []

    def __call__(self, p, e):
        return tf.zeros_like(e['locs1'][:, 0])

class TrendPrior(Kernel):
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
        return v['alpha'] * tf.einsum('ba,ca->bc', F1, F2)

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

class SquaredExponential(Kernel):
    def __init__(self, sill='sill', range='range', scale=None, metric=None):
        fa = dict(sill=sill, range=range)
        autoinputs = scale_to_metric(scale, metric)
        super().__init__(fa, dict(d2=autoinputs))

    def vars(self):
        return ppp(self.fa['sill']) + ppp(self.fa['range'])

    def call(self, p, e):
        v = get_parameter_values(self.fa, p)
        return v['sill'] * tf.exp(-0.5 * e['d2'] / tf.square(v['range']))

    def reg(self, p):
        v = get_parameter_values(self.fa, p)
        return v['range']

class GammaExponential(Kernel):
    def __init__(self, range='range', sill='sill', gamma='gamma', scale=None, metric=None):
        fa = dict(sill=sill, range=range, gamma=gamma, scale=scale)
        autoinputs = scale_to_metric(scale, metric)
        super().__init__(fa, dict(d2=autoinputs))

    def vars(self):
        return ppp(self.fa['sill']) + ppp(self.fa['range']) + bpp(self.fa['gamma'], 0., 2.)

    def call(self, p, e):
        v = get_parameter_values(self.fa, p)
        return v['sill'] * gamma_exp(e['d2'] / tf.square(v['range']), v['gamma'])

    def reg(self, p):
        v = get_parameter_values(self.fa, p)
        return v['range']

class Wiener(Kernel):
    def __init__(self, axis, start):

        self.axis = axis
        self.start = start

        # Include the element of scale corresponding to the axis of
        # integration as an explicit formal argument.
        fa = dict()

        super().__init__(fa, dict(locs1='locs1', locs2='locs2'))

    def vars(self):
        return []

    def call(self, p, e):
        v = get_parameter_values(self.fa, p)
        x1 = e['locs1'][..., self.axis]
        x2 = e['locs2'][..., self.axis]
        k = tf.minimum(ed(x1, 1), ed(x2, 0)) - self.start
        return k

    def reg(self, p):
        return 0.

class IntSquaredExponential(Kernel):
    def __init__(self, axis, start, range='range'):

        self.axis = axis
        self.start = start

        # Include the element of scale corresponding to the axis of
        # integration as an explicit formal argument.
        fa = dict(range=range)

        super().__init__(fa, dict(locs1='locs1', locs2='locs2'))

    def vars(self):
        return ppp(self.fa['range'])

    def call(self, p, e):
        v = get_parameter_values(self.fa, p)
        x1 = tf.pad(e['locs1'][..., self.axis] - self.start, [[1, 0]])
        x2 = tf.pad(e['locs2'][..., self.axis] - self.start, [[1, 0]])

        r = v['range']
        sdiff = (ed(x1, 1) - ed(x2, 0)) / (r * np.sqrt(2.))
        k = -tf.square(r) * (np.sqrt(np.pi) * sdiff * tf.math.erf(sdiff) + tf.exp(-tf.square(sdiff)))
        k -= k[0:1, :]
        k -= k[:, 0:1]
        k = k[1:, 1:]

        return k

    def reg(self, p):
        return 0.

class IntExponential(Kernel):
    def __init__(self, axis, start, range='range'):

        self.axis = axis
        self.start = start

        # Include the element of scale corresponding to the axis of
        # integration as an explicit formal argument.
        fa = dict(range=range)

        super().__init__(fa, dict(locs1='locs1', locs2='locs2'))

    def vars(self):
        return ppp(self.fa['range'])

    def call(self, p, e):
        v = get_parameter_values(self.fa, p)
        x1 = tf.pad(e['locs1'][..., self.axis] - self.start, [[1, 0]])
        x2 = tf.pad(e['locs2'][..., self.axis] - self.start, [[1, 0]])

        r = v['range']
        sdiff = tf.abs(ed(x1, 1) - ed(x2, 0)) / r
        k = -tf.square(r) * (sdiff + tf.exp(-sdiff))
        k -= k[0:1, :]
        k -= k[:, 0:1]
        k = k[1:, 1:]

        return k

    def reg(self, p):
        return 0.

class Noise(Kernel):
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
        return C

    def reg(self, p):
        return 0.

class Delta(Kernel):
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

        return v['dsill'] * tf.cast(tf.equal(d2, 0.), tf.float32)

    def reg(self, p):
        return 0.

class Stack(Kernel):
    def __init__(self, parts: List[Kernel]):
        self.parts = parts
        super().__init__({}, dict(locs1='locs1', locs2='locs2', parts=parts))

    def vars(self):
        return [p for part in self.parts for p in part.vars()]

    def __add__(self, other):
        if isinstance(other, Kernel):
            return Stack(self.parts + [other])
    
    def call(self, p, e):
        return tf.reduce_sum(e['parts'], axis=0)

    def report(self, p):
        return ' '.join(part.report(p) for part in self.parts)

    def reg(self, p):
        return tf.reduce_sum([part.reg(p) for part in self.parts], axis=0)

class Product(Kernel):
    def __init__(self, parts: List[Kernel]):
        self.parts = parts
        super().__init__({}, dict(locs1='locs1', locs2='locs2', parts=parts))

    def vars(self):
        return [p for part in self.parts for p in part.vars()]

    def __mul__(self, other):
        if isinstance(other, Kernel):
            return Product(self.parts + [other])
    
    def call(self, p, e):
        return tf.reduce_sum(e['parts'], axis=0)

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
        noise: Kernel
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
