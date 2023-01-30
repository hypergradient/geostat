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
class CovarianceFunction(Op):
    def __add__(self, other):
        return Stack([self]) + other

    def report(self, p):
        string = ', '.join('%s %4.2f' % (v.name, p[v.name]) for v in self.vars())
        return '[' + string + ']'

    def reg(self, p):
        pass

class TrendPrior(CovarianceFunction):
    def __init__(self, featurizer, alpha='alpha'):
        fa = dict(alpha=alpha)
        self.featurizer = featurizer
        super().__init__(fa, dict(locs='locs'))

    def vars(self):
        return ppp(self.fa['alpha'])

    def __call__(self, p, e):
        v = get_parameter_values(self.fa, p)
        F = tf.cast(self.featurizer(e['locs']), tf.float32)
        return v['alpha'] * tf.einsum('ba,ca->bc', F, F)

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

class SquaredExponential(CovarianceFunction):
    def __init__(self, sill='sill', range='range', scale=None, metric=None):
        fa = dict(sill=sill, range=range)
        autoinputs = scale_to_metric(scale, metric)
        super().__init__(fa, dict(d2=autoinputs))

    def vars(self):
        return ppp(self.fa['sill']) + ppp(self.fa['range'])

    def __call__(self, p, e):
        v = get_parameter_values(self.fa, p)
        return v['sill'] * tf.exp(-e['d2'] / tf.square(v['range']))

    def reg(self, p):
        v = get_parameter_values(self.fa, p)
        return v['range']

class GammaExponential(CovarianceFunction):
    def __init__(self, range='range', sill='sill', gamma='gamma', scale=None, metric=None):
        fa = dict(sill=sill, range=range, gamma=gamma, scale=scale)
        autoinputs = scale_to_metric(scale, metric)
        super().__init__(fa, dict(d2=autoinputs))

    def vars(self):
        return ppp(self.fa['sill']) + ppp(self.fa['range']) + bpp(self.fa['gamma'], 0., 2.)

    def __call__(self, p, e):
        v = get_parameter_values(self.fa, p)
        return v['sill'] * gamma_exp(e['d2'] / tf.square(v['range']), v['gamma'])

    def reg(self, p):
        v = get_parameter_values(self.fa, p)
        return v['range']

class Noise(CovarianceFunction):
    def __init__(self, nugget='nugget'):
        fa = dict(nugget=nugget)
        super().__init__(fa, dict(locs='locs'))

    def vars(self):
        return ppp(self.fa['nugget'])

    def __call__(self, p, e):
        v = get_parameter_values(self.fa, p)

        return v['nugget'] * tf.eye(tf.shape(e['locs'])[0])

    def reg(self, p):
        return 0.

class Delta(CovarianceFunction):
    def __init__(self, dsill='dsill', axes=None):
        fa = dict(dsill=dsill)
        self.axes = axes
        super().__init__(fa, dict(pa_d2='per_axis_dist2'))

    def vars(self):
        return ppp(self.fa['dsill'])

    def __call__(self, p, e):
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

class Stack(CovarianceFunction):
    def __init__(self, parts: List[CovarianceFunction]):
        self.parts = parts
        super().__init__({}, dict(parts=parts))

    def vars(self):
        return [p for part in self.parts for p in part.vars()]

    def __add__(self, other):
        if isinstance(other, CovarianceFunction):
            return Stack(self.parts + [other])
    
    def __call__(self, p, e):
        return tf.reduce_sum([x for x in e['parts']], axis=0)

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
        offset: Union[float, Callable],
        noise: CovarianceFunction
    ):
        self.coefs = coefs
        self.offset = offset
        self.noise = noise
        super().__init__({}, self.noise)

    def vars(self):
        vv = [p for c in self.coefs for p in upp(c)]
        vv += self.noise.vars()
        return vv

    def mu(self, locs):
        if callable(self.offset):
            return self.offset(*tf.unstack(locs, axis=1))
        else:
            return self.offset * tf.ones_like(locs[..., 0], tf.float32)

    def __call__(self, p, e):
        """
        Dummy.
        """
        return 0.
