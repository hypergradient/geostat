from dataclasses import dataclass
from typing import Callable, Dict, List, Union

# Tensorflow is extraordinarily noisy. Catch warnings during import.
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow as tf

from .params import get_parameter_values, ppp, upp, bpp
from .util import einsum_abc_c_ab

@dataclass
class CovarianceFunction:
    fa: Dict[str, object] # Formal arguments.

    def __add__(self, other):
        return Stack([self]) + other

    def vars(self):
        pass

    def matrix(self, x, d2, p):
        pass

    def report(self, p):
        string = ', '.join('%s %4.2f' % (v.name, p[v.name]) for v in self.vars())
        return '[' + string + ']'

    def reg(self, p):
        pass

    def __tf_tracing_type__(self, context):
        return SingletonTraceType(type(self))

class Trend(CovarianceFunction):
    def __init__(self, featurizer, alpha='alpha', axes=None):
        fa = dict(alpha=alpha)
        self.featurizer = featurizer
        super().__init__(fa)

    def vars(self):
        return ppp(self.fa['alpha'])

    def matrix(self, x, d2, p):
        v = get_parameter_values(self.fa, p)
        F = tf.cast(self.featurizer(x), tf.float32)
        return v['alpha'] * tf.einsum('ba,ca->bc', F, F)

    def reg(self, p):
        return 0.

def scale_to_metric(scale, metric):
    assert not (scale is None and metric is None)
    if scale is not None:
        metric = Scaled(scale)
    elif metric is None:
        metric = Euclidean()
    return metric

class SquaredExponential(CovarianceFunction):
    def __init__(self, sill='sill', range='range', scale=None, metric=None):
        self.metric = scale_to_metric(scale, metric)
        fa = dict(sill=sill, range=range)
        super().__init__(fa)

    def vars(self):
        return ppp(self.fa['sill']) + ppp(self.fa['range'])

    def matrix(self, x, d2, p):
        v = get_parameter_values(self.fa, p)

        if v['scale'] is not None:
            scale = v['scale']
        else:
            scale = tf.ones_like(d2[0, 0, :])

        d2 = einsum_abc_c_ab(d2, tf.square(scale / v['range']))
        return v['sill'] * tf.exp(-self.metric.out.d2)

    def reg(self, p):
        v = get_parameter_values(self.fa, p)
        return v['range']

class GammaExponential(CovarianceFunction):
    def __init__(self, range='range', sill='sill', gamma='gamma', scale=None):
        self.metric = scale_to_metric(scale, metric)
        fa = dict(sill=sill, range=range, gamma=gamma, scale=scale)
        super().__init__(fa)

    def vars(self):
        return get_scale_vars(self.fa['scale']) + \
            ppp(self.fa['sill']) + ppp(self.fa['range']) + bpp(self.fa['gamma'], 0., 2.)

    def matrix(self, x, d2, p):
        v = get_parameter_values(self.fa, p)

        if v['scale'] is not None:
            scale = v['scale']
        else:
            scale = tf.ones_like(d2[0, 0, :])
            
        d2 = einsum_abc_c_ab(d2, tf.square(scale / v['range']))
        return v['sill'] * gamma_exp(d2, v['gamma'])

    def reg(self, p):
        v = get_parameter_values(self.fa, p)
        return v['range']

class Noise(CovarianceFunction):
    def __init__(self, nugget='nugget'):
        fa = dict(nugget=nugget)
        super().__init__(fa)

    def vars(self):
        return ppp(self.fa['nugget'])

    def matrix(self, x, d2, p):
        v = get_parameter_values(self.fa, p)

        return v['nugget'] * tf.eye(tf.shape(x)[0])

    def reg(self, p):
        return 0.

class Delta(CovarianceFunction):
    def __init__(self, dsill='dsill', axes=None):
        fa = dict(dsill=dsill)
        self.axes = axes
        super().__init__(fa)

    def vars(self):
        return ppp(self.fa['dsill'])

    def matrix(self, x, d2, p):
        v = get_parameter_values(self.fa, p)

        if self.axes is not None:
            n = tf.shape(x)[-1]
            mask = tf.math.bincount(self.axes, minlength=n, maxlength=n, dtype=tf.float32)
            d2 = einsum_abc_c_ab(d2, mask)
        else:
            d2 = tf.reduce_sum(d2, axis=-1)

        return v['dsill'] * tf.cast(tf.equal(d2, 0.), tf.float32)

    def reg(self, p):
        return 0.

class Stack(CovarianceFunction):
    def __init__(self, parts: List[CovarianceFunction]):
        self.parts = parts
        super().__init__({})

    def vars(self):
        return [p for part in self.parts for p in part.vars()]

    def __add__(self, other):
        if isinstance(other, CovarianceFunction):
            return Stack(self.parts + [other])
    
    def matrix(self, x, d2, p):
        return tf.reduce_sum([part.matrix(x, d2, p) for part in self.parts], axis=0)

    def report(self, p):
        return ' '.join(part.report(p) for part in self.parts)

    def reg(self, p):
        return tf.reduce_sum([part.reg(p) for part in self.parts], axis=0)

def e(x, a=-1):
    return tf.expand_dims(x, a)

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

@dataclass
class Observation:
    coefs: List
    offset: Union[float, Callable]
    noise: CovarianceFunction

    def vars(self):
        vv = [p for c in self.coefs for p in upp(c)]
        vv += self.noise.vars()
        return vv

    def mu(self, locs):
        if callable(self.offset):
            return self.offset(*tf.unstack(locs, axis=1))
        else:
            return self.offset * tf.ones_like(locs[..., 0], tf.float32)

    def __tf_tracing_type__(self, context):
        return SingletonTraceType(type(self))

class SingletonTraceType(tf.types.experimental.TraceType):
  """
  A trace type to override TF's default behavior, which is 
  to treat dataclass-based onjects as dicts.
  """

  def __init__(self, classtype):
     self.classtype = classtype
     pass

  def is_subtype_of(self, other):
     return type(other) is SingletonTraceType \
         and self.classtype is other.classtype

  def most_specific_common_supertype(self, others):
     return self if all(self == other for other in others) else None

  def __eq__(self, other):
     return isinstance(other, SingletonTraceType) and self.classtype == other.classtype

  def __hash__(self):
     return hash(self.classtype)
