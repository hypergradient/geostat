from typing import List

# Tensorflow is extraordinarily noisy. Catch warnings during import.
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow as tf

def dedup(a: List) -> List:
    return list(dict.fromkeys(a))

class CovarianceFunction:
    def __add__(self, other):
        return Stack([self]) + other

    def vars(self):
        pass

    def matrix(self, x, p):
        pass

    def report(self, p):
        string = ', '.join('%s %4.2f' % (v, p[v]) for v in self.vars())
        return '[' + string + ']'

    def reg(self, p):
        pass

class SquaredExponential(CovarianceFunction):
    def __init__(self, range='range', sill='sill', scale=None):
        self.range = range
        self.sill = sill
        self.scale = scale
        super().__init__()

    def vars(self):
        if self.scale is not None:
            scale_variables = [s for s in self.scale if isinstance(s, str)]
        else:
            scale_variables = []
        return dedup(scale_variables + [self.range, self.sill])

    def matrix(self, x, p):
        if self.scale is not None:
            scale_tensor = [p.get(s, s) for s in self.scale]
            x *= scale_tensor
        d2 = tf.reduce_sum(tf.square(e(x, 0) - e(x, 1)), axis=-1)
        return p[self.sill] * tf.exp(-d2 / tf.square(p[self.range]))

    def reg(self, p):
        return p[self.range]

class GammaExponential(CovarianceFunction):
    def __init__(self, range='range', sill='sill', gamma='gamma', scale=None):
        self.range = range
        self.sill = sill
        self.gamma = gamma
        self.scale = scale
        super().__init__()

    def vars(self):
        if self.scale is not None:
            scale_variables = [s for s in self.scale if isinstance(s, str)]
        else:
            scale_variables = []
        return dedup(scale_variables + [self.range, self.sill, self.gamma])

    def matrix(self, x, p):
        if self.scale is not None:
            scale_tensor = [p.get(s, s) for s in self.scale]
            missing = list(filter(lambda s: isinstance(s, str), scale_tensor))
            if missing:
                raise ValueError("parameter(s) [%s] not found" % ', '.join(missing))
            x *= scale_tensor
        d2 = tf.reduce_sum(tf.square(e(x, 0) - e(x, 1)), axis=-1)
        return p[self.sill] * gamma_exp(d2 / tf.square(p[self.range]), p[self.gamma])

    def reg(self, p):
        return p[self.range]

class Noise(CovarianceFunction):
    def __init__(self, nugget='nugget'):
        self.nugget = nugget
        super().__init__()

    def vars(self):
        return [self.nugget]

    def matrix(self, x, p):
        return p[self.nugget] * tf.eye(x.shape[0])

    def reg(self, p):
        return 0.

class Stack(CovarianceFunction):
    def __init__(self, parts: List[CovarianceFunction]):
        self.parts = parts
        super().__init__()

    def vars(self):
        return dedup([v for part in self.parts for v in part.vars()])

    def __add__(self, other):
        if isinstance(other, CovarianceFunction):
            return Stack(self.parts + [other])
    
    def matrix(self, x, p):
        return tf.reduce_sum([part.matrix(x, p) for part in self.parts], axis=0)

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
    axes_that_are_one = tf.where(tf.equal(shape, 1))
    x = tf.reduce_sum(x, axis=axes_that_are_one, keepdims=True)
    return x

def gamma_exp(d2, gamma):
    return tf.exp(-safepow(tf.maximum(d2, 0.0), 0.5 * gamma))
