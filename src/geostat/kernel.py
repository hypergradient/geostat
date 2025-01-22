from typing import List
import numpy as np

# Tensorflow is extraordinarily noisy. Catch warnings during import.
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow as tf
    from tensorflow.linalg import LinearOperatorFullMatrix as LOFullMatrix
    from tensorflow.linalg import LinearOperatorBlockDiag as LOBlockDiag

from .op import Op
from .metric import Euclidean, PerAxisDist2, ed
from .param import ppp, upp, bpp, ppp_list
from .mean import get_trend_coefs

__all__ = ['Kernel']

def block_diag(blocks):
    """Return a dense block-diagonal matrix."""
    return LOBlockDiag([LOFullMatrix(b) for b in blocks]).to_dense()

class Kernel(Op):
    def __init__(self, fa, autoinputs):
        if 'offset' not in autoinputs: autoinputs['offset'] = 'offset'
        if 'locs1' not in autoinputs: autoinputs['locs1'] = 'locs1'
        if 'locs2' not in autoinputs: autoinputs['locs2'] = 'locs2'
        super().__init__(fa, autoinputs)

    def __add__(self, other):
        if other is None:
            return self
        else:
            return Stack([self]) + other

    def __mul__(self, other):
        return Product([self]) * other

    def call(self, e):
        """
        Returns tuple `(mean, covariance)` for locations.
        Return values may be unbroadcasted.
        """
        pass

    def __call__(self, e):
        """
        Returns tuple `(mean, covariance)` for locations.
        Return values have correct shapes.
        """
        C = self.call(e)
        if C is None: C = 0.
        n1 = tf.shape(e['locs1'])[0]
        n2 = tf.shape(e['locs2'])[0]
        C = tf.broadcast_to(C, [n1, n2])
        return C

    def report(self):
        import sys
        sys.exit(1)
        string = ', '.join('%s %4.2f' % (v.name, p[v.name]) for v in self.vars())
        return '[' + string + ']'

class TrendPrior(Kernel):
    def __init__(self, featurizer, alpha):
        fa = dict(alpha=alpha)
        self.featurizer = featurizer
        super().__init__(fa, dict(locs1='locs1', locs2='locs2'))

    def vars(self):
        return ppp(self.fa['alpha'])

    def call(self, e):
        F1 = tf.cast(self.featurizer(e['locs1']), tf.float32)
        F2 = tf.cast(self.featurizer(e['locs2']), tf.float32)
        return e['alpha'] * tf.einsum('ba,ca->bc', F1, F2)

def scale_to_metric(scale, metric):
    assert scale is None or metric is None
    if metric is None:
        if scale is None:
            metric = 'euclidean'
        else:
            metric = Euclidean(scale)
    return metric

class Constant(Kernel):
    def __init__(self, sill):
        fa = dict(sill=sill)
        super().__init__(fa, dict())

    def vars(self):
        return ppp(self.fa['sill'])

    def call(self, e):
        return e['sill']

class SquaredExponential(Kernel):
    def __init__(self, sill, range, scale=None, metric=None):
        fa = dict(sill=sill, range=range)
        autoinputs = scale_to_metric(scale, metric)
        super().__init__(fa, dict(d2=autoinputs))

    def vars(self):
        return ppp(self.fa['sill']) | ppp(self.fa['range'])

    def call(self, e):
        return e['sill'] * tf.exp(-0.5 * e['d2'] / tf.square(e['range']))

class GammaExponential(Kernel):
    def __init__(self, range, sill, gamma, scale=None, metric=None):
        fa = dict(sill=sill, range=range, gamma=gamma, scale=scale)
        autoinputs = scale_to_metric(scale, metric)
        super().__init__(fa, dict(d2=autoinputs))

    def vars(self):
        return ppp(self.fa['sill']) | ppp(self.fa['range']) | bpp(self.fa['gamma'], 0., 2.)

    def call(self, e):
        return e['sill'] * gamma_exp(e['d2'] / tf.square(e['range']), e['gamma'])

@tf.custom_gradient
def ramp(x):
    ax = tf.abs(x)
    def grad(upstream):
        return upstream * tf.where(ax < 1., -tf.sign(x), 0.)
    return tf.maximum(0., 1. - ax), grad

class Ramp(Kernel):
    def __init__(self, range, sill, scale=None, metric=None):
        fa = dict(sill=sill, range=range, scale=scale)
        autoinputs = scale_to_metric(scale, metric)
        super().__init__(fa, dict(d2=autoinputs))

    def vars(self):
        return ppp(self.fa['sill']) | ppp(self.fa['range'])

    def call(self, e):
        return e['sill'] * ramp(tf.sqrt(e['d2']) / e['range'])

# @tf.custom_gradient
# def rampstack(x, sills, ranges):
#     """
#     `x` has arbitrary shape [...].
#     `sills` and `ranges` both have shape [K].
#     """
#     ax = ed(tf.abs(x)) # [..., 1]
#     y = sills * tf.maximum(0., 1. - ax / ranges) # [..., K]
#     def grad(upstream):
#         ax = ed(tf.abs(x)) # [..., 1]
#         y = sills * tf.maximum(0., 1. - ax / ranges) # [..., K]
#         K = tf.shape(sills)[0]
#         grad_x = upstream * tf.reduce_sum(tf.where(ax < ranges, -tf.sign(ed(x)), 0.), -1) # [...]
#         grad_sills = tf.reduce_sum(tf.reshape(ed(upstream) * y, [-1, K]), 0) # [K}
#         grad_ranges = tf.where(ax < ranges, sills * ax / tf.square(ranges), 0.) # [..., K}
#         grad_ranges = tf.reduce_sum(tf.reshape(ed(upstream) * grad_ranges, [-1, K]), 0) # [K]
#         return grad_x, grad_sills, grad_ranges
#     return tf.reduce_sum(y, -1), grad

@tf.custom_gradient
def rampstack(x, sills, ranges):
    """
    `x` has arbitrary shape [...], but must be non-negative.
    `sills` and `ranges` both have shape [K].
    """
    ax = ed(tf.abs(x)) # [..., 1]
    y = sills * tf.maximum(0., 1. - ax / ranges) # [..., K]
    def grad(upstream):
        ax = ed(tf.abs(x)) # [..., 1]
        y = sills * tf.maximum(0., 1. - ax / ranges) # [..., K]
        K = tf.shape(sills)[0]
        small = ax < ranges
        grad_x = upstream * tf.reduce_sum(tf.where(small, -tf.sign(ed(x)) * (sills / ranges), 0.), -1) # [...]
        grad_sills = tf.einsum('ak,a->k', tf.reshape(y, [-1, K]), tf.reshape(upstream, [-1]))
        grad_ranges = tf.where(small, ax * (sills / tf.square(ranges)), 0.) # [..., K}
        grad_ranges = tf.einsum('ak,a->k', tf.reshape(grad_ranges, [-1, K]), tf.reshape(upstream, [-1]))
        return grad_x, grad_sills, grad_ranges
    return tf.reduce_sum(y, -1), grad

class RampStack(Kernel):
    def __init__(self, range, sill, scale=None, metric=None):
        fa = dict(sill=sill, range=range, scale=scale)
        autoinputs = scale_to_metric(scale, metric)
        super().__init__(fa, dict(d2=autoinputs))

    def vars(self):
        return ppp_list(self.fa['sill']) | ppp_list(self.fa['range'])

    def call(self, e):
        if isinstance(e['sill'], (tuple, list)):
            e['sill'] = tf.stack(e['sill'])
        if isinstance(e['range'], (tuple, list)):
            e['range'] = tf.stack(e['range'])

        return rampstack(tf.sqrt(e['d2']), e['sill'], e['range'])

@tf.recompute_grad
def smooth_convex(x, sills, ranges):
    """
    `x` has arbitrary shape [...], but must be non-negative.
    `sills` and `ranges` both have shape [K].
    """
    r2 = ranges
    r1 = tf.pad(ranges[:-1], [[1, 0]])
    ex = ed(x)
    ax = tf.abs(ex)
    rx = ax / r2 - 1.

    c1 = 2. / (r1 + r2)
    c2 = 1. / (1. - tf.square(r1/r2))

    # i1 = tf.cast(ax <= r1, tf.float32) # Indicates x <= r1.
    # i2 = tf.cast(ax <= r2, tf.float32) * (1. - i1) # Indicates r1 < x <= r2.
    # v = i1 * (1. - c1 * ax) + i2 * c2 * tf.square(rx)

    v = tf.where(ax <= r1, 1. - c1 * ax, c2 * tf.square(rx))
    v = tf.where(ax <= r2, v, 0.)

    y = tf.einsum('...k,k->...', v, sills)
    return y

@tf.custom_gradient
def smooth_convex_grad(x, sills, ranges):
    """
    `x` has arbitrary shape [...], but must be non-negative.
    `sills` and `ranges` both have shape [K].
    """
    r2 = ranges
    r1 = tf.pad(ranges[:-1], [[1, 0]])
    ex = ed(x)
    ax = tf.abs(ex)
    rx = ax / r2 - 1.

    c1 = 2. / (r1 + r2)
    c2 = 1. / (1. - tf.square(r1/r2))

    v = tf.where(ax <= r1, 1. - c1 * ax, c2 * tf.square(rx))
    v = tf.where(ax <= r2, v, 0.)

    y = tf.einsum('...k,k->...', v, sills)

    def grad(upstream):
        r2 = ranges
        r1 = tf.pad(ranges[:-1], [[1, 0]])
        ex = ed(x)
        ax = tf.abs(ex)
        rx = ax / r2 - 1.
        i1 = tf.cast(ax <= r1, tf.float32) # Indicates x <= r1.
        i2 = tf.cast(ax <= r2, tf.float32) * (1. - i1) # Indicates r1 < x <= r2.

        c1 = 2. / (r1 + r2)
        c2 = 1. / (1. - tf.square(r1 / r2))
        c3 = 1. / (r2 - tf.square(r1) / r2)

        v = i1 * (1. - c1 * ax) + i2 * c2 * tf.square(rx)

        sx = tf.sign(ex)

        K = tf.shape(sills)[0]
        gx = sx * sills * (i1 * -c1 + i2 * rx * (2 * c3))
        grad_x = upstream * tf.reduce_sum(gx, -1) # [...]

        grad_sills = tf.einsum('ak,a->k', tf.reshape(v, [-1, K]), tf.reshape(upstream, [-1]))

        u = 2 / tf.square(r1 + r2) * ax * i1
        yr1 = u + i2 * tf.square(rx * c3) * 2 * r1
        yr2 = u - 2 * i2 * (rx * c3 + tf.square(rx * c2) / r2)
        yr1 = sills * tf.reshape(yr1, [-1, K])
        yr2 = sills * tf.reshape(yr2, [-1, K])
        yr = tf.pad(yr1[:, 1:], [[0, 0], [0, 1]]) + yr2
        grad_ranges = tf.einsum('ak,a->k', yr, tf.reshape(upstream, [-1]))

        return grad_x, grad_sills, grad_ranges

    return y, grad

class SmoothConvex(Kernel):
    def __init__(self, range, sill, scale=None, metric=None):
        fa = dict(sill=sill, range=range, scale=scale)
        autoinputs = scale_to_metric(scale, metric)
        super().__init__(fa, dict(d2=autoinputs))

    def vars(self):
        return ppp_list(self.fa['sill']) | ppp_list(self.fa['range'])

    def call(self, e):
        if isinstance(e['sill'], (tuple, list)):
            e['sill'] = tf.stack(e['sill'])
        if isinstance(e['range'], (tuple, list)):
            e['range'] = tf.stack(e['range'])

        return smooth_convex(tf.sqrt(e['d2']), e['sill'], e['range'])

@tf.recompute_grad
def quadstack(x, sills, ranges):
    """
    `x` has arbitrary shape [...], but must be non-negative.
    `sills` and `ranges` both have shape [K].
    """
    ex = ed(x)
    ax = tf.maximum(0., 1. - tf.abs(ex) / ranges) # [..., 1]
    y = sills * tf.square(ax) # [..., K]
    return tf.reduce_sum(y, -1)

class QuadStack(Kernel):
    def __init__(self, range, sill, scale=None, metric=None):
        fa = dict(sill=sill, range=range, scale=scale)
        autoinputs = scale_to_metric(scale, metric)
        super().__init__(fa, dict(d2=autoinputs))

    def vars(self):
        return ppp_list(self.fa['sill']) | ppp_list(self.fa['range'])

    def call(self, e):
        if isinstance(e['sill'], (tuple, list)):
            e['sill'] = tf.stack(e['sill'])
        if isinstance(e['range'], (tuple, list)):
            e['range'] = tf.stack(e['range'])

        return quadstack(tf.sqrt(e['d2']), e['sill'], e['range'])

class Wiener(Kernel):
    def __init__(self, axis, start):

        self.axis = axis
        self.start = start

        # Include the element of scale corresponding to the axis of
        # integration as an explicit formal argument.
        fa = dict()

        super().__init__(fa, dict(locs1='locs1', locs2='locs2'))

    def vars(self):
        return {}

    def call(self, e):
        x1 = e['locs1'][..., self.axis]
        x2 = e['locs2'][..., self.axis]
        k = tf.maximum(0., tf.minimum(ed(x1, 1), ed(x2, 0)) - self.start)
        return k

class IntSquaredExponential(Kernel):
    def __init__(self, axis, start, range):

        self.axis = axis
        self.start = start

        # Include the element of scale corresponding to the axis of
        # integration as an explicit formal argument.
        fa = dict(range=range)

        super().__init__(fa, dict(locs1='locs1', locs2='locs2'))

    def vars(self):
        return ppp(self.fa['range'])

    def call(self, e):
        x1 = tf.pad(e['locs1'][..., self.axis] - self.start, [[1, 0]])
        x2 = tf.pad(e['locs2'][..., self.axis] - self.start, [[1, 0]])

        r = e['range']
        sdiff = (ed(x1, 1) - ed(x2, 0)) / (r * np.sqrt(2.))
        k = -tf.square(r) * (np.sqrt(np.pi) * sdiff * tf.math.erf(sdiff) + tf.exp(-tf.square(sdiff)))
        k -= k[0:1, :]
        k -= k[:, 0:1]
        k = k[1:, 1:]
        k = tf.maximum(0., k)

        return k

class IntExponential(Kernel):
    def __init__(self, axis, start, range):

        self.axis = axis
        self.start = start

        # Include the element of scale corresponding to the axis of
        # integration as an explicit formal argument.
        fa = dict(range=range)

        super().__init__(fa, dict(locs1='locs1', locs2='locs2'))

    def vars(self):
        return ppp(self.fa['range'])

    def call(self, e):
        x1 = tf.pad(e['locs1'][..., self.axis] - self.start, [[1, 0]])
        x2 = tf.pad(e['locs2'][..., self.axis] - self.start, [[1, 0]])

        r = e['range']
        sdiff = tf.abs(ed(x1, 1) - ed(x2, 0)) / r
        k = -tf.square(r) * (sdiff + tf.exp(-sdiff))
        k -= k[0:1, :]
        k -= k[:, 0:1]
        k = k[1:, 1:]
        k = tf.maximum(0., k)

        return k

class Noise(Kernel):
    def __init__(self, nugget):
        fa = dict(nugget=nugget)
        super().__init__(fa, dict(locs1='locs1', locs2='locs2'))

    def vars(self):
        return ppp(self.fa['nugget'])

    def call(self, e):

        indices1 = tf.range(tf.shape(e['locs1'])[0])
        indices2 = tf.range(tf.shape(e['locs2'])[0]) + e['offset']
        C = tf.where(tf.equal(tf.expand_dims(indices1, -1), indices2), e['nugget'], 0.)
        return C

class Delta(Kernel):
    def __init__(self, sill, axes=None):
        fa = dict(sill=sill)
        self.axes = axes
        super().__init__(fa, dict(pa_d2='per_axis_dist2'))

    def vars(self):
        return ppp(self.fa['sill'])

    def call(self, e):

        if self.axes is not None:
            n = tf.shape(e['pa_d2'])[-1]
            mask = tf.math.bincount(self.axes, minlength=n, maxlength=n, dtype=tf.float32)
            d2 = tf.einsum('abc,c->ab', e['pa_d2'], mask)
        else:
            d2 = tf.reduce_sum(e['pa_d2'], axis=-1)

        return e['sill'] * tf.cast(tf.equal(d2, 0.), tf.float32)

class Mix(Kernel):
    def __init__(self, inputs, weights=None):
        self.inputs = inputs
        fa = {}
        ai = dict(cats1='cats1', cats2='cats2')

        # Special case if weights is not given.
        if weights is not None:
            fa['weights'] = weights
            ai['inputs'] = inputs

        super().__init__(fa, ai)

    def gather_vars(self, cache=None):
        """Make a special version of gather_vars because
           we want to gather variables from `inputs`
           even when it's not in autoinputs"""
        vv = super().gather_vars(cache)
        for iput in self.inputs:
            cache[id(self)] |= iput.gather_vars(cache)
        return cache[id(self)]

    def vars(self):
        if 'weights' in self.fa:
            return {k: p for row in self.fa['weights']
                      for k, p in get_trend_coefs(row).items()}
        else:
            return {}

    def call(self, e):
        if 'weights' in e:
            weights = []
            for row in e['weights']:
                if isinstance(row, (tuple, list)):
                    row = tf.stack(row)
                    weights.append(row)
            weights = tf.stack(weights)
            C = tf.stack(e['inputs'], axis=-1) # [locs, locs, numinputs].
            Aaug1 = tf.gather(weights, e['cats1']) # [locs, numinputs].
            Aaug2 = tf.gather(weights, e['cats2']) # [locs, numinputs].
            outer = tf.einsum('ac,bc->abc', Aaug1, Aaug2) # [locs, locs, numinputs].
            C = tf.einsum('abc,abc->ab', C, outer) # [locs, locs].
            return C
        else:
            # When weights is not given, exploit the fact that we don't have
            # to compute every element in component covariance matrices.
            N = len(self.inputs)
            catcounts1 = tf.math.bincount(e['cats1'], minlength=N, maxlength=N)
            catcounts2 = tf.math.bincount(e['cats2'], minlength=N, maxlength=N)
            catindices1 = tf.math.cumsum(catcounts1, exclusive=True)
            catindices2 = tf.math.cumsum(catcounts2, exclusive=True)
            catdiffs = tf.unstack(catindices2 - catindices1, num=N)
            locsegs1 = tf.split(e['locs1'], catcounts1, num=N)
            locsegs2 = tf.split(e['locs2'], catcounts2, num=N)

            # TODO: Check that the below is still correct.
            CC = [] # Observation noise submatrices.
            for sublocs1, sublocs2, catdiff, iput in zip(locsegs1, locsegs2, catdiffs, self.inputs):
                cache = dict(
                    offset = e['offset'] + catdiff,
                    locs1 = sublocs1,
                    locs2 = sublocs2)
                cache['per_axis_dist2'] = PerAxisDist2().run(cache)
                cache['euclidean'] = Euclidean().run(cache)
                Csub = iput.run(cache)
                CC.append(Csub)

            return block_diag(CC)

class Stack(Kernel):
    def __init__(self, parts: List[Kernel]):
        self.parts = parts
        super().__init__({}, dict(locs1='locs1', locs2='locs2', parts=parts))

    def vars(self):
        return {k: p for part in self.parts for k, p in part.vars().items()}

    def __add__(self, other):
        if isinstance(other, Kernel):
            return Stack(self.parts + [other])
    
    def call(self, e):
        return tf.reduce_sum(e['parts'], axis=0)

    def report(self):
        return ' '.join(part.report(p) for part in self.parts)

class Product(Kernel):
    def __init__(self, parts: List[Kernel]):
        self.parts = parts
        super().__init__({}, dict(locs1='locs1', locs2='locs2', parts=parts))

    def vars(self):
        return {k: p for part in self.parts for k, p in part.vars().items()}

    def __mul__(self, other):
        if isinstance(other, Kernel):
            return Product(self.parts + [other])
    
    def call(self, e):
        return tf.reduce_prod(e['parts'], axis=0)

    def report(self):
        return ' '.join(part.report(p) for part in self.parts)

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
    xrank = len(x.shape)
    rank = len(shape)
    excess_rank = tf.maximum(0, xrank - rank)
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
        vv = {k: p for c in self.coefs for k, p in upp(c)}
        vv |= self.noise.vars()
        return vv

    def __call__(self, e):
        """
        Dummy.
        """
        return 0.
