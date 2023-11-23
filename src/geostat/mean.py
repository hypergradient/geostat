from dataclasses import dataclass
from typing import Callable, List, Union
import numpy as np

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow as tf

from .op import Op
from .param import get_parameter_values, ppp, upp, bpp

__all__ = ['Mean', 'Trend']

# TODO: in `call` methods, call get_parameter_values beforehand?

def get_trend_coefs(beta):
    if isinstance(beta, (list, tuple)):
        return [p for s in beta for p in upp(s)]
    elif isinstance(beta, str):
        return upp(beta)
    else:
        return []

class Mean(Op):
    def __init__(self, fa, autoinputs):
        if 'locs1' not in autoinputs: autoinputs['locs1'] = 'locs1'
        super().__init__(fa, autoinputs)

    def __add__(self, other):
        if isinstance(other, ZeroTrend):
            return self
        elif isinstance(self, ZeroTrend):
            return other
        else:
            return Stack([self]) + other

    def call(self, p, e):
        pass

    def __call__(self, p, e):
        """
        Returns tuple `(mean, covariance)` for locations.
        Return values have correct shapes.
        """
        M = self.call(p, e)
        if M is None: M = 0.
        n1 = tf.shape(e['locs1'])[0]
        M = tf.broadcast_to(M, [n1])
        return M

class Trend(Mean):
    def __init__(self, featurizer, beta='beta'):
        self.featurizer = featurizer
        super().__init__(dict(beta=beta), dict(locs1='locs1'))

    def vars(self):
        return get_trend_coefs(self.fa['beta'])

    def call(self, p, e):
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

class Mix(Mean):
    def __init__(self, inputs, weights=None):
        fa = {}
        if weights is not None: fa['weights'] = weights
        super().__init__(fa, dict(inputs=inputs, cats1='cats1'))

    def vars(self):
        if 'weights' in self.fa:
            return [p for row in self.fa['weights']
                      for p in get_trend_coefs(row)]
        else:
            return []

    def call(self, p, e):
        M = tf.stack(e['inputs'], axis=-1) # [locs, numinputs].

        # Transform M with weights, if given.
        v = get_parameter_values(self.fa, p)
        if 'weights' in v:
            weights = []
            for row in v['weights']:
                if isinstance(row, (tuple, list)):
                    row = tf.stack(row)
                    weights.append(row)
            weights = tf.stack(weights)
            M = tf.einsum('lh,sh->ls', M, weights)

        return tf.gather(M, e['cats1'], batch_dims=1) # [locs]

class Stack(Mean):
    def __init__(self, parts: List[Mean]):
        self.parts = parts
        super().__init__({}, dict(locs1='locs1', locs2='locs2', parts=parts))

    def vars(self):
        return [p for part in self.parts for p in part.vars()]

    def __add__(self, other):
        if isinstance(other, Mean):
            return Stack(self.parts + [other])

    def call(self, p, e):
        return tf.reduce_sum(e['parts'], axis=0)

    def report(self, p):
        return ' '.join(part.report(p) for part in self.parts)

    def reg(self, p):
        return tf.reduce_sum([part.reg(p) for part in self.parts], axis=0)
