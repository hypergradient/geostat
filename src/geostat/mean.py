from dataclasses import dataclass
from typing import Callable, List, Union
import numpy as np

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow as tf

from .op import Op
from .param import Parameter, ppp, upp, bpp

__all__ = ['Mean', 'Trend']

def get_trend_coefs(beta):
    if isinstance(beta, (list, tuple)):
        return {k: p for s in beta for k, p in upp(s).items()}
    elif isinstance(beta, Parameter):
        return upp(beta)
    else:
        return {}

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

    def call(self, e):
        pass

    def __call__(self, e):
        """
        Returns tuple `(mean, covariance)` for locations.
        Return values have correct shapes.
        """
        M = self.call(e)
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

    def call(self, e):
        x = tf.cast(self.featurizer(e['locs1']), tf.float32)
        if isinstance(e['beta'], (tuple, list)):
            e['beta'] = tf.stack(e['beta'])
        return tf.einsum('ab,b->a', x, e['beta']) # [locs1]

class ZeroTrend(Op):
    def __init__(self):
        super().__init__({}, dict(locs1='locs1'))

    def vars(self):
        return {}

    def __call__(self, e):
        return tf.zeros_like(e['locs1'][:, 0])

class Mix(Mean):
    def __init__(self, inputs, weights=None):
        fa = {}
        if weights is not None: fa['weights'] = weights
        super().__init__(fa, dict(inputs=inputs, cats1='cats1'))

    def vars(self):
        if 'weights' in self.fa:
            return {k: p for row in self.fa['weights']
                      for k, p in get_trend_coefs(row).items()}
        else:
            return {}

    def call(self, e):
        M = tf.stack(e['inputs'], axis=-1) # [locs, numinputs].

        # Transform M with weights, if given.
        if 'weights' in e:
            weights = []
            for row in e['weights']:
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
        return {k: p for part in self.parts for k, p in part.vars().items()}

    def __add__(self, other):
        if isinstance(other, Mean):
            return Stack(self.parts + [other])

    def call(self, e):
        return tf.reduce_sum(e['parts'], axis=0)

    def report(self, p):
        return ' '.join(part.report(p) for part in self.parts)
