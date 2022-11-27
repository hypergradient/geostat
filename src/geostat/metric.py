from dataclasses import dataclass
from typing import Dict

# Tensorflow is extraordinarily noisy. Catch warnings during import.
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import tensorflow as tf

from .op import Op
from .param import get_parameter_values, ppp, upp, bpp

def ed(x, a=-1):
    return tf.expand_dims(x, a)

class Metric(Op):
    pass

def get_scale_vars(scale):
    if scale is not None:
        return [p for s in scale for p in ppp(s)]
    else:
        return []

class Euclidean(Metric):
    def __init__(self, scale=None):
        fa = dict(scale=scale)
        super().__init__(fa, [])

    def vars(self):
        return get_scale_vars(self.fa['scale'])

    def __call__(self, p, **e):
        x = e['locs']
        v = get_parameter_values(self.fa, p)
        d2 = tf.square(ed(x, 0) - ed(x, 1))
        if v['scale'] is not None:
            return tf.einsum('abc,c->ab', d2, tf.square(v['scale']))
        else:
            return tf.reduce_sum(d2, axis=-1)
