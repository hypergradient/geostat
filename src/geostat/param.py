from dataclasses import dataclass, replace
from typing import Dict
import tensorflow as tf
import numpy as np
from scipy.special import logit

@dataclass
class Bound:
    lo: float
    hi: float

    def bounding(self):
        return \
            ('u' if self.lo == float('-inf') else 'b') + \
            ('u' if self.hi == float('inf') else 'b')

    def get_underlying_parameter(self, v, name=None):
        b = self.bounding()
        if b == 'bb':
            init = logit((v - self.lo) / (self.hi - self.lo))
        elif b == 'bu':
            init = np.log(v - self.lo)
        elif b == 'ub':
            init = -np.log(self.hi - v)
        else:
            init = v
        return tf.Variable(init, name=name, dtype=tf.float32)

    def get_surface_parameter(self, v):
        b = self.bounding()
        if b == 'bb':
            v = tf.math.sigmoid(v) * (self.hi - self.lo) + self.lo
        elif b == 'bu':
            v = tf.exp(v) + self.lo
        elif b == 'ub':
            v = self.hi - tf.exp(-v)
        else:
            v = v
        return v

@dataclass
class ParameterSpace:
    bounds: Dict[str, Bound]
    
    def get_underlying(self, parameters):
        up = {}
        for name, v in parameters.items():
            up[name] = self.bounds[name].get_underlying_parameter(v, name)
        return up

    def get_surface(self, parameters, numpy=False):
        sp = {}
        for name, v in parameters.items():
            v = self.bounds[name].get_surface_parameter(v)
            if numpy: v = v.numpy()
            sp[name] = v
        return sp
