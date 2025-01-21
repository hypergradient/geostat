from types import SimpleNamespace
from dataclasses import dataclass
import tensorflow as tf
import numpy as np
from scipy.special import logit


__all__ = ['Parameter', 'Parameters']

def Parameters(**kw):
    return SimpleNamespace(**{k:Parameter(k, v) for k, v in kw.items()})

@dataclass
class Parameter:
    name: str
    value: float
    lo: float = np.nan
    hi: float = np.nan
    underlying: tf.Variable = None

    def update_bounds(self, lo: float, hi: float):
        if np.isnan(self.lo):
            self.lo = lo
        else:
            assert self.lo == lo, f'Conflicting bounds for parameter {self.name}'
        if np.isnan(self.hi):
            self.hi = hi
        else:
            assert self.hi == hi, f'Conflicting bounds for parameter {self.name}'

    def bounding(self):
        return \
            ('u' if self.lo == float('-inf') else 'b') + \
            ('u' if self.hi == float('inf') else 'b')

    def create_tf_variable(self):
        """Create TF variable for underlying parameter or update it"""
        # Create underlying parameter.
        b = self.bounding()
        if b == 'bb':
            init = logit((self.value - self.lo) / (self.hi - self.lo))
        elif b == 'bu':
            init = np.log(self.value - self.lo)
        elif b == 'ub':
            init = -np.log(self.hi - self.value)
        else:
            init = self.value

        if self.underlying is None:
            self.underlying = tf.Variable(init, name=self.name, dtype=tf.float32)
        else:
            self.underlying.assign(init)

    def surface(self):
        """ Create tensor for surface parameter"""
        # Create surface parameter.
        b = self.bounding()
        v = self.underlying
        if b == 'bb':
            v = tf.math.sigmoid(v) * (self.hi - self.lo) + self.lo
        elif b == 'bu':
            v = tf.exp(v) + self.lo
        elif b == 'ub':
            v = self.hi - tf.exp(-v)
        else:
            v = v + tf.constant(0.)
        return v

    def update_value(self):
        self.value = self.surface().numpy()

# TODO: cache surface somehow
# TODO: replace with map nested?
def get_parameter_values(blob: object):
    """
    For each Parameter encountered in the nested blob,
    replace it with its surface tensor.
    """
    if isinstance(blob, dict):
        return {k: get_parameter_values(a) for k, a in blob.items()}
    elif isinstance(blob, (list, tuple)):
        return [get_parameter_values(a) for a in blob]
    elif isinstance(blob, Parameter):
        return blob.surface()
    elif isinstance(blob, str):
        raise ValueError(f'Bad parameter {blob} is a string')
    else:
        return blob

def ppp(param):
    """Positive paper parameter (maybe)."""
    if isinstance(param, Parameter):
        param.update_bounds(0., np.inf)
        return {param.name: param}
    else:
        return {}

def upp(param):
    """Unbounded paper parameter (maybe)."""
    if isinstance(param, Parameter):
        param.update_bounds(-np.inf, np.inf)
        return {param.name: param}
    else:
        return {}

def bpp(param, lo, hi):
    """Bounded paper parameter (maybe)."""
    if isinstance(param, Parameter):
        param.update_bounds(lo, hi)
        return {param.name: param}
    else:
        return {}

def ppp_list(beta):
    if isinstance(beta, (list, tuple)):
        return {k: p for s in beta for k, p in ppp(s).items()}
    elif isinstance(beta, Parameter):
        return ppp(beta)
    else:
        return {}
