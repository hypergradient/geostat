from dataclasses import dataclass
from .params import get_parameter_values, ppp, upp, bpp

class Metric:
    def __init__(self):
        self.out = {}
    def __call__(self, a, b):
        pass
    def run(self, d2):
        """
        d2 holds squared distances. It has shape [N, N, K] where:
          - N is number of observations
          - K is number of input dimentions
        """
        pass

def get_scale_vars(scale):
    if scale is not None:
        return [p for s in scale for p in ppp(s)]
    else:
        return []

class ScaledEuclidean(Metric):
    def __init__(self, scale):
        fa = dict(scale=scale)
        super().__init__(fa)

    def vars(self):
        return get_scale_vars(self.fa['scale'])

    def run(self, d2):
        if v['scale'] is not None:
            scale = v['scale']
        else:
            scale = tf.ones_like(d2[0, 0, :])

        self.out['d2'] = einsum('abc,c->ab', d2, tf.square(scale / v['range']))

class Euclidean(Metric):
    def __init__(self, scale):
        fa = dict(scale=scale)
        super().__init__(fa)

    def vars(self):
        return []

    def run(self, d2):
        self.out['d2'] = einsum('abc,c->ab', d2, tf.square(scale / v['range']))
