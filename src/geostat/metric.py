from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
import jax
import jax.numpy as jnp

from .op import Op
from .param import ppp, upp, bpp

def ed(x, a=-1):
    return jnp.expand_dims(x, a)

class PerAxisDist2(Op):
    def __init__(self):
        super().__init__({}, dict(locs1='locs1', locs2='locs2'))

    def __call__(self, e):
        x1 = e['locs1']
        x2 = e['locs2']
        return jnp.square(ed(x1, 1) - ed(x2, 0))

class Metric(Op):
    pass

def get_scale_vars(scale):
    if scale is not None:
        return {k: p for s in scale for k, p in ppp(s).items()}
    else:
        return {}

class Euclidean(Metric):
    def __init__(self, scale=None):
        fa = dict(scale=scale)
        super().__init__(fa, dict(pa_d2='per_axis_dist2'))

    def vars(self):
        return get_scale_vars(self.fa['scale'])

    def __call__(self, e):
        if e['scale'] is not None:
            return jnp.einsum('abc,c->ab', e['pa_d2'], jnp.square(jnp.array(e['scale'])))
        else:
            return jnp.sum(e['pa_d2'], axis=-1)

class Poincare(Metric):
    def __init__(self, xform: Callable, zoff='zoff', scale=None):
        fa = dict(zoff=zoff, scale=scale)
        self.xform = xform
        super().__init__(fa, dict(locs1='locs1', locs2='locs2'))

    def vars(self):
        return ppp(self.fa['zoff']) | get_scale_vars(self.fa['scale'])

    def __call__(self, e):
        xlocs1 = jnp.stack(self.xform(*jnp.unstack(e['locs1'], axis=1)), axis=1)
        xlocs2 = jnp.stack(self.xform(*jnp.unstack(e['locs2'], axis=1)), axis=1)
        zoff = e['zoff']
        print('---- zoff')
        jax.debug.print('{zoff}', zoff=zoff)

        # Maybe scale locations and zoff.
        if e['scale'] is not None:
            scale = jnp.array(e['scale'])
            print('---- scale')
            jax.debug.print('{scale}', scale=scale)
            xlocs1 *= scale
            xlocs2 *= scale
            zoff *= scale[0]
        
        z1 = xlocs1[:, 0] + zoff
        z2 = xlocs2[:, 0] + zoff
        zz = ed(z1, -1) * z2
        print('---- zz')
        jax.debug.print('z1 {a} {b}', a=z1.min(), b=z1.max())
        jax.debug.print('z2 {a} {b}', a=z2.min(), b=z2.max())
        jax.debug.print('zz {a} {b}', a=zz.min(), b=zz.max())

        d2 = jnp.sum(jnp.square(ed(xlocs1, 1) - ed(xlocs2, 0)), axis=-1)
        d2 = jnp.arcsinh(0.5 * jnp.sqrt(d2 / zz))
        d2 = jnp.square(2.0 * zoff * d2)

        jax.debug.print('d2 {a} {b}', a=d2.min(), b=d2.max())

        return d2

def is_distance_matrix(m):
    assert len(m.shape) == 2
    assert m.shape[0] == m.shape[1]

    a = np.ones_like(m) * float('inf')
    for i in range(m.shape[0]):
        a = np.minimum(a, m[i:i+1, :] + m[:, i:i+1])

    # print(np.nonzero(~np.equal(a, m)))
    return np.allclose(a, m)
