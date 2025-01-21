import numpy as np
import tensorflow as tf
from geostat.kernel import quadstack
from geostat.metric import ed

def vanilla_quadstack(x, sills, ranges):
    """
    `x` has arbitrary shape [...], but must be non-negative.
    `sills` and `ranges` both have shape [K].
    """
    r2 = ranges
    r1 = tf.pad(ranges[:-1], [[1, 0]])
    ex = ed(x)
    ax = tf.abs(ex)
    i1 = tf.cast(ax <= r1, tf.float32) # Indicates x <= r1.
    i2 = tf.cast(ax <= r2, tf.float32) * (1. - i1) # Indicates r1 < x <= r2.

    c1 = 2. / (r1 + r2)
    c2 = 1. / (1. - tf.square(r1/r2))

    y = sills * (i1 * (1. - c1 * ax) + i2 * c2 * tf.square(ax / r2 - 1))
    return tf.reduce_sum(y, -1)

def test_quadstack():

    sills = tf.Variable([0.1, 0.2, 0.5, 0.7], dtype=tf.float32)
    ranges = tf.Variable([1, 2, 5, float('inf')], dtype=tf.float32)
    x = tf.Variable([0.5, 1.5, 2.5, 5.5], dtype=tf.float32)

    with tf.GradientTape(persistent=True) as g:
        g.watch(sills)
        g.watch(ranges)
        g.watch(x)
        y1 = quadstack(x, sills, ranges)
        y2 = vanilla_quadstack(x, sills, ranges)

    print('---')
    print(y1)
    print(y2)
    print('---')
    print(g.jacobian(y1, sills))
    print(g.jacobian(y2, sills))
    print('---')
    print(g.jacobian(y1, ranges))
    print(g.jacobian(y2, ranges))
    print('---')
    print(g.jacobian(y1, x))
    print(g.jacobian(y2, x))
