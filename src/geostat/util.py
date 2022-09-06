import tensorflow as tf

def einsum_abc_c_ab(x, y):
    """
    Workaround for:

        tf.einsum('abc,c->ab', x, y)
    """

    A, B, C = tf.unstack(tf.shape(x))
    x = tf.reshape(x, [-1, C]) @ tf.reshape(y, [C, 1])
    x = tf.reshape(x, [A, B])
    return x
