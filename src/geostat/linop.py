import numpy as np
import tensorflow as tf

from dataclasses import dataclass
from tensorflow.linalg import LinearOperator as LO
from tensorflow.linalg import LinearOperatorBlockDiag as LOBlockDiag
from tensorflow.linalg import LinearOperatorComposition as LOComposition
from tensorflow.linalg import LinearOperatorDiag as LODiag
from tensorflow.linalg import LinearOperatorFullMatrix as LOFullMatrix
from tensorflow.linalg import LinearOperatorPermutation as LOPermutation

logdet = tf.linalg.logdet

def e(x, axis=-1):
    return tf.expand_dims(x, axis)

def t(x, perm=None, conjugate=False):
    return tf.transpose(x, perm, conjugate)

def reshape(x, adjoint_arg):
    s = tf.shape(x)
    x = tf.reshape(x, [-1, s[-2], s[-1]])
    batch_dims = s[:-2]
    if adjoint_arg:
        x = tf.transpose(x, [2, 0, 1], conjugate=True)
    else:
        x = tf.transpose(x, [1, 0, 2])
    inner_dim = tf.shape(x)[-1]
    x = tf.reshape(x, [tf.shape(x)[0], -1])
    return x, batch_dims, inner_dim

def revert(x, batch_dims, inner_dim):
    x = tf.reshape(x, [tf.shape(x)[0], -1, inner_dim])
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, tf.concat([batch_dims, [-1, inner_dim]], axis=0))
    return x

class LODiagBlockDiag(LO):
    def __init__(self, blocksizes: np.ndarray, diag, const, dtype=tf.float32):
        self.blocksizes = np.array(blocksizes)
        self.size = np.sum(blocksizes)
        self.segment_ids = np.repeat(range(len(blocksizes)), blocksizes)
        self.diag = tf.cast(diag, dtype)
        self.const = tf.cast(const, dtype)
        super().__init__(dtype=dtype, is_square=True, is_self_adjoint=True)

    def to_block_diag(self):
        blocks = []
        for s, d, c in zip(self.blocksizes, self.diag, self.const):
            blocks.append(LOFullMatrix(tf.eye(s) * d + c))
        return LOBlockDiag(blocks)

    def _to_dense(self):
        return self.to_block_diag().to_dense()

    def _shape(self):
        return tf.TensorShape([self.size, self.size])

    def _matmul(self, x, adjoint=False, adjoint_arg=False):
        x, bdims, idim = reshape(x, adjoint_arg)
        a = e(self.const) * tf.math.segment_sum(x, self.segment_ids)
        b = e(self.diag)
        x = tf.gather(a, self.segment_ids) + x * tf.gather(b, self.segment_ids)
        return revert(x, bdims, idim)

    def _inverse(self, adjoint=False):
        n = tf.constant(self.blocksizes, dtype=self._dtype)
        d, c = self.diag, self.const
        ci = -c / (d * (d + n * c))
        di = 1 / d
        return LODiagBlockDiag(self.blocksizes, di, ci)

    def _solve(self, x, adjoint=False, adjoint_arg=False):
        return self._inverse().matmul(x, adjoint_arg=adjoint_arg)

    def _log_abs_determinant(self):
        n = tf.constant(self.blocksizes, dtype=self._dtype)
        d, c = self.diag, self.const
        return tf.reduce_sum(n * tf.math.log(d) + tf.math.log(1. + c * n / d), axis=0)

@dataclass
class LOLowRank(LO):
    """
    Implements Q^T R^-1 Q + N.
    R and N should be symmetric.
    """
    Q: tf.Tensor
    Ri: LO
    N: LO
    dtype = tf.float32

    def __post_init__(self):
        super().__init__(dtype=self.dtype, is_square=True, is_self_adjoint=True)

    def _to_dense(self):
        Q, Ri, N = self.Q, self.Ri, self.N
        return t(Q) @ Ri.solve(Q) + N.to_dense()

    def _shape(self):
        return self.N.shape

    def _matmul(self, x, adjoint=False, adjoint_arg=False):
        Q, Ri, N = self.Q, self.Ri, self.N
        return t(Q) @ Ri.solve(Q @ x) + N @ x

    def _solve(self, x, adjoint=False, adjoint_arg=False):
        x, bdims, idim = reshape(x, adjoint_arg)
        Q, Ri, N = self.Q, self.Ri, self.N
        A = Ri.to_dense() + Q @ N.solve(Q, adjoint_arg=True)
        B = tf.linalg.solve(A, Q @ N.solve(x, adjoint_arg=adjoint_arg))
        x = N.solve(x - t(Q) @ B)
        return revert(x, bdims, idim)

    def _log_abs_determinant(self):
        Q, Ri, N = self.Q, self.Ri, self.N
        A = Ri.to_dense() + Q @ N.solve(Q, adjoint_arg=True)
        return -Ri.log_abs_determinant() + N.log_abs_determinant() + tf.linalg.logdet(A)
