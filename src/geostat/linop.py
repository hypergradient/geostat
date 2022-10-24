import numpy as np
import tensorflow as tf

from tensorflow.linalg import LinearOperator as LO
from tensorflow.linalg import LinearOperatorBlockDiag as LOBlockDiag
from tensorflow.linalg import LinearOperatorComposition as LOComposition
from tensorflow.linalg import LinearOperatorDiag as LODiag
from tensorflow.linalg import LinearOperatorFullMatrix as LOFullMatrix
from tensorflow.linalg import LinearOperatorPermutation as LOPermutation

def e(x, axis=-1):
    return tf.expand_dims(x, axis)

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
        s = tf.shape(x)
        x = tf.reshape(x, [-1, s[-2], s[-1]])
        if adjoint_arg:
            x = tf.transpose(x, [2, 0, 1], conjugate=True)
        else:
            x = tf.transpose(x, [1, 0, 2])
        a = e(e(self.const)) * tf.math.segment_sum(x, self.segment_ids)
        b = e(e(self.diag))
        x = tf.gather(a, self.segment_ids) + x * tf.gather(b, self.segment_ids)
        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, s)
        return x
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

class LOLowRank(LO):
    def __init__(self, Q: tf.Tensor, R: LO, N: LO, dtype=tf.float32):
        """
        Implements Q^T R^-1 Q + N.
        R and N should be symmetric.
        """
    def _to_dense(self):
        return tf.einsum('ab,ac->bc', Q, R.solve(Q)) + N.to_dense()
    def _shape(self):
        return N.shape()
    def _matmul(self, x, adjoint=False, adjoint_arg=False):
        return tf.einsum('ab,ac->bc', Q, R.solve(Q @ x)) + N @ x
    def _solve(self, x, adjoint=False, adjoint_arg=False):
        A = R.to_dense() + Q @ N.solve(Q, adjoint_arg=True)
        B = tf.linalg.solve(A, Q @ N.solve(x, adjoint_arg=adjoint_arg))
        B
