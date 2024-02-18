import tensorflow as tf

from tensorflow.linalg import LinearOperator as LO
from tensorflow.linalg import LinearOperatorBlockDiag as LOBlockDiag
from tensorflow.linalg import LinearOperatorFullMatrix as LOFull


def cov(X):
    return LOFull(X, is_self_adjoint=True, is_positive_definite=True, is_square=True)

def lowrankcov(A, B):
    """
    A.shape is [k, n].
    B.shape is [n, n] and easily invertible.
    




    return Lo
