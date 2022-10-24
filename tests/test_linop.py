import geostat.linop as lo
import numpy as np
import tensorflow as tf

def test_diag_block_diag():

    a = lo.LODiagBlockDiag([2, 3, 4], [1., 1., 1.], [0.2, 0.3, 0.4])
    check(a)

def dense(a):
    return lo.LOFullMatrix(a.to_dense())

def check(a):
    # Mat-vec multiply.
    n = a.domain_dimension
    b = np.random.uniform(-1., 1., [n, 1]).astype(np.float32)
    assert(np.allclose(a @ b, dense(a) @ b))

    # Mat-mat multiply.
    n = a.domain_dimension
    b = np.random.uniform(-1., 1., [n, 3]).astype(np.float32)
    assert(np.allclose(a @ b, dense(a) @ b))

    # Batch mat-mat multiply.
    n = a.domain_dimension
    b = np.random.uniform(-1., 1., [3, n, 3]).astype(np.float32)
    b[0, :, :] = 0.
    assert(np.allclose(a @ b, dense(a) @ b))

    # Mat-vec solve.
    n = a.domain_dimension
    b = np.random.uniform(-1., 1., [n, 1]).astype(np.float32)
    assert(np.allclose(a.solve(b), dense(a).solve(b)))

    # Mat-mat solve.
    n = a.domain_dimension
    b = np.random.uniform(-1., 1., [n, 3]).astype(np.float32)
    assert(np.allclose(a.solve(b), dense(a).solve(b)))

    # Batch mat-mat solve.
    n = a.domain_dimension
    b = np.random.uniform(-1., 1., [3, n, 3]).astype(np.float32)
    b[0, :, :] = 0.
    assert(np.allclose(a.solve(b), dense(a).solve(b)))

    # Inverse
    assert(np.allclose(a.inverse().to_dense(), dense(a).inverse().to_dense()))

    # Log abs determinant
    assert(np.allclose(a.log_abs_determinant(), dense(a).log_abs_determinant()))

    # Shape abs determinant
    assert(np.allclose(a.shape.as_list(), dense(a).shape.as_list()))
