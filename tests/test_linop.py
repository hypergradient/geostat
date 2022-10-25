import geostat.linop as lo
import numpy as np
import tensorflow as tf

def test_diag_block_diag():
    A = lo.LODiagBlockDiag([2, 3, 4], [1., 1., 1.], [0.2, 0.3, 0.4])
    check(A)

def test_low_rank():
    num = 4
    x = np.arange(num).astype(np.float32)
    d2 = np.square(x[:, np.newaxis] - x[np.newaxis, :]) / 5.
    C = np.exp(-d2)
    R = lo.LOFullMatrix(C[:3, :3])
    Q = C[:3, :]
    N = lo.LODiag(0.1 * np.ones([num], dtype=np.float32))
    A = lo.LOLowRank(Q, R, N)
    check(A)

def allclose(a, b):
    return np.allclose(a, b, atol=1e-4)

def check(a):
    a2 = lo.LOFullMatrix(a.to_dense())

    # Shape
    assert(allclose(a.shape.as_list(), a2.shape.as_list()))

    # Mat-vec multiply.
    n = a.domain_dimension
    b = np.random.uniform(-1., 1., [n, 1]).astype(np.float32)
    assert(allclose(a @ b, a2 @ b))

    # Mat-mat multiply.
    n = a.domain_dimension
    b = np.random.uniform(-1., 1., [n, 3]).astype(np.float32)
    assert(allclose(a @ b, a2 @ b))

    # Batch mat-mat multiply.
    n = a.domain_dimension
    b = np.random.uniform(-1., 1., [3, n, 3]).astype(np.float32)
    b[0, :, :] = 0.
    assert(allclose(a @ b, a2 @ b))

    # Mat-vec solve.
    n = a.domain_dimension
    b = np.random.uniform(-1., 1., [n, 1]).astype(np.float32)
    assert(allclose(a.solve(b), a2.solve(b)))

    # Mat-mat solve.
    n = a.domain_dimension
    b = np.random.uniform(-1., 1., [n, 3]).astype(np.float32)
    assert(allclose(a.solve(b), a2.solve(b)))

    # Batch mat-mat solve.
    n = a.domain_dimension
    b = np.random.uniform(-1., 1., [3, n, 3]).astype(np.float32)
    b[0, :, :] = 0.
    assert(allclose(a.solve(b), a2.solve(b)))

    # Inverse
    assert(allclose(a.inverse().to_dense(), a2.inverse().to_dense()))

    # Log abs determinant
    assert(allclose(a.log_abs_determinant(), a2.log_abs_determinant()))
