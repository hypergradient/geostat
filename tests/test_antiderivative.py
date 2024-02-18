import numpy as np
import tensorflow as tf
from geostat import GP, Model, Mesh, NormalizingFeaturizer, Parameters
import geostat.kernel as krn


def test_int_sq_exp():

    np.random.seed(2)
    tf.random.set_seed(2)

    numsteps = 160
    xdim = 4.
    ydim = 0.5

    mesh = Mesh.from_bounds([0., 0., xdim, ydim], nx=numsteps)
    def trend_terms(x, y): return x, y, x*x, x*y, y*y
    featurizer = NormalizingFeaturizer(trend_terms, mesh.locations())

    mesh_vals = Model(
        GP(0, krn.SquaredExponential(range=0.1, sill=1., scale=[1., 1.]) + krn.Noise(nugget=1e-4))
    ).generate(mesh.locations()).vals

    vmin, vmax = mesh_vals.min(), mesh_vals.max()
    meshx, meshy, mesh_vals_2d = mesh.slice(mesh_vals) # Each return value is a 2d array.

    int_vals = np.cumsum(mesh_vals_2d, axis=-1) / numsteps * xdim # Numerical integration on x-axis.

    sample_indices = np.random.choice(len(int_vals.ravel()), [500], replace=False)
    locs = np.stack([meshx.ravel()[sample_indices], meshy.ravel()[sample_indices]], axis=-1)
    vals = int_vals.ravel()[sample_indices]

    p = Parameters(x_range=1., y_range=1., sill=2.)

    kernel = krn.SquaredExponential(scale=[0., 1.], range=p.y_range, sill=p.sill) * \
        krn.IntSquaredExponential(axis=0, start=0., range=p.x_range) + \
        krn.Noise(nugget=1e-4)

    model = Model(
        GP(0, kernel),
        verbose=True).fit(locs, vals, iters=2000, step_size=1e-1)

    assert np.allclose(
        [vars(p)[name].value for name in ['x_range', 'y_range', 'sill']],
        [0.1, 0.1, 1.],
        rtol=0.5)

def test_int_exp():

    np.random.seed(2)
    tf.random.set_seed(2)

    numsteps = 160
    xdim = 4.
    ydim = 0.5

    mesh = Mesh.from_bounds([0., 0., xdim, ydim], nx=numsteps)
    def trend_terms(x, y): return x, y, x*x, x*y, y*y
    featurizer = NormalizingFeaturizer(trend_terms, mesh.locations())

    kernel1 = krn.SquaredExponential(range=0.1, sill=1., scale=[0., 1.]) * \
        krn.GammaExponential(scale=[1., 0.], range=0.1, sill=1., gamma=1.) + krn.Noise(nugget=1e-4)

    mesh_vals = Model(GP(0, kernel1)).generate(mesh.locations()).vals

    vmin, vmax = mesh_vals.min(), mesh_vals.max()
    meshx, meshy, mesh_vals_2d = mesh.slice(mesh_vals) # Each return value is a 2d array.

    int_vals = np.cumsum(mesh_vals_2d, axis=-1) / numsteps * xdim # Numerical integration on x-axis.

    sample_indices = np.random.choice(len(int_vals.ravel()), [500], replace=False)
    locs = np.stack([meshx.ravel()[sample_indices], meshy.ravel()[sample_indices]], axis=-1)
    vals = int_vals.ravel()[sample_indices]

    p = Parameters(x_range=1., y_range=1., sill=2.)

    kernel2 = \
        krn.SquaredExponential(scale=[0., 1.], range=p.y_range, sill=p.sill) * \
        krn.IntExponential(axis=0, start=0., range=p.x_range) + \
        krn.Noise(nugget=1e-3)

    model = Model(GP(0, kernel2)).fit(locs, vals, iters=2000, step_size=1e-1)

    assert np.allclose(
        [vars(p)[name].value for name in ['x_range', 'y_range', 'sill']],
        [0.1, 0.1, 1.],
        rtol=0.5)

def test_wiener():

    np.random.seed(2)
    tf.random.set_seed(2)

    numsteps = 80
    xdim = 2.

    mesh = Mesh.from_bounds([0., 0., xdim, 1.], nx=numsteps)
    def trend_terms(x, y): return x, y, x*x, x*y, y*y
    featurizer = NormalizingFeaturizer(trend_terms, mesh.locations())

    kernel1 = \
        krn.SquaredExponential(scale=[0., 1.], sill=1., range=0.3) * \
        krn.SquaredExponential(scale=[1., 0.], sill=1., range=1e-4) + krn.Noise(nugget=1e-4)

    mesh_vals = Model(GP(0, kernel1)).generate(mesh.locations()).vals

    vmin, vmax = mesh_vals.min(), mesh_vals.max()
    meshx, meshy, mesh_vals_2d = mesh.slice(mesh_vals) # Each return value is a 2d array.

    # Numerical integration on x-axis. Scaling is tricky.
    int_vals = np.cumsum(mesh_vals_2d, axis=-1) / np.sqrt(numsteps / xdim)

    sample_indices = np.random.choice(len(int_vals.ravel()), [500], replace=False)
    locs = np.stack([meshx.ravel()[sample_indices], meshy.ravel()[sample_indices]], axis=-1)
    vals = int_vals.ravel()[sample_indices]

    p = Parameters(range=1., sill=2.)

    kernel2 = krn.SquaredExponential(range=p.range, sill=p.sill, scale=[0., 1.]) * \
              krn.Wiener(axis=0, start=0.) + krn.Noise(nugget=1e-4)

    model = Model(GP(0, kernel2)).fit(locs, vals, iters=2000, step_size=1e-1)

    assert np.allclose(
        [vars(p)[name].value for name in ['range', 'sill']],
        [0.3, 1.],
        rtol=0.5)
