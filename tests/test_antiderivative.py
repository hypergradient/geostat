import numpy as np
import tensorflow as tf
from geostat import gp, Model, Mesh, NormalizingFeaturizer


def test_int_sq_exp():

    np.random.seed(1)
    tf.random.set_seed(1)

    numsteps = 160
    xdim = 4.
    ydim = 0.5

    mesh = Mesh.from_bounds([0., 0., xdim, ydim], nx=numsteps)
    def trend_terms(x, y): return x, y, x*x, x*y, y*y
    featurizer = NormalizingFeaturizer(trend_terms, mesh.locations())

    mesh_vals = Model(
        gp.SquaredExponential(scale=[1., 1.]) + gp.Noise(nugget=1e-4),
        parameters = dict(range=0.2, sill=1.),
        verbose=True).generate(mesh.locations()).vals

    vmin, vmax = mesh_vals.min(), mesh_vals.max()
    meshx, meshy, mesh_vals_2d = mesh.slice(mesh_vals) # Each return value is a 2d array.

    int_vals = np.cumsum(mesh_vals_2d, axis=-1) / numsteps * xdim # Numerical integration on x-axis.

    sample_indices = np.random.choice(len(int_vals.ravel()), [500], replace=False)
    locs = np.stack([meshx.ravel()[sample_indices], meshy.ravel()[sample_indices]], axis=-1)
    vals = int_vals.ravel()[sample_indices]

    model = Model(
        gp.SquaredExponential(scale=[0., 1.], range='y_range') * \
        gp.IntSquaredExponential(axis=0, start=0., range='x_range') + \
        gp.Noise(nugget=1e-4),
        parameters = dict(x_range=1., y_range=1., sill=2.),
        verbose=True).fit(locs, vals, iters=1000, step_size=1e-1)

def test_int_exp():

    np.random.seed(1)
    tf.random.set_seed(1)

    numsteps = 160
    xdim = 4.
    ydim = 0.5

    mesh = Mesh.from_bounds([0., 0., xdim, ydim], nx=numsteps)
    def trend_terms(x, y): return x, y, x*x, x*y, y*y
    featurizer = NormalizingFeaturizer(trend_terms, mesh.locations())

    mesh_vals = Model(
        gp.SquaredExponential(scale=[0., 1.]) *
        gp.GammaExponential(scale=[1., 0.], sill=1., gamma=1.) + gp.Noise(nugget=1e-4),
        parameters = dict(range=0.2, sill=1.),
        verbose=True).generate(mesh.locations()).vals

    vmin, vmax = mesh_vals.min(), mesh_vals.max()
    meshx, meshy, mesh_vals_2d = mesh.slice(mesh_vals) # Each return value is a 2d array.

    int_vals = np.cumsum(mesh_vals_2d, axis=-1) / numsteps * xdim # Numerical integration on x-axis.

    sample_indices = np.random.choice(len(int_vals.ravel()), [500], replace=False)
    locs = np.stack([meshx.ravel()[sample_indices], meshy.ravel()[sample_indices]], axis=-1)
    vals = int_vals.ravel()[sample_indices]

    model = Model(
        gp.SquaredExponential(scale=[0., 1.], range='y_range') * \
        gp.IntExponential(axis=0, start=0., range='x_range') + \
        gp.Noise(nugget=1e-3),
        parameters = dict(x_range=1., y_range=1., sill=2.),
        verbose=True).fit(locs, vals, iters=1000, step_size=1e-1)

def test_wiener():

    np.random.seed(1)
    tf.random.set_seed(1)

    numsteps = 80
    xdim = 2.

    mesh = Mesh.from_bounds([0., 0., xdim, 1.], nx=numsteps)
    def trend_terms(x, y): return x, y, x*x, x*y, y*y
    featurizer = NormalizingFeaturizer(trend_terms, mesh.locations())

    mesh_vals = Model(
        gp.SquaredExponential(scale=[0., 1.], sill=1., range=0.3) *
        gp.SquaredExponential(scale=[1., 0.], sill=1., range=1e-4) + gp.Noise(nugget=1e-4),
        parameters = dict(),
        verbose=True).generate(mesh.locations()).vals

    vmin, vmax = mesh_vals.min(), mesh_vals.max()
    meshx, meshy, mesh_vals_2d = mesh.slice(mesh_vals) # Each return value is a 2d array.

    # Numerical integration on x-axis. Scaling is tricky.
    int_vals = np.cumsum(mesh_vals_2d, axis=-1) / np.sqrt(numsteps / xdim)

    sample_indices = np.random.choice(len(int_vals.ravel()), [500], replace=False)
    locs = np.stack([meshx.ravel()[sample_indices], meshy.ravel()[sample_indices]], axis=-1)
    vals = int_vals.ravel()[sample_indices]

    model = Model(
        gp.SquaredExponential(scale=[0., 1.]) * gp.Wiener(axis=0, start=0.) + gp.Noise(nugget=1e-4),
        parameters = dict(range=1., sill=2.),
        verbose=True).fit(locs, vals, iters=1000, step_size=1e-1)

