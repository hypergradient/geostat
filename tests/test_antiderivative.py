import numpy as np
import tensorflow as tf
from geostat import gp, Model, Mesh, NormalizingFeaturizer


def test_gp2d():

    np.random.seed(1)
    tf.random.set_seed(1)

    numsteps = 80
    xwidth = 2.

    mesh = Mesh.from_bounds([0., 0., xwidth, 1.], nx=numsteps)
    def trend_terms(x, y): return x, y, x*x, x*y, y*y
    featurizer = NormalizingFeaturizer(trend_terms, mesh.locations())

    mesh_vals = Model(
        gp.GammaExponential(scale=[1., 1.]),
        parameters = dict(range=0.33, sill=1., gamma=1.99),
        verbose=True).generate(mesh.locations()).vals

    vmin, vmax = mesh_vals.min(), mesh_vals.max()
    meshx, meshy, mesh_vals_2d = mesh.slice(mesh_vals) # Each return value is a 2d array.

    int_vals = np.cumsum(mesh_vals_2d, axis=-1) / numsteps * xwidth # Numerical integration on x-axis.

    print(meshx[:5, :5])
    print(meshy[:5, :5])
    print(int_vals[:5, :5])

    sample_indices = np.random.choice(len(int_vals.ravel()), [500], replace=False)
    locs = np.stack([meshx.ravel()[sample_indices], meshy.ravel()[sample_indices]], axis=-1)
    vals = int_vals.ravel()[sample_indices]

    model = Model(
        gp.GammaExponentialInt(axis=0, start=0., scale=['sx', 1.], gamma=tf.constant(1.99)) + \
        gp.Noise(),
        parameters = dict(range=1., sill=2., nugget=0.01, sx=1.),
        verbose=True).fit(locs, vals, iters=1500)
