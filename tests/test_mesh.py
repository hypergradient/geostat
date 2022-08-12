import numpy as np
from geostat import Mesh
from shapely.geometry import Polygon

def test_mesh_from_convex_hull():

    # Create 100 random locations in a square centered on the origin.
    locs = np.random.uniform(-1., 1., [100, 2])

    mesh = Mesh.from_convex_hull(locs, 20)

    values = np.random.uniform(-10., 10., [mesh.locations().shape[0]])

    x, y, u = mesh.slice(values)

    print(np.sum(np.isnan(u)))

    assert x.shape == y.shape == u.shape

def test_mesh_from_polygon():

    polygon = Polygon([(0., 0.), (1., 1.), (1., 0.)])

    mesh = Mesh.from_polygon(polygon, 20)

    values = np.random.uniform(-10., 10., [mesh.locations().shape[0]])

    x, y, u = mesh.slice(values)

    print(np.sum(np.isnan(u)))

    assert x.shape == y.shape == u.shape
