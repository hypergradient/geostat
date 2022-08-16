from dataclasses import dataclass, replace
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from shapely.geometry import MultiPoint, Point, Polygon, shape

__all__ = ['Mesh']

@dataclass
class Mesh:

    x: np.ndarray
    y: np.ndarray
    meshdf: pd.DataFrame

    @staticmethod
    def from_bounds(bounds, nx=None, ny=None):
        # Get bounding box.
        x0, y0, x1, y1 = bounds

        # Figure out pitch.
        if nx is None and ny is None:
            raise ValueError('Either nx or ny (or both) must be set to an integer')
        elif nx is not None and nx < 1:
            raise ValueError('nx or ny, if set, must be 1 or greater')
        elif ny is not None and ny < 1:
            raise ValueError('nx or ny, if set, must be 1 or greater')
        elif nx is not None and ny is not None:
            xpitch = (x1 - x0) / nx
            ypitch = (y1 - y0) / ny
        elif ny is None:
            xpitch = ypitch = (x1 - x0) / nx
        elif nx is None:
            xpitch = ypitch = (y1 - y0) / ny

        nx = int((x1 - x0) // xpitch)
        margin = (x1 - x0 - nx * xpitch) / 2
        x = np.linspace(x0 + margin, x1 - margin, nx)

        ny = int((y1 - y0) // xpitch)
        margin = (y1 - y0 - ny * xpitch) / 2
        y = np.linspace(y0 + margin, y1 - margin, ny)

        meshx, meshy = np.meshgrid(x, y)
        ix, iy = np.meshgrid(np.arange(len(x)), np.arange(len(y)))

        meshdf = pd.DataFrame().assign(
            x = meshx.ravel(),
            y = meshy.ravel(),
            ix = ix.ravel(),
            iy = iy.ravel())

        return Mesh(x, y, meshdf)

    @staticmethod
    def from_convex_hull(locs, nx=None, ny=None):
        mp = MultiPoint(locs)
        mesh = Mesh.from_bounds(mp.bounds, nx, ny)
        hull = mp.convex_hull
        mask = [hull.contains(Point(p)) for p in mesh.locations()]
        return replace(mesh, meshdf = mesh.meshdf.iloc[mask])

    @staticmethod
    def from_polygon(polygon, nx=None, ny=None):
        mesh = Mesh.from_bounds(polygon.bounds, nx, ny)
        mask = [polygon.contains(Point(p)) for p in mesh.locations()]
        return replace(mesh, meshdf = mesh.meshdf.iloc[mask])

    def locations(self, proj=None):
        loc = self.meshdf[['x', 'y']].values
        if proj is not None:
            loc = (loc @ np.eye(2, 3) + [0, 0, 1]) @ proj
        return loc

    def slice(self, values):
        meshx, meshy = np.meshgrid(self.x, self.y)
        out = np.full([len(self.y), len(self.x)], float('nan'))
        out[(self.meshdf['iy'], self.meshdf['ix'])] = values
        return meshx, meshy, out
