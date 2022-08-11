import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from shapely.geometry import MultiPoint, Point, shape

__all__ = ['Mesh']

class Mesh:
    def __init__(self, locs, nx=None, ny=None):

        self.locs = locs

        # Get bounding box.
        x0, y0 = locs.min(axis=0)
        x1, y1 = locs.max(axis=0)

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
        self.x = np.linspace(x0 + margin, x1 - margin, nx)

        ny = int((y1 - y0) // xpitch)
        margin = (y1 - y0 - ny * xpitch) / 2
        self.y = np.linspace(y0 + margin, y1 - margin, ny)

        meshx, meshy = np.meshgrid(self.x, self.y)
        ix, iy = np.meshgrid(np.arange(len(self.x)), np.arange(len(self.y)))

        self.meshdf = pd.DataFrame().assign(
            x = meshx.ravel(),
            y = meshy.ravel(),
            ix = ix.ravel(),
            iy = iy.ravel())

    def convex_hull(self):
        hull = MultiPoint(self.locs).convex_hull
        mask = [hull.contains(Point(p)) for p in self.locations()]
        self.meshdf = self.meshdf.iloc[mask]
        return self

    def locations(self):
        return self.meshdf[['x', 'y']].values

    def slice(self, values):
        meshx, meshy = np.meshgrid(self.x, self.y)
        out = np.full([len(self.y), len(self.x)], float('nan'))
        out[(self.meshdf['iy'], self.meshdf['ix'])] = values
        return meshx, meshy, out
