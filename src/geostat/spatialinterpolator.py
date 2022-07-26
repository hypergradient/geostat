import numpy as np
# import geopandas as gpd
# from shapely.geometry import Point

class SpatialInterpolator:
     def __init__(self, projection=None):
         self.projection = projection

     def predict(self, x, **kw):
         pass

     def project(self, x: np.ndarray) -> np.ndarray:
         if self.projection is None:
             return x
         else:
             return self.projection(x)

     def convex_hull_grid(self, spacing, lon, lat, z=None):

        '''
        This function replaces manual workflows in gis using
        the minimum bounding geometry tool to make a custom
        extent/bounds on a spatial dataset. It also adds on
        a depth series for 3d data and projects if desired.


        Parameters:
                spacing : int
                    The spacing of the grid locations produced. The bigger
                    the number, the closer the spacing and the denser
                    the dataset created.

                lon : array
                    The longitude coordinate of the input (x1) data to
                    be encompassed.

                lat : array
                    The latitude coordinate of the input (x1) data to
                    be encompassed.

                z : array, opt
                    The depths to make a depth series at each xy coordinate.
                    Example:  z = np.arange(-100, -5, 10) would make a depth
                    series at each xy coordinate from -100 to -5 by 10.
                    Default is None.

        Returns:
                x2 :  pandas dataframe
                    Locations to make GP predictions.


        '''

        # Make df then geodf of input data.
        df = pd.DataFrame()
        df['lon'] = lon
        df['lat'] = lat
        df['geometry'] = df.apply(lambda row: Point(row.lon, row.lat), axis=1)
        df_shp  = gpd.GeoDataFrame(df).set_crs('EPSG:4269')

        # Make square grid.
        loni = np.linspace(np.min(lon), np.max(lon), spacing)
        lati = np.linspace(np.min(lat), np.max(lat), spacing)
        gridlon, gridlat = np.meshgrid(loni, lati)

        # Grid df then geodf.
        df_grid = pd.DataFrame()
        df_grid['gridlon'] = gridlon.flatten()
        df_grid['gridlat'] = gridlat.flatten()
        df_grid['geometry'] = df_grid.apply(lambda row: Point(row.gridlon, row.gridlat), axis=1)
        df_grid  = gpd.GeoDataFrame(df_grid).set_crs('EPSG:4269')

        # Clip.
        hull = df_shp.unary_union.convex_hull
        clipped = gpd.clip(df_grid, hull)

        # Lon/lat.
        x2_array = np.array([clipped.gridlon.to_numpy(), clipped.gridlat.to_numpy()]).T

        # Project.
        x2 = self.project(x2_array)

        return x2
