import pandas as pd
import numpy as np
from geostat import GP
import pyproj

def test_gp():
    # Define trend features.
    def featurization(df):
        x, y = df['x'].values, df['y'].values
        return np.stack([x, y], axis=1)

    # Define projection.
    xform = pyproj.Transformer.from_crs('epsg:4269', 'epsg:3310').transform
    def projection(df):
        x, y = xform(df['lat'], df['lon'])
        return pd.DataFrame().assign(x = x * 1e-3, y = y * 1e-3)

    # Data.
    df = pd.DataFrame().assign(
        lat = [35.0, 35.0, 35.0, 36.0, 36.0, 36.0, 37.0, 37.0, 37.0],
        lon = [-120.0, -121.0, -122.0, -120.0, -121.0, -122.0, -120.0, -121.0, -122.0],
        u = [5., 5., 5., 5., 6., 6., 5., 6., 7.])

    # Fit GP.
    gp = GP(df[['lat', 'lon']], df['u'],
            projection = projection,
            featurization = featurization,
            covariance_func = 'squared-exp',
            parameter0 = dict(vrange=50, sill=500, nugget=50),
            hyperparameters = dict(alpha=df['u'].values.ptp()**2, reg=1, train_iters=1000),
            verbose=True)

    # Interpolate using GP.
    df2 = pd.DataFrame().assign(
        lat = [35.0, 35.5, 35.5, 36.5],
        lon = [-120.0, -120.0, -121.0, -121.0])

    mean, var = gp.predict(df2)
    
    print(mean, var)
