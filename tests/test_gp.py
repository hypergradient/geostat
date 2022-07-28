import pandas as pd
import numpy as np
from geostat import GP, use_xform
import pyproj

def test_gp():
    # Define trend features.
    def featurization(df):
        x, y = df['x'].values, df['y'].values
        return x

    # Define projection.
    xform = pyproj.Transformer.from_crs('epsg:4269', 'epsg:3310').transform

    # Data.
    df = pd.DataFrame().assign(
        lat = [35.0, 35.0, 35.0, 36.0, 36.0, 36.0, 37.0, 37.0, 37.0],
        lon = [-120.0, -121.0, -122.0, -120.0, -121.0, -122.0, -120.0, -121.0, -122.0],
        u = [5., 5., 5., 5., 7., 7., 5., 7., 7.])

    # Fit GP.
    gp = GP(df[['lat', 'lon']], df['u'],
            projection = use_xform(xform, in_coords = ['lat', 'lon'], out_coords = ['x', 'y'], rescale=1e-3),
            featurization = featurization,
            covariance_func = 'squared-exp',
            parameter0 = dict(vrange=50., sill=1., nugget=1.),
            hyperparameters = dict(alpha=df['u'].values.ptp()**2, reg=0, train_iters=1000),
            verbose=True)

    # Interpolate using GP.
    df2 = pd.DataFrame().assign(
        lat = [35.0, 35.5, 35.5, 36.5],
        lon = [-120.0, -120.0, -121.0, -121.0])

    mean, var = gp.predict(df2)
    
    print(mean, var)
