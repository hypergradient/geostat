import pandas as pd
from geostat import GP
import pyproj

def test_gp():
    # Define trend features.
    def featurization(df):
        x, y = df['x'], df['y']
        return x, y, x*x, x*y, y*y

    # Define projection.
    xform = pyproj.Transformer.from_crs('epsg:4269', 'epsg:3310').transform
    def project(df):
        x, y = xform(df['lat'], df['lon'])
        df.assign(x = x * 1e-3, y = y * 1e-3)

    # Data.
    df = pd.DataFrame().assign(
        lat = [35.0, 35.0, 36.0, 36.0, 37.0, 37.0],
        lon = [-120.0, -121.0, -120.0, -121.0, -120.0, -121.0],
        u = [5., 5., 5., 6., 5., 7.])

    # Fit GP.
    gp = GP(df[['lat', 'lon']], df['u'],
            projection = project,
            featurization = featurization,
            covariance_func = 'gamma-exp',
            parameter0 = dict(vrange=50, sill=500, nugget=50, gamma=1.0),
            hyperparameters = dict(alpha=u1.ptp()**2, reg=1, train_iters=300),
            verbose=True)

    # Interpolate using GP.
    df2 = pd.DataFrame().assign(
        lat = [35.5, 35.5, 36.5],
        lon = [-120.0, -121.0, -121.0])

    df2 = gp.predict(df2)
    
    print(df2['mean'], df2['var'])
