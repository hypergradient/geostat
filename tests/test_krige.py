import numpy as np
import pandas as pd
from geostat import Krige

def featurization(x1):
    return x1[:, 0], x1[:, 1]

def project(x):
    return 

def test_krige():
    df = pd.DataFrame().assign(
        lat = [35.0, 35.0, 36.0, 36.0, 37.0, 37.0, 38.0, 38.0, 39.0, 39.0],
        lon = [-120.0, -121.0, -120.0, -121.0, -120.0, -121.0, -120.0, -121.0, -120.0, -121.0],
        u = [5., 5., 5., 6., 5., 7., 6., 6., 6., 7.])

    x1 = df[['lon', 'lat']].to_numpy()

    u1 = df[['u']].to_numpy()

    # A silly projection for this unit test.
    def projection(x):
        lon, lat = np.moveaxis(x, -1, 0)
        x1 = np.cos(lat / 180 * np.pi) * lon / 360 * 40000
        x2 = lat / 360 * 40000
        return np.stack([x1, x2], axis=-1)

    krige = Krige(x1, u1, 2,
            variogram_func='linear',
            projection=projection,
            featurization=featurization,
            show_plots=False,
            verbose=True)

    df2 = pd.DataFrame().assign(
        lat = [35.5, 35.5, 36.5],
        lon = [-120.0, -121.0, -121.0])

    x2 = df2[['lon', 'lat']].to_numpy()

    u2_mean, u2_var = krige.predict(x2)
    
    print(u2_mean, u2_var)
