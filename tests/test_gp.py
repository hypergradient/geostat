import pandas as pd
from geostat import GP
import pyproj

def featurization(x1):
    return x1[:, 0], x1[:, 1]

def project(x):
    return 

def test_gp():
    df = pd.DataFrame().assign(
        lat = [35.0, 35.0, 36.0, 36.0, 37.0, 37.0],
        lon = [-120.0, -121.0, -120.0, -121.0, -120.0, -121.0],
        u = [5., 5., 5., 6., 5., 7.])

    x1 = df[['lon', 'lat']].to_numpy()

    u1 = df[['u']].to_numpy()

    # Projection
    p1 = pyproj.Proj(proj='latlong', datum='NAD83')
    p2 = pyproj.Proj('EPSG:3310')
    project = pyproj.Transformer.from_proj(p1, p2)
    to_km = lambda x, y: (x * 1e-3, y * 1e-3)

    gp = GP(x1, u1,
            covariance_func='gamma-exp',
            parameter0=dict(vrange=50, sill=500, nugget=50, gamma=1.0),
            projection=lambda x, y: to_km(*project.transform(x, y)),
            featurization=featurization,
            train_epochs=300,
            hyperparameters=dict(alpha=u1.ptp()**2, reg=1),
            verbose=True)

    df2 = pd.DataFrame().assign(
        lat = [35.5, 35.5, 36.5],
        lon = [-120.0, -121.0, -121.0])

    x2 = df2[['lon', 'lat']].to_numpy()

    u2_mean, u2_var = gp.predict(x2, batch_size=500)
    
    print(u2_mean, u2_var)
