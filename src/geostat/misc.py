import pandas as pd

__all__ = ['use_xform']

def use_xform(
    xform,
    in_coords = ['lat', 'lon'],
    out_coords = ['x', 'y'],
    rescale = [1., 1.]
):
    def f(df):
        loc_cols = xform(*[df[name] for name in in_coords])
        if isinstance(rescale, (int, float)):
            loc_map = {name: col * rescale for name, col in zip(out_coords, loc_cols)}
        else:
            loc_map = {name: col * s for name, col, s in zip(out_coords, loc_cols, rescale)}
        return pd.DataFrame().assign(**loc_map)

    return f
