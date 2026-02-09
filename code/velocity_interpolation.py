"""
Velocity interpolation utilities for FETCH Range Calculation analysis.
"""

import pandas as pd
from scipy.interpolate import interp1d


def interpolate_velocity(time_series: pd.Series, velocity_df: pd.DataFrame) -> pd.Series:
    velocity_df = velocity_df.copy()
    velocity_df["Record Time"] = pd.to_datetime(velocity_df["Record Time"])
    time_numeric = velocity_df["Record Time"].astype("int64")

    interpolator = interp1d(
        time_numeric,
        velocity_df["SoundSpeed"],
        fill_value="extrapolate",
        bounds_error=False,
    )

    return pd.Series(
        interpolator(pd.to_datetime(time_series).astype("int64")),
        index=pd.to_datetime(time_series),
    )
