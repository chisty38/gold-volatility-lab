import pandas as pd

FEATURE_COLUMNS = [
    "vol_now",
    "vol_lag_1h",
    "vol_lag_2h",
    "avg_3_hourly",
    "avg_6_hourly",
    "avg_24_hourly",
]


def build_features(df: pd.DataFrame, require_consecutive: bool = True) -> pd.DataFrame:
    """Build past-only features from consecutive hourly observations.

    `df["vol_now"]` is the volatility of the hour ending at each timestamp.
    Every feature at time t uses only rows up to and including t.

    Real markets pause overnight and at weekends. Pass
    `require_consecutive=False` to treat rows as consecutive trading hours,
    so a lag can reach back across a pause.
    """
    if "vol_now" not in df.columns:
        raise ValueError("Missing required column: vol_now")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected a DatetimeIndex")

    if not df.index.is_monotonic_increasing or not df.index.is_unique:
        raise ValueError("Timestamps must be sorted and unique")

    intervals = df.index.to_series().diff().dropna()
    if require_consecutive and not intervals.eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Expected consecutive hourly observations")

    volatility = df["vol_now"]

    features = pd.DataFrame(index=df.index)
    features["vol_now"] = volatility
    features["vol_lag_1h"] = volatility.shift(1)
    features["vol_lag_2h"] = volatility.shift(2)
    # Averages of separate, non-overlapping hourly estimates.
    features["avg_3_hourly"] = volatility.rolling(3).mean()
    features["avg_6_hourly"] = volatility.rolling(6).mean()
    features["avg_24_hourly"] = volatility.rolling(24).mean()

    return features
