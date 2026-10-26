"""Gold futures data: download, cache and hourly examples.

Source: Yahoo Finance 1-minute bars for COMEX gold futures (GC=F). Yahoo
keeps only about 30 days of 1-minute history, so every download is merged
into a local cache and the dataset grows each time it is refreshed.

Raw data stays in data/ and is not committed (Yahoo's terms do not allow
redistribution).
"""

import hashlib
import json
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import build_features
from .volatility import HORIZON, realized_volatility

SYMBOL = "GC=F"
DEFAULT_CACHE = Path("data") / "gc_f_1m.csv"

# COMEX gold trades 18:00-17:00 New York time, Sunday to Friday.
EXCHANGE_TZ = "America/New_York"

# Yahoo serves 1-minute bars for roughly the last 30 days, at most 7 days
# per request.
HISTORY_DAYS = 29
CHUNK_DAYS = 7

# A return spanning a longer pause than this crosses the daily break or a
# weekend. It is not a one-minute return, so it is discarded.
MAX_RETURN_GAP = pd.Timedelta(minutes=10)

# An hour counts only if almost all of its minutes traded.
MIN_RETURNS_PER_HOUR = 50

# Sessions averaged for the time-of-day feature.
SEASONAL_SESSIONS = 5


@dataclass(frozen=True)
class MergeStats:
    """What a download changed in the cache."""

    added: int
    revised: int
    total: int


def fetch_minutes(
    symbol: str = SYMBOL, days: int = HISTORY_DAYS, now: float | None = None
) -> pd.DataFrame:
    """Download recent 1-minute bars from Yahoo Finance."""
    now = int(now or time.time())
    start = now - days * 86400
    frames = []

    while start < now:
        end = min(start + CHUNK_DAYS * 86400, now)
        url = (
            "https://query1.finance.yahoo.com/v8/finance/chart/"
            f"{symbol}?interval=1m&period1={start}&period2={end}"
        )
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)

        result = payload["chart"]["result"]
        if not result:
            raise RuntimeError(f"Yahoo returned no data: {payload['chart']}")

        result = result[0]
        quote = result["indicators"]["quote"][0]
        frames.append(
            pd.DataFrame(
                {
                    column: quote[column]
                    for column in ["open", "high", "low", "close", "volume"]
                },
                index=pd.to_datetime(result.get("timestamp", []), unit="s", utc=True),
            )
        )
        start = end

    return clean_minutes(pd.concat(frames), now=now)


def clean_minutes(bars: pd.DataFrame, now: float | None = None) -> pd.DataFrame:
    """Keep finished, minute-aligned bars that have a price."""
    bars = bars.dropna(subset=["close"])
    # Yahoo appends a live quote stamped mid-minute; it is not a bar.
    bars = bars[bars.index.second == 0]

    if now is not None:
        # The bar for the current minute is still forming.
        current_minute = pd.Timestamp(now, unit="s", tz="UTC").floor("min")
        bars = bars[bars.index < current_minute]

    bars = bars[~bars.index.duplicated(keep="last")].sort_index()
    bars.index.name = "time"
    return bars


def merge_cache(
    cached: pd.DataFrame, fresh: pd.DataFrame
) -> tuple[pd.DataFrame, MergeStats]:
    """Combine cached and fresh bars; fresh values win on overlap."""
    overlap = cached.index.intersection(fresh.index)
    revised = int((cached.loc[overlap, "close"] != fresh.loc[overlap, "close"]).sum())

    combined = pd.concat([cached, fresh])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()

    stats = MergeStats(
        added=len(fresh.index.difference(cached.index)),
        revised=revised,
        total=len(combined),
    )
    return combined, stats


def load_cache(path: Path = DEFAULT_CACHE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `gold-vol fetch` from the repository root."
        )
    # The default float parser can change the last digits of a price, which
    # would make unchanged bars look revised on the next download.
    return pd.read_csv(
        path, index_col="time", parse_dates=["time"], float_precision="round_trip"
    )


def update_cache(path: Path = DEFAULT_CACHE) -> tuple[pd.DataFrame, MergeStats]:
    fresh = fetch_minutes()
    if path.exists():
        bars, stats = merge_cache(load_cache(path), fresh)
    else:
        bars = fresh
        stats = MergeStats(added=len(fresh), revised=0, total=len(fresh))

    path.parent.mkdir(parents=True, exist_ok=True)
    bars.to_csv(path)
    return bars, stats


def file_fingerprint(path: Path) -> str:
    """SHA-256 of a file, so every report names the exact data it used."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def minute_returns(bars: pd.DataFrame) -> pd.Series:
    """One-minute log returns, missing where trading paused."""
    returns = np.log(bars["close"]).diff()
    pause = bars.index.to_series().diff() > MAX_RETURN_GAP
    return returns.mask(pause)


def hourly_examples(bars: pd.DataFrame) -> pd.DataFrame:
    """One row per forecast time t, at the end of a complete clock hour.

    Rows follow trading time: the hour before a daily break or weekend is
    followed by the first hour after it. `actual` is the volatility of the
    clock hour starting at t, and exists only when that hour traded fully.
    """
    returns = minute_returns(bars)
    # A bar stamped 10:59 closes at 11:00, so hour 10 is known at 11:00.
    hour_start = returns.index.floor("h")
    by_hour = returns.groupby(hour_start)

    counts = by_hour.count()
    complete = counts[counts >= MIN_RETURNS_PER_HOUR].index
    vol_by_start = (by_hour.std(ddof=0) * np.sqrt(HORIZON))[complete]

    hourly = pd.DataFrame(
        {"vol_now": vol_by_start.to_numpy()},
        index=vol_by_start.index + pd.Timedelta(hours=1),
    )
    hourly.index.name = "time"

    # The answer key: volatility of the hour starting at t.
    hourly["actual"] = vol_by_start.reindex(hourly.index).to_numpy()

    # Volatility over the last 6 and 24 hours of trading-time returns.
    valid = returns.dropna()
    for name, hours in [("rv_6h", 6), ("rv_24h", 24)]:
        rolling = realized_volatility(valid, hours * HORIZON)
        last_in_hour = rolling.groupby(valid.index.floor("h")).last()
        hourly[name] = last_in_hour.reindex(vol_by_start.index).to_numpy()

    seasonal = seasonal_features(vol_by_start, hourly.index)
    features = build_features(hourly, require_consecutive=False)

    examples = pd.concat(
        [features, seasonal, hourly[["actual", "rv_6h", "rv_24h"]]],
        axis=1,
    )
    return examples.dropna()


def seasonal_features(
    vol_by_start: pd.Series, forecast_times: pd.DatetimeIndex
) -> pd.DataFrame:
    """Past volatility of the clock hour being forecast.

    For a forecast at t, the target hour starts at t. These features look
    only at earlier sessions' hours with the same New York clock time.
    """
    history = pd.DataFrame(
        {
            "start": vol_by_start.index,
            "clock_hour": vol_by_start.index.tz_convert(EXCHANGE_TZ).hour,
            "seasonal_1d": vol_by_start.to_numpy(),
        }
    )
    history["seasonal_5d"] = history.groupby("clock_hour")["seasonal_1d"].transform(
        lambda v: v.rolling(SEASONAL_SESSIONS).mean()
    )

    targets = pd.DataFrame(
        {
            "start": forecast_times,
            "clock_hour": forecast_times.tz_convert(EXCHANGE_TZ).hour,
        }
    )

    # Strictly earlier hours only: the target hour itself is excluded.
    matched = pd.merge_asof(
        targets.sort_values("start"),
        history.sort_values("start"),
        on="start",
        by="clock_hour",
        allow_exact_matches=False,
        direction="backward",
    ).set_index("start")

    return matched.loc[forecast_times, ["seasonal_1d", "seasonal_5d"]].set_axis(
        forecast_times
    )
