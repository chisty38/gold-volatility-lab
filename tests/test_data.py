import numpy as np
import pandas as pd
import pytest

from gold_volatility.data import (
    EXCHANGE_TZ,
    clean_minutes,
    file_fingerprint,
    hourly_examples,
    load_cache,
    merge_cache,
    minute_returns,
    update_cache,
)
from gold_volatility.features import build_features


def test_clean_minutes_keeps_only_finished_bars():
    index = pd.DatetimeIndex(
        [
            "2026-10-20 10:00:00",
            "2026-10-20 10:01:00",
            "2026-10-20 10:01:00",
            "2026-10-20 10:02:00",
            "2026-10-20 10:02:30",
            "2026-10-20 10:03:00",
        ],
        tz="UTC",
    )
    raw = pd.DataFrame({"close": [1.0, np.nan, 2.0, 3.0, 3.5, 4.0]}, index=index)
    now = pd.Timestamp("2026-10-20 10:03:20", tz="UTC").timestamp()

    cleaned = clean_minutes(raw, now=now)

    # Null close, mid-minute quote and the still-forming 10:03 bar are
    # gone; the duplicate stamp keeps its priced row.
    assert list(cleaned["close"]) == [1.0, 2.0, 3.0]


def test_merge_cache_prefers_fresh_values():
    cached = pd.DataFrame(
        {"close": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2026-10-01 00:00", "2026-10-01 00:01"], tz="UTC"),
    )
    fresh = pd.DataFrame(
        {"close": [20.0, 3.0]},
        index=pd.DatetimeIndex(["2026-10-01 00:01", "2026-10-01 00:02"], tz="UTC"),
    )

    merged, stats = merge_cache(cached, fresh)

    assert list(merged["close"]) == [1.0, 20.0, 3.0]
    assert (stats.added, stats.revised, stats.total) == (1, 1, 3)


def test_cache_round_trip_and_fingerprint(tmp_path, bars):
    path = tmp_path / "cache.csv"
    bars.head(100).to_csv(path)

    loaded = load_cache(path)

    pd.testing.assert_index_equal(loaded.index, bars.head(100).index)
    assert len(file_fingerprint(path)) == 64
    assert file_fingerprint(path) == file_fingerprint(path)


def test_missing_cache_explains_how_to_fetch(tmp_path):
    with pytest.raises(FileNotFoundError, match="gold-vol fetch"):
        load_cache(tmp_path / "missing.csv")


def test_returns_do_not_bridge_market_pauses(bars):
    returns = minute_returns(bars)
    local = returns.index.tz_convert(EXCHANGE_TZ)
    session_open = (local.hour == 18) & (local.minute == 0)

    assert returns[session_open].isna().all()
    assert returns[~session_open].iloc[1:].notna().all()


def test_targets_exist_only_for_traded_hours(examples):
    local = examples.index.tz_convert(EXCHANGE_TZ)

    # Forecasts at 17:00 would target the daily break.
    assert not (local.hour == 17).any()
    assert not (local.dayofweek == 5).any()


def test_actual_is_next_hour_volatility(examples):
    t = examples.index[10]
    next_hour = t + pd.Timedelta(hours=1)

    assert next_hour in examples.index
    assert examples.loc[t, "actual"] == pytest.approx(
        examples.loc[next_hour, "vol_now"]
    )


def test_seasonal_feature_is_previous_session_same_hour(examples):
    for t in examples.index[-30:]:
        day_before = t - pd.Timedelta(days=1)
        if day_before in examples.index:
            assert examples.loc[t, "seasonal_1d"] == pytest.approx(
                examples.loc[day_before, "actual"]
            )


def test_seasonal_feature_sees_the_busy_morning(examples):
    local = examples.index.tz_convert(EXCHANGE_TZ)
    busy = examples.loc[local.hour == 8, "seasonal_5d"].mean()
    quiet = examples.loc[local.hour == 13, "seasonal_5d"].mean()

    assert busy > 2 * quiet


def test_future_prices_do_not_change_earlier_features(bars, examples):
    cutoff = examples.index[len(examples) // 2]

    changed = bars.copy()
    changed.loc[changed.index >= cutoff, "close"] *= np.exp(
        np.random.default_rng(1).normal(0, 0.01, (changed.index >= cutoff).sum())
    )

    features = examples.columns.drop("actual")
    original = examples.loc[:cutoff, features]
    modified = hourly_examples(changed).loc[:cutoff, features]

    pd.testing.assert_frame_equal(original, modified)


def test_trading_time_features_allow_pauses():
    index = pd.DatetimeIndex(
        ["2026-10-23 20:00", "2026-10-23 21:00", "2026-10-25 23:00"],
        tz="UTC",
    )
    data = pd.DataFrame({"vol_now": [1.0, 2.0, 3.0]}, index=index)

    features = build_features(data, require_consecutive=False)

    assert features["vol_lag_1h"].iloc[-1] == 2.0
    with pytest.raises(ValueError, match="consecutive hourly"):
        build_features(data)


def test_update_cache_creates_then_extends(monkeypatch, tmp_path, bars):
    path = tmp_path / "data" / "cache.csv"
    downloads = iter([bars.iloc[:500], bars.iloc[400:600]])
    monkeypatch.setattr("gold_volatility.data.fetch_minutes", lambda: next(downloads))

    _, first = update_cache(path)
    merged, second = update_cache(path)

    assert (first.added, first.total) == (500, 500)
    assert (second.added, second.revised, second.total) == (100, 0, 600)
    pd.testing.assert_series_equal(
        merged["close"], bars["close"].iloc[:600], check_freq=False
    )
