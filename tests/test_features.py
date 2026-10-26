import numpy as np
import pandas as pd
import pytest

from gold_volatility.features import FEATURE_COLUMNS, build_features


def sample_data():
    return pd.DataFrame(
        {"vol_now": np.arange(1, 49, dtype=float)},
        index=pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC"),
    )


def test_known_feature_values():
    features = build_features(sample_data())

    # At row 23, available values are 1 through 24.
    row = features.iloc[23]

    assert row["vol_now"] == 24
    assert row["vol_lag_1h"] == 23
    assert row["vol_lag_2h"] == 22
    assert row["avg_3_hourly"] == pytest.approx(23)
    assert row["avg_6_hourly"] == pytest.approx(21.5)
    assert row["avg_24_hourly"] == pytest.approx(12.5)


def test_future_changes_do_not_change_earlier_features():
    original = sample_data()
    modified = original.copy()

    cutoff = original.index[30]
    modified.loc[modified.index > cutoff, "vol_now"] = 9999

    original_features = build_features(original)
    modified_features = build_features(modified)

    pd.testing.assert_frame_equal(
        original_features.loc[:cutoff],
        modified_features.loc[:cutoff],
    )


def test_insufficient_history_stays_missing():
    features = build_features(sample_data())

    assert features["avg_24_hourly"].iloc[:23].isna().all()
    assert features["avg_24_hourly"].iloc[23] == pytest.approx(12.5)


def test_missing_hour_is_rejected():
    data = sample_data().drop(sample_data().index[10])

    with pytest.raises(ValueError, match="consecutive hourly"):
        build_features(data)


def test_target_is_not_included_as_a_feature():
    data = sample_data()
    data["actual"] = 12345

    features = build_features(data)

    assert list(features.columns) == FEATURE_COLUMNS
    assert "actual" not in features.columns


def test_missing_volatility_column_is_rejected():
    data = sample_data().rename(columns={"vol_now": "other"})

    with pytest.raises(ValueError, match="vol_now"):
        build_features(data)


def test_non_time_index_is_rejected():
    data = sample_data().reset_index(drop=True)

    with pytest.raises(ValueError, match="DatetimeIndex"):
        build_features(data)


def test_unsorted_timestamps_are_rejected():
    data = sample_data().iloc[::-1]

    with pytest.raises(ValueError, match="sorted and unique"):
        build_features(data)
