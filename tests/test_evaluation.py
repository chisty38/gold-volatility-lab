import numpy as np
import pandas as pd
import pytest

from gold_volatility.evaluation import (
    CONFIG,
    block_bootstrap_ci,
    fit,
    predict,
    score,
    split_validation,
    trading_session,
    walk_forward,
    walk_forward_folds,
)
from gold_volatility.volatility import HORIZON, realized_volatility


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(0)
    n = 200
    vol = np.exp(rng.normal(np.log(0.002), 0.3, n))
    columns = {
        name: vol * rng.uniform(0.8, 1.2, n)
        for name in {c for cols in CONFIG["feature_sets"].values() for c in cols}
    }
    return pd.DataFrame({**columns, "actual": vol})


def test_realized_volatility_known_value():
    # Alternating +r and -r returns have population std exactly r.
    returns = pd.Series([0.001, -0.001] * 30)
    vol = realized_volatility(returns, HORIZON)

    assert vol.iloc[: HORIZON - 1].isna().all()
    assert vol.iloc[-1] == pytest.approx(0.001 * np.sqrt(HORIZON))


@pytest.mark.parametrize("features", list(CONFIG["feature_sets"]))
@pytest.mark.parametrize("target", CONFIG["targets"])
def test_fit_and_predict_give_positive_forecasts(frame, features, target):
    candidate = {"features": features, "target": target, "alpha": 1.0}

    predictions = predict(candidate, fit(candidate, frame), frame)

    assert predictions.shape == (len(frame),)
    assert np.isfinite(predictions).all()
    assert (predictions >= 0).all()


def test_score_known_values():
    frame = pd.DataFrame({"actual": [0.01, 0.02, 0.03, 0.04]})
    predictions = np.array([0.02, 0.02, 0.03, 0.05])

    result = score(frame, predictions)

    assert result["forecasts"] == 4
    assert result["mae_pp"] == pytest.approx(0.5)
    assert result["worst_pp"] == pytest.approx(1.0)


def test_bootstrap_interval_brackets_the_mean():
    rng = np.random.default_rng(0)
    differences = rng.normal(-0.5, 1.0, 230)

    low, high = block_bootstrap_ci(differences, np.random.default_rng(1))

    assert low < differences.mean() < high
    assert high < 0


def test_trading_session_follows_the_comex_day():
    times = pd.DatetimeIndex(
        [
            "2026-10-18 23:00",  # Sunday 19:00 New York: Monday's session
            "2026-10-19 20:00",  # Monday 16:00 New York: still Monday
            "2026-10-19 23:00",  # Monday 19:00 New York: Tuesday's session
        ],
        tz="UTC",
    )

    sessions = [str(s) for s in trading_session(times)]

    assert sessions == ["2026-10-19", "2026-10-19", "2026-10-20"]


def test_folds_train_only_on_earlier_sessions(long_examples):
    folds = list(walk_forward_folds(long_examples))
    all_sessions = sorted(trading_session(long_examples.index).unique())

    assert [s for s, _, _ in folds] == all_sessions[CONFIG["min_train_sessions"] :]
    for session, history, test in folds:
        assert (trading_session(history.index) < session).all()
        assert (trading_session(test.index) == session).all()
        assert history.index.max() < test.index.min()


def test_validation_is_the_most_recent_history(long_examples):
    _, history, _ = next(walk_forward_folds(long_examples))

    train, validation = split_validation(history)
    validation_sessions = trading_session(validation.index).unique()

    assert len(validation_sessions) == CONFIG["validation_sessions"]
    assert train.index.max() < validation.index.min()
    assert len(train) + len(validation) == len(history)


def test_walk_forward_ignores_later_sessions(long_examples):
    # Forecasts for a session must not change if later sessions are
    # removed: no model may have seen them.
    full = walk_forward(long_examples)
    cutoff = sorted(full["session"].unique())[2]
    earlier = long_examples[trading_session(long_examples.index) <= cutoff]

    truncated = walk_forward(earlier)

    pd.testing.assert_frame_equal(full[full["session"] <= cutoff], truncated)
