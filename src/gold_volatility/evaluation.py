"""Walk-forward evaluation of next-hour volatility forecasts.

Before each trading session, Ridge settings are chosen on the most recent
past sessions, the model is refit on all past sessions, and that session's
hours are forecast. No forecast ever uses a model that saw its session or
anything later.
"""

import json
import platform
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from matplotlib.figure import Figure
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from . import __version__
from .data import (
    DEFAULT_CACHE,
    EXCHANGE_TZ,
    SYMBOL,
    file_fingerprint,
    hourly_examples,
    load_cache,
)
from .volatility import HORIZON

DEFAULT_REPORTS = Path("reports")

CONFIG = {
    "symbol": SYMBOL,
    # Sessions of history required before the first forecast session.
    "min_train_sessions": 8,
    # The most recent past sessions, used to choose Ridge settings.
    "validation_sessions": 3,
    "alphas": [0.01, 0.1, 1.0, 10.0, 100.0],
    "feature_sets": {
        "lags": ["vol_now", "vol_lag_1h", "vol_lag_2h"],
        "har_avg": ["vol_now", "avg_6_hourly", "avg_24_hourly"],
        "har_direct": ["vol_now", "rv_6h", "rv_24h"],
        # Time of day: the same clock hour in recent sessions.
        "seasonal": ["vol_now", "seasonal_5d"],
        "har_seasonal": ["vol_now", "avg_24_hourly", "seasonal_5d"],
    },
    "targets": ["original", "log"],
    # Volatility of exactly zero has no logarithm, so floor it first.
    "log_floor": 1e-8,
    "baselines": {
        "previous_hour": "vol_now",
        "avg_3_hourly": "avg_3_hourly",
        "same_hour_5d": "seasonal_5d",
    },
    # One block per trading session keeps intraday patterns together
    # when resampling test errors.
    "bootstrap_block": 23,
    "bootstrap_samples": 2000,
    "bootstrap_seed": 0,
}

Candidate = dict[str, str | float]

CANDIDATES: list[Candidate] = [
    {"features": features, "target": target, "alpha": alpha}
    for features in CONFIG["feature_sets"]
    for target in CONFIG["targets"]
    for alpha in CONFIG["alphas"]
]


def trading_session(index: pd.DatetimeIndex) -> pd.Index:
    """Session date of the hour starting at each forecast time.

    A COMEX session opens at 18:00 New York time and is named after the
    following day, so shifting by 6 hours maps it onto one calendar date.
    """
    local = index.tz_convert(EXCHANGE_TZ) + pd.Timedelta(hours=6)
    return pd.Index(local.date, name="session")


def walk_forward_folds(
    examples: pd.DataFrame,
) -> Iterator[tuple[date, pd.DataFrame, pd.DataFrame]]:
    """Yield (session, history, test) for each session after warm-up."""
    sessions = trading_session(examples.index)
    ordered = sorted(sessions.unique())

    for session in ordered[CONFIG["min_train_sessions"] :]:
        history = examples[sessions < session]
        test = examples[sessions == session]

        # Each target hour ends before the test session's first target.
        horizon = pd.Timedelta(minutes=HORIZON)
        assert history.index.max() + horizon <= test.index.min()
        yield session, history, test


def split_validation(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Older sessions for fitting, the most recent ones for validation."""
    sessions = trading_session(history.index)
    ordered = sorted(sessions.unique())
    first_validation = ordered[-CONFIG["validation_sessions"]]
    return (
        history[sessions < first_validation],
        history[sessions >= first_validation],
    )


def choose_candidate(history: pd.DataFrame) -> Candidate:
    """Ridge settings with the lowest validation MAE, using history only."""
    train, validation = split_validation(history)
    maes = [
        score(validation, predict(c, fit(c, train), validation))["mae_pp"]
        for c in CANDIDATES
    ]
    return CANDIDATES[int(np.argmin(maes))]


def to_log(values):
    return np.log(np.maximum(values, CONFIG["log_floor"]))


def fit(candidate: Candidate, frame: pd.DataFrame) -> Pipeline:
    # StandardScaler learns feature scales from the fitting data only.
    # Ridge learns regression weights with a penalty on large weights.
    model = make_pipeline(StandardScaler(), Ridge(alpha=candidate["alpha"]))
    features = frame[CONFIG["feature_sets"][candidate["features"]]]
    target = frame["actual"]

    if candidate["target"] == "log":
        features, target = to_log(features), to_log(target)

    model.fit(features, target)
    return model


def predict(candidate: Candidate, model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    features = frame[CONFIG["feature_sets"][candidate["features"]]]

    if candidate["target"] == "log":
        # exp() of a log-space mean estimates a median rather than a
        # mean. That is acceptable for MAE, but not unbiased for the mean.
        return np.exp(model.predict(to_log(features)))

    # Volatility cannot be negative.
    return np.maximum(model.predict(features), 0.0)


def score(frame: pd.DataFrame, predictions: np.ndarray) -> dict[str, float]:
    errors = np.abs(frame["actual"].to_numpy() - predictions)
    return {
        "forecasts": len(errors),
        "mae_pp": errors.mean() * 100,
        "worst_pp": errors.max() * 100,
    }


def label(candidate: Candidate) -> str:
    return (
        f"ridge[{candidate['features']}, {candidate['target']}, "
        f"alpha={candidate['alpha']}]"
    )


def block_bootstrap_ci(differences: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """95% interval for a mean, resampling consecutive blocks."""
    block = CONFIG["bootstrap_block"]
    n = len(differences)
    starts = rng.integers(
        0,
        n - block + 1,
        size=(CONFIG["bootstrap_samples"], int(np.ceil(n / block))),
    )
    samples = [
        np.concatenate([differences[s : s + block] for s in row])[:n].mean()
        for row in starts
    ]
    return np.percentile(samples, [2.5, 97.5])


def walk_forward(examples: pd.DataFrame) -> pd.DataFrame:
    """Out-of-sample forecasts for every session after warm-up."""
    parts = []

    for session, history, test in walk_forward_folds(examples):
        chosen = choose_candidate(history)
        model = fit(chosen, history)

        part = pd.DataFrame({"actual": test["actual"]})
        part["chosen"] = predict(chosen, model, test)
        for name, column in CONFIG["baselines"].items():
            part[name] = test[column]
        part["session"] = session
        part["chosen_settings"] = label(chosen)
        parts.append(part)

    return pd.concat(parts)


def summarize(
    predictions: pd.DataFrame, price: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pooled and per-session results for every method."""
    methods = ["chosen", *CONFIG["baselines"]]

    summary = pd.DataFrame(
        [
            {"method": m, **score(predictions, predictions[m].to_numpy())}
            for m in methods
        ]
    ).set_index("method")
    # Volatility is a fraction of price; in dollars at the latest price.
    summary["mae_usd"] = summary["mae_pp"] / 100 * price

    # Paired comparison against previous hour, with uncertainty.
    rng = np.random.default_rng(CONFIG["bootstrap_seed"])
    errors = predictions[methods].sub(predictions["actual"], axis=0).abs()
    reference = errors["previous_hour"].to_numpy()
    for m in methods:
        difference = errors[m].to_numpy() - reference
        low, high = block_bootstrap_ci(difference, rng)
        summary.loc[m, "vs_previous_hour_pp"] = difference.mean() * 100
        summary.loc[m, "ci_low_pp"] = low * 100
        summary.loc[m, "ci_high_pp"] = high * 100

    # Per-session results: how often does each method beat previous hour?
    folds = errors.groupby(predictions["session"]).mean() * 100
    for m in methods:
        summary.loc[m, "sessions_won"] = (folds[m] < folds["previous_hour"]).sum()
    folds["forecasts"] = predictions.groupby("session").size()
    folds["chosen_settings"] = predictions.groupby("session")["chosen_settings"].first()

    return summary, folds


def run(cache: Path = DEFAULT_CACHE, reports: Path = DEFAULT_REPORTS) -> pd.DataFrame:
    """Run the walk-forward evaluation and write every report."""
    reports.mkdir(parents=True, exist_ok=True)

    bars = load_cache(cache)
    price = float(bars["close"].iloc[-1])
    examples = hourly_examples(bars)
    predictions = walk_forward(examples)
    summary, folds = summarize(predictions, price)
    sessions = predictions["session"].unique()

    print(
        f"{CONFIG['symbol']}: {len(examples)} hourly examples "
        f"({examples.index[0]:%Y-%m-%d} to {examples.index[-1]:%Y-%m-%d}). "
        f"Walk-forward over {len(sessions)} sessions, "
        f"{len(predictions)} out-of-sample forecasts."
    )
    print("\nPER SESSION, MAE in percentage points:")
    print(folds.round(4).to_string())
    print("\nOUT OF SAMPLE, all sessions pooled:")
    print(summary.round(4).to_string())
    print(
        f"\nmae_usd: average forecast error in dollars per ounce at the latest "
        f"price (${price:,.2f}). vs_previous_hour_pp: mean change in absolute "
        "error against the previous-hour forecast (negative is better), with a "
        "95% block-bootstrap interval. sessions_won: sessions where the method "
        "beat the previous hour."
    )

    predictions.to_csv(reports / "real_predictions.csv")
    folds.to_csv(reports / "real_folds.csv")
    summary.to_csv(reports / "real_metrics.csv")

    record = {
        "config": CONFIG,
        "data": {
            "cache_sha256": file_fingerprint(cache),
            "first_example": str(examples.index[0]),
            "last_example": str(examples.index[-1]),
            "examples": len(examples),
            "latest_price": price,
            "test_sessions": [str(s) for s in sessions],
            "out_of_sample_forecasts": len(predictions),
        },
        "chosen_settings": folds["chosen_settings"].value_counts().to_dict(),
        "summary": summary.reset_index().to_dict(orient="records"),
        "versions": {
            "gold_volatility": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }
    (reports / "real_run.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )

    plot(examples, predictions, reports / "real_chart.png")
    print(f"\nReports written to {reports}/")
    return summary


def plot(examples: pd.DataFrame, predictions: pd.DataFrame, path: Path) -> None:
    # The target hour starts at the forecast time.
    clock_hour = examples.index.tz_convert(EXCHANGE_TZ).hour
    profile = examples["actual"].groupby(clock_hour).median() * 100

    # A Figure built directly needs no GUI backend, so this runs headless.
    fig = Figure(figsize=(14, 4))
    left, right = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 2.5]})

    left.bar(profile.index, profile.to_numpy(), color="#4a78b5")
    left.set_xlabel("Hour starting (New York time)")
    left.set_ylabel("Median hourly volatility (%)")
    left.set_title("Gold volatility by time of day")
    left.set_xticks(range(0, 24, 3))

    # Plot against forecast number so weekends don't leave long gaps.
    shown = predictions.reset_index(drop=True)
    x = shown.index
    right.plot(x, shown["actual"] * 100, label="Actual", color="black", linewidth=1)
    right.plot(x, shown["previous_hour"] * 100, label="Previous hour", alpha=0.7)
    right.plot(x, shown["chosen"] * 100, label="Walk-forward Ridge", alpha=0.8)

    starts = shown.groupby("session").head(1)
    for position in starts.index[1:]:
        right.axvline(position, color="grey", linewidth=0.5, alpha=0.5)
    right.set_xticks(starts.index)
    right.set_xticklabels([f"{s:%b %d}" for s in starts["session"]], rotation=45)
    right.set_xlim(x[0], x[-1])
    right.set_ylabel("Hourly volatility (%)")
    right.set_title("Out-of-sample forecasts, one model refit per session")
    right.legend(loc="upper left")

    fig.suptitle(f"Real {CONFIG['symbol']} gold futures, 1-minute data")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
