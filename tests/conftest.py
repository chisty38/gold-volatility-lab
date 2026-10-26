import numpy as np
import pandas as pd
import pytest

from gold_volatility.data import EXCHANGE_TZ, hourly_examples


def trading_minutes(start="2026-10-04 18:00", days=9):
    """Minute stamps following the COMEX schedule, in UTC.

    Trading runs 18:00-17:00 New York time, Sunday evening to Friday.
    """
    local = pd.date_range(
        pd.Timestamp(start, tz=EXCHANGE_TZ),
        periods=days * 24 * 60,
        freq="min",
    )
    daily_break = local.hour == 17
    weekend = (
        (local.dayofweek == 5)
        | ((local.dayofweek == 4) & (local.hour >= 17))
        | ((local.dayofweek == 6) & (local.hour < 18))
    )
    return local[~daily_break & ~weekend].tz_convert("UTC")


def simulated_bars(days):
    """Random-walk prices on the trading schedule, offline."""
    index = trading_minutes(days=days)
    rng = np.random.default_rng(0)
    # Busier New York mornings, like real gold.
    sigma = np.where(index.tz_convert(EXCHANGE_TZ).hour.isin([8, 9]), 0.0006, 0.0002)
    close = 4000 * np.exp(np.cumsum(rng.normal(0, sigma)))
    return pd.DataFrame({"close": close}, index=index.rename("time"))


@pytest.fixture(scope="session")
def bars():
    return simulated_bars(days=9)


@pytest.fixture(scope="session")
def examples(bars):
    return hourly_examples(bars)


@pytest.fixture(scope="session")
def long_bars():
    # Enough sessions for feature warm-up plus several walk-forward folds.
    return simulated_bars(days=26)


@pytest.fixture(scope="session")
def long_examples(long_bars):
    return hourly_examples(long_bars)
