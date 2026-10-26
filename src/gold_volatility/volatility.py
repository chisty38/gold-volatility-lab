import numpy as np
import pandas as pd

# Minutes per forecast: each forecast covers the following hour.
HORIZON = 60


def realized_volatility(log_returns: pd.Series, window: int) -> pd.Series:
    """Volatility of the last `window` returns, scaled to one hour.

    At time t, this uses returns ending at t.
    """
    return log_returns.rolling(window).std(ddof=0) * np.sqrt(HORIZON)
