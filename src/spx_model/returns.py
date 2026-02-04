from __future__ import annotations

import numpy as np
import pandas as pd


def daily_returns(prices: pd.Series) -> pd.Series:
    """
    Close-to-close arithmetic returns: (P_t / P_{t-1}) - 1
    First value will be NaN.
    """
    if prices.empty:
        return prices.copy()

    r = prices.pct_change()
    r.name = "daily_return"
    return r


def log_returns(prices: pd.Series) -> pd.Series:
    """
    Close-to-close log returns: ln(P_t / P_{t-1})
    First value will be NaN.
    """
    if prices.empty:
        return prices.copy()

    lr = np.log(prices / prices.shift(1))
    lr.name = "log_return"
    return lr


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """
    Convert a return series into cumulative growth of $1:
      (1 + r).cumprod() - 1
    Assumes returns are arithmetic returns.
    """
    if returns.empty:
        return returns.copy()

    cr = (1.0 + returns).cumprod() - 1.0
    cr.name = "cumulative_return"
    return cr
