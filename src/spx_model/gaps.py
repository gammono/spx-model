from __future__ import annotations

import pandas as pd


def gap_return(close: pd.Series, open_: pd.Series) -> pd.Series:
    """
    Overnight gap return: (Open_t / Close_{t-1}) - 1

    Inputs should be indexed by date. We align Open_t with Close_{t-1} by shifting close.
    The first date will be NaN because there is no prior close.
    """
    g = (open_ / close.shift(1)) - 1.0
    g.name = "gap_return"
    return g


def intraday_return(open_: pd.Series, close: pd.Series) -> pd.Series:
    """Intraday return: (Close_t / Open_t) - 1"""
    r = (close / open_) - 1.0
    r.name = "intraday_return"
    return r


def daily_range(open_: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
    """Daily range scaled by open: (High_t - Low_t) / Open_t"""
    rng = (high - low) / open_
    rng.name = "daily_range"
    return rng
