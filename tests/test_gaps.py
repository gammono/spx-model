import numpy as np
import pandas as pd

from spx_model.gaps import gap_return, intraday_return, daily_range


def test_gap_return_aligns_with_previous_close():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    close = pd.Series([100.0, 110.0], index=idx)
    open_ = pd.Series([101.0, 120.0], index=idx)

    g = gap_return(close, open_)
    assert np.isnan(g.loc["2024-01-02"])
    assert np.isclose(g.loc["2024-01-03"], (120.0 / 100.0) - 1.0)


def test_intraday_return():
    idx = pd.to_datetime(["2024-01-02"])
    open_ = pd.Series([100.0], index=idx)
    close = pd.Series([110.0], index=idx)

    r = intraday_return(open_, close)
    assert np.isclose(r.iloc[0], 0.10)


def test_daily_range_scaled_by_open():
    idx = pd.to_datetime(["2024-01-02"])
    open_ = pd.Series([100.0], index=idx)
    high = pd.Series([110.0], index=idx)
    low = pd.Series([90.0], index=idx)

    rng = daily_range(open_, high, low)
    assert np.isclose(rng.iloc[0], 0.20)
