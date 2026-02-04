import numpy as np
import pandas as pd

from spx_model.returns import daily_returns, log_returns, cumulative_returns


def test_daily_returns_simple_sequence():
    prices = pd.Series([100.0, 110.0, 121.0])
    r = daily_returns(prices)
    assert np.isnan(r.iloc[0])
    assert np.isclose(r.iloc[1], 0.10)
    assert np.isclose(r.iloc[2], 0.10)


def test_log_returns_simple_sequence():
    prices = pd.Series([100.0, 110.0])
    lr = log_returns(prices)
    assert np.isnan(lr.iloc[0])
    assert np.isclose(lr.iloc[1], np.log(1.1))


def test_cumulative_returns_two_steps():
    returns = pd.Series([np.nan, 0.10, 0.10])
    cr = cumulative_returns(returns)
    # cumprod propagates NaN at first element; second is 10%, third is 21%
    assert np.isnan(cr.iloc[0])
    assert np.isclose(cr.iloc[1], 0.10)
    assert np.isclose(cr.iloc[2], 0.21)
