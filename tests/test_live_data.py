import os
import pytest

from spx_model.live_data import fetch_spx_last_12_months


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_DATA") != "1",
    reason="Set RUN_LIVE_DATA=1 to run live-data integration tests",
)


def test_fetch_spx_last_12_months_has_reasonable_shape():
    df = fetch_spx_last_12_months()

    # Basic expectations (not too strict)
    assert len(df) >= 200  # ~252 trading days/year; allow for holidays/source quirks
    assert set(df.columns) == {"open", "high", "low", "close"}

    # Sanity: high >= low, prices positive
    assert (df["open"] > 0).all()
    assert (df["close"] > 0).all()
    assert (df["high"] >= df["low"]).all()
