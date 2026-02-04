from __future__ import annotations

from datetime import date, timedelta
from io import StringIO
import urllib.request

import pandas as pd


STOOQ_SPX_DAILY_CSV_URL = "https://stooq.com/q/d/l/?s=^spx&i=d"


def fetch_spx_last_12_months() -> pd.DataFrame:
    """
    Fetch last ~12 months of SPX daily OHLC from Stooq.

    Returns a DataFrame with columns: date, open, high, low, close
    (lowercase), indexed by date.
    """
    with urllib.request.urlopen(STOOQ_SPX_DAILY_CSV_URL, timeout=20) as resp:
        raw = resp.read().decode("utf-8")

    df = pd.read_csv(StringIO(raw))
    df.columns = [c.strip().lower() for c in df.columns]

    # Stooq columns typically: Date, Open, High, Low, Close, Volume
    # We only need OHLC.
    keep = ["date", "open", "high", "low", "close"]
    missing = set(keep) - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected Stooq schema, missing: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values("date")

    cutoff = pd.Timestamp(date.today() - timedelta(days=365))
    df = df[df["date"] >= cutoff].copy()

    return df.set_index("date")[keep[1:]]
