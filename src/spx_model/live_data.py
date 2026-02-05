from __future__ import annotations

from datetime import date, timedelta
from io import StringIO
from pathlib import Path
import urllib.request

import pandas as pd


STOOQ_SPX_DAILY_CSV_URL = "https://stooq.com/q/d/l/?s=^spx&i=d"


def fetch_spx_last_12_months() -> pd.DataFrame:
    """
    Fetch last ~12 months of SPX daily OHLC from Stooq.

    Returns a DataFrame indexed by date with columns: open, high, low, close
    """
    with urllib.request.urlopen(STOOQ_SPX_DAILY_CSV_URL, timeout=20) as resp:
        raw = resp.read().decode("utf-8")

    df = pd.read_csv(StringIO(raw))
    df.columns = [c.strip().lower() for c in df.columns]

    keep = ["date", "open", "high", "low", "close"]
    missing = set(keep) - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected Stooq schema, missing: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values("date")

    cutoff = pd.Timestamp(date.today() - timedelta(days=365))
    df = df[df["date"] >= cutoff].copy()

    df = df.set_index("date")[["open", "high", "low", "close"]]
    return df


def get_cached_spx_last_12_months(
    cache_path: Path = Path("data/spx_last12m.csv"),
    max_age_hours: int = 24,
) -> pd.DataFrame:
    """
    Load SPX last-12-months OHLC from a local CSV cache if it's fresh enough,
    otherwise fetch fresh data and overwrite the cache.

    Cache format: CSV with columns Date, Open, High, Low, Close (capitalized),
    matching our ingestion expectations.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        mtime = pd.Timestamp(cache_path.stat().st_mtime, unit="s")
        age = pd.Timestamp.now() - mtime
        if age <= pd.Timedelta(hours=max_age_hours):
            df = pd.read_csv(cache_path)
            df.columns = [c.strip().lower() for c in df.columns]
            df["date"] = pd.to_datetime(df["date"], errors="raise")
            return df.sort_values("date").set_index("date")[["open", "high", "low", "close"]]

    # Fetch fresh + write cache
    df = fetch_spx_last_12_months()

    out = df.reset_index().rename(
        columns={"date": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close"}
    )
    out.to_csv(cache_path, index=False)

    return df
