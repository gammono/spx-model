from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass(frozen=True)
class OHLC:
    """Daily OHLC time series indexed by date."""
    open: pd.Series
    high: pd.Series
    low: pd.Series
    close: pd.Series


REQUIRED_COLUMNS = {"date", "open", "high", "low", "close"}


def load_spx_ohlc_csv(path: Path) -> OHLC:
    """
    Load SPX daily OHLC data from a CSV file.

    Expectations:
    - Columns: Date, Open, High, Low, Close (case-insensitive)
    - Date parseable to datetime
    - No duplicate dates
    - Sorted by date ascending (we enforce sorting)
    - Prices must be positive
    - Basic OHLC sanity checks (high >= open/close, low <= open/close)
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        bad_rows = df[df["date"].isna()].index.tolist()[:5]
        raise ValueError(f"Unparseable date values at rows: {bad_rows} (showing up to 5)")

    if df["date"].duplicated().any():
        dupes = df.loc[df["date"].duplicated(), "date"].dt.strftime("%Y-%m-%d").tolist()[:5]
        raise ValueError(f"Duplicate dates found: {dupes} (showing up to 5)")

    df = df.sort_values("date").set_index("date")

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            bad = df[df[col].isna()].index[:5].strftime("%Y-%m-%d").tolist()
            raise ValueError(f"Non-numeric {col} values on dates: {bad} (showing up to 5)")
        if (df[col] <= 0).any():
            bad = df[df[col] <= 0].index[:5].strftime("%Y-%m-%d").tolist()
            raise ValueError(f"Non-positive {col} values on dates: {bad} (showing up to 5)")

    if (df["high"] < df[["open", "close"]].max(axis=1)).any():
        bad = df[df["high"] < df[["open", "close"]].max(axis=1)].index[:5].strftime("%Y-%m-%d").tolist()
        raise ValueError(f"High lower than open/close on dates: {bad} (showing up to 5)")

    if (df["low"] > df[["open", "close"]].min(axis=1)).any():
        bad = df[df["low"] > df[["open", "close"]].min(axis=1)].index[:5].strftime("%Y-%m-%d").tolist()
        raise ValueError(f"Low higher than open/close on dates: {bad} (showing up to 5)")

    return OHLC(
        open=df["open"].copy(),
        high=df["high"].copy(),
        low=df["low"].copy(),
        close=df["close"].copy(),
    )
