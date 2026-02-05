from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from spx_model.gaps import gap_return, intraday_return, daily_range
from spx_model.live_data import get_cached_spx_last_12_months


@dataclass(frozen=True)
class GapBuckets:
    edges: tuple[float, ...] = (
        -np.inf,
        -0.02,
        -0.01,
        -0.005,
        0.0,
        0.005,
        0.01,
        0.02,
        np.inf,
    )
    labels: tuple[str, ...] = (
        "< -2.0%",
        "-2.0% to -1.0%",
        "-1.0% to -0.5%",
        "-0.5% to 0.0%",
        "0.0% to 0.5%",
        "0.5% to 1.0%",
        "1.0% to 2.0%",
        ">= 2.0%",
    )


def summarize_by_gap_bucket(df: pd.DataFrame, buckets: GapBuckets) -> pd.DataFrame:
    df = df.dropna(
        subset=[
            "gap_return",
            "intraday_return",
            "daily_range",
            "next_day_return",
        ]
    ).copy()

    df["gap_bucket"] = pd.cut(
        df["gap_return"],
        bins=list(buckets.edges),
        labels=list(buckets.labels),
        right=False,
        include_lowest=True,
    )

    grouped = df.groupby("gap_bucket", observed=True)

    out = pd.DataFrame(
        {
            "n": grouped.size(),
            "gap_mean_%": grouped["gap_return"].mean() * 100.0,
            "intraday_mean_%": grouped["intraday_return"].mean() * 100.0,
            "intraday_median_%": grouped["intraday_return"].median() * 100.0,
            "range_mean_%": grouped["daily_range"].mean() * 100.0,
            "range_median_%": grouped["daily_range"].median() * 100.0,
            "next_day_mean_%": grouped["next_day_return"].mean() * 100.0,
            "next_day_median_%": grouped["next_day_return"].median() * 100.0,
        }
    )

    return out.reset_index()


def main() -> None:
    df_live = get_cached_spx_last_12_months(
        Path("data/spx_last12m.csv"),
        max_age_hours=24,
    )

    df = pd.DataFrame(
        {
            "open": df_live["open"],
            "high": df_live["high"],
            "low": df_live["low"],
            "close": df_live["close"],
        }
    )

    df["gap_return"] = gap_return(df["close"], df["open"])
    df["intraday_return"] = intraday_return(df["open"], df["close"])
    df["daily_range"] = daily_range(df["open"], df["high"], df["low"])
    df["next_day_return"] = (df["close"].shift(-1) / df["close"]) - 1.0

    summary = summarize_by_gap_bucket(df, GapBuckets())

    pd.set_option("display.width", 140)
    pd.set_option("display.max_rows", 200)

    pretty = summary.copy()
    for col in [
        "gap_mean_%",
        "intraday_mean_%",
        "intraday_median_%",
        "range_mean_%",
        "range_median_%",
        "next_day_mean_%",
        "next_day_median_%",
    ]:
        pretty[col] = pretty[col].round(4)

    print("\nSPX gap bucket summary (same-day intraday, range, and next-day return):\n")
    print(pretty.to_string(index=False))

    print("\nSample computed rows:\n")
    cols = [
        "open",
        "high",
        "low",
        "close",
        "gap_return",
        "intraday_return",
        "daily_range",
        "next_day_return",
    ]
    print(df[cols].tail(10).to_string())


if __name__ == "__main__":
    main()

