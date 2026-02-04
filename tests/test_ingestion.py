from pathlib import Path
import pytest

from spx_model.ingestion import load_spx_ohlc_csv


def write_csv(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "spx.csv"
    p.write_text(content)
    return p


def test_load_spx_ohlc_happy_path(tmp_path: Path):
    p = write_csv(
        tmp_path,
        "Date,Open,High,Low,Close\n"
        "2024-01-02,100,110,95,105\n"
        "2024-01-03,105,112,104,111\n",
    )
    ohlc = load_spx_ohlc_csv(p)
    assert float(ohlc.close.iloc[0]) == 105.0
    assert len(ohlc.close) == 2


def test_missing_required_column_raises(tmp_path: Path):
    p = write_csv(
        tmp_path,
        "Date,Open,High,Low\n"
        "2024-01-02,100,110,95\n",
    )
    with pytest.raises(ValueError, match="Missing required columns"):
        load_spx_ohlc_csv(p)


def test_duplicate_dates_raises(tmp_path: Path):
    p = write_csv(
        tmp_path,
        "Date,Open,High,Low,Close\n"
        "2024-01-02,100,110,95,105\n"
        "2024-01-02,101,111,96,106\n",
    )
    with pytest.raises(ValueError, match="Duplicate dates"):
        load_spx_ohlc_csv(p)


def test_non_positive_price_raises(tmp_path: Path):
    p = write_csv(
        tmp_path,
        "Date,Open,High,Low,Close\n"
        "2024-01-02,0,110,95,105\n",
    )
    with pytest.raises(ValueError, match="Non-positive open"):
        load_spx_ohlc_csv(p)


def test_high_low_sanity_checks(tmp_path: Path):
    # high < close should fail
    p = write_csv(
        tmp_path,
        "Date,Open,High,Low,Close\n"
        "2024-01-02,100,104,95,105\n",
    )
    with pytest.raises(ValueError, match="High lower than open/close"):
        load_spx_ohlc_csv(p)
