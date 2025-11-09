import datetime as dt
import sys
import types

dummy_downloader = types.ModuleType("sec_edgar_downloader")
dummy_downloader.Downloader = object  # type: ignore[attr-defined]
sys.modules.setdefault("sec_edgar_downloader", dummy_downloader)

tqdm_module = types.ModuleType("tqdm")
tqdm_module.tqdm = lambda iterable=None, **kwargs: iterable
sys.modules.setdefault("tqdm", tqdm_module)

from tools.finance_real_eval import _strict_time_split


def test_strict_time_split_enforces_issuer_ratio():
    rows = [
        {"ticker": "AAA", "filed": dt.date(2018, 6, 1), "text": ""},
        {"ticker": "AAA", "filed": dt.date(2020, 6, 1), "text": ""},
        {"ticker": "AAA", "filed": dt.date(2021, 6, 1), "text": ""},
        {"ticker": "BBB", "filed": dt.date(2019, 5, 1), "text": ""},
        {"ticker": "BBB", "filed": dt.date(2021, 5, 1), "text": ""},
        {"ticker": "CCC", "filed": dt.date(2022, 7, 1), "text": ""},
        {"ticker": "DDD", "filed": dt.date(2023, 3, 1), "text": ""},
    ]

    train_idx, test_idx, metadata = _strict_time_split(rows, min_test=2, issuer_ratio=0.75)

    assert metadata["cutoff_year"] == 2021
    assert metadata["initial_cutoff_year"] == 2021
    assert metadata["adjustments"] == []

    train_tickers = {rows[idx]["ticker"] for idx in train_idx}
    test_tickers = {rows[idx]["ticker"] for idx in test_idx}

    assert train_tickers == {"AAA", "BBB"}
    assert test_tickers == {"CCC", "DDD"}
    assert metadata["issuer_ratio"] == 1.0
    assert len(test_idx) == 2
