import pandas as pd

from tools import finance_dedup


def _make_manifest(paths, tickers, dates):
    return pd.DataFrame(
        {
            "path": paths,
            "ticker": tickers,
            "filed": dates,
            "form": ["10-K"] * len(paths),
        }
    )


def test_md5_duplicate_removed():
    manifest = _make_manifest(
        [
            "tests/finance/fixtures/phase1_docs/baseline.html",
            "tests/finance/fixtures/phase1_docs/duplicate.html",
        ],
        ["AAA", "AAA"],
        ["2018-06-01", "2019-06-01"],
    )
    kept, quarantine, decisions = finance_dedup.deduplicate_manifest(manifest, threshold=0.85)

    assert len(kept) == 1
    assert len(quarantine) == 1
    assert any(decision.method == "md5_raw" for decision in decisions)
    assert quarantine.iloc[0]["dedup_method"] == "md5_raw"


def test_tfidf_duplicate_removed():
    manifest = _make_manifest(
        [
            "tests/finance/fixtures/phase1_docs/baseline.html",
            "tests/finance/fixtures/phase1_docs/tfidf_variant.html",
        ],
        ["BBB", "BBB"],
        ["2020-05-10", "2021-04-02"],
    )
    kept, quarantine, decisions = finance_dedup.deduplicate_manifest(manifest, threshold=0.85)

    assert len(kept) == 1
    assert len(quarantine) == 1
    assert any(decision.method == "tfidf" for decision in decisions)
    assert quarantine.iloc[0]["dedup_method"] == "tfidf"


def test_minhash_duplicate_removed(monkeypatch):
    manifest = _make_manifest(
        [
            "tests/finance/fixtures/phase1_docs/minhash_a.html",
            "tests/finance/fixtures/phase1_docs/minhash_b.html",
        ],
        ["CCC", "CCC"],
        ["2021-07-01", "2022-07-01"],
    )

    def empty_tfidf(records, threshold):  # type: ignore[unused-arg]
        return []

    monkeypatch.setattr(finance_dedup, "_tfidf_near_duplicates", empty_tfidf)
    kept, quarantine, decisions = finance_dedup.deduplicate_manifest(manifest, threshold=0.85)

    assert len(kept) == 1
    assert len(quarantine) == 1
    assert any(decision.method == "minhash" for decision in decisions)
    assert quarantine.iloc[0]["dedup_method"] == "minhash"
