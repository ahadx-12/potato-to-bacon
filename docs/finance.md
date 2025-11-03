# Real-world SEC data workflows

This repository ships a pair of helpers that make it easy to refresh the real
SEC manifests used in our finance smoke tests.

## Download the latest filings

```
python tools/fetch_sec_real.py
```

The downloader keeps historical rows in `reports/realworld/manifest.csv` and
only appends new entries when it encounters a previously unseen accession. Each
row includes a digest of the primary filing along with the accession identifier
and inferred CIK so downstream jobs can de-duplicate or validate downloads.

## Label filings for downstream analysis

```
python tools/sec_manifest.py [--leverage path/to/leverage.csv] [--threshold 4.5]
```

The manifest enricher tags filings as `distressed`, `control`, or `unknown`
using a watchlist of issuers and, when provided, a leverage CSV with
`ticker`/`net_debt_to_ebitda` columns. Quality flags note whether the
classification relied on a watchlist hit, a leverage rule, or lacks sufficient
data. Flags accumulate (semicolon separated) so manual reviewers can layer on
additional annotations without losing the automated provenance.

## Incremental updates

Both helpers rewrite `reports/realworld/manifest.csv` in-place. They preserve
previous rows and quality signals, enabling longitudinal comparisons across
filings as the manifest evolves.
