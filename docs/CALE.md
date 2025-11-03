# CALE Validation Workflow

The CALE command line entry point orchestrates the end-to-end validation
pipeline that powers the finance extraction and law consistency checks.  Use it
to download a focused cohort of SEC filings, score covenant-rich sections, run
CALE's legal reasoning engine, and emit a consolidated JSON report.

```bash
python -m potatobacon.cli.cale validate \
    --ticker MPW --ticker BBBY --ticker AAPL \
    --report-path reports/cale/latest.json
```

## Prerequisites

* Python 3.11 or newer.
* Network access for reaching `https://data.sec.gov`.
* The `requests` dependency (installed through the project requirements).
* When running from a source checkout, export `PYTHONPATH=src` so the
  `potatobacon` package is importable.
* A descriptive User-Agent string (override with `--user-agent`) so SEC traffic
  is compliant with EDGAR guidelines.

The command automatically shares CALE's bootstrap services between finance and
law analyses.  No additional service startup is required.

## What the CLI Does

1. **Manifest assembly** – For each ticker the CLI queries the SEC submissions
   API, selects the latest 10-K and 10-Q filings prior to the supplied
   `--event-date` (defaults to today), and persists the manifest under
   `data/cache/manifest/`.
2. **Document parsing** – HTML filings are parsed into block-structured
   documents.  Parsed structures live under `data/cache/docs/` and are
   invalidated automatically when the underlying HTML file changes.
3. **Finance extraction** – Covenant sections are ranked via
   `potatobacon.cale.finance.sectionizer`.  Each section is analysed with
   `extract_numeric_covenants` to surface key metrics.
4. **Law checks** – Obligation/permission sentence pairs are mined with
   `tools.finance_extract.extract_pairs_from_html`.  Each pair is passed to
   `CALEEngine.suggest` to compute conflict scores and proposed amendments.
5. **Reporting** – Results are merged into a single JSON payload containing a
   manifest, per-filing finance summaries, and top-ranked law conflicts.  If
   `--report-path` is omitted the report is printed to stdout but not written to
disk.

Typical runs against the default six tickers complete in 1–2 minutes once the
caches are warm.  The initial run may take longer due to SEC downloads and
embedding generation.

## Cache Behaviour

* **Manifest cache**: keyed by ticker cohort, event date, and user agent.  It
  expires after six hours.  Use `--refresh-cache` to force re-computation.
* **Document cache**: keyed by source path, mtime, and file size.  Regenerated
  automatically if the HTML changes.
* **Embedding cache**: stored under `data/cache/embeddings/` with stable hashes
  of the embedded text.  Subsequent runs reuse vectors for identical sentences.

All caches reside in `data/cache/`.  Delete the directory or pass
`--refresh-cache` to invalidate everything for a run.

## Interpreting the Report

The JSON report contains three primary sections:

* `summary` – high-level metrics such as the number of filings processed,
  aggregate obligation/permission pairs analysed, the average conflict
  intensity, and (if present) the most severe conflict identified.
* `manifest` – the normalized list of filings, including accession number,
  primary document, and local path.
* `filings` – one object per filing with ranked finance sections and the top
  law conflicts.  Each conflict entry includes the original sentences, CALE's
  conflict metrics, amendment suggestions, and any numeric covenant signals.

A simple way to inspect the report after the run is to open the generated JSON:

```bash
jq '.summary, .filings[0]' reports/cale/latest.json
```

For iterative experimentation rerun the CLI with different `--ticker`,
`--pair-limit`, or `--section-limit` values.  Caches keep subsequent runs fast,
and `--refresh-cache` resets the pipeline when you need a clean slate.
