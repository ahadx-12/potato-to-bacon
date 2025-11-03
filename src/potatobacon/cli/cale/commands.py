"""Command line entry points for CALE validation workflows."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Tuple

import click

ROOT = Path(__file__).resolve()
for _ in range(5):
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from .orchestrator import CALEValidator, DEFAULT_TICKERS, DEFAULT_USER_AGENT

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


@click.group()
def cli() -> None:
    """CALE command suite."""


@cli.command()
@click.option(
    "--ticker",
    "tickers",
    multiple=True,
    help="Ticker symbol to include. Provide multiple --ticker flags to expand the cohort.",
)
@click.option(
    "--event-date",
    default=None,
    help="ISO-8601 date used to select filings (latest filings prior to this date are used).",
)
@click.option(
    "--manifest-limit",
    type=int,
    default=None,
    help="Maximum number of filings to analyse (after caching).",
)
@click.option(
    "--section-limit",
    type=int,
    default=3,
    show_default=True,
    help="Number of top covenant sections to include per filing in the report.",
)
@click.option(
    "--pair-limit",
    type=int,
    default=5,
    show_default=True,
    help="Number of law conflict pairs to include per filing in the report.",
)
@click.option(
    "--user-agent",
    default=DEFAULT_USER_AGENT,
    show_default=True,
    help="User-Agent header to send to the SEC API.",
)
@click.option(
    "--report-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path for writing the consolidated JSON report.",
)
@click.option(
    "--refresh-cache",
    is_flag=True,
    help="Ignore cached manifest, document and embedding artifacts for this run.",
)
def validate(
    tickers: Tuple[str, ...],
    event_date: str | None,
    manifest_limit: int | None,
    section_limit: int,
    pair_limit: int,
    user_agent: str,
    report_path: Path | None,
    refresh_cache: bool,
) -> None:
    """Fetch SEC filings, extract finance signals and run CALE law checks."""

    cohort = tuple(tickers) if tickers else DEFAULT_TICKERS
    validator = CALEValidator(
        tickers=cohort,
        event_date=event_date,
        user_agent=user_agent,
        read_cache=not refresh_cache,
    )
    report = validator.generate_report(
        section_limit=section_limit,
        pair_limit=pair_limit,
        manifest_limit=manifest_limit,
    )
    summary = report["summary"]
    click.echo(
        "[CALE] Processed {filings} filings (pairs analysed: {pairs}). Avg CI={avg:.3f}".format(
            filings=summary.get("filings_processed", 0),
            pairs=summary.get("pairs_analyzed", 0),
            avg=summary.get("avg_conflict_intensity", 0.0),
        )
    )
    max_conflict = summary.get("max_conflict")
    if isinstance(max_conflict, dict):
        click.echo(
            "[CALE] Highest conflict {score:.3f} for {ticker} {form} ({filed}).".format(
                score=max_conflict.get("score", 0.0),
                ticker=max_conflict.get("ticker", "?"),
                form=max_conflict.get("form", "?"),
                filed=max_conflict.get("filed", "?"),
            )
        )
    if report_path:
        validator.write_report(report, report_path)
    else:
        click.echo(click.style("[CALE] Report not written (use --report-path to persist)", fg="yellow"))
