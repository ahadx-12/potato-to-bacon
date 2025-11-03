"""Command-line interface for Potato-to-Bacon."""

from __future__ import annotations

import json

import click

from ..cale.engine import CALEEngine


DEFAULT_RULE1_TEXT = "Organizations MUST collect personal data IF consent."
DEFAULT_RULE2_TEXT = "Security agencies MUST NOT collect personal data IF emergency."
DEFAULT_JURISDICTION = "Canada.Federal"


@click.group()
def cli() -> None:
    """Potato-to-Bacon command suite."""


@cli.group()
def law() -> None:
    """CALE helpers."""


@law.command("sanity-check")
@click.option(
    "--rule1",
    "rule1_text",
    default=DEFAULT_RULE1_TEXT,
    show_default=True,
    help="Text of the first rule.",
)
@click.option(
    "--rule2",
    "rule2_text",
    default=DEFAULT_RULE2_TEXT,
    show_default=True,
    help="Text of the second rule.",
)
@click.option(
    "--jurisdiction",
    "jurisdiction",
    default=DEFAULT_JURISDICTION,
    show_default=True,
    help="Jurisdiction shared by both rules.",
)
@click.option(
    "--rule1-statute",
    default="CLI Statute 1",
    show_default=True,
    help="Statute name for the first rule.",
)
@click.option(
    "--rule1-section",
    default="1",
    show_default=True,
    help="Section identifier for the first rule.",
)
@click.option(
    "--rule1-year",
    default=2000,
    show_default=True,
    type=int,
    help="Enactment year for the first rule.",
)
@click.option(
    "--rule2-statute",
    default="CLI Statute 2",
    show_default=True,
    help="Statute name for the second rule.",
)
@click.option(
    "--rule2-section",
    default="2",
    show_default=True,
    help="Section identifier for the second rule.",
)
@click.option(
    "--rule2-year",
    default=2001,
    show_default=True,
    type=int,
    help="Enactment year for the second rule.",
)
def sanity_check(
    rule1_text: str,
    rule2_text: str,
    jurisdiction: str,
    rule1_statute: str,
    rule1_section: str,
    rule1_year: int,
    rule2_statute: str,
    rule2_section: str,
    rule2_year: int,
) -> None:
    """Run the CALE engine for the provided rule pair and emit structured JSON."""

    engine = CALEEngine()

    rule1_payload = {
        "id": "CLI_RULE_1",
        "text": rule1_text,
        "jurisdiction": jurisdiction,
        "statute": rule1_statute,
        "section": rule1_section,
        "enactment_year": rule1_year,
    }
    rule2_payload = {
        "id": "CLI_RULE_2",
        "text": rule2_text,
        "jurisdiction": jurisdiction,
        "statute": rule2_statute,
        "section": rule2_section,
        "enactment_year": rule2_year,
    }

    result = engine.suggest(rule1_payload, rule2_payload)
    payload = {
        "rule1": rule1_payload,
        "rule2": rule2_payload,
        **result,
    }

    click.echo(json.dumps(payload, indent=2))


if __name__ == "__main__":
    cli()
