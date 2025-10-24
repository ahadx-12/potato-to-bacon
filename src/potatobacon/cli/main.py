"""Command-line interface for Potato-to-Bacon."""

from __future__ import annotations

import json

import click

from ..cale.bootstrap import build_services


@click.group()
def cli() -> None:
    """Potato-to-Bacon command suite."""


@cli.group()
def law() -> None:
    """CALE helpers."""


@law.command("sanity-check")
def sanity_check() -> None:
    """Run a deterministic CALE amendment suggestion over a canned example."""

    services = build_services()

    t1 = "Citizens MUST present ID when voting in federal elections."
    t2 = "Citizens MAY vote without presenting ID if known to election officials."

    rule1 = services.parser.parse(
        t1,
        {
            "id": "CLI_SANITY_R1",
            "jurisdiction": "Canada",
            "statute": "Elections Act",
            "section": "1",
            "enactment_year": 2000,
        },
    )
    rule2 = services.parser.parse(
        t2,
        {
            "id": "CLI_SANITY_R2",
            "jurisdiction": "Canada",
            "statute": "Elections Act",
            "section": "1B",
            "enactment_year": 2014,
        },
    )

    rule1 = services.feature_engine.populate(rule1)
    rule2 = services.feature_engine.populate(rule2)

    conflict = services.checker.check_conflict(rule1, rule2)
    analysis = services.calculator.compute_multiperspective(rule1, rule2, conflict)
    output = services.suggester.suggest_amendment(rule1, rule2, analysis)

    click.echo(json.dumps(output, indent=2))


if __name__ == "__main__":
    cli()
