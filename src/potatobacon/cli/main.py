"""Command line interface for Potato-to-Bacon."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Tuple

import typer

from potatobacon.cale.runtime import bootstrap, get_services
from potatobacon.cale.train import CALETrainer, LegalConflictDataset
from potatobacon.cale.types import LegalRule

app = typer.Typer(help="Utilities for working with Potato-to-Bacon")
law_app = typer.Typer(help="CALE training and evaluation utilities")
app.add_typer(law_app, name="law")


def _ensure_services():
    services = bootstrap()
    if services is None:
        try:
            services = get_services()
        except RuntimeError as exc:  # pragma: no cover - defensive
            raise typer.BadParameter(
                "CALE runtime not initialised; unset CALE_DISABLE_STARTUP_INIT"
            ) from exc
    return services


def _load_demo_dataset(services) -> LegalConflictDataset:
    dataset = LegalConflictDataset(
        csv_path=Path("data/cale/expert_labels.csv"),
        corpus=services.corpus,
        symbolic=services.symbolic,
    )
    if len(dataset) == 0:
        raise typer.Exit(code=1)
    return dataset


def _feature_dim(corpus: List[LegalRule]) -> int:
    if not corpus:
        raise typer.BadParameter("CALE corpus is empty")
    return len(corpus[0].feature_vector)


@law_app.command("train")
def train(
    epochs: int = typer.Option(10, help="Number of training epochs"),
    ssl: bool = typer.Option(False, "--ssl/--no-ssl", help="Enable semi-supervised loss"),
    graph: bool = typer.Option(False, "--graph/--no-graph", help="Enable graph regularisation"),
    use_demo: bool = typer.Option(
        False, "--use-demo/--no-demo", help="Train using bundled demo dataset"
    ),
):
    """Train the CALE model and persist the weights to ``models/``."""

    services = _ensure_services()
    if use_demo:
        dataset = _load_demo_dataset(services)
    else:
        raise typer.BadParameter("Provide --use-demo or a custom dataset implementation")

    try:
        trainer = CALETrainer(_feature_dim(services.corpus))
    except RuntimeError as exc:  # pragma: no cover - optional dependency missing
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    history = trainer.train(
        dataset,
        symbolic=services.symbolic,
        corpus=services.corpus,
        num_epochs=epochs,
        use_ssl=ssl,
        use_graph=graph,
    )
    typer.echo("Training complete. Final losses:")
    for key, values in history.items():
        if values:
            typer.echo(f"  {key}: {values[-1]:.4f}")
    typer.echo("Weights saved to models/cale_weights.pt")


@law_app.command("eval")
def evaluate(
    pairs: int = typer.Option(10, help="Number of random pairs to evaluate")
) -> None:
    """Run a lightweight evaluation using the CCS calculator."""

    services = _ensure_services()
    rng = random.Random(42)
    if pairs <= 0:
        raise typer.BadParameter("pairs must be positive")
    sample_pairs: List[Tuple[LegalRule, LegalRule]] = []
    for _ in range(pairs):
        r1, r2 = rng.sample(services.corpus, 2)
        sample_pairs.append((r1, r2))

    results = []
    for r1, r2 in sample_pairs:
        ci = services.symbolic.check_conflict(r1, r2)
        analysis = services.ccs.compute_multiperspective(r1, r2, ci)
        results.append(
            {
                "rule1": r1.id,
                "rule2": r2.id,
                "ccs_pragmatic": analysis.CCS_pragmatic,
                "ci": analysis.CI,
            }
        )
    typer.echo(json.dumps(results, indent=2))


if __name__ == "__main__":
    app()
