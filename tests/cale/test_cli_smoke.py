from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("typer")
pytest.importorskip("torch")

from typer.testing import CliRunner

from potatobacon.cli.main import app


def test_cli_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "law" in result.stdout


def test_cli_train_demo() -> None:
    weights_path = Path("models/cale_weights.pt")
    if weights_path.exists():
        weights_path.unlink()

    runner = CliRunner()
    result = runner.invoke(app, ["law", "train", "--epochs", "1", "--use-demo"], catch_exceptions=False)
    assert result.exit_code == 0
    assert weights_path.exists()
