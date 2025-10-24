from __future__ import annotations

from click.testing import CliRunner

from potatobacon.cli.main import cli


def test_cli_help() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "law" in result.stdout


def test_cli_sanity_check() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["law", "sanity-check"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "suggestions" in result.stdout
