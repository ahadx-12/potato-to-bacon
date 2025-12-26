#!/usr/bin/env python3
"""Deterministic test runner with optional coverage."""

from __future__ import annotations

import subprocess
import sys


def _has_pytest_cov() -> bool:
    try:
        import pytest_cov  # noqa: F401

        return True
    except Exception:
        pass

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and "cov" in result.stdout


def main() -> int:
    base_cmd = [sys.executable, "-m", "pytest", "-q", "--maxfail=1", "--disable-warnings"]
    if _has_pytest_cov():
        base_cmd.extend(["--cov=src/potatobacon", "--cov-report=term-missing"])

    result = subprocess.run(base_cmd, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
