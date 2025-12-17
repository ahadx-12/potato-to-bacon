from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List

from potatobacon.law.solver_z3 import PolicyAtom
from potatobacon.tariff.context_registry import _atom_to_dict, _rule_obj_to_atom


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and register a tariff law context")
    parser.add_argument("--context-id", required=True, help="Unique context identifier")
    parser.add_argument("--jurisdiction", required=True, help="Jurisdiction label")
    parser.add_argument("--effective-from", required=True, help="ISO date the context becomes active")
    parser.add_argument("--effective-to", default=None, help="ISO date the context expires")
    parser.add_argument("--description", required=True, help="Human-readable context description")
    parser.add_argument("--rules-in", required=True, help="Path to JSON rule file")
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "contexts"),
        help="Output directory for manifests and rules",
    )
    return parser.parse_args()


def _load_and_validate_rules(path: Path) -> List[PolicyAtom]:
    with path.open("r", encoding="utf-8") as handle:
        raw_rules: List[Any] = json.load(handle)

    atoms: list[PolicyAtom] = []
    for idx, rule in enumerate(raw_rules):
        try:
            atoms.append(_rule_obj_to_atom(rule))
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid rule at index {idx}: {exc}") from exc
    return atoms


def main() -> None:
    args = _parse_args()

    rules_path = Path(args.rules_in)
    if not rules_path.exists():
        raise FileNotFoundError(rules_path)

    atoms = _load_and_validate_rules(rules_path)
    rule_payload = [_atom_to_dict(atom) for atom in atoms]

    out_dir = Path(args.out_dir)
    rules_out_dir = out_dir / "rules"
    manifests_out_dir = out_dir / "manifests"
    rules_out_dir.mkdir(parents=True, exist_ok=True)
    manifests_out_dir.mkdir(parents=True, exist_ok=True)

    rules_out_path = rules_out_dir / f"{args.context_id}_rules.json"
    with rules_out_path.open("w", encoding="utf-8") as handle:
        json.dump(rule_payload, handle, indent=2)

    manifest = {
        "context_id": args.context_id,
        "domain": "tariff",
        "jurisdiction": args.jurisdiction,
        "effective_from": args.effective_from,
        "effective_to": args.effective_to,
        "description": args.description,
        "loader": {
            "type": "json_rules",
            "rules_file": f"rules/{rules_out_path.name}",
        },
    }

    manifest_out_path = manifests_out_dir / f"{args.context_id}.json"
    with manifest_out_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


if __name__ == "__main__":
    main()
