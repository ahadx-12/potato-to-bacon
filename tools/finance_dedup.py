#!/usr/bin/env python3
"""Deduplicate SEC manifests and maintain a reusable LSH cache."""

from __future__ import annotations

import argparse
import csv
import json
import re
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import pandas as pd

DEFAULT_MANIFEST = Path("data/sec_real/manifest.csv")
DEFAULT_OUTPUT = Path("data/sec_real/manifest_dedup.csv")
DEFAULT_QUARANTINE = Path("data/sec_real/manifest_quarantine.csv")
DEFAULT_CACHE = Path("data/sec_real/lsh_cache.json")
DEFAULT_MINHASH_PERMUTATIONS = 64
DEFAULT_LSH_BAND_SIZE = 8
DEFAULT_DUPLICATE_THRESHOLD = 0.9
DEFAULT_MIN_SPLIT = 80

TEXT_ENCODING_FALLBACKS = ("utf-8", "latin-1", "cp1252")
TOKEN_PATTERN = re.compile(r"[a-z0-9]{3,}")


@dataclass
class ManifestRow:
    data: Dict[str, str]

    @property
    def md5(self) -> str:
        return self.data.get("md5", "")

    @property
    def path(self) -> Path:
        return Path(self.data.get("path", ""))

    @property
    def filed(self) -> str:
        return self.data.get("filed", "")

    @property
    def label(self) -> str:
        return self.data.get("label", "") or "unknown"

    @property
    def key(self) -> str:
        return self.md5 or f"{self.data.get('ticker','')}::{self.path.name}"


@dataclass(frozen=True)
class DedupDecision:
    method: str
    primary: str
    duplicate: str
    score: float


class SimpleMinHashLSH:
    def __init__(
        self,
        cache_path: Path,
        num_perm: int = DEFAULT_MINHASH_PERMUTATIONS,
        band_size: int = DEFAULT_LSH_BAND_SIZE,
    ) -> None:
        if num_perm % band_size != 0:
            raise ValueError("num_perm must be divisible by band_size")
        self.cache_path = cache_path
        self.num_perm = num_perm
        self.band_size = band_size
        self._signatures: Dict[str, List[int]] = {}
        self._bands: Dict[str, Set[str]] = defaultdict(set)
        self._load()

    # -- cache management -------------------------------------------------
    def _load(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            with self.cache_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return
        signatures = payload.get("signatures", {}) if isinstance(payload, dict) else {}
        for doc_id, sig in signatures.items():
            if not isinstance(sig, list):
                continue
            signature = [int(value) for value in sig]
            self._signatures[doc_id] = signature
            self._index_signature(doc_id, signature)

    def save(self) -> None:
        if not self._signatures:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "num_perm": self.num_perm,
                    "band_size": self.band_size,
                    "signatures": self._signatures,
                },
                handle,
                indent=2,
                sort_keys=True,
            )

    # -- LSH primitives ---------------------------------------------------
    def _index_signature(self, doc_id: str, signature: Sequence[int]) -> None:
        for offset in range(0, self.num_perm, self.band_size):
            band = signature[offset : offset + self.band_size]
            band_key = f"{offset}:{','.join(map(str, band))}"
            self._bands[band_key].add(doc_id)

    def _compute_signature(self, shingles: Sequence[int]) -> Tuple[int, ...]:
        if not shingles:
            return tuple([2**32 - 1] * self.num_perm)
        signature: List[int] = []
        for idx in range(self.num_perm):
            values = [(hash((idx, shingle)) & 0xFFFFFFFF) for shingle in shingles]
            signature.append(min(values) if values else 2**32 - 1)
        return tuple(signature)

    def add(self, doc_id: str, shingles: Sequence[int]) -> None:
        signature = list(self._compute_signature(shingles))
        self._signatures[doc_id] = signature
        self._index_signature(doc_id, signature)

    def query(self, shingles: Sequence[int]) -> Set[str]:
        signature = self._compute_signature(shingles)
        candidates: Set[str] = set()
        for offset in range(0, self.num_perm, self.band_size):
            band = signature[offset : offset + self.band_size]
            band_key = f"{offset}:{','.join(map(str, band))}"
            candidates.update(self._bands.get(band_key, set()))
        return candidates


def _load_manifest_rows(manifest_path: Path) -> Tuple[List[ManifestRow], List[str]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest {manifest_path} does not exist")
    df = pd.read_csv(manifest_path)
    records = []
    for row in df.to_dict("records"):
        records.append(ManifestRow(data={key: str(value) if not pd.isna(value) else "" for key, value in row.items()}))
    records.sort(key=lambda item: (item.filed, item.data.get("ticker", ""), item.data.get("form", "")))
    return records, list(df.columns)


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    for encoding in TEXT_ENCODING_FALLBACKS:
        try:
            return path.read_text(encoding=encoding)
        except Exception:
            continue
    return ""


def _tokenize(text: str) -> List[str]:
    lowered = text.lower()
    return TOKEN_PATTERN.findall(lowered)


def _shingles(tokens: Sequence[str], size: int = 5) -> List[int]:
    if len(tokens) < size:
        return []
    shingles: List[int] = []
    for idx in range(len(tokens) - size + 1):
        chunk = " ".join(tokens[idx : idx + size])
        shingles.append(hash(chunk) & 0xFFFFFFFF)
    return shingles


def _jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _cosine(counter_a: Counter[str], counter_b: Counter[str]) -> float:
    if not counter_a or not counter_b:
        return 0.0
    intersection = set(counter_a) & set(counter_b)
    dot = sum(counter_a[token] * counter_b[token] for token in intersection)
    norm_a = sum(value * value for value in counter_a.values()) ** 0.5
    norm_b = sum(value * value for value in counter_b.values()) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _tfidf_near_duplicates(records: Sequence[Dict[str, object]], threshold: float) -> List[Tuple[int, int, float]]:
    matches: List[Tuple[int, int, float]] = []
    for idx, left in enumerate(records):
        left_counts: Counter[str] = left["counts"]  # type: ignore[assignment]
        for jdx in range(idx + 1, len(records)):
            right = records[jdx]
            right_counts: Counter[str] = right["counts"]  # type: ignore[assignment]
            score = _cosine(left_counts, right_counts)
            if score >= threshold:
                matches.append((idx, jdx, score))
    return matches


def _minhash_near_duplicates(records: Sequence[Dict[str, object]], threshold: float) -> List[Tuple[int, int, float]]:
    matches: List[Tuple[int, int, float]] = []
    for idx, left in enumerate(records):
        left_shingles: Sequence[int] = left["shingles"]  # type: ignore[assignment]
        left_tokens = set(left.get("tokens", []))
        for jdx in range(idx + 1, len(records)):
            right = records[jdx]
            right_shingles: Sequence[int] = right["shingles"]  # type: ignore[index]
            right_tokens = set(right.get("tokens", []))
            shingle_score = _jaccard(left_shingles, right_shingles)
            token_score = (
                len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
                if left_tokens and right_tokens
                else 0.0
            )
            score = max(shingle_score, token_score)
            if score >= threshold:
                matches.append((idx, jdx, score))
    return matches


def _apply_quality_flag(row: Dict[str, str], reason: str) -> None:
    existing = [flag for flag in (row.get("quality_flag") or "").split(";") if flag]
    if reason not in existing:
        existing.append(reason)
    row["quality_flag"] = ";".join(existing)


def _ensure_columns(row: Dict[str, str]) -> None:
    for column in ("quality_flag", "split"):
        row.setdefault(column, "")


def _assign_splits(rows: List[Dict[str, str]], min_split: int = DEFAULT_MIN_SPLIT) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["filed"] = pd.to_datetime(df.get("filed", pd.Series([], dtype=str)), errors="coerce")
    df = df.sort_values(["label", "filed", "ticker", "accession"], na_position="last")
    assignments: Dict[int, str] = {}
    counts = {"train": 0, "test": 0}
    label_indices: Dict[str, List[int]] = {}
    for label, group in df.groupby("label"):
        indices = list(group.index)
        label_indices[label] = indices
        label_total = len(indices)
        if label_total == 0:
            continue
        label_test_target = max(1 if label_total > 1 else 0, int(round(label_total * 0.3)))
        label_train_target = max(label_total - label_test_target, 0)
        if label_train_target == 0 and label_total > 1:
            label_train_target = label_total - 1
            label_test_target = 1
        for position, idx in enumerate(indices):
            if position < label_train_target:
                assignments[idx] = "train"
                counts["train"] += 1
            else:
                assignments[idx] = "test"
                counts["test"] += 1

    total = len(df)
    min_split = min(min_split, total // 2) if total < min_split * 2 else min_split

    def _adjust(target_split: str) -> None:
        other_split = "test" if target_split == "train" else "train"
        deficit = min_split - counts[target_split]
        if deficit <= 0:
            return
        for label, indices in label_indices.items():
            for idx in indices:
                if assignments[idx] == other_split:
                    # prevent draining a label entirely from the other split
                    label_other_count = sum(1 for j in indices if assignments[j] == other_split)
                    if label_other_count <= 1:
                        continue
                    assignments[idx] = target_split
                    counts[target_split] += 1
                    counts[other_split] -= 1
                    deficit -= 1
                    if deficit <= 0:
                        return

    if counts["train"] < min_split:
        _adjust("train")
    if counts["test"] < min_split:
        _adjust("test")

    df["split"] = df.index.map(assignments).fillna("train")
    keyed = {
        (str(row.get("accession", "")), str(row.get("md5", ""))): split
        for (_, row), split in zip(df.iterrows(), df["split"].tolist())
    }
    for row in rows:
        key = (row.get("accession", ""), row.get("md5", ""))
        row["split"] = keyed.get(key, "train")


def _prepare_records(manifest: "pd.DataFrame") -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for idx, row in manifest.reset_index(drop=True).iterrows():
        path = Path(str(row.get("path", "")))
        text = _read_text(path)
        tokens = _tokenize(text)
        counts = Counter(tokens)
        shingles = _shingles(tokens)
        md5 = hashlib.md5(text.encode("utf-8")).hexdigest() if text else ""
        records.append(
            {
                "index": idx,
                "row": {key: (str(value) if not pd.isna(value) else "") for key, value in row.items()},
                "path": path,
                "text": text,
                "tokens": tokens,
                "counts": counts,
                "shingles": shingles,
                "md5": md5,
                "key": f"{row.get('ticker', '')}::{path.name}",
            }
        )
    return records


def deduplicate_manifest(
    manifest: "pd.DataFrame",
    threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
) -> Tuple["pd.DataFrame", "pd.DataFrame", Sequence[DedupDecision]]:
    records = _prepare_records(manifest)
    kept: List[Dict[str, object]] = []
    quarantine: List[Dict[str, object]] = []
    decisions: List[DedupDecision] = []
    seen_md5: Dict[str, Dict[str, object]] = {}

    tfidf_matches = _tfidf_near_duplicates(records, threshold)
    minhash_matches = _minhash_near_duplicates(records, max(0.5, threshold - 0.1))

    tfidf_lookup = {(i, j): score for i, j, score in tfidf_matches}
    minhash_lookup = {(i, j): score for i, j, score in minhash_matches}

    def _mark_duplicate(record: Dict[str, object], method: str, primary: Dict[str, object], score: float) -> None:
        row = dict(record["row"])
        row["dedup_method"] = method
        row["duplicate_of"] = primary.get("key", "")
        row["dedup_score"] = round(float(score), 4)
        quarantine.append(row)
        decisions.append(
            DedupDecision(
                method=method,
                primary=str(primary.get("key", "")),
                duplicate=str(record.get("key", "")),
                score=float(score),
            )
        )

    for idx, record in enumerate(records):
        md5 = record["md5"]
        if md5 and md5 in seen_md5:
            _mark_duplicate(record, "md5_raw", seen_md5[md5], 1.0)
            continue

        duplicate_marked = False
        for kept_record in kept:
            pair = (min(idx, kept_record["index"]), max(idx, kept_record["index"]))
            if pair in tfidf_lookup:
                _mark_duplicate(record, "tfidf", kept_record, tfidf_lookup[pair])
                duplicate_marked = True
                break
            if pair in minhash_lookup:
                _mark_duplicate(record, "minhash", kept_record, minhash_lookup[pair])
                duplicate_marked = True
                break
        if duplicate_marked:
            continue

        kept.append(record)
        if md5:
            seen_md5[md5] = record

    base_columns = list(manifest.columns)
    extra_columns = ["dedup_method", "duplicate_of", "dedup_score"]

    kept_rows = []
    for record in kept:
        row = dict(record["row"])
        for column in extra_columns:
            row.setdefault(column, "")
        kept_rows.append(row)

    quarantine_rows = []
    for row in quarantine:
        for column in extra_columns:
            row.setdefault(column, "")
        quarantine_rows.append(row)

    kept_df = pd.DataFrame(kept_rows, columns=base_columns + extra_columns)
    quarantine_df = pd.DataFrame(quarantine_rows, columns=base_columns + extra_columns)
    return kept_df, quarantine_df, decisions


def _write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quarantine", type=Path, default=DEFAULT_QUARANTINE)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--duplicate-threshold", type=float, default=DEFAULT_DUPLICATE_THRESHOLD)
    parser.add_argument("--min-split", type=int, default=DEFAULT_MIN_SPLIT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path: Path = args.manifest
    cache_path: Path = args.cache

    rows, base_fieldnames = _load_manifest_rows(manifest_path)
    lsh = SimpleMinHashLSH(cache_path)

    kept: List[Dict[str, str]] = []
    quarantine: List[Dict[str, str]] = []
    row_lookup: Dict[str, Dict[str, str]] = {}
    shingles_cache: Dict[str, List[int]] = {}

    for manifest_row in rows:
        data = dict(manifest_row.data)
        _ensure_columns(data)
        text = _read_text(manifest_row.path)
        if not text:
            _apply_quality_flag(data, "missing-content")
        tokens = _tokenize(text)
        shingles = _shingles(tokens)
        doc_id = manifest_row.key
        if not shingles:
            _apply_quality_flag(data, "insufficient-shingles")
        candidates = lsh.query(shingles)
        duplicate_of = None
        for candidate in candidates:
            if candidate == doc_id:
                continue
            reference = row_lookup.get(candidate)
            if not reference:
                continue
            reference_shingles = shingles_cache.get(candidate)
            if reference_shingles is None:
                reference_text = _read_text(Path(reference.get("path", "")))
                reference_tokens = _tokenize(reference_text)
                reference_shingles = _shingles(reference_tokens)
                shingles_cache[candidate] = reference_shingles
            similarity = _jaccard(shingles, reference_shingles)
            if similarity >= args.duplicate_threshold:
                duplicate_of = candidate
                break
        if duplicate_of:
            data["duplicate_of"] = duplicate_of
            data["duplicate_reason"] = f"jaccard>={args.duplicate_threshold}"
            quarantine.append(data)
            continue
        kept.append(data)
        row_lookup[doc_id] = data
        shingles_cache[doc_id] = shingles
        lsh.add(doc_id, shingles)

    _assign_splits(kept, min_split=args.min_split)
    if kept:
        additional_columns = sorted({key for row in kept for key in row.keys()} - set(base_fieldnames))
        fieldnames = list(base_fieldnames) + [column for column in additional_columns if column not in base_fieldnames]
        if "split" not in fieldnames:
            fieldnames.append("split")
    else:
        fieldnames = list(base_fieldnames)
    quarantine_fields = list({key for row in quarantine for key in row.keys()}) if quarantine else fieldnames

    if kept:
        _write_csv(Path(args.out), kept, fieldnames)
    if quarantine:
        _write_csv(Path(args.quarantine), quarantine, quarantine_fields)
    lsh.save()

    print(f"deduplicated manifest written to {Path(args.out).resolve()}")
    if quarantine:
        print(f"quarantined {len(quarantine)} suspected duplicates")
    print(f"training split: {(len([row for row in kept if row.get('split') == 'train']))}")
    print(f"test split: {(len([row for row in kept if row.get('split') == 'test']))}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI helper
    raise SystemExit(main())
