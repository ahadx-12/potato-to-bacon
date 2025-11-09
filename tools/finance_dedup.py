#!/usr/bin/env python3
"""SEC finance manifest deduplication with multi-modal similarity checks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

MAX_HTML_CHARS = 3_500_000
LARGE_PRIME = 4_294_967_311  # largest 32-bit prime
NUM_PERMUTATIONS = 64
NUM_BANDS = 8
ROWS_PER_BAND = NUM_PERMUTATIONS // NUM_BANDS


@dataclass
class FilingRecord:
    idx: int
    path: Path
    ticker: str
    filed: pd.Timestamp
    form: str
    row: pd.Series
    md5_raw: str
    md5_normalised: str
    text: str
    shingles: Optional[Tuple[int, ...]] = None
    signature: Optional[np.ndarray] = None

    @property
    def row_id(self) -> str:
        filed = self.filed.strftime("%Y-%m-%d") if not pd.isna(self.filed) else "unknown"
        stem = self.path.stem
        return f"{self.ticker}:{filed}:{stem}".lower()


@dataclass
class DedupDecision:
    kept_id: str
    removed_id: str
    kept_path: str
    removed_path: str
    method: str
    score: float
    rationale: str


def _load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "path" not in df.columns:
        raise ValueError("Manifest must contain a 'path' column")
    if df.empty:
        print("[warn] manifest is empty after verifying file paths", file=sys.stderr)
        return df.reset_index(drop=True)
    df = df[df["path"].notna()]
    if df.empty:
        print("[warn] manifest has no rows with valid paths", file=sys.stderr)
        return df.reset_index(drop=True)
    df["path"] = df["path"].astype(str)
    df = df[df["path"].map(lambda p: Path(p).exists())]
    if df.empty:
        print("[warn] manifest paths do not exist locally", file=sys.stderr)
    return df.reset_index(drop=True)


def _read_html(path: Path) -> str:
    html = path.read_text(encoding="utf-8", errors="ignore")
    if len(html) > MAX_HTML_CHARS:
        html = html[:MAX_HTML_CHARS]
    return html


def _normalise_html(html: str) -> str:
    return re.sub(r"\s+", "", html).lower()


def _strip_html(html: str) -> str:
    text = re.sub(r"<script.*?>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style.*?>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"&amp;", "&", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _hash_string(value: str) -> int:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest, 16) & 0xFFFFFFFF


def _generate_permutations(num_perm: int = NUM_PERMUTATIONS) -> np.ndarray:
    rng = np.random.default_rng(20240417)
    a = rng.integers(1, LARGE_PRIME - 1, size=num_perm, dtype=np.int64)
    b = rng.integers(0, LARGE_PRIME - 1, size=num_perm, dtype=np.int64)
    return np.stack([a, b], axis=1)


PERMUTATIONS = _generate_permutations()


def _build_shingles(text: str) -> Tuple[int, ...]:
    tokens = re.findall(r"\w+", text.lower())
    shingles: List[int] = []
    for n in range(3, 6):
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            shingle = " ".join(tokens[i : i + n])
            shingles.append(_hash_string(shingle))
    if not shingles:
        return tuple()
    return tuple(sorted(set(shingles)))


def _minhash_signature(shingles: Tuple[int, ...]) -> np.ndarray:
    if not shingles:
        return np.full(NUM_PERMUTATIONS, LARGE_PRIME, dtype=np.uint64)
    shingles_arr = np.fromiter(shingles, dtype=np.uint64)
    a = PERMUTATIONS[:, 0].astype(np.uint64)
    b = PERMUTATIONS[:, 1].astype(np.uint64)
    signatures = (np.minimum.reduce(((a[:, None] * shingles_arr) + b[:, None]) % LARGE_PRIME, axis=1)).astype(
        np.uint64
    )
    return signatures


def _minhash_lsh_candidates(signatures: Dict[int, np.ndarray]) -> Iterator[Tuple[int, int]]:
    buckets: Dict[Tuple[int, bytes], List[int]] = defaultdict(list)
    for idx, signature in signatures.items():
        for band in range(NUM_BANDS):
            start = band * ROWS_PER_BAND
            end = start + ROWS_PER_BAND
            key = signature[start:end].tobytes()
            buckets[(band, key)].append(idx)
    for (_, _), indices in buckets.items():
        if len(indices) < 2:
            continue
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                yield indices[i], indices[j]


def _pair_identifier(record: FilingRecord) -> str:
    if "filing_id" in record.row and not pd.isna(record.row["filing_id"]):
        return str(record.row["filing_id"])
    return record.row_id


def _choose_preferred_idx(left_idx: int, right_idx: int, records: Sequence[FilingRecord]) -> Tuple[int, int]:
    left = records[left_idx]
    right = records[right_idx]
    left_filed = left.filed
    right_filed = right.filed
    if pd.isna(left_filed) and pd.isna(right_filed):
        return (left_idx, right_idx) if left.row_id <= right.row_id else (right_idx, left_idx)
    if pd.isna(left_filed):
        return right_idx, left_idx
    if pd.isna(right_filed):
        return left_idx, right_idx
    if left_filed == right_filed:
        return (left_idx, right_idx) if left.row_id <= right.row_id else (right_idx, left_idx)
    return (left_idx, right_idx) if left_filed <= right_filed else (right_idx, left_idx)


def _tfidf_near_duplicates(records: List[FilingRecord], threshold: float) -> List[Tuple[int, int, float]]:
    if len(records) < 2:
        return []
    texts = [record.text for record in records]
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    nn = NearestNeighbors(metric="cosine", algorithm="brute", radius=1.0 - threshold)
    nn.fit(matrix)
    pairs: List[Tuple[int, int, float]] = []
    for idx, sparse_row in enumerate(matrix):
        distances, neighbours = nn.radius_neighbors(sparse_row, return_distance=True)
        if not len(neighbours):
            continue
        for dist, neighbour in zip(distances[0], neighbours[0]):
            if neighbour <= idx:
                continue
            similarity = 1.0 - float(dist)
            if similarity >= threshold:
                pairs.append((idx, int(neighbour), similarity))
    return pairs


def _minhash_near_duplicates(records: List[FilingRecord], threshold: float) -> List[Tuple[int, int, float]]:
    if len(records) < 2:
        return []
    signatures: Dict[int, np.ndarray] = {}
    shingles_cache: Dict[int, Tuple[int, ...]] = {}
    for idx, record in enumerate(records):
        shingles = record.shingles if record.shingles is not None else _build_shingles(record.text)
        record.shingles = shingles
        signature = record.signature if record.signature is not None else _minhash_signature(shingles)
        record.signature = signature
        signatures[idx] = signature
        shingles_cache[idx] = shingles

    seen_pairs: set[Tuple[int, int]] = set()
    results: List[Tuple[int, int, float]] = []
    for left_idx, right_idx in _minhash_lsh_candidates(signatures):
        key = (min(left_idx, right_idx), max(left_idx, right_idx))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        left_shingles = shingles_cache[left_idx]
        right_shingles = shingles_cache[right_idx]
        if not left_shingles or not right_shingles:
            continue
        left_set = set(left_shingles)
        right_set = set(right_shingles)
        intersection = len(left_set & right_set)
        union = len(left_set | right_set)
        similarity = intersection / union if union else 0.0
        if similarity >= threshold:
            results.append((left_idx, right_idx, similarity))
    return results


def max_cross_tfidf_similarity(
    train_texts: Sequence[str], test_texts: Sequence[str], max_features: int = 10000
) -> Tuple[float, Tuple[int, int]]:
    if not train_texts or not test_texts:
        return (float("nan"), (-1, -1))
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    all_texts = list(train_texts) + list(test_texts)
    matrix = vectorizer.fit_transform(all_texts)
    train_matrix = matrix[: len(train_texts)]
    test_matrix = matrix[len(train_texts) :]
    sims = cosine_similarity(test_matrix, train_matrix)
    if sims.size == 0:
        return (float("nan"), (-1, -1))
    max_pos = np.unravel_index(np.argmax(sims), sims.shape)
    max_val = float(sims[max_pos])
    return max_val, (int(max_pos[1]), int(max_pos[0]))


def max_cross_minhash_similarity(train_texts: Sequence[str], test_texts: Sequence[str]) -> Tuple[float, Tuple[int, int]]:
    if not train_texts or not test_texts:
        return (float("nan"), (-1, -1))
    train_shingles = [_build_shingles(text) for text in train_texts]
    test_shingles = [_build_shingles(text) for text in test_texts]
    max_score = float("nan")
    best_pair = (-1, -1)
    for test_idx, test_shingle in enumerate(test_shingles):
        test_set = set(test_shingle)
        if not test_set:
            continue
        for train_idx, train_shingle in enumerate(train_shingles):
            train_set = set(train_shingle)
            if not train_set:
                continue
            intersection = len(test_set & train_set)
            union = len(test_set | train_set)
            score = intersection / union if union else 0.0
            if math.isnan(max_score) or score > max_score:
                max_score = score
                best_pair = (train_idx, test_idx)
    return max_score, best_pair


def _prepare_records(df: pd.DataFrame) -> List[FilingRecord]:
    records: List[FilingRecord] = []
    for idx, row in df.iterrows():
        path = Path(str(row["path"]))
        html = _read_html(path)
        md5_raw = hashlib.md5(html.encode("utf-8")).hexdigest()
        normalised = _normalise_html(html)
        md5_normalised = hashlib.md5(normalised.encode("utf-8")).hexdigest()
        text = _strip_html(html)
        filed_raw = row.get("filed")
        filed_ts: pd.Timestamp
        try:
            filed_ts = pd.to_datetime(filed_raw)
        except Exception:
            filed_ts = pd.NaT
        ticker = str(row.get("ticker", "")).strip().upper()
        form = str(row.get("form", "")).strip()
        records.append(
            FilingRecord(
                idx=idx,
                path=path,
                ticker=ticker,
                filed=filed_ts,
                form=form,
                row=row,
                md5_raw=md5_raw,
                md5_normalised=md5_normalised,
                text=text,
            )
        )
    return records


def _apply_duplicate_rules(records: List[FilingRecord], threshold: float) -> Tuple[List[FilingRecord], List[DedupDecision]]:
    keep_flags = [True] * len(records)
    decisions: List[DedupDecision] = []

    def _mark_duplicate(keep_idx: int, drop_idx: int, method: str, score: float, rationale: str) -> None:
        if drop_idx == keep_idx or not keep_flags[drop_idx]:
            return
        keep_record = records[keep_idx]
        drop_record = records[drop_idx]
        if not keep_flags[keep_idx]:
            return
        keep_flags[drop_idx] = False
        decisions.append(
            DedupDecision(
                kept_id=_pair_identifier(keep_record),
                removed_id=_pair_identifier(drop_record),
                kept_path=str(keep_record.path),
                removed_path=str(drop_record.path),
                method=method,
                score=score,
                rationale=rationale,
            )
        )

    # Raw MD5 duplicates
    seen_md5: Dict[str, int] = {}
    for idx, record in enumerate(records):
        key = record.md5_raw
        if key in seen_md5:
            keep_idx, drop_idx = _choose_preferred_idx(seen_md5[key], idx, records)
            seen_md5[key] = keep_idx
            _mark_duplicate(keep_idx, drop_idx, "md5_raw", 1.0, "identical raw HTML")
        else:
            seen_md5[key] = idx

    # Normalised MD5 duplicates
    seen_norm: Dict[str, int] = {}
    for idx, record in enumerate(records):
        if not keep_flags[idx]:
            continue
        key = record.md5_normalised
        if key in seen_norm:
            keep_idx, drop_idx = _choose_preferred_idx(seen_norm[key], idx, records)
            seen_norm[key] = keep_idx
            _mark_duplicate(keep_idx, drop_idx, "md5_normalised", 1.0, "identical normalised HTML")
        else:
            seen_norm[key] = idx

    # TF-IDF duplicates
    kept_indices = [i for i, flag in enumerate(keep_flags) if flag]
    kept_records = [records[i] for i in kept_indices]
    tfidf_pairs = _tfidf_near_duplicates(kept_records, threshold)
    for left_local, right_local, score in tfidf_pairs:
        left_idx = kept_indices[left_local]
        right_idx = kept_indices[right_local]
        if not keep_flags[left_idx] or not keep_flags[right_idx]:
            continue
        keep_idx, drop_idx = _choose_preferred_idx(left_idx, right_idx, records)
        _mark_duplicate(keep_idx, drop_idx, "tfidf", float(score), "cosine similarity above threshold")

    kept_indices = [i for i, flag in enumerate(keep_flags) if flag]
    kept_records = [records[i] for i in kept_indices]
    minhash_pairs = _minhash_near_duplicates(kept_records, threshold)
    for left_local, right_local, score in minhash_pairs:
        left_idx = kept_indices[left_local]
        right_idx = kept_indices[right_local]
        if not keep_flags[left_idx] or not keep_flags[right_idx]:
            continue
        keep_idx, drop_idx = _choose_preferred_idx(left_idx, right_idx, records)
        _mark_duplicate(keep_idx, drop_idx, "minhash", float(score), "3-5 gram MinHash similarity")

    kept_records_final = [records[i] for i, flag in enumerate(keep_flags) if flag]
    return kept_records_final, decisions


def deduplicate_manifest(df: pd.DataFrame, threshold: float = 0.85) -> Tuple[pd.DataFrame, pd.DataFrame, List[DedupDecision]]:
    records = _prepare_records(df)
    kept_records, decisions = _apply_duplicate_rules(records, threshold)
    kept_indices = [record.idx for record in kept_records]
    kept_df = df.loc[kept_indices].copy()
    kept_df.reset_index(drop=True, inplace=True)

    if decisions:
        removal_rows = []
        for decision in decisions:
            match = next((record for record in records if str(record.path) == decision.removed_path), None)
            if match is None:
                continue
            row_dict = match.row.to_dict()
            row_dict.update(
                {
                    "duplicate_of": decision.kept_id,
                    "duplicate_path": decision.kept_path,
                    "dedup_method": decision.method,
                    "dedup_score": decision.score,
                    "dedup_rationale": decision.rationale,
                }
            )
            removal_rows.append(row_dict)
        quarantine_df = pd.DataFrame(removal_rows)
    else:
        quarantine_df = pd.DataFrame(columns=list(df.columns) + [
            "duplicate_of",
            "duplicate_path",
            "dedup_method",
            "dedup_score",
            "dedup_rationale",
        ])
    return kept_df, quarantine_df, decisions


def _write_csv(path: Path, df: pd.DataFrame, *, header: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        columns = header if header is not None else df.columns.tolist()
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
    else:
        df.to_csv(path, index=False, quoting=csv.QUOTE_NONNUMERIC)


def _decisions_to_json(decisions: List[DedupDecision]) -> List[Dict[str, object]]:
    return [
        {
            "kept_id": decision.kept_id,
            "removed_id": decision.removed_id,
            "kept_path": decision.kept_path,
            "removed_path": decision.removed_path,
            "method": decision.method,
            "score": decision.score,
            "rationale": decision.rationale,
        }
        for decision in decisions
    ]


def _infer_quarantine_path(manifest_path: Path) -> Path:
    stem = manifest_path.stem
    if stem.endswith("_dedup"):
        return manifest_path.with_name(stem.replace("_dedup", "_quarantine") + manifest_path.suffix)
    return manifest_path.with_name(stem + "_quarantine" + manifest_path.suffix)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Input manifest CSV")
    parser.add_argument("--out", type=Path, required=True, help="Path to deduplicated manifest")
    parser.add_argument(
        "--quarantine",
        type=Path,
        default=None,
        help="Optional path to quarantine CSV (duplicates). Defaults next to output.",
    )
    parser.add_argument("--thresh", type=float, default=0.85, help="Similarity threshold (default: 0.85)")
    parser.add_argument(
        "--decisions-json",
        type=Path,
        default=None,
        help="Optional JSON file capturing dedup decisions for downstream reporting.",
    )
    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    original_columns = manifest.columns.tolist()
    kept_df, quarantine_df, decisions = deduplicate_manifest(manifest, threshold=args.thresh)

    quarantine_path = args.quarantine or _infer_quarantine_path(args.out)
    _write_csv(args.out, kept_df, header=original_columns)
    quarantine_header = quarantine_df.columns.tolist() if not quarantine_df.empty else original_columns + [
        "duplicate_of",
        "duplicate_path",
        "dedup_method",
        "dedup_score",
        "dedup_rationale",
    ]
    _write_csv(quarantine_path, quarantine_df, header=quarantine_header)

    decisions_payload = _decisions_to_json(decisions)
    if args.decisions_json is not None:
        args.decisions_json.parent.mkdir(parents=True, exist_ok=True)
        with args.decisions_json.open("w", encoding="utf-8") as handle:
            json.dump(decisions_payload, handle, indent=2)

    print(json.dumps(
        {
            "input_rows": int(len(manifest)),
            "retained_rows": int(len(kept_df)),
            "quarantined_rows": int(len(quarantine_df)),
            "threshold": args.thresh,
            "decisions": decisions_payload,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()

