"""SQLite-backed persistence for arbitrage assets and jobs."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_DB_PATH = Path(os.getenv("CALE_STORAGE_PATH", "data/persistence.db"))


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


class PersistenceStore:
    """Minimal persistence layer for assets and jobs."""

    def __init__(self, db_path: Path):
        _ensure_parent(db_path)
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS assets (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    asset_id TEXT UNIQUE,
                    jurisdiction TEXT,
                    created_at TEXT,
                    payload TEXT,
                    metrics_summary TEXT,
                    provenance_chain TEXT,
                    dependency_graph TEXT,
                    engine_version TEXT,
                    manifest_hash TEXT,
                    run_id TEXT
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_assets_jurisdiction_created
                ON assets(jurisdiction, created_at)
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    type TEXT,
                    status TEXT,
                    result TEXT,
                    error TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    run_id TEXT
                )
                """
            )

    def save_asset(
        self,
        asset_id: str,
        jurisdiction: str,
        payload: Dict[str, Any],
        metrics_summary: Dict[str, Any],
        provenance_chain: Any,
        dependency_graph: Any,
        engine_version: Optional[str],
        manifest_hash: Optional[str],
        run_id: Optional[str],
        created_at: Optional[datetime] = None,
    ) -> str:
        created = created_at or datetime.now(timezone.utc)
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO assets (
                    asset_id, jurisdiction, created_at, payload, metrics_summary,
                    provenance_chain, dependency_graph, engine_version, manifest_hash, run_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    asset_id,
                    jurisdiction,
                    created.isoformat(),
                    json.dumps(payload),
                    json.dumps(metrics_summary),
                    json.dumps(provenance_chain),
                    json.dumps(dependency_graph),
                    engine_version,
                    manifest_hash,
                    run_id,
                ),
            )
        return asset_id

    def list_assets(
        self,
        jurisdiction: Optional[str],
        from_date: Optional[datetime],
        limit: int = 10,
        cursor: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[int]]:
        where: List[str] = []
        params: List[Any] = []
        if jurisdiction:
            where.append("jurisdiction = ?")
            params.append(jurisdiction)
        if from_date:
            where.append("created_at >= ?")
            params.append(from_date.isoformat())
        if cursor is not None:
            where.append("seq > ?")
            params.append(cursor)
        where_clause = f"WHERE {' AND '.join(where)}" if where else ""
        query = (
            "SELECT seq, asset_id, jurisdiction, created_at, metrics_summary, provenance_chain,"
            " dependency_graph, engine_version, manifest_hash, run_id FROM assets "
            f"{where_clause} ORDER BY seq ASC LIMIT ?"
        )
        params.append(limit + 1)
        with self._lock, self._conn:
            rows = list(self._conn.execute(query, params))

        next_cursor: Optional[int] = None
        if len(rows) > limit:
            next_cursor = rows[-1]["seq"]
            rows = rows[:limit]

        items: List[Dict[str, Any]] = []
        for row in rows:
            items.append(
                {
                    "id": row["asset_id"],
                    "jurisdiction": row["jurisdiction"],
                    "created_at": row["created_at"],
                    "metrics": json.loads(row["metrics_summary"] or "{}"),
                    "provenance_chain": json.loads(row["provenance_chain"] or "[]"),
                    "dependency_graph": json.loads(row["dependency_graph"] or "{}"),
                    "engine_version": row["engine_version"],
                    "manifest_hash": row["manifest_hash"],
                    "run_id": row["run_id"],
                    "_cursor": row["seq"],
                }
            )
        return items, next_cursor

    def get_asset(self, asset_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT * FROM assets WHERE asset_id = ?", (asset_id,)
            ).fetchone()
        if not row:
            return None
        return {
            "id": row["asset_id"],
            "jurisdiction": row["jurisdiction"],
            "created_at": row["created_at"],
            "payload": json.loads(row["payload"] or "{}"),
            "metrics": json.loads(row["metrics_summary"] or "{}"),
            "provenance_chain": json.loads(row["provenance_chain"] or "[]"),
            "dependency_graph": json.loads(row["dependency_graph"] or "{}"),
            "engine_version": row["engine_version"],
            "manifest_hash": row["manifest_hash"],
            "run_id": row["run_id"],
        }

    def save_job(self, job: Any) -> None:
        payload = asdict(job)
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO jobs (job_id, type, status, result, error, created_at, updated_at, run_id)
                VALUES (:id, :type, :status, :result, :error, :created_at, :updated_at, :run_id)
                """,
                {
                    **payload,
                    "result": json.dumps(payload.get("result")),
                    "error": payload.get("error"),
                    "created_at": payload.get("created_at").isoformat(),
                    "updated_at": payload.get("updated_at").isoformat(),
                },
            )

    def load_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if not row:
            return None
        return {
            "id": row["job_id"],
            "type": row["type"],
            "status": row["status"],
            "result": json.loads(row["result"] or "{}"),
            "error": row["error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "run_id": row["run_id"],
        }


_GLOBAL_STORE: Optional[PersistenceStore] = None


def get_store(path: Optional[str] = None) -> PersistenceStore:
    """Return a singleton persistence store."""

    global _GLOBAL_STORE
    target = Path(path) if path else DEFAULT_DB_PATH
    if _GLOBAL_STORE is None or _GLOBAL_STORE.db_path != target:
        _GLOBAL_STORE = PersistenceStore(target)
    return _GLOBAL_STORE

