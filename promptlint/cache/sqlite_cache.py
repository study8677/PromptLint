from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, Optional


class CacheStore:
    """SQLite-backed cache for model runs and embeddings."""

    def __init__(self, path: str) -> None:
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS runs (key TEXT PRIMARY KEY, payload TEXT)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, payload TEXT)"
        )
        self._conn.commit()

    async def get_run(self, key: str) -> Optional[Dict[str, Any]]:
        return await asyncio.to_thread(self._get_payload, "runs", key)

    async def set_run(self, key: str, payload: Dict[str, Any]) -> None:
        await asyncio.to_thread(self._set_payload, "runs", key, payload)

    async def get_embedding(self, key: str) -> Optional[Dict[str, Any]]:
        return await asyncio.to_thread(self._get_payload, "embeddings", key)

    async def set_embedding(self, key: str, payload: Dict[str, Any]) -> None:
        await asyncio.to_thread(self._set_payload, "embeddings", key, payload)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _get_payload(self, table: str, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cursor = self._conn.execute(
                f"SELECT payload FROM {table} WHERE key = ?", (key,)
            )
            row = cursor.fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            return None

    def _set_payload(self, table: str, key: str, payload: Dict[str, Any]) -> None:
        serialized = json.dumps(payload)
        with self._lock:
            self._conn.execute(
                f"INSERT OR REPLACE INTO {table} (key, payload) VALUES (?, ?)" ,
                (key, serialized),
            )
            self._conn.commit()
