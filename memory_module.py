"""
M.A.G.I.C. Core – Memory Module
Persistent store for experiences, insights, and task outcomes.
Uses SQLite for durability + simple JSON serialisation.
"""

from __future__ import annotations
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class MemoryModule:
    """
    Stores learned information, insights, and experiences.
    Supports both in-memory operation and durable SQLite persistence.
    """

    def __init__(
        self,
        learned_information: Optional[List[Any]] = None,
        db_path: str = "magic_memory.db",
        use_persistence: bool = True,
    ):
        self.learned_information: List[Dict[str, Any]] = []
        self.db_path = Path(db_path)
        self.use_persistence = use_persistence

        if use_persistence:
            self._init_db()
            self._load_from_db()

        if learned_information:
            for item in learned_information:
                self.store_information(item)

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    task_signature TEXT,
                    content TEXT,
                    tags TEXT
                )
                """
            )
            conn.commit()

    def _load_from_db(self) -> None:
        if not self.db_path.exists():
            return
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT timestamp, task_signature, content, tags FROM experiences ORDER BY id"
            ).fetchall()
            for ts, sig, content, tags in rows:
                self.learned_information.append(
                    {
                        "timestamp": ts,
                        "task_signature": sig,
                        "content": json.loads(content),
                        "tags": json.loads(tags) if tags else [],
                    }
                )

    def store_information(
        self,
        information: Any,
        task_signature: str = "general",
        tags: Optional[List[str]] = None,
    ) -> None:
        """Persist an experience or insight."""
        entry = {
            "timestamp": time.time(),
            "task_signature": task_signature,
            "content": information,
            "tags": tags or [],
        }
        self.learned_information.append(entry)

        if self.use_persistence:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO experiences (timestamp, task_signature, content, tags) VALUES (?, ?, ?, ?)",
                    (
                        entry["timestamp"],
                        task_signature,
                        json.dumps(information, default=str),
                        json.dumps(tags or []),
                    ),
                )
                conn.commit()

    def retrieve_information(
        self,
        task_signature: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Retrieve experiences, optionally filtered by task signature."""
        results = self.learned_information
        if task_signature:
            results = [e for e in results if e.get("task_signature") == task_signature]
        return results[-limit:]

    def search_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        return [e for e in self.learned_information if tag in e.get("tags", [])]

    def clear(self) -> None:
        self.learned_information.clear()
        if self.use_persistence and self.db_path.exists():
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM experiences")
                conn.commit()
