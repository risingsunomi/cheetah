from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Optional, TypedDict


class ChatMessage(TypedDict):
    role: str
    content: str
    timestamp: str


class ChatLogSummary(TypedDict):
    id: int
    name: str
    model_id: str
    updated_at: str


class ChatLogStorage:
    """Simple SQLite-backed storage for chat transcripts."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        base = db_path or (Path.home() / ".cache" / "tiny_cheetah")
        if base.suffix != ".db":
            base = base / "chat_logs.db"
        self._db_path = base
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    model_id TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    log_id INTEGER NOT NULL REFERENCES chat_logs(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def list_logs(self) -> List[ChatLogSummary]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, name, model_id, updated_at
                FROM chat_logs
                ORDER BY updated_at DESC, id DESC
                """
            ).fetchall()
            return [
                ChatLogSummary(
                    id=row["id"],
                    name=row["name"],
                    model_id=row["model_id"],
                    updated_at=row["updated_at"],
                )
                for row in rows
            ]

    def get_log(self, log_id: int) -> Optional[ChatLogSummary]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, name, model_id, updated_at
                FROM chat_logs
                WHERE id = ?
                """,
                (log_id,),
            ).fetchone()
            if row is None:
                return None
            return ChatLogSummary(
                id=row["id"],
                name=row["name"],
                model_id=row["model_id"],
                updated_at=row["updated_at"],
            )

    def create_log(self, name: str, model_id: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO chat_logs (name, model_id)
                VALUES (?, ?)
                """,
                (name, model_id),
            )
            conn.commit()
            return int(cur.lastrowid)

    def upsert_log(self, name: str, model_id: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO chat_logs (name, model_id)
                VALUES (?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    model_id = excluded.model_id,
                    updated_at = datetime('now')
                """,
                (name, model_id),
            )
            conn.commit()
            if cur.lastrowid:
                return int(cur.lastrowid)
            row = conn.execute(
                "SELECT id FROM chat_logs WHERE name = ?",
                (name,),
            ).fetchone()
            return int(row["id"]) if row is not None else 0

    def append_message(self, log_id: int, role: str, content: str, timestamp: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages (log_id, role, content, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (log_id, role, content, timestamp),
            )
            conn.execute(
                "UPDATE chat_logs SET updated_at = datetime('now') WHERE id = ?",
                (log_id,),
            )
            conn.commit()

    def get_messages(self, log_id: int, limit: Optional[int] = None) -> List[ChatMessage]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, timestamp
                FROM chat_messages
                WHERE log_id = ?
                ORDER BY id ASC
                """,
                (log_id,),
            ).fetchall()
        messages = [
            ChatMessage(
                role=row["role"],
                content=row["content"],
                timestamp=row["timestamp"],
            )
            for row in rows
        ]
        if limit is not None and len(messages) > limit:
            return messages[-limit:]
        return messages

    def rename_log(self, log_id: int, name: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE chat_logs SET name = ?, updated_at = datetime('now') WHERE id = ?",
                (name, log_id),
            )
            conn.commit()

    def set_log_model(self, log_id: int, model_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE chat_logs
                SET model_id = ?, updated_at = datetime('now')
                WHERE id = ?
                """,
                (model_id, log_id),
            )
            conn.commit()

    def delete_log(self, log_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chat_logs WHERE id = ?", (log_id,))
            conn.commit()
