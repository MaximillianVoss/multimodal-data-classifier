from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS model_registry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    modality TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    accuracy REAL NOT NULL,
    weighted_f1 REAL NOT NULL,
    artifact_path TEXT NOT NULL,
    trained_at TEXT NOT NULL,
    UNIQUE(modality, model_name, model_version)
);

CREATE TABLE IF NOT EXISTS classification_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    modality TEXT NOT NULL,
    source_type TEXT NOT NULL,
    input_preview TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS classification_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id INTEGER NOT NULL,
    predicted_label TEXT NOT NULL,
    confidence REAL NOT NULL,
    processing_time_ms INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(request_id) REFERENCES classification_requests(id) ON DELETE CASCADE
);
"""


class Database:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def register_model(
        self,
        modality: str,
        model_name: str,
        model_version: str,
        accuracy: float,
        weighted_f1: float,
        artifact_path: str,
        trained_at: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO model_registry (
                    modality, model_name, model_version, accuracy, weighted_f1, artifact_path, trained_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(modality, model_name, model_version) DO UPDATE SET
                    accuracy = excluded.accuracy,
                    weighted_f1 = excluded.weighted_f1,
                    artifact_path = excluded.artifact_path,
                    trained_at = excluded.trained_at
                """,
                (modality, model_name, model_version, accuracy, weighted_f1, artifact_path, trained_at),
            )

    def replace_model_registry(self, models: list[dict[str, object]]) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM model_registry")
            connection.executemany(
                """
                INSERT INTO model_registry (
                    modality, model_name, model_version, accuracy, weighted_f1, artifact_path, trained_at
                )
                VALUES (:modality, :model_name, :model_version, :accuracy, :weighted_f1, :artifact_path, :trained_at)
                """,
                models,
            )

    def log_prediction(
        self,
        *,
        modality: str,
        source_type: str,
        input_preview: str,
        predicted_label: str,
        confidence: float,
        processing_time_ms: int,
        model_name: str,
        model_version: str,
        created_at: str,
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO classification_requests (modality, source_type, input_preview, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (modality, source_type, input_preview, created_at),
            )
            request_id = int(cursor.lastrowid)
            connection.execute(
                """
                INSERT INTO classification_results (
                    request_id, predicted_label, confidence, processing_time_ms,
                    model_name, model_version, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    predicted_label,
                    confidence,
                    processing_time_ms,
                    model_name,
                    model_version,
                    created_at,
                ),
            )
            return request_id

    def get_history(self, limit: int = 20) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    r.id,
                    r.created_at,
                    r.modality,
                    r.source_type,
                    r.input_preview,
                    result.predicted_label,
                    result.confidence,
                    result.processing_time_ms,
                    result.model_name,
                    result.model_version
                FROM classification_requests AS r
                JOIN classification_results AS result ON result.request_id = r.id
                ORDER BY r.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_models(self) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT modality, model_name, model_version, accuracy, weighted_f1, artifact_path, trained_at
                FROM model_registry
                ORDER BY modality ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]
