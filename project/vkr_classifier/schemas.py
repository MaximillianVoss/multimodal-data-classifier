from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TextClassificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=5, description="Текст для классификации")


class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    modality: Literal["text", "image"]
    label: str
    confidence: float
    processing_time_ms: int
    model_name: str
    model_version: str
    probabilities: dict[str, float]


class HistoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    created_at: str
    modality: str
    source_type: str
    input_preview: str
    predicted_label: str
    confidence: float
    processing_time_ms: int
    model_name: str
    model_version: str


class ModelInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    modality: str
    model_name: str
    model_version: str
    accuracy: float
    weighted_f1: float
    artifact_path: str
    trained_at: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
    models_ready: bool


class BatchItemEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_name: str
    relative_path: str
    modality: str | None
    predicted_label: str | None
    confidence: float | None
    processing_time_ms: int | None
    model_name: str | None
    model_version: str | None
    status: str
    note: str


class BatchRunResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: int
    source_name: str
    total_files: int
    processed_files: int
    skipped_files: int
    created_at: str
    output_archive_path: str
    label_distribution: dict[str, int]
    items: list[BatchItemEntry]


class BatchHistoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    source_name: str
    total_files: int
    processed_files: int
    skipped_files: int
    output_archive_path: str
    created_at: str
