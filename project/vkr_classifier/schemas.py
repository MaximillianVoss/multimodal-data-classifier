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

