from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from threading import Lock
from time import perf_counter

import numpy as np
from PIL import Image

from vkr_classifier.config import Settings, get_settings
from vkr_classifier.data.image_generator import image_to_vector
from vkr_classifier.database import Database
from vkr_classifier.models.image_classifier import ImageModelArtifact, load_image_model
from vkr_classifier.models.text_classifier import TextModelArtifact, load_text_model
from vkr_classifier.training import generate_training_assets


class ClassifierService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_directories()
        self.database = Database(self.settings.database_path)
        self.database.initialize()
        self.text_artifact: TextModelArtifact | None = None
        self.image_artifact: ImageModelArtifact | None = None
        self._ready_lock = Lock()

    @property
    def is_ready(self) -> bool:
        return self.text_artifact is not None and self.image_artifact is not None

    def ensure_ready(self) -> None:
        if self.is_ready:
            return

        with self._ready_lock:
            if self.is_ready:
                return
            generate_training_assets(self.settings, force=False)
            self.text_artifact = load_text_model(self.settings.text_model_path)
            self.image_artifact = load_image_model(self.settings.image_model_path)

    def _require_ready(self) -> None:
        if not self.is_ready:
            self.ensure_ready()

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _truncate_text(value: str, limit: int = 120) -> str:
        compact = " ".join(value.split())
        return compact[:limit] + ("..." if len(compact) > limit else "")

    @staticmethod
    def _build_probabilities(classes: np.ndarray, scores: np.ndarray) -> dict[str, float]:
        return {
            str(label): round(float(probability), 4)
            for label, probability in zip(classes, scores, strict=False)
        }

    def classify_text(self, text: str) -> dict[str, object]:
        self._require_ready()
        if self.text_artifact is None:
            raise RuntimeError("Текстовая модель не инициализирована.")

        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("Текст не должен быть пустым.")

        started = perf_counter()
        probabilities = self.text_artifact.pipeline.predict_proba([cleaned_text])[0]
        classes = self.text_artifact.pipeline.named_steps["classifier"].classes_
        predicted_index = int(np.argmax(probabilities))
        processing_time_ms = int((perf_counter() - started) * 1000)
        response = {
            "modality": "text",
            "label": str(classes[predicted_index]),
            "confidence": round(float(probabilities[predicted_index]), 4),
            "processing_time_ms": processing_time_ms,
            "model_name": self.text_artifact.model_name,
            "model_version": self.text_artifact.model_version,
            "probabilities": self._build_probabilities(classes, probabilities),
        }
        self.database.log_prediction(
            modality="text",
            source_type="manual_text",
            input_preview=self._truncate_text(cleaned_text),
            predicted_label=str(response["label"]),
            confidence=float(response["confidence"]),
            processing_time_ms=processing_time_ms,
            model_name=self.text_artifact.model_name,
            model_version=self.text_artifact.model_version,
            created_at=self._timestamp(),
        )
        return response

    def classify_image(self, image: Image.Image) -> dict[str, object]:
        self._require_ready()
        if self.image_artifact is None:
            raise RuntimeError("Модель изображений не инициализирована.")

        started = perf_counter()
        vector = image_to_vector(image, self.image_artifact.image_size).reshape(1, -1)
        probabilities = self.image_artifact.classifier.predict_proba(vector)[0]
        classes = self.image_artifact.classifier.classes_
        predicted_index = int(np.argmax(probabilities))
        processing_time_ms = int((perf_counter() - started) * 1000)
        response = {
            "modality": "image",
            "label": str(classes[predicted_index]),
            "confidence": round(float(probabilities[predicted_index]), 4),
            "processing_time_ms": processing_time_ms,
            "model_name": self.image_artifact.model_name,
            "model_version": self.image_artifact.model_version,
            "probabilities": self._build_probabilities(classes, probabilities),
        }
        self.database.log_prediction(
            modality="image",
            source_type="uploaded_image",
            input_preview=f"image_{response['label']}",
            predicted_label=str(response["label"]),
            confidence=float(response["confidence"]),
            processing_time_ms=processing_time_ms,
            model_name=self.image_artifact.model_name,
            model_version=self.image_artifact.model_version,
            created_at=self._timestamp(),
        )
        return response

    def classify_image_bytes(self, payload: bytes) -> dict[str, object]:
        image = Image.open(BytesIO(payload))
        return self.classify_image(image)

    def get_history(self, limit: int = 20) -> list[dict[str, object]]:
        self._require_ready()
        return self.database.get_history(limit=limit)

    def get_models(self) -> list[dict[str, object]]:
        self._require_ready()
        return self.database.get_models()
