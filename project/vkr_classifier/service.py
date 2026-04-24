from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import csv
import shutil
from threading import Lock
from time import perf_counter
import tempfile

import numpy as np
from PIL import Image

from vkr_classifier.batch_processing import (
    BatchItemResult,
    build_output_archive,
    classify_modality,
    ensure_unique_path,
    extract_text_from_document,
    safe_extract_archive,
)
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

    @staticmethod
    def _prepare_image(image: Image.Image | str | Path) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        with Image.open(image) as source:
            return source.copy()

    def _predict_text(self, cleaned_text: str) -> dict[str, object]:
        if self.text_artifact is None:
            raise RuntimeError("Текстовая модель не инициализирована.")

        started = perf_counter()
        probabilities = self.text_artifact.pipeline.predict_proba([cleaned_text])[0]
        classes = self.text_artifact.pipeline.named_steps["classifier"].classes_
        predicted_index = int(np.argmax(probabilities))
        processing_time_ms = int((perf_counter() - started) * 1000)
        return {
            "modality": "text",
            "label": str(classes[predicted_index]),
            "confidence": round(float(probabilities[predicted_index]), 4),
            "processing_time_ms": processing_time_ms,
            "model_name": self.text_artifact.model_name,
            "model_version": self.text_artifact.model_version,
            "probabilities": self._build_probabilities(classes, probabilities),
        }

    def _predict_image(self, image: Image.Image) -> dict[str, object]:
        if self.image_artifact is None:
            raise RuntimeError("Модель изображений не инициализирована.")

        started = perf_counter()
        vector = image_to_vector(image, self.image_artifact.image_size).reshape(1, -1)
        probabilities = self.image_artifact.classifier.predict_proba(vector)[0]
        classes = self.image_artifact.classifier.classes_
        predicted_index = int(np.argmax(probabilities))
        processing_time_ms = int((perf_counter() - started) * 1000)
        return {
            "modality": "image",
            "label": str(classes[predicted_index]),
            "confidence": round(float(probabilities[predicted_index]), 4),
            "processing_time_ms": processing_time_ms,
            "model_name": self.image_artifact.model_name,
            "model_version": self.image_artifact.model_version,
            "probabilities": self._build_probabilities(classes, probabilities),
        }

    def _log_prediction(
        self,
        *,
        modality: str,
        source_type: str,
        input_preview: str,
        response: dict[str, object],
    ) -> None:
        self.database.log_prediction(
            modality=modality,
            source_type=source_type,
            input_preview=input_preview,
            predicted_label=str(response["label"]),
            confidence=float(response["confidence"]),
            processing_time_ms=int(response["processing_time_ms"]),
            model_name=str(response["model_name"]),
            model_version=str(response["model_version"]),
            created_at=self._timestamp(),
        )

    def classify_text(
        self,
        text: str,
        *,
        source_type: str = "manual_text",
        log_request: bool = True,
    ) -> dict[str, object]:
        self._require_ready()

        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("Текст не должен быть пустым.")

        response = self._predict_text(cleaned_text)
        if log_request:
            self._log_prediction(
                modality="text",
                source_type=source_type,
                input_preview=self._truncate_text(cleaned_text),
                response=response,
            )
        return response

    def classify_image(
        self,
        image: Image.Image | str | Path,
        *,
        source_type: str = "uploaded_image",
        log_request: bool = True,
    ) -> dict[str, object]:
        self._require_ready()

        prepared_image = self._prepare_image(image)
        response = self._predict_image(prepared_image)
        if log_request:
            self._log_prediction(
                modality="image",
                source_type=source_type,
                input_preview=f"image_{response['label']}",
                response=response,
            )
        return response

    def classify_image_bytes(
        self,
        payload: bytes,
        *,
        source_type: str = "uploaded_image",
        log_request: bool = True,
    ) -> dict[str, object]:
        image = Image.open(BytesIO(payload))
        return self.classify_image(image, source_type=source_type, log_request=log_request)

    def classify_archive(
        self,
        archive_path: str | Path,
        *,
        source_name: str | None = None,
    ) -> dict[str, object]:
        self._require_ready()

        source_path = Path(archive_path)
        archive_name = source_name or source_path.name
        if source_path.suffix.lower() != ".zip":
            raise ValueError("Для пакетной обработки необходимо загрузить ZIP-архив.")

        created_at = self._timestamp()
        batch_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.settings.batch_reports_dir / f"batch_{batch_stamp}"
        sorted_dir = run_dir / "sorted_documents"
        summary_csv = run_dir / "summary.csv"
        run_dir.mkdir(parents=True, exist_ok=True)
        sorted_dir.mkdir(parents=True, exist_ok=True)

        items: list[BatchItemResult] = []
        label_distribution: Counter[str] = Counter()

        with tempfile.TemporaryDirectory(prefix="archive_extract_") as temporary_dir:
            extracted_root = Path(temporary_dir)
            safe_extract_archive(source_path, extracted_root)
            all_files = sorted(path for path in extracted_root.rglob("*") if path.is_file())
            if not all_files:
                raise ValueError("Архив не содержит файлов для обработки.")

            for file_path in all_files:
                relative_path = file_path.relative_to(extracted_root).as_posix()
                modality = classify_modality(file_path)
                if modality is None:
                    items.append(
                        BatchItemResult(
                            file_name=file_path.name,
                            relative_path=relative_path,
                            modality=None,
                            predicted_label=None,
                            confidence=None,
                            processing_time_ms=None,
                            model_name=None,
                            model_version=None,
                            status="skipped",
                            note="Формат файла не поддерживается пакетной обработкой.",
                        )
                    )
                    continue

                try:
                    if modality == "text":
                        content = extract_text_from_document(file_path).strip()
                        if not content:
                            raise ValueError("Не удалось извлечь текст из документа.")
                        result = self.classify_text(
                            content,
                            source_type="batch_archive_text",
                            log_request=False,
                        )
                    else:
                        result = self.classify_image(
                            file_path,
                            source_type="batch_archive_image",
                            log_request=False,
                        )
                except Exception as error:
                    items.append(
                        BatchItemResult(
                            file_name=file_path.name,
                            relative_path=relative_path,
                            modality=modality,
                            predicted_label=None,
                            confidence=None,
                            processing_time_ms=None,
                            model_name=None,
                            model_version=None,
                            status="skipped",
                            note=f"Ошибка обработки: {error}",
                        )
                    )
                    continue

                predicted_label = str(result["label"])
                label_distribution[predicted_label] += 1
                target_dir = sorted_dir / predicted_label
                target_dir.mkdir(parents=True, exist_ok=True)
                target_path = ensure_unique_path(target_dir / file_path.name)
                shutil.copy2(file_path, target_path)

                items.append(
                    BatchItemResult(
                        file_name=file_path.name,
                        relative_path=relative_path,
                        modality=modality,
                        predicted_label=predicted_label,
                        confidence=float(result["confidence"]),
                        processing_time_ms=int(result["processing_time_ms"]),
                        model_name=str(result["model_name"]),
                        model_version=str(result["model_version"]),
                        status="processed",
                        note=f"Файл помещен в каталог {predicted_label}.",
                    )
                )

        with summary_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "file_name",
                    "relative_path",
                    "modality",
                    "predicted_label",
                    "confidence",
                    "processing_time_ms",
                    "model_name",
                    "model_version",
                    "status",
                    "note",
                ],
            )
            writer.writeheader()
            for item in items:
                writer.writerow(asdict(item))

        output_archive = build_output_archive(
            run_dir,
            self.settings.batch_exports_dir / f"document_routing_{batch_stamp}.zip",
        )

        processed_files = sum(1 for item in items if item.status == "processed")
        skipped_files = len(items) - processed_files
        run_id = self.database.log_batch_run(
            source_name=archive_name,
            total_files=len(items),
            processed_files=processed_files,
            skipped_files=skipped_files,
            output_archive_path=output_archive.relative_to(self.settings.project_root).as_posix(),
            created_at=created_at,
            items=[asdict(item) for item in items],
        )

        return {
            "run_id": run_id,
            "source_name": archive_name,
            "total_files": len(items),
            "processed_files": processed_files,
            "skipped_files": skipped_files,
            "created_at": created_at,
            "output_archive_path": str(output_archive),
            "label_distribution": dict(label_distribution),
            "items": [asdict(item) for item in items],
        }

    def get_history(self, limit: int = 20) -> list[dict[str, object]]:
        self._require_ready()
        return self.database.get_history(limit=limit)

    def get_batch_history(self, limit: int = 10) -> list[dict[str, object]]:
        self._require_ready()
        return self.database.get_batch_history(limit=limit)

    def get_models(self) -> list[dict[str, object]]:
        self._require_ready()
        return self.database.get_models()
