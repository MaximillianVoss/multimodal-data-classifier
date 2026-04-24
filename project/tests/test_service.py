from __future__ import annotations

from pathlib import Path

import pytest

from vkr_classifier.config import Settings
from vkr_classifier.data.image_generator import create_document_image
from vkr_classifier.service import ClassifierService


def test_text_classification_detects_invoice(service: ClassifierService) -> None:
    result = service.classify_text(
        "Счет выставлен поставщиком на оплату услуг. В документе указаны реквизиты, сумма и срок оплаты."
    )
    assert result["label"] == "Счет"
    assert float(result["confidence"]) > 0.6


def test_image_classification_detects_order_layout(service: ClassifierService, settings: Settings) -> None:
    image = create_document_image("Приказ", seed=2024, image_size=settings.image_size)
    result = service.classify_image(image)
    assert result["label"] == "Приказ"
    assert float(result["confidence"]) > 0.6


def test_image_classification_accepts_file_path(service: ClassifierService, settings: Settings) -> None:
    image = create_document_image("Договор", seed=2025, image_size=settings.image_size)
    file_path = settings.demo_examples_dir / "test_contract.png"
    image.save(file_path)

    result = service.classify_image(file_path)
    assert result["label"] == "Договор"
    assert float(result["confidence"]) > 0.6


def test_training_outputs_are_created(settings: Settings, service: ClassifierService) -> None:
    assert settings.text_model_path.exists()
    assert settings.image_model_path.exists()
    assert settings.text_confusion_figure.exists()
    assert settings.image_confusion_figure.exists()
    assert settings.model_comparison_figure.exists()


def test_model_registry_contains_only_current_models_with_relative_paths(service: ClassifierService) -> None:
    models = service.get_models()
    assert len(models) == 2

    paths = {item["modality"]: str(item["artifact_path"]) for item in models}
    assert paths["text"] == "artifacts/models/text_classifier.joblib"
    assert paths["image"] == "artifacts/models/image_classifier.joblib"


def test_batch_archive_classification_returns_summary(
    service: ClassifierService,
    batch_archive_path,
) -> None:
    result = service.classify_archive(batch_archive_path)

    assert result["source_name"] == batch_archive_path.name
    assert result["total_files"] == 3
    assert result["processed_files"] == 2
    assert result["skipped_files"] == 1
    assert Path(result["output_archive_path"]).exists()
    assert set(result["label_distribution"].keys()) == {"Договор", "Приказ"}


def test_ensure_ready_does_not_reload_models_when_service_is_already_ready(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    service = ClassifierService(settings)
    calls = {"generate": 0, "text": 0, "image": 0}
    text_artifact = object()
    image_artifact = object()

    def fake_generate(*args, **kwargs) -> None:
        calls["generate"] += 1

    def fake_load_text(*args, **kwargs):
        calls["text"] += 1
        return text_artifact

    def fake_load_image(*args, **kwargs):
        calls["image"] += 1
        return image_artifact

    monkeypatch.setattr("vkr_classifier.service.generate_training_assets", fake_generate)
    monkeypatch.setattr("vkr_classifier.service.load_text_model", fake_load_text)
    monkeypatch.setattr("vkr_classifier.service.load_image_model", fake_load_image)

    service.ensure_ready()
    service.ensure_ready()

    assert service.text_artifact is text_artifact
    assert service.image_artifact is image_artifact
    assert calls == {"generate": 1, "text": 1, "image": 1}
