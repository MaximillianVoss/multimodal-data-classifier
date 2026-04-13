from __future__ import annotations

import pytest

from vkr_classifier.config import Settings
from vkr_classifier.data.image_generator import create_shape_image
from vkr_classifier.service import ClassifierService


def test_text_classification_detects_technology(service: ClassifierService) -> None:
    result = service.classify_text(
        "Инженерная команда реализовала интерфейс работы с API и улучшила надежность цифровой платформы."
    )
    assert result["label"] == "Технологии"
    assert float(result["confidence"]) > 0.6


def test_image_classification_detects_triangle(service: ClassifierService, settings: Settings) -> None:
    image = create_shape_image("Треугольник", seed=2024, image_size=settings.image_size)
    result = service.classify_image(image)
    assert result["label"] == "Треугольник"
    assert float(result["confidence"]) > 0.6


def test_image_classification_accepts_file_path(service: ClassifierService, settings: Settings) -> None:
    image = create_shape_image("Квадрат", seed=2025, image_size=settings.image_size)
    file_path = settings.demo_examples_dir / "test_square.png"
    image.save(file_path)

    result = service.classify_image(file_path)
    assert result["label"] == "Квадрат"
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
