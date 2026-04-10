from __future__ import annotations

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


def test_training_outputs_are_created(settings: Settings, service: ClassifierService) -> None:
    assert settings.text_model_path.exists()
    assert settings.image_model_path.exists()
    assert settings.text_confusion_figure.exists()
    assert settings.image_confusion_figure.exists()
    assert settings.model_comparison_figure.exists()

