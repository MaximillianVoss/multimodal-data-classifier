from __future__ import annotations

from vkr_classifier.config import Settings
from vkr_classifier.data.image_generator import save_demo_examples
from vkr_classifier.database import Database
from vkr_classifier.diagrams import generate_documentation_figures
from vkr_classifier.models.image_classifier import (
    ImageModelArtifact,
    load_image_model,
    save_image_model,
    train_image_model,
)
from vkr_classifier.models.text_classifier import (
    TextModelArtifact,
    load_text_model,
    save_text_model,
    train_text_model,
)
from vkr_classifier.reporting import export_reports


def generate_training_assets(
    settings: Settings,
    force: bool = True,
) -> tuple[TextModelArtifact, ImageModelArtifact]:
    settings.ensure_directories()
    database = Database(settings.database_path)
    database.initialize()

    if force or not settings.text_model_path.exists():
        text_artifact = train_text_model(settings)
        save_text_model(text_artifact, settings.text_model_path)
    else:
        text_artifact = load_text_model(settings.text_model_path)

    if force or not settings.image_model_path.exists():
        image_artifact = train_image_model(settings)
        save_image_model(image_artifact, settings.image_model_path)
    else:
        image_artifact = load_image_model(settings.image_model_path)

    export_reports(settings, text_artifact, image_artifact)
    generate_documentation_figures(settings)
    save_demo_examples(settings.demo_examples_dir, settings.image_labels, settings.image_size)

    database.register_model(
        modality="text",
        model_name=text_artifact.model_name,
        model_version=text_artifact.model_version,
        accuracy=text_artifact.metrics["accuracy"],
        weighted_f1=text_artifact.metrics["f1_score"],
        artifact_path=str(settings.text_model_path),
        trained_at=text_artifact.trained_at,
    )
    database.register_model(
        modality="image",
        model_name=image_artifact.model_name,
        model_version=image_artifact.model_version,
        accuracy=image_artifact.metrics["accuracy"],
        weighted_f1=image_artifact.metrics["f1_score"],
        artifact_path=str(settings.image_model_path),
        trained_at=image_artifact.trained_at,
    )
    return text_artifact, image_artifact
