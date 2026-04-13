from __future__ import annotations

from pathlib import Path

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


def _all_exist(paths: tuple[Path, ...]) -> bool:
    return all(path.exists() for path in paths)


def _report_assets_ready(settings: Settings) -> bool:
    return _all_exist(
        (
            settings.text_metrics_path,
            settings.image_metrics_path,
            settings.summary_metrics_path,
            settings.text_report_path,
            settings.image_report_path,
            settings.model_comparison_figure,
            settings.text_confusion_figure,
            settings.image_confusion_figure,
        )
    )


def _documentation_figures_ready(settings: Settings) -> bool:
    return _all_exist(
        (
            settings.use_case_figure,
            settings.architecture_figure,
            settings.database_figure,
            settings.workflow_figure,
            settings.interaction_figure,
        )
    )


def _demo_examples_ready(settings: Settings) -> bool:
    expected_files = tuple(
        settings.demo_examples_dir / f"{index:02d}_{label}.png"
        for index, label in enumerate(settings.image_labels, start=1)
    )
    return _all_exist(expected_files)


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

    if force or not _report_assets_ready(settings):
        export_reports(settings, text_artifact, image_artifact)
    if force or not _documentation_figures_ready(settings):
        generate_documentation_figures(settings)
    if force or not _demo_examples_ready(settings):
        save_demo_examples(settings.demo_examples_dir, settings.image_labels, settings.image_size)

    database.replace_model_registry(
        [
            {
                "modality": "text",
                "model_name": text_artifact.model_name,
                "model_version": text_artifact.model_version,
                "accuracy": text_artifact.metrics["accuracy"],
                "weighted_f1": text_artifact.metrics["f1_score"],
                "artifact_path": settings.text_model_path.relative_to(settings.project_root).as_posix(),
                "trained_at": text_artifact.trained_at,
            },
            {
                "modality": "image",
                "model_name": image_artifact.model_name,
                "model_version": image_artifact.model_version,
                "accuracy": image_artifact.metrics["accuracy"],
                "weighted_f1": image_artifact.metrics["f1_score"],
                "artifact_path": settings.image_model_path.relative_to(settings.project_root).as_posix(),
                "trained_at": image_artifact.trained_at,
            },
        ]
    )
    return text_artifact, image_artifact
