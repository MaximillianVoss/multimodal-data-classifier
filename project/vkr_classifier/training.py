from __future__ import annotations

from pathlib import Path

from vkr_classifier.batch_processing import create_demo_archive
from vkr_classifier.config import Settings
from vkr_classifier.data.image_generator import save_demo_examples
from vkr_classifier.data.text_samples import build_demo_text_map
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


def _demo_archive_ready(settings: Settings) -> bool:
    return settings.demo_archive_path.exists()


def _text_model_needs_refresh(settings: Settings) -> bool:
    if not settings.text_model_path.exists():
        return True
    artifact = load_text_model(settings.text_model_path)
    return (
        artifact.labels != list(settings.text_labels)
        or artifact.model_name != settings.text_model_name
        or artifact.model_version != settings.text_model_version
    )


def _image_model_needs_refresh(settings: Settings) -> bool:
    if not settings.image_model_path.exists():
        return True
    artifact = load_image_model(settings.image_model_path)
    return (
        artifact.labels != list(settings.image_labels)
        or artifact.model_name != settings.image_model_name
        or artifact.model_version != settings.image_model_version
        or artifact.image_size != settings.image_size
    )


def generate_training_assets(
    settings: Settings,
    force: bool = True,
) -> tuple[TextModelArtifact, ImageModelArtifact]:
    settings.ensure_directories()
    database = Database(settings.database_path)
    database.initialize()
    models_refreshed = False

    if force or _text_model_needs_refresh(settings):
        text_artifact = train_text_model(settings)
        save_text_model(text_artifact, settings.text_model_path)
        models_refreshed = True
    else:
        text_artifact = load_text_model(settings.text_model_path)

    if force or _image_model_needs_refresh(settings):
        image_artifact = train_image_model(settings)
        save_image_model(image_artifact, settings.image_model_path)
        models_refreshed = True
    else:
        image_artifact = load_image_model(settings.image_model_path)

    if force or models_refreshed or not _report_assets_ready(settings):
        export_reports(settings, text_artifact, image_artifact)
    if force or models_refreshed or not _documentation_figures_ready(settings):
        generate_documentation_figures(settings)

    if force or models_refreshed or not _demo_examples_ready(settings):
        image_examples = save_demo_examples(settings.demo_examples_dir, settings.image_labels, settings.image_size)
    else:
        image_examples = {
            label: str(settings.demo_examples_dir / f"{index:02d}_{label}.png")
            for index, label in enumerate(settings.image_labels, start=1)
        }

    if force or models_refreshed or not _demo_archive_ready(settings):
        create_demo_archive(
            settings.demo_archive_path,
            text_examples=build_demo_text_map(),
            image_examples={label: Path(path) for label, path in image_examples.items()},
        )

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
