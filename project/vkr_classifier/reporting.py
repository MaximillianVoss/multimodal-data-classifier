from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from vkr_classifier.config import Settings
from vkr_classifier.models.image_classifier import ImageModelArtifact
from vkr_classifier.models.text_classifier import TextModelArtifact


def _save_confusion_matrix(
    matrix: list[list[int]],
    labels: list[str],
    title: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    heatmap = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(heatmap, ax=axis)
    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=30, ha="right")
    axis.set_yticklabels(labels)
    axis.set_title(title)
    axis.set_xlabel("Предсказанный класс")
    axis.set_ylabel("Истинный класс")

    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            axis.text(col_index, row_index, str(value), ha="center", va="center", color="black")

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _save_model_comparison(
    text_artifact: TextModelArtifact,
    image_artifact: ImageModelArtifact,
    output_path: Path,
) -> None:
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    text_values = [text_artifact.metrics[name] for name in metrics]
    image_values = [image_artifact.metrics[name] for name in metrics]

    positions = range(len(metrics))
    width = 0.35

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar([item - width / 2 for item in positions], text_values, width=width, label="Тексты")
    axis.bar([item + width / 2 for item in positions], image_values, width=width, label="Изображения")
    axis.set_ylim(0, 1.05)
    axis.set_xticks(list(positions))
    axis.set_xticklabels(["Accuracy", "Precision", "Recall", "F1"])
    axis.set_ylabel("Значение метрики")
    axis.set_title("Сравнение качества моделей")
    axis.legend()
    axis.grid(axis="y", linestyle="--", alpha=0.3)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _class_report_to_frame(class_report: dict[str, dict[str, float] | float]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for label, metrics in class_report.items():
        if isinstance(metrics, dict):
            row = {"label": label}
            row.update(metrics)
            rows.append(row)
    frame = pd.DataFrame(rows)
    numeric_columns = [column for column in frame.columns if column != "label"]
    frame[numeric_columns] = frame[numeric_columns].round(4)
    return frame


def export_reports(
    settings: Settings,
    text_artifact: TextModelArtifact,
    image_artifact: ImageModelArtifact,
) -> None:
    settings.ensure_directories()

    settings.text_metrics_path.write_text(
        json.dumps(text_artifact.metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    settings.image_metrics_path.write_text(
        json.dumps(image_artifact.metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_frame = pd.DataFrame(
        [
            {"model": "Текстовая модель", **text_artifact.metrics},
            {"model": "Модель изображений", **image_artifact.metrics},
        ]
    ).round(4)
    summary_frame.to_csv(settings.summary_metrics_path, index=False, encoding="utf-8-sig")

    _class_report_to_frame(text_artifact.class_report).to_csv(
        settings.text_report_path,
        index=False,
        encoding="utf-8-sig",
    )
    _class_report_to_frame(image_artifact.class_report).to_csv(
        settings.image_report_path,
        index=False,
        encoding="utf-8-sig",
    )

    _save_confusion_matrix(
        matrix=text_artifact.confusion,
        labels=text_artifact.labels,
        title="Матрица ошибок текстовой модели",
        output_path=settings.text_confusion_figure,
    )
    _save_confusion_matrix(
        matrix=image_artifact.confusion,
        labels=image_artifact.labels,
        title="Матрица ошибок модели изображений",
        output_path=settings.image_confusion_figure,
    )
    _save_model_comparison(
        text_artifact=text_artifact,
        image_artifact=image_artifact,
        output_path=settings.model_comparison_figure,
    )
