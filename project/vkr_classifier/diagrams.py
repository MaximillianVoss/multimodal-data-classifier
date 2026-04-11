from __future__ import annotations

import matplotlib
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch

matplotlib.use("Agg")

from matplotlib import pyplot as plt

from vkr_classifier.config import Settings


def _rounded_box(axis, xy, width, height, text, fontsize=11, facecolor="#eaf3ff") -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.6,
        edgecolor="#1f4e79",
        facecolor=facecolor,
    )
    axis.add_patch(box)
    axis.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize)


def _arrow(axis, start, end) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=18,
            linewidth=1.5,
            color="#1f4e79",
        )
    )


def generate_documentation_figures(settings: Settings) -> None:
    settings.ensure_directories()
    _build_use_case(settings)
    _build_architecture(settings)
    _build_database_schema(settings)
    _build_workflow(settings)
    _build_interaction(settings)


def _build_use_case(settings: Settings) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 6)
    axis.axis("off")

    axis.text(1.1, 3.0, "Пользователь", fontsize=12, weight="bold", ha="center")
    axis.plot([1.1, 1.1], [2.1, 3.5], color="black", linewidth=2)
    axis.plot([0.75, 1.45], [3.1, 3.1], color="black", linewidth=2)
    axis.plot([1.1, 0.7], [2.1, 1.4], color="black", linewidth=2)
    axis.plot([1.1, 1.5], [2.1, 1.4], color="black", linewidth=2)

    use_cases = [
        (6.5, 4.6, "Классификация\nтекста"),
        (6.5, 3.3, "Классификация\nизображения"),
        (6.5, 2.0, "Просмотр истории\nзапросов"),
        (6.5, 0.8, "Просмотр метрик\nмодели"),
    ]
    for x, y, text in use_cases:
        ellipse = Ellipse((x, y), width=3.4, height=0.9, edgecolor="#1f4e79", facecolor="#f6fbff", linewidth=1.6)
        axis.add_patch(ellipse)
        axis.text(x, y, text, ha="center", va="center", fontsize=11)
        _arrow(axis, (1.5, 3.0), (x - 1.8, y))

    figure.tight_layout()
    figure.savefig(settings.use_case_figure, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _build_architecture(settings: Settings) -> None:
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.set_xlim(0, 11)
    axis.set_ylim(0, 6)
    axis.axis("off")

    _rounded_box(axis, (0.5, 2.1), 1.6, 1.1, "Пользователь")
    _rounded_box(axis, (2.8, 2.1), 2.0, 1.1, "Gradio UI\n(low-code слой)")
    _rounded_box(axis, (5.4, 3.5), 2.1, 1.1, "FastAPI API")
    _rounded_box(axis, (5.4, 1.8), 2.1, 1.1, "Сервисный слой")
    _rounded_box(axis, (8.1, 3.7), 2.1, 1.0, "Модель текста")
    _rounded_box(axis, (8.1, 2.3), 2.1, 1.0, "Модель изображений")
    _rounded_box(axis, (8.1, 0.9), 2.1, 1.0, "SQLite")

    _arrow(axis, (2.1, 2.65), (2.8, 2.65))
    _arrow(axis, (4.8, 2.65), (5.4, 4.0))
    _arrow(axis, (4.8, 2.65), (5.4, 2.35))
    _arrow(axis, (6.45, 3.5), (6.45, 2.9))
    _arrow(axis, (7.5, 4.0), (8.1, 4.2))
    _arrow(axis, (7.5, 2.35), (8.1, 2.8))
    _arrow(axis, (7.5, 2.0), (8.1, 1.4))

    figure.tight_layout()
    figure.savefig(settings.architecture_figure, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _build_database_schema(settings: Settings) -> None:
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.set_xlim(0, 11)
    axis.set_ylim(0, 6)
    axis.axis("off")

    _rounded_box(
        axis,
        (0.5, 1.3),
        3.0,
        3.6,
        "classification_requests\n\nid (PK)\nmodality\nsource_type\ninput_preview\ncreated_at",
        fontsize=10,
        facecolor="#fff7e6",
    )
    _rounded_box(
        axis,
        (4.0, 1.3),
        3.0,
        3.6,
        "classification_results\n\nid (PK)\nrequest_id (FK)\npredicted_label\nconfidence\nprocessing_time_ms\nmodel_name\nmodel_version\ncreated_at",
        fontsize=10,
        facecolor="#e9f7ef",
    )
    _rounded_box(
        axis,
        (7.5, 1.6),
        3.0,
        3.0,
        "model_registry\n\nid (PK)\nmodality\nmodel_name\nmodel_version\naccuracy\nweighted_f1\nartifact_path\ntrained_at",
        fontsize=10,
        facecolor="#eef3ff",
    )
    _arrow(axis, (3.5, 3.1), (4.0, 3.1))
    _arrow(axis, (7.0, 3.1), (7.5, 3.1))

    figure.tight_layout()
    figure.savefig(settings.database_figure, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _build_workflow(settings: Settings) -> None:
    figure, axis = plt.subplots(figsize=(11.2, 4.8))
    axis.set_xlim(0, 11.2)
    axis.set_ylim(0, 4.8)
    axis.axis("off")

    steps = [
        (0.4, "1. Ввод текста\nили изображения"),
        (2.55, "2. Предобработка\nданных"),
        (4.7, "3. Запуск\nмодели"),
        (6.85, "4. Сохранение\nрезультата"),
        (9.0, "5. Вывод\nпрогноза"),
    ]
    for x, text in steps:
        _rounded_box(axis, (x, 1.7), 1.8, 1.25, text, fontsize=11, facecolor="#f7fbff")

    for index in range(len(steps) - 1):
        _arrow(axis, (steps[index][0] + 1.8, 2.32), (steps[index + 1][0], 2.32))

    figure.tight_layout()
    figure.savefig(settings.workflow_figure, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _build_interaction(settings: Settings) -> None:
    figure, axis = plt.subplots(figsize=(11.2, 5.0))
    axis.set_xlim(0, 11.2)
    axis.set_ylim(0, 5.0)
    axis.axis("off")

    top_boxes = [
        (0.45, "Пользователь"),
        (2.6, "Gradio UI"),
        (4.75, "FastAPI API"),
        (6.9, "Сервисный слой"),
        (9.05, "ML-модель"),
    ]
    for x, text in top_boxes:
        _rounded_box(axis, (x, 3.2), 1.7, 0.95, text, fontsize=10, facecolor="#f7fbff")

    _arrow(axis, (2.15, 3.68), (2.6, 3.68))
    _arrow(axis, (4.3, 3.68), (4.75, 3.68))
    _arrow(axis, (6.45, 3.68), (6.9, 3.68))
    _arrow(axis, (8.6, 3.68), (9.05, 3.68))

    bottom_boxes = [
        (2.6, "Отображение\nв UI"),
        (4.75, "JSON-ответ"),
        (6.9, "Класс и\nвероятность"),
    ]
    for x, text in bottom_boxes:
        _rounded_box(axis, (x, 1.1), 1.7, 0.95, text, fontsize=10, facecolor="#eef4ff")

    _arrow(axis, (9.9, 3.2), (7.75, 2.05))
    _arrow(axis, (6.9, 1.58), (6.45, 1.58))
    _arrow(axis, (4.75, 1.58), (4.3, 1.58))
    _arrow(axis, (2.6, 1.58), (2.15, 1.58))
    _arrow(axis, (1.3, 1.58), (1.3, 3.2))

    figure.tight_layout()
    figure.savefig(settings.interaction_figure, dpi=220, bbox_inches="tight")
    plt.close(figure)
