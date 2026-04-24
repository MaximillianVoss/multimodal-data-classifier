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
    figure, axis = plt.subplots(figsize=(11.4, 6.2))
    axis.set_xlim(0, 11.4)
    axis.set_ylim(0, 6)
    axis.axis("off")

    axis.text(1.1, 3.0, "Пользователь", fontsize=12, weight="bold", ha="center")
    axis.plot([1.1, 1.1], [2.1, 3.5], color="black", linewidth=2)
    axis.plot([0.75, 1.45], [3.1, 3.1], color="black", linewidth=2)
    axis.plot([1.1, 0.7], [2.1, 1.4], color="black", linewidth=2)
    axis.plot([1.1, 1.5], [2.1, 1.4], color="black", linewidth=2)

    use_cases = [
        (7.3, 5.0, "Классификация\nтекста документа"),
        (7.3, 3.8, "Классификация\nскана документа"),
        (7.3, 2.6, "Пакетная сортировка\nZIP-архива"),
        (7.3, 1.4, "Получение архива\nс результатами"),
        (7.3, 0.3, "Просмотр истории и\nметрик моделей"),
    ]
    for x, y, text in use_cases:
        ellipse = Ellipse((x, y), width=4.1, height=0.92, edgecolor="#1f4e79", facecolor="#f6fbff", linewidth=1.6)
        axis.add_patch(ellipse)
        axis.text(x, y, text, ha="center", va="center", fontsize=11)
        _arrow(axis, (1.55, 3.0), (x - 2.1, y))

    figure.tight_layout()
    figure.savefig(settings.use_case_figure, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _build_architecture(settings: Settings) -> None:
    figure, axis = plt.subplots(figsize=(12.0, 6.5))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 6)
    axis.axis("off")

    _rounded_box(axis, (0.45, 2.1), 1.7, 1.1, "Пользователь")
    _rounded_box(axis, (2.5, 2.1), 2.2, 1.1, "Gradio UI\n(low-code слой)")
    _rounded_box(axis, (5.2, 3.85), 2.3, 1.0, "FastAPI API")
    _rounded_box(axis, (5.2, 2.15), 2.3, 1.0, "Сервисный слой")
    _rounded_box(axis, (5.2, 0.45), 2.3, 1.0, "Архивный модуль")
    _rounded_box(axis, (8.25, 4.05), 2.35, 0.92, "Текстовая модель")
    _rounded_box(axis, (8.25, 2.85), 2.35, 0.92, "Модель макета\nдокумента")
    _rounded_box(axis, (8.25, 1.65), 2.35, 0.92, "SQLite")
    _rounded_box(axis, (8.25, 0.45), 2.35, 0.92, "Файловое хранилище\nрезультатов")

    _arrow(axis, (2.15, 2.65), (2.5, 2.65))
    _arrow(axis, (4.7, 2.65), (5.2, 4.3))
    _arrow(axis, (4.7, 2.65), (5.2, 2.65))
    _arrow(axis, (4.7, 2.65), (5.2, 0.95))
    _arrow(axis, (6.35, 3.85), (6.35, 3.15))
    _arrow(axis, (7.5, 4.35), (8.25, 4.45))
    _arrow(axis, (7.5, 2.65), (8.25, 3.1))
    _arrow(axis, (7.5, 2.55), (8.25, 2.1))
    _arrow(axis, (7.5, 0.95), (8.25, 0.9))

    figure.tight_layout()
    figure.savefig(settings.architecture_figure, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _build_database_schema(settings: Settings) -> None:
    figure, axis = plt.subplots(figsize=(12.6, 6.6))
    axis.set_xlim(0, 12.6)
    axis.set_ylim(0, 6)
    axis.axis("off")

    _rounded_box(
        axis,
        (0.35, 3.0),
        2.7,
        2.35,
        "classification_requests\n\nid (PK)\nmodality\nsource_type\ninput_preview\ncreated_at",
        fontsize=10,
        facecolor="#fff7e6",
    )
    _rounded_box(
        axis,
        (3.35, 2.7),
        2.85,
        2.95,
        "classification_results\n\nid (PK)\nrequest_id (FK)\npredicted_label\nconfidence\nprocessing_time_ms\nmodel_name\nmodel_version\ncreated_at",
        fontsize=10,
        facecolor="#e9f7ef",
    )
    _rounded_box(
        axis,
        (6.55, 3.05),
        2.95,
        2.3,
        "model_registry\n\nid (PK)\nmodality\nmodel_name\nmodel_version\naccuracy\nweighted_f1\nartifact_path\ntrained_at",
        fontsize=10,
        facecolor="#eef3ff",
    )
    _rounded_box(
        axis,
        (2.1, 0.35),
        2.85,
        1.95,
        "batch_runs\n\nid (PK)\nsource_name\ntotal_files\nprocessed_files\nskipped_files\noutput_archive_path\ncreated_at",
        fontsize=10,
        facecolor="#fff0f5",
    )
    _rounded_box(
        axis,
        (5.65, 0.35),
        3.35,
        1.95,
        "batch_items\n\nid (PK)\nrun_id (FK)\nfile_name\nrelative_path\nmodality\npredicted_label\nstatus",
        fontsize=10,
        facecolor="#f3f8ff",
    )

    _arrow(axis, (3.05, 4.18), (3.35, 4.18))
    _arrow(axis, (4.78, 2.7), (3.9, 2.3))
    _arrow(axis, (4.95, 1.32), (5.65, 1.32))
    _arrow(axis, (6.2, 4.18), (6.55, 4.18))

    figure.tight_layout()
    figure.savefig(settings.database_figure, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _build_workflow(settings: Settings) -> None:
    figure, axis = plt.subplots(figsize=(12.4, 5.0))
    axis.set_xlim(0, 12.4)
    axis.set_ylim(0, 5.0)
    axis.axis("off")

    steps = [
        (0.35, "1. Загрузка\nдокумента\nили архива"),
        (2.3, "2. Определение\nмодальности\nи маршрута"),
        (4.25, "3. Извлечение\nтекста или\nподготовка скана"),
        (6.2, "4. Прогноз\nтипа\nдокумента"),
        (8.15, "5. Запись\nв БД и\nраскладка файлов"),
        (10.1, "6. Выдача\nрезультата,\nCSV и ZIP"),
    ]
    for x, text in steps:
        _rounded_box(axis, (x, 1.65), 1.72, 1.45, text, fontsize=10.5, facecolor="#f7fbff")

    for index in range(len(steps) - 1):
        _arrow(axis, (steps[index][0] + 1.72, 2.37), (steps[index + 1][0], 2.37))

    figure.tight_layout()
    figure.savefig(settings.workflow_figure, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _build_interaction(settings: Settings) -> None:
    figure, axis = plt.subplots(figsize=(12.2, 5.2))
    axis.set_xlim(0, 12.2)
    axis.set_ylim(0, 5.0)
    axis.axis("off")

    top_boxes = [
        (0.35, "Пользователь"),
        (2.25, "Gradio UI"),
        (4.15, "FastAPI API"),
        (6.05, "Сервисный слой"),
        (7.95, "ML-модели"),
        (9.85, "SQLite и\nархивный модуль"),
    ]
    for x, text in top_boxes:
        _rounded_box(axis, (x, 3.2), 1.55, 0.95, text, fontsize=9.8, facecolor="#f7fbff")

    _arrow(axis, (1.9, 3.68), (2.25, 3.68))
    _arrow(axis, (3.8, 3.68), (4.15, 3.68))
    _arrow(axis, (5.7, 3.68), (6.05, 3.68))
    _arrow(axis, (7.6, 3.68), (7.95, 3.68))
    _arrow(axis, (9.5, 3.68), (9.85, 3.68))

    bottom_boxes = [
        (2.25, "Отображение\nрезультата"),
        (4.15, "JSON-ответ"),
        (6.05, "Класс,\nвероятность,\nпути файлов"),
        (7.95, "Логи,\nCSV,\nZIP-архив"),
    ]
    for x, text in bottom_boxes:
        _rounded_box(axis, (x, 1.0), 1.55, 1.02, text, fontsize=9.5, facecolor="#eef4ff")

    _arrow(axis, (10.62, 3.2), (8.72, 2.02))
    _arrow(axis, (7.95, 1.53), (7.6, 1.53))
    _arrow(axis, (6.05, 1.53), (5.7, 1.53))
    _arrow(axis, (4.15, 1.53), (3.8, 1.53))
    _arrow(axis, (2.25, 1.53), (1.9, 1.53))
    _arrow(axis, (1.1, 1.53), (1.1, 3.2))

    figure.tight_layout()
    figure.savefig(settings.interaction_figure, dpi=220, bbox_inches="tight")
    plt.close(figure)
