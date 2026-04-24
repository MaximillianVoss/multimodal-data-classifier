from __future__ import annotations

from pathlib import Path

import gradio as gr
import pandas as pd

from vkr_classifier.data.text_samples import build_demo_texts
from vkr_classifier.service import ClassifierService


def _result_markdown(result: dict[str, object]) -> str:
    return (
        f"**Класс:** {result['label']}\n\n"
        f"**Уверенность:** {float(result['confidence']):.2%}\n\n"
        f"**Модель:** {result['model_name']} {result['model_version']}\n\n"
        f"**Время обработки:** {result['processing_time_ms']} мс"
    )


def _history_frame(service: ClassifierService) -> pd.DataFrame:
    records = service.get_history(limit=12)
    if not records:
        return pd.DataFrame(
            columns=[
                "Дата",
                "Тип",
                "Источник",
                "Входные данные",
                "Класс",
                "Уверенность",
                "Время, мс",
            ]
        )

    frame = pd.DataFrame(records)
    frame = frame.rename(
        columns={
            "created_at": "Дата",
            "modality": "Тип",
            "source_type": "Источник",
            "input_preview": "Входные данные",
            "predicted_label": "Класс",
            "confidence": "Уверенность",
            "processing_time_ms": "Время, мс",
        }
    )
    frame["Уверенность"] = frame["Уверенность"].map(lambda value: round(float(value), 4))
    return frame[["Дата", "Тип", "Источник", "Входные данные", "Класс", "Уверенность", "Время, мс"]]


def _batch_history_frame(service: ClassifierService) -> pd.DataFrame:
    records = service.get_batch_history(limit=8)
    if not records:
        return pd.DataFrame(
            columns=[
                "Дата",
                "Архив",
                "Всего файлов",
                "Обработано",
                "Пропущено",
                "Архив результата",
            ]
        )

    frame = pd.DataFrame(records)
    frame = frame.rename(
        columns={
            "created_at": "Дата",
            "source_name": "Архив",
            "total_files": "Всего файлов",
            "processed_files": "Обработано",
            "skipped_files": "Пропущено",
            "output_archive_path": "Архив результата",
        }
    )
    return frame[["Дата", "Архив", "Всего файлов", "Обработано", "Пропущено", "Архив результата"]]


def _batch_items_frame(result: dict[str, object]) -> pd.DataFrame:
    items = result.get("items", [])
    if not items:
        return pd.DataFrame(
            columns=[
                "Файл",
                "Путь в архиве",
                "Модальность",
                "Класс",
                "Уверенность",
                "Статус",
                "Комментарий",
            ]
        )

    frame = pd.DataFrame(items)
    frame = frame.rename(
        columns={
            "file_name": "Файл",
            "relative_path": "Путь в архиве",
            "modality": "Модальность",
            "predicted_label": "Класс",
            "confidence": "Уверенность",
            "status": "Статус",
            "note": "Комментарий",
        }
    )
    if "Уверенность" in frame.columns:
        frame["Уверенность"] = frame["Уверенность"].fillna(0).map(lambda value: round(float(value), 4))
    return frame[["Файл", "Путь в архиве", "Модальность", "Класс", "Уверенность", "Статус", "Комментарий"]]


def _metrics_markdown(service: ClassifierService) -> str:
    blocks = []
    for item in service.get_models():
        blocks.append(
            "\n".join(
                [
                    f"**{item['modality'].upper()}**",
                    f"- Модель: {item['model_name']} {item['model_version']}",
                    f"- Accuracy: {float(item['accuracy']):.4f}",
                    f"- Weighted F1: {float(item['weighted_f1']):.4f}",
                    f"- Артефакт: `{item['artifact_path']}`",
                ]
            )
        )
    return "\n\n".join(blocks)


def _batch_summary_markdown(result: dict[str, object]) -> str:
    distribution = result.get("label_distribution", {})
    distribution_lines = "\n".join(
        f"- {label}: {count}" for label, count in sorted(distribution.items())
    ) or "- Подходящие файлы не найдены"

    return (
        f"**Архив:** {result['source_name']}\n\n"
        f"**Всего файлов:** {result['total_files']}\n\n"
        f"**Обработано:** {result['processed_files']}\n\n"
        f"**Пропущено:** {result['skipped_files']}\n\n"
        f"**Распределение по классам:**\n{distribution_lines}"
    )


def build_ui(service: ClassifierService) -> gr.Blocks:
    service.ensure_ready()
    text_examples = [[item] for item in build_demo_texts()]
    image_examples = [[str(path)] for path in sorted(Path(service.settings.demo_examples_dir).glob("*.png"))]
    archive_examples = (
        [[str(service.settings.demo_archive_path)]]
        if service.settings.demo_archive_path.exists()
        else []
    )

    def classify_text_for_ui(text: str):
        result = service.classify_text(text)
        return _result_markdown(result), result["probabilities"], _history_frame(service)

    def classify_image_for_ui(image):
        result = service.classify_image(image)
        return _result_markdown(result), result["probabilities"], _history_frame(service)

    def classify_archive_for_ui(archive_path: str):
        if not archive_path:
            raise gr.Error("Необходимо выбрать ZIP-архив документов.")
        result = service.classify_archive(archive_path, source_name=Path(archive_path).name)
        return (
            _batch_summary_markdown(result),
            _batch_items_frame(result),
            result["output_archive_path"],
            _batch_history_frame(service),
        )

    with gr.Blocks(title="Система пакетной классификации документов") as demo:
        gr.Markdown(
            """
            # Система классификации и пакетной сортировки документов
            Приложение определяет тип документа по тексту, по скану и умеет разложить ZIP-архив по категориям
            `Договор`, `Счет`, `Приказ`, `Служебная записка`, `Отчет`.
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Tab("Классификация текста"):
                    text_input = gr.Textbox(
                        lines=8,
                        label="Текст документа для анализа",
                        placeholder="Вставьте текст договора, счета, приказа, служебной записки или отчета.",
                    )
                    text_button = gr.Button("Классифицировать текст документа", variant="primary")
                    text_summary = gr.Markdown()
                    text_scores = gr.Label(label="Распределение вероятностей")
                    gr.Examples(examples=text_examples, inputs=text_input, label="Примеры документов")

                with gr.Tab("Классификация скана"):
                    image_input = gr.Image(type="filepath", label="Скан или изображение документа")
                    image_button = gr.Button("Классифицировать скан документа", variant="primary")
                    image_summary = gr.Markdown()
                    image_scores = gr.Label(label="Распределение вероятностей")
                    gr.Examples(examples=image_examples, inputs=image_input, label="Примеры сканов документов")

                with gr.Tab("Пакетная сортировка архива"):
                    gr.Markdown(
                        """
                        Загрузите ZIP-архив с файлами `txt`, `md`, `docx`, `png`, `jpg`, `jpeg`, `bmp` или `webp`.
                        Система классифицирует содержимое, сформирует сводный отчет и вернет архив с разложением по типам документов.
                        """
                    )
                    archive_input = gr.File(
                        type="filepath",
                        file_types=[".zip"],
                        label="ZIP-архив для пакетной сортировки",
                    )
                    archive_button = gr.Button("Обработать архив документов", variant="primary")
                    archive_summary = gr.Markdown()
                    archive_items = gr.Dataframe(
                        label="Результаты обработки архива",
                        interactive=False,
                        wrap=True,
                    )
                    archive_output = gr.File(label="Сформированный архив с результатами")
                    if archive_examples:
                        gr.Examples(
                            examples=archive_examples,
                            inputs=archive_input,
                            label="Демонстрационный архив",
                        )

            with gr.Column(scale=2):
                gr.Markdown("## Метрики обученных моделей")
                gr.Markdown(_metrics_markdown(service))
                history_table = gr.Dataframe(
                    value=_history_frame(service),
                    label="История одиночных запросов",
                    interactive=False,
                    wrap=True,
                )
                batch_history_table = gr.Dataframe(
                    value=_batch_history_frame(service),
                    label="История пакетной сортировки",
                    interactive=False,
                    wrap=True,
                )
                refresh_button = gr.Button("Обновить историю")

        with gr.Row():
            gr.Image(value=str(service.settings.text_confusion_figure), label="Матрица ошибок текстовой модели")
            gr.Image(value=str(service.settings.image_confusion_figure), label="Матрица ошибок модели сканов")
            gr.Image(value=str(service.settings.model_comparison_figure), label="Сравнение моделей")

        text_button.click(
            fn=classify_text_for_ui,
            inputs=text_input,
            outputs=[text_summary, text_scores, history_table],
        )
        image_button.click(
            fn=classify_image_for_ui,
            inputs=image_input,
            outputs=[image_summary, image_scores, history_table],
        )
        archive_button.click(
            fn=classify_archive_for_ui,
            inputs=archive_input,
            outputs=[archive_summary, archive_items, archive_output, batch_history_table],
        )
        refresh_button.click(
            fn=lambda: (_history_frame(service), _batch_history_frame(service)),
            outputs=[history_table, batch_history_table],
        )

    return demo
