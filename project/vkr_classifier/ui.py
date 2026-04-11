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


def build_ui(service: ClassifierService) -> gr.Blocks:
    service.ensure_ready()
    text_examples = [[item] for item in build_demo_texts()]
    image_examples = [[str(path)] for path in sorted(Path(service.settings.demo_examples_dir).glob("*.png"))]

    def classify_text_for_ui(text: str):
        result = service.classify_text(text)
        return _result_markdown(result), result["probabilities"], _history_frame(service)

    def classify_image_for_ui(image):
        result = service.classify_image(image)
        return _result_markdown(result), result["probabilities"], _history_frame(service)

    with gr.Blocks(title="Система классификации данных") as demo:
        gr.Markdown(
            """
            # Система классификации изображений и текстовых данных
            Веб-интерфейс построен на low-code платформе Gradio и работает поверх серверной части FastAPI.
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Tab("Классификация текста"):
                    text_input = gr.Textbox(
                        lines=6,
                        label="Текст для анализа",
                        placeholder="Введите новостной или тематический текст на русском языке.",
                    )
                    text_button = gr.Button("Классифицировать текст", variant="primary")
                    text_summary = gr.Markdown()
                    text_scores = gr.Label(label="Распределение вероятностей")
                    gr.Examples(examples=text_examples, inputs=text_input, label="Примеры текстов")

                with gr.Tab("Классификация изображения"):
                    image_input = gr.Image(type="pil", label="Изображение для анализа")
                    image_button = gr.Button("Классифицировать изображение", variant="primary")
                    image_summary = gr.Markdown()
                    image_scores = gr.Label(label="Распределение вероятностей")
                    gr.Examples(examples=image_examples, inputs=image_input, label="Примеры фигур")

            with gr.Column(scale=2):
                gr.Markdown("## Метрики обученных моделей")
                gr.Markdown(_metrics_markdown(service))
                history_table = gr.Dataframe(
                    value=_history_frame(service),
                    label="История последних запросов",
                    interactive=False,
                    wrap=True,
                )
                refresh_button = gr.Button("Обновить историю")

        with gr.Row():
            gr.Image(value=str(service.settings.text_confusion_figure), label="Матрица ошибок текстовой модели")
            gr.Image(value=str(service.settings.image_confusion_figure), label="Матрица ошибок модели изображений")
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
        refresh_button.click(fn=lambda: _history_frame(service), outputs=history_table)

    return demo
