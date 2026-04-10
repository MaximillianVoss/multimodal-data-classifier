from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent
DOCS_DIR = WORKSPACE_DIR / "docs"
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vkr_classifier.config import get_settings  # noqa: E402


OUTPUT_NAME = "Презентация ВКР.pptx"
TEMPLATE_NAME = "Шаблон презентации.pptx"
TITLE_COLOR = RGBColor(59, 55, 152)
TEXT_COLOR = RGBColor(44, 44, 44)
ACCENT = RGBColor(92, 113, 196)
CARD_FILL = RGBColor(247, 248, 252)
CARD_ALT = RGBColor(236, 241, 255)


def remove_slide(prs: Presentation, index: int) -> None:
    slide_id_list = prs.slides._sldIdLst  # type: ignore[attr-defined]
    slide = slide_id_list[index]
    rel_id = slide.rId
    prs.part.drop_rel(rel_id)
    slide_id_list.remove(slide)


def set_text_run_style(run, *, size: int, color: RGBColor = TEXT_COLOR, bold: bool = False) -> None:
    run.font.name = "Montserrat"
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold


def set_title(shape, text: str, font_size: int = 28) -> None:
    text_frame = shape.text_frame
    text_frame.clear()
    paragraph = text_frame.paragraphs[0]
    paragraph.text = text
    paragraph.alignment = PP_ALIGN.LEFT
    for run in paragraph.runs:
        set_text_run_style(run, size=font_size, color=TITLE_COLOR, bold=True)


def set_body(shape, lines: list[str], *, font_size: int = 20, color: RGBColor = TEXT_COLOR, bold_first: bool = False) -> None:
    text_frame = shape.text_frame
    text_frame.clear()
    text_frame.word_wrap = True
    for index, line in enumerate(lines):
        paragraph = text_frame.paragraphs[0] if index == 0 else text_frame.add_paragraph()
        paragraph.text = line
        paragraph.level = 0
        paragraph.space_after = Pt(6)
        for run in paragraph.runs:
            set_text_run_style(run, size=font_size, color=color, bold=bold_first and index == 0)


def add_textbox(
    slide,
    left,
    top,
    width,
    height,
    lines: list[str],
    *,
    font_size: int = 20,
    color: RGBColor = TEXT_COLOR,
    bold_first: bool = False,
):
    textbox = slide.shapes.add_textbox(left, top, width, height)
    set_body(textbox, lines, font_size=font_size, color=color, bold_first=bold_first)
    return textbox


def add_card(
    slide,
    left,
    top,
    width,
    height,
    title: str,
    body_lines: list[str],
    *,
    fill_color: RGBColor = CARD_FILL,
    title_size: int = 17,
    body_size: int = 14,
) -> None:
    shape = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.color.rgb = ACCENT
    shape.line.width = Pt(1.25)

    text_frame = shape.text_frame
    text_frame.clear()
    text_frame.word_wrap = True
    text_frame.margin_left = Pt(10)
    text_frame.margin_right = Pt(10)
    text_frame.margin_top = Pt(10)
    text_frame.margin_bottom = Pt(8)

    title_paragraph = text_frame.paragraphs[0]
    title_paragraph.text = title
    title_paragraph.space_after = Pt(6)
    for run in title_paragraph.runs:
        set_text_run_style(run, size=title_size, color=TITLE_COLOR, bold=True)

    for line in body_lines:
        paragraph = text_frame.add_paragraph()
        paragraph.text = line
        paragraph.space_after = Pt(3)
        for run in paragraph.runs:
            set_text_run_style(run, size=body_size)


def add_stat_card(
    slide,
    left,
    top,
    width,
    height,
    title: str,
    value_lines: list[str],
    *,
    fill_color: RGBColor = CARD_ALT,
) -> None:
    add_card(
        slide,
        left,
        top,
        width,
        height,
        title,
        value_lines,
        fill_color=fill_color,
        title_size=18,
        body_size=16,
    )


def clear_content_placeholders(slide) -> None:
    for index in (0, 2):
        if index < len(slide.shapes):
            shape = slide.shapes[index]
            if hasattr(shape, "text_frame"):
                shape.text_frame.clear()


def set_slide_number(shape, number: int) -> None:
    set_body(shape, [str(number)], font_size=14, color=TITLE_COLOR)


def build_presentation() -> Path:
    settings = get_settings(PROJECT_DIR)
    summary = pd.read_csv(settings.summary_metrics_path)
    image_metrics = summary[summary["model"] == "Модель изображений"].iloc[0]
    text_metrics = summary[summary["model"] == "Текстовая модель"].iloc[0]

    presentation = Presentation(DOCS_DIR / TEMPLATE_NAME)

    for index in range(len(presentation.slides) - 1, 8, -1):
        remove_slide(presentation, index)
    remove_slide(presentation, 0)

    title_slide = presentation.slides[0]
    set_title(
        title_slide.shapes[3],
        "Разработка системы классификации изображений и текстовых данных на языке Python с фронтендом на low-code платформе",
        font_size=24,
    )
    set_body(title_slide.shapes[1], ["Студент: Медведев Илья", "Группа: не указана"], font_size=18)
    set_body(title_slide.shapes[0], ["Научный руководитель: данные уточняются"], font_size=16)

    relevance_slide = presentation.slides[1]
    clear_content_placeholders(relevance_slide)
    set_title(relevance_slide.shapes[1], "Актуальность темы")
    add_card(
        relevance_slide,
        Inches(0.62),
        Inches(1.55),
        Inches(2.9),
        Inches(1.55),
        "Данные растут",
        ["Текстовые и графические файлы требуют автоматической сортировки."],
        fill_color=CARD_FILL,
        title_size=16,
        body_size=13,
    )
    add_card(
        relevance_slide,
        Inches(3.67),
        Inches(1.55),
        Inches(3.0),
        Inches(1.55),
        "Мультимодальность",
        ["Современным системам нужен единый контур для текста и изображений."],
        fill_color=CARD_ALT,
        title_size=15,
        body_size=13,
    )
    add_card(
        relevance_slide,
        Inches(6.82),
        Inches(1.55),
        Inches(2.9),
        Inches(1.55),
        "Python и ML",
        ["Библиотеки машинного обучения позволяют быстро собрать прикладной прототип."],
        fill_color=CARD_FILL,
        title_size=16,
        body_size=13,
    )
    add_card(
        relevance_slide,
        Inches(9.87),
        Inches(1.55),
        Inches(2.75),
        Inches(1.55),
        "Low-code UI",
        ["Gradio дает low-code интерфейс без ручной frontend-разработки."],
        fill_color=CARD_ALT,
        title_size=16,
        body_size=13,
    )
    add_card(
        relevance_slide,
        Inches(0.95),
        Inches(3.6),
        Inches(11.15),
        Inches(1.35),
        "Цель работы",
        ["Разработать локальное приложение для классификации текста и изображений с API, low-code интерфейсом и тестами."],
        fill_color=CARD_ALT,
        title_size=20,
        body_size=17,
    )
    set_body(
        relevance_slide.shapes[2],
        ["Результатом стал единый Python-проект, готовый к запуску в PyCharm."],
        font_size=12,
    )
    set_slide_number(relevance_slide.shapes[3], 2)

    stack_slide = presentation.slides[2]
    clear_content_placeholders(stack_slide)
    set_title(stack_slide.shapes[1], "Выбранные технические решения")
    tech_cards = [
        ("Python 3.12", ["единая среда разработки и запуска"]),
        ("FastAPI", ["серверная часть и REST API"]),
        ("Gradio", ["low-code интерфейс"]),
        ("scikit-learn", ["обучение и инференс моделей"]),
        ("SQLite", ["история запросов и метаданные"]),
        ("pytest", ["автоматические проверки проекта"]),
    ]
    positions = [
        (Inches(0.8), Inches(1.55)),
        (Inches(4.15), Inches(1.55)),
        (Inches(7.5), Inches(1.55)),
        (Inches(0.8), Inches(3.35)),
        (Inches(4.15), Inches(3.35)),
        (Inches(7.5), Inches(3.35)),
    ]
    for (title, body), (left, top) in zip(tech_cards, positions, strict=False):
        add_card(stack_slide, left, top, Inches(2.95), Inches(1.45), title, body, fill_color=CARD_FILL)
    set_body(
        stack_slide.shapes[2],
        ["Проект открывается в PyCharm и запускается локально одной командой."],
        font_size=12,
    )
    set_slide_number(stack_slide.shapes[3], 3)

    architecture_slide = presentation.slides[3]
    clear_content_placeholders(architecture_slide)
    set_title(architecture_slide.shapes[1], "Архитектурное проектирование")
    architecture_slide.shapes.add_picture(
        str(settings.architecture_figure),
        left=Inches(0.9),
        top=Inches(1.5),
        width=Inches(11.45),
        height=Inches(3.55),
    )
    add_card(
        architecture_slide,
        Inches(0.95),
        Inches(5.3),
        Inches(3.55),
        Inches(0.78),
        "UI",
        ["Gradio принимает ввод пользователя"],
        fill_color=CARD_FILL,
        title_size=16,
        body_size=12,
    )
    add_card(
        architecture_slide,
        Inches(4.8),
        Inches(5.3),
        Inches(3.05),
        Inches(0.78),
        "API",
        ["FastAPI маршрутизирует запросы"],
        fill_color=CARD_ALT,
        title_size=16,
        body_size=12,
    )
    add_card(
        architecture_slide,
        Inches(8.15),
        Inches(5.3),
        Inches(4.05),
        Inches(0.78),
        "Модели и данные",
        ["scikit-learn и SQLite"],
        fill_color=CARD_FILL,
        title_size=16,
        body_size=12,
    )
    set_body(
        architecture_slide.shapes[2],
        ["Схема показывает полный путь данных: от интерфейса до модели и БД."],
        font_size=12,
    )
    set_slide_number(architecture_slide.shapes[3], 4)

    implementation_slide = presentation.slides[4]
    clear_content_placeholders(implementation_slide)
    set_title(implementation_slide.shapes[1], "Программная реализация")
    implementation_picture = implementation_slide.shapes.add_picture(
        str(settings.screenshots_dir / "ui_text_prediction.png"),
        left=Inches(4.85),
        top=Inches(1.38),
        width=Inches(7.4),
        height=Inches(4.82),
    )
    implementation_picture.crop_bottom = 0.47
    implementation_picture.crop_left = 0.02
    implementation_picture.crop_right = 0.02
    implementation_picture.crop_top = 0.06
    add_card(
        implementation_slide,
        Inches(0.75),
        Inches(1.75),
        Inches(3.95),
        Inches(1.68),
        "Что реализовано",
        ["REST API и сервисный слой", "две модели классификации", "SQLite-журнал запросов"],
        fill_color=CARD_FILL,
    )
    add_card(
        implementation_slide,
        Inches(0.75),
        Inches(3.72),
        Inches(3.95),
        Inches(1.52),
        "Пользовательские сценарии",
        ["ввод текста", "загрузка изображения", "просмотр метрик и истории"],
        fill_color=CARD_ALT,
    )
    set_body(
        implementation_slide.shapes[2],
        ["Интерфейс доступен по адресу /ui и не требует отдельной клиентской сборки."],
        font_size=12,
    )
    set_slide_number(implementation_slide.shapes[3], 5)

    results_slide = presentation.slides[5]
    clear_content_placeholders(results_slide)
    set_title(results_slide.shapes[1], "Результаты экспериментальных исследований")
    results_slide.shapes.add_picture(
        str(settings.model_comparison_figure),
        left=Inches(6.0),
        top=Inches(1.55),
        width=Inches(6.05),
        height=Inches(3.85),
    )
    add_stat_card(
        results_slide,
        Inches(0.75),
        Inches(1.55),
        Inches(4.9),
        Inches(1.5),
        "Текстовая модель",
        [f"Accuracy: {text_metrics['accuracy']:.4f}", f"F1-score: {text_metrics['f1_score']:.4f}"],
        fill_color=CARD_ALT,
    )
    add_stat_card(
        results_slide,
        Inches(0.75),
        Inches(3.15),
        Inches(4.9),
        Inches(1.5),
        "Модель изображений",
        [f"Accuracy: {image_metrics['accuracy']:.4f}", f"F1-score: {image_metrics['f1_score']:.4f}"],
        fill_color=CARD_FILL,
    )
    add_textbox(
        results_slide,
        Inches(0.82),
        Inches(5.05),
        Inches(5.0),
        Inches(0.65),
        ["6 автотестов, покрытие 87%, среднее время инференса: 1 мс и 25.5 мс."],
        font_size=13,
        color=TITLE_COLOR,
        bold_first=True,
    )
    set_body(
        results_slide.shapes[2],
        ["Полученные метрики подтверждают устойчивую работу обеих моделей в интерактивном режиме."],
        font_size=12,
    )
    set_slide_number(results_slide.shapes[3], 6)

    value_slide = presentation.slides[6]
    clear_content_placeholders(value_slide)
    set_title(value_slide.shapes[1], "Практическая значимость работы")
    benefit_cards = [
        ("Локальный запуск", ["Не нужен внешний сервер БД или отдельная клиентская сборка."]),
        ("Воспроизводимость", ["Модели, графики и документы собираются из одного репозитория."]),
        ("Расширяемость", ["Можно подключить реальные датасеты и добавить новые классы."]),
        ("Готовность к защите", ["Есть код, тесты, записка, презентация и интерфейс."]),
    ]
    positions = [
        (Inches(0.8), Inches(1.65)),
        (Inches(6.75), Inches(1.65)),
        (Inches(0.8), Inches(3.55)),
        (Inches(6.75), Inches(3.55)),
    ]
    for index, ((title, body), (left, top)) in enumerate(zip(benefit_cards, positions, strict=False)):
        add_card(
            value_slide,
            left,
            top,
            Inches(5.0),
            Inches(1.45),
            title,
            body,
            fill_color=CARD_ALT if index % 2 else CARD_FILL,
            title_size=18,
            body_size=14,
        )
    set_body(
        value_slide.shapes[2],
        ["Проект можно использовать как учебный и демонстрационный стенд мультимодальной классификации данных."],
        font_size=12,
    )
    set_slide_number(value_slide.shapes[3], 7)

    thanks_slide = presentation.slides[7]
    set_title(thanks_slide.shapes[3], "Спасибо за внимание", font_size=26)
    set_body(thanks_slide.shapes[1], ["Студент: Медведев Илья", "Группа: не указана"], font_size=18)
    set_body(thanks_slide.shapes[0], ["Научный руководитель: данные уточняются"], font_size=16)

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DOCS_DIR / OUTPUT_NAME
    presentation.save(output_path)
    return output_path


if __name__ == "__main__":
    result = build_presentation()
    print(result)
