from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def _draw_text_line(
    draw: ImageDraw.ImageDraw,
    left: int,
    top: int,
    width: int,
    *,
    height: int = 4,
    fill: int = 45,
) -> None:
    draw.rounded_rectangle((left, top, left + width, top + height), radius=1, fill=fill)


def _draw_paragraph(
    draw: ImageDraw.ImageDraw,
    rng: np.random.Generator,
    left: int,
    top: int,
    max_width: int,
    lines: int,
    *,
    line_step: int = 12,
    min_ratio: float = 0.58,
    max_ratio: float = 0.98,
) -> int:
    current_top = top
    for index in range(lines):
        width = int(max_width * float(rng.uniform(min_ratio, max_ratio)))
        if index == lines - 1:
            width = int(width * 0.8)
        _draw_text_line(draw, left, current_top, width)
        current_top += line_step
    return current_top


def _draw_signature(draw: ImageDraw.ImageDraw, left: int, top: int, width: int) -> None:
    draw.line((left, top, left + width, top), fill=70, width=2)
    _draw_text_line(draw, left + 5, top + 8, int(width * 0.5), height=3, fill=90)


def _draw_table(
    draw: ImageDraw.ImageDraw,
    left: int,
    top: int,
    width: int,
    height: int,
    rows: int,
    cols: int,
) -> None:
    draw.rectangle((left, top, left + width, top + height), outline=40, width=2)
    row_step = height / rows
    col_step = width / cols
    for row in range(1, rows):
        y = int(top + row * row_step)
        draw.line((left, y, left + width, y), fill=70, width=1)
    for col in range(1, cols):
        x = int(left + col * col_step)
        draw.line((x, top, x, top + height), fill=70, width=1)


def _draw_bar_chart(
    draw: ImageDraw.ImageDraw,
    rng: np.random.Generator,
    left: int,
    top: int,
    width: int,
    height: int,
    bars: int,
) -> None:
    draw.rectangle((left, top, left + width, top + height), outline=45, width=2)
    base_y = top + height - 8
    step = width // (bars + 1)
    bar_width = max(6, step // 2)
    for index in range(bars):
        bar_height = int(height * float(rng.uniform(0.25, 0.78)))
        x = left + step * (index + 1) - bar_width // 2
        draw.rectangle((x, base_y - bar_height, x + bar_width, base_y), fill=55)


def _create_page(rng: np.random.Generator) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    canvas = Image.new("L", (480, 640), color=250)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((18, 18, 462, 622), outline=180, width=2)
    if rng.random() > 0.45:
        draw.rectangle((24, 24, 466, 626), outline=225, width=1)
    return canvas, draw


def _draw_contract(draw: ImageDraw.ImageDraw, rng: np.random.Generator) -> None:
    _draw_text_line(draw, 150, 48, 190, height=6, fill=35)
    _draw_text_line(draw, 175, 66, 140, height=4, fill=50)
    _draw_paragraph(draw, rng, 58, 110, 360, 4)
    _draw_paragraph(draw, rng, 58, 178, 360, 5)
    _draw_paragraph(draw, rng, 58, 258, 360, 4)
    _draw_text_line(draw, 68, 380, 120, height=4, fill=70)
    _draw_text_line(draw, 288, 380, 120, height=4, fill=70)
    _draw_signature(draw, 78, 520, 130)
    _draw_signature(draw, 282, 520, 130)


def _draw_invoice(draw: ImageDraw.ImageDraw, rng: np.random.Generator) -> None:
    draw.rectangle((52, 50, 428, 112), outline=45, width=2)
    _draw_text_line(draw, 68, 68, 150, height=6, fill=35)
    _draw_text_line(draw, 280, 68, 110, height=6, fill=35)
    _draw_text_line(draw, 68, 88, 130, fill=70)
    _draw_text_line(draw, 280, 88, 95, fill=70)
    _draw_table(draw, 52, 148, 376, 235, rows=7, cols=4)
    draw.rectangle((278, 410, 428, 492), outline=40, width=2)
    _draw_text_line(draw, 292, 426, 92, fill=50)
    _draw_text_line(draw, 292, 448, 104, fill=35)
    _draw_text_line(draw, 292, 470, 78, fill=50)
    _draw_signature(draw, 286, 548, 110)


def _draw_order(draw: ImageDraw.ImageDraw, rng: np.random.Generator) -> None:
    _draw_text_line(draw, 186, 42, 108, height=4, fill=60)
    _draw_text_line(draw, 168, 70, 144, height=8, fill=30)
    _draw_text_line(draw, 148, 92, 184, height=4, fill=55)
    for index, top in enumerate((150, 214, 278, 342), start=1):
        draw.ellipse((58, top - 2, 70, top + 10), outline=50, width=1)
        _draw_paragraph(draw, rng, 82, top, 300, 3 if index < 4 else 2)
    _draw_signature(draw, 286, 534, 118)


def _draw_memo(draw: ImageDraw.ImageDraw, rng: np.random.Generator) -> None:
    _draw_text_line(draw, 56, 54, 140, height=6, fill=35)
    draw.rectangle((272, 46, 404, 116), outline=45, width=2)
    _draw_text_line(draw, 284, 60, 84, fill=55)
    _draw_text_line(draw, 284, 80, 96, fill=55)
    _draw_text_line(draw, 284, 100, 72, fill=55)
    _draw_text_line(draw, 58, 136, 250, height=4, fill=55)
    _draw_paragraph(draw, rng, 58, 176, 350, 4)
    _draw_paragraph(draw, rng, 58, 250, 350, 3)
    _draw_text_line(draw, 58, 372, 150, fill=70)
    _draw_signature(draw, 268, 530, 124)


def _draw_report(draw: ImageDraw.ImageDraw, rng: np.random.Generator) -> None:
    _draw_text_line(draw, 168, 46, 144, height=7, fill=35)
    _draw_text_line(draw, 128, 66, 224, height=4, fill=55)
    _draw_bar_chart(draw, rng, 58, 110, 166, 118, bars=5)
    _draw_table(draw, 252, 110, 158, 118, rows=5, cols=3)
    _draw_paragraph(draw, rng, 58, 270, 352, 4)
    _draw_paragraph(draw, rng, 58, 336, 352, 3)
    _draw_text_line(draw, 58, 456, 210, fill=55)
    _draw_text_line(draw, 58, 478, 156, fill=55)
    _draw_signature(draw, 280, 546, 116)


DOCUMENT_LAYOUTS = {
    "Договор": _draw_contract,
    "Счет": _draw_invoice,
    "Приказ": _draw_order,
    "Служебная записка": _draw_memo,
    "Отчет": _draw_report,
}


def create_document_image(
    label: str,
    seed: int,
    image_size: tuple[int, int] = (48, 64),
) -> Image.Image:
    if label not in DOCUMENT_LAYOUTS:
        raise ValueError(f"Неизвестный класс документа: {label}")

    rng = np.random.default_rng(seed)
    canvas, draw = _create_page(rng)
    DOCUMENT_LAYOUTS[label](draw, rng)

    array = np.asarray(canvas, dtype=np.float32)
    shadow = np.linspace(0.0, float(rng.uniform(6.0, 14.0)), array.shape[0], dtype=np.float32)[:, None]
    array = np.clip(array - shadow, 0, 255)
    array = np.clip(array + rng.normal(loc=0.0, scale=5.5, size=array.shape), 0, 255)

    image = Image.fromarray(array.astype(np.uint8), mode="L")
    if rng.random() > 0.35:
        image = image.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.15, 0.8))))
    return image.resize(image_size, Image.Resampling.LANCZOS)


def create_shape_image(
    label: str,
    seed: int,
    image_size: tuple[int, int] = (48, 64),
) -> Image.Image:
    return create_document_image(label=label, seed=seed, image_size=image_size)


def image_to_vector(image: Image.Image, image_size: tuple[int, int]) -> np.ndarray:
    prepared = image.convert("L").resize(image_size, Image.Resampling.LANCZOS)
    array = np.asarray(prepared, dtype=np.float32) / 255.0
    return array.reshape(-1)


def build_image_dataset(
    seed: int,
    labels: tuple[str, ...],
    samples_per_class: int = 180,
    image_size: tuple[int, int] = (48, 64),
) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    vectors: list[np.ndarray] = []
    target: list[str] = []

    for label in labels:
        for _ in range(samples_per_class):
            image_seed = int(rng.integers(0, 10_000_000))
            image = create_document_image(label=label, seed=image_seed, image_size=image_size)
            vectors.append(image_to_vector(image, image_size))
            target.append(label)

    return np.vstack(vectors), target


def save_demo_examples(
    output_dir: Path,
    labels: tuple[str, ...],
    image_size: tuple[int, int],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    for index, label in enumerate(labels, start=1):
        image = create_document_image(label=label, seed=index * 97, image_size=image_size)
        file_path = output_dir / f"{index:02d}_{label}.png"
        image.save(file_path)
        mapping[label] = str(file_path)
    return mapping
