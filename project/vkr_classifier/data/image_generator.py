from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def _regular_polygon(
    center: tuple[float, float],
    radius: float,
    sides: int,
    rotation: float,
) -> list[tuple[float, float]]:
    cx, cy = center
    return [
        (
            cx + radius * math.cos(rotation + 2 * math.pi * step / sides),
            cy + radius * math.sin(rotation + 2 * math.pi * step / sides),
        )
        for step in range(sides)
    ]


def _star_polygon(
    center: tuple[float, float],
    outer_radius: float,
    inner_radius: float,
    rotation: float,
) -> list[tuple[float, float]]:
    cx, cy = center
    points: list[tuple[float, float]] = []
    for step in range(10):
        radius = outer_radius if step % 2 == 0 else inner_radius
        angle = rotation + step * math.pi / 5
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return points


def create_shape_image(
    label: str,
    seed: int,
    image_size: tuple[int, int] = (32, 32),
) -> Image.Image:
    rng = np.random.default_rng(seed)
    canvas_size = 128
    background = int(rng.integers(5, 35))
    canvas = Image.new("L", (canvas_size, canvas_size), color=background)
    draw = ImageDraw.Draw(canvas)

    cx = canvas_size / 2 + float(rng.uniform(-8, 8))
    cy = canvas_size / 2 + float(rng.uniform(-8, 8))
    radius = float(rng.uniform(28, 42))
    fill = int(rng.integers(185, 245))
    outline = min(fill + 10, 255)
    rotation = float(rng.uniform(0, math.pi))

    if label == "Круг":
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=fill, outline=outline)
    elif label == "Квадрат":
        points = _regular_polygon((cx, cy), radius, sides=4, rotation=rotation)
        draw.polygon(points, fill=fill, outline=outline)
    elif label == "Треугольник":
        points = _regular_polygon((cx, cy), radius, sides=3, rotation=rotation)
        draw.polygon(points, fill=fill, outline=outline)
    elif label == "Звезда":
        points = _star_polygon((cx, cy), outer_radius=radius, inner_radius=radius * 0.45, rotation=rotation)
        draw.polygon(points, fill=fill, outline=outline)
    else:
        raise ValueError(f"Неизвестный класс изображения: {label}")

    if rng.random() > 0.4:
        blur_radius = float(rng.uniform(0.2, 1.5))
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    array = np.asarray(canvas, dtype=np.float32)
    noise = rng.normal(loc=0.0, scale=10.0, size=array.shape)
    array = np.clip(array + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(array, mode="L")
    return image.resize(image_size, Image.Resampling.LANCZOS)


def image_to_vector(image: Image.Image, image_size: tuple[int, int]) -> np.ndarray:
    prepared = image.convert("L").resize(image_size, Image.Resampling.LANCZOS)
    array = np.asarray(prepared, dtype=np.float32) / 255.0
    return array.reshape(-1)


def build_image_dataset(
    seed: int,
    labels: tuple[str, ...],
    samples_per_class: int = 180,
    image_size: tuple[int, int] = (32, 32),
) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    vectors: list[np.ndarray] = []
    target: list[str] = []

    for label in labels:
        for _ in range(samples_per_class):
            image_seed = int(rng.integers(0, 10_000_000))
            image = create_shape_image(label=label, seed=image_seed, image_size=image_size)
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
        image = create_shape_image(label=label, seed=index * 97, image_size=image_size)
        file_path = output_dir / f"{index:02d}_{label}.png"
        image.save(file_path)
        mapping[label] = str(file_path)
    return mapping

