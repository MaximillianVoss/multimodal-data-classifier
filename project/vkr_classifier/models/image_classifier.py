from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from vkr_classifier.config import Settings
from vkr_classifier.data.image_generator import build_image_dataset, image_to_vector


@dataclass(slots=True)
class ImageModelArtifact:
    classifier: KNeighborsClassifier
    labels: list[str]
    metrics: dict[str, float]
    class_report: dict[str, dict[str, float] | float]
    confusion: list[list[int]]
    training_size: int
    test_size: int
    image_size: tuple[int, int]
    trained_at: str
    model_name: str
    model_version: str


def train_image_model(settings: Settings) -> ImageModelArtifact:
    features, labels = build_image_dataset(
        seed=settings.random_seed,
        labels=settings.image_labels,
        samples_per_class=240,
        image_size=settings.image_size,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=settings.random_seed,
        stratify=labels,
    )

    classifier = KNeighborsClassifier(
        n_neighbors=1,
        weights="distance",
    )
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)

    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    confusion = confusion_matrix(y_test, predictions, labels=list(settings.image_labels))
    weighted = report["weighted avg"]
    metrics = {
        "accuracy": float(report["accuracy"]),
        "precision": float(weighted["precision"]),
        "recall": float(weighted["recall"]),
        "f1_score": float(weighted["f1-score"]),
    }

    return ImageModelArtifact(
        classifier=classifier,
        labels=list(settings.image_labels),
        metrics=metrics,
        class_report=report,
        confusion=confusion.tolist(),
        training_size=len(x_train),
        test_size=len(x_test),
        image_size=settings.image_size,
        trained_at=datetime.now(timezone.utc).isoformat(),
        model_name=settings.image_model_name,
        model_version=settings.image_model_version,
    )


def prepare_image_vector(artifact: ImageModelArtifact, image: Image.Image) -> list[list[float]]:
    vector = image_to_vector(image=image, image_size=artifact.image_size)
    return [vector.tolist()]


def save_image_model(artifact: ImageModelArtifact, path: Path) -> None:
    joblib.dump(artifact, path)


def load_image_model(path: Path) -> ImageModelArtifact:
    return joblib.load(path)
