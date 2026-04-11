from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from vkr_classifier.config import Settings
from vkr_classifier.data.text_samples import build_text_dataset


@dataclass(slots=True)
class TextModelArtifact:
    pipeline: Pipeline
    labels: list[str]
    metrics: dict[str, float]
    class_report: dict[str, dict[str, float] | float]
    confusion: list[list[int]]
    training_size: int
    test_size: int
    trained_at: str
    model_name: str
    model_version: str


def train_text_model(settings: Settings) -> TextModelArtifact:
    texts, labels = build_text_dataset(seed=settings.random_seed)
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=settings.random_seed,
        stratify=labels,
    )

    pipeline = Pipeline(
        steps=[
            ("vectorizer", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("classifier", LogisticRegression(max_iter=2500, C=3.0)),
        ]
    )
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)

    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    confusion = confusion_matrix(y_test, predictions, labels=list(settings.text_labels))
    weighted = report["weighted avg"]
    metrics = {
        "accuracy": float(report["accuracy"]),
        "precision": float(weighted["precision"]),
        "recall": float(weighted["recall"]),
        "f1_score": float(weighted["f1-score"]),
    }

    return TextModelArtifact(
        pipeline=pipeline,
        labels=list(settings.text_labels),
        metrics=metrics,
        class_report=report,
        confusion=confusion.tolist(),
        training_size=len(x_train),
        test_size=len(x_test),
        trained_at=datetime.now(timezone.utc).isoformat(),
        model_name=settings.text_model_name,
        model_version=settings.text_model_version,
    )


def save_text_model(artifact: TextModelArtifact, path: Path) -> None:
    joblib.dump(artifact, path)


def load_text_model(path: Path) -> TextModelArtifact:
    return joblib.load(path)

