from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Settings:
    project_root: Path
    host: str = "127.0.0.1"
    port: int = 8000
    random_seed: int = 42
    image_size: tuple[int, int] = (48, 64)
    text_labels: tuple[str, ...] = ("Договор", "Счет", "Приказ", "Служебная записка", "Отчет")
    image_labels: tuple[str, ...] = ("Договор", "Счет", "Приказ", "Служебная записка", "Отчет")
    text_model_name: str = "TF-IDF + Logistic Regression Document Classifier"
    image_model_name: str = "Random Forest Document Layout Classifier"
    text_model_version: str = "2.0.0"
    image_model_version: str = "2.0.0"
    artifacts_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    tables_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)
    screenshots_dir: Path = field(init=False)
    batch_reports_dir: Path = field(init=False)
    batch_exports_dir: Path = field(init=False)
    database_path: Path = field(init=False)
    text_model_path: Path = field(init=False)
    image_model_path: Path = field(init=False)
    text_metrics_path: Path = field(init=False)
    image_metrics_path: Path = field(init=False)
    summary_metrics_path: Path = field(init=False)
    text_report_path: Path = field(init=False)
    image_report_path: Path = field(init=False)
    model_comparison_figure: Path = field(init=False)
    text_confusion_figure: Path = field(init=False)
    image_confusion_figure: Path = field(init=False)
    use_case_figure: Path = field(init=False)
    architecture_figure: Path = field(init=False)
    database_figure: Path = field(init=False)
    workflow_figure: Path = field(init=False)
    interaction_figure: Path = field(init=False)
    demo_examples_dir: Path = field(init=False)
    demo_archive_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root)
        self.artifacts_dir = self.project_root / "artifacts"
        self.models_dir = self.artifacts_dir / "models"
        self.tables_dir = self.artifacts_dir / "tables"
        self.figures_dir = self.artifacts_dir / "figures"
        self.screenshots_dir = self.artifacts_dir / "screenshots"
        self.batch_reports_dir = self.artifacts_dir / "batch_reports"
        self.batch_exports_dir = self.artifacts_dir / "batch_exports"
        self.demo_examples_dir = self.artifacts_dir / "demo_examples"
        self.demo_archive_path = self.batch_exports_dir / "demo_document_archive.zip"
        self.database_path = self.project_root / "classifier_history.sqlite3"
        self.text_model_path = self.models_dir / "text_classifier.joblib"
        self.image_model_path = self.models_dir / "image_classifier.joblib"
        self.text_metrics_path = self.tables_dir / "text_metrics.json"
        self.image_metrics_path = self.tables_dir / "image_metrics.json"
        self.summary_metrics_path = self.tables_dir / "summary_metrics.csv"
        self.text_report_path = self.tables_dir / "text_classification_report.csv"
        self.image_report_path = self.tables_dir / "image_classification_report.csv"
        self.model_comparison_figure = self.figures_dir / "model_comparison.png"
        self.text_confusion_figure = self.figures_dir / "text_confusion_matrix.png"
        self.image_confusion_figure = self.figures_dir / "image_confusion_matrix.png"
        self.use_case_figure = self.figures_dir / "use_case_diagram.png"
        self.architecture_figure = self.figures_dir / "system_architecture.png"
        self.database_figure = self.figures_dir / "database_schema.png"
        self.workflow_figure = self.figures_dir / "request_workflow.png"
        self.interaction_figure = self.figures_dir / "interaction_scheme.png"

    def ensure_directories(self) -> None:
        for path in (
            self.artifacts_dir,
            self.models_dir,
            self.tables_dir,
            self.figures_dir,
            self.screenshots_dir,
            self.batch_reports_dir,
            self.batch_exports_dir,
            self.demo_examples_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def get_settings(project_root: Path | None = None) -> Settings:
    root = project_root or Path(__file__).resolve().parents[1]
    settings = Settings(project_root=root)
    settings.ensure_directories()
    return settings
