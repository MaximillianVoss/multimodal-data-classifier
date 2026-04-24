from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
import zipfile

from docx import Document


TEXT_FILE_EXTENSIONS = {".txt", ".md", ".docx"}
IMAGE_FILE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(slots=True)
class BatchItemResult:
    file_name: str
    relative_path: str
    modality: str | None
    predicted_label: str | None
    confidence: float | None
    processing_time_ms: int | None
    model_name: str | None
    model_version: str | None
    status: str
    note: str


def classify_modality(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in TEXT_FILE_EXTENSIONS:
        return "text"
    if suffix in IMAGE_FILE_EXTENSIONS:
        return "image"
    return None


def extract_text_from_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        document = Document(path)
        return "\n".join(paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip())
    return path.read_text(encoding="utf-8", errors="ignore")


def iter_supported_files(root_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in root_dir.rglob("*")
        if path.is_file() and classify_modality(path) is not None and "__MACOSX" not in path.parts
    )


def safe_extract_archive(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            target_path = (destination / member.filename).resolve()
            if not str(target_path).startswith(str(destination.resolve())):
                raise ValueError("Архив содержит небезопасные пути.")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target_path.open("wb") as target:
                shutil.copyfileobj(source, target)


def ensure_unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def build_output_archive(source_dir: Path, output_archive_path: Path) -> Path:
    output_archive_path.parent.mkdir(parents=True, exist_ok=True)
    base_name = output_archive_path.with_suffix("")
    archive_path = Path(shutil.make_archive(str(base_name), "zip", root_dir=source_dir))
    if archive_path != output_archive_path:
        archive_path.replace(output_archive_path)
    return output_archive_path


def create_demo_archive(
    output_archive_path: Path,
    *,
    text_examples: dict[str, str],
    image_examples: dict[str, Path],
) -> Path:
    output_archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="demo_docs_") as temporary_dir:
        root_dir = Path(temporary_dir)
        for label, text in text_examples.items():
            file_name = f"{label.lower().replace(' ', '_')}.txt"
            (root_dir / file_name).write_text(text, encoding="utf-8")

        for label, image_path in image_examples.items():
            target_name = f"{label.lower().replace(' ', '_')}.png"
            shutil.copy2(image_path, root_dir / target_name)

        return build_output_archive(root_dir, output_archive_path)
