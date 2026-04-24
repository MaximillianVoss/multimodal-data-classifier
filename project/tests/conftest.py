from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
import zipfile

import pytest
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from vkr_classifier.config import Settings
from vkr_classifier.data.image_generator import create_document_image
from vkr_classifier.main import create_application
from vkr_classifier.service import ClassifierService


@pytest.fixture(scope="session")
def settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    project_root = tmp_path_factory.mktemp("classifier_project")
    config = Settings(project_root=project_root)
    config.ensure_directories()
    return config


@pytest.fixture(scope="session")
def service(settings: Settings) -> ClassifierService:
    instance = ClassifierService(settings)
    instance.ensure_ready()
    return instance


@pytest.fixture(scope="session")
def client(settings: Settings) -> TestClient:
    app = create_application(settings=settings, include_ui=False)
    return TestClient(app)


@pytest.fixture()
def invoice_image_bytes(settings: Settings) -> bytes:
    image = create_document_image("Счет", seed=1234, image_size=settings.image_size)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture()
def batch_archive_path(settings: Settings) -> Path:
    archive_path = settings.batch_exports_dir / "test_batch_input.zip"
    text_path = settings.demo_examples_dir / "test_contract.txt"
    image_path = settings.demo_examples_dir / "test_order.png"

    text_path.write_text(
        (
            "Настоящий договор заключен между заказчиком и исполнителем. "
            "Предмет договора включает сопровождение информационной системы. "
            "Стоимость услуг фиксируется в приложении и оплачивается по этапам."
        ),
        encoding="utf-8",
    )
    create_document_image("Приказ", seed=4321, image_size=settings.image_size).save(image_path)

    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(text_path, arcname="incoming/contract.txt")
        archive.write(image_path, arcname="incoming/order.png")
        archive.writestr("incoming/ignore.bin", b"raw")

    return archive_path
