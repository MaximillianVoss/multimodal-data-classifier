from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vkr_classifier.config import Settings  # noqa: E402
from vkr_classifier.data.image_generator import create_shape_image  # noqa: E402
from vkr_classifier.main import create_application  # noqa: E402
from vkr_classifier.service import ClassifierService  # noqa: E402


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
def star_image_bytes(settings: Settings) -> bytes:
    image = create_shape_image("Звезда", seed=1234, image_size=settings.image_size)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

