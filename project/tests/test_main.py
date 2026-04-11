from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from vkr_classifier.config import get_settings
from vkr_classifier.main import _resolve_port, create_application


def test_root_redirects_to_docs_when_ui_is_disabled(settings) -> None:
    app = create_application(settings=settings, include_ui=False)
    client = TestClient(app)
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/docs"


def test_resolve_port_returns_preferred_port_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("vkr_classifier.main._can_bind", lambda host, port: port == 8000)
    assert _resolve_port("127.0.0.1", 8000) == 8000


def test_resolve_port_uses_next_available_port(monkeypatch: pytest.MonkeyPatch) -> None:
    available_ports = {8002}
    monkeypatch.setattr("vkr_classifier.main._can_bind", lambda host, port: port in available_ports)
    assert _resolve_port("127.0.0.1", 8000, attempts=5) == 8002


def test_resolve_port_raises_when_no_ports_are_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("vkr_classifier.main._can_bind", lambda host, port: False)
    with pytest.raises(RuntimeError):
        _resolve_port("127.0.0.1", 8000, attempts=2)


def test_default_settings_use_project_directory() -> None:
    settings = get_settings()
    assert settings.project_root.name == "project"
