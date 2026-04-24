from __future__ import annotations

from fastapi.testclient import TestClient


def test_text_endpoint_returns_prediction(client: TestClient) -> None:
    response = client.post(
        "/api/text/classify",
        json={"text": "Приказом утверждается состав рабочей группы и сроки исполнения. Контроль исполнения возлагается на заместителя руководителя."},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] == "Приказ"
    assert payload["modality"] == "text"


def test_image_endpoint_returns_prediction(client: TestClient, invoice_image_bytes: bytes) -> None:
    response = client.post(
        "/api/image/classify",
        files={"file": ("invoice.png", invoice_image_bytes, "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] == "Счет"
    assert payload["modality"] == "image"


def test_history_endpoint_contains_logged_requests(client: TestClient) -> None:
    response = client.get("/api/history?limit=5")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) >= 2


def test_batch_archive_endpoint_returns_summary(client: TestClient, batch_archive_path) -> None:
    response = client.post(
        "/api/batch/classify-archive",
        files={"file": (batch_archive_path.name, batch_archive_path.read_bytes(), "application/zip")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["processed_files"] == 2
    assert payload["skipped_files"] == 1
    assert payload["source_name"] == batch_archive_path.name
