from __future__ import annotations

from fastapi.testclient import TestClient


def test_text_endpoint_returns_prediction(client: TestClient) -> None:
    response = client.post(
        "/api/text/classify",
        json={"text": "Банк пересмотрел прогноз денежного потока и улучшил устойчивость финансовой модели."},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] == "Финансы"
    assert payload["modality"] == "text"


def test_image_endpoint_returns_prediction(client: TestClient, star_image_bytes: bytes) -> None:
    response = client.post(
        "/api/image/classify",
        files={"file": ("star.png", star_image_bytes, "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] == "Звезда"
    assert payload["modality"] == "image"


def test_history_endpoint_contains_logged_requests(client: TestClient) -> None:
    response = client.get("/api/history?limit=5")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) >= 2
