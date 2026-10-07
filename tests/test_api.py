"""HTTP API tests. The model is not loaded; ``run_verify`` is stubbed."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from attire_verification.api import app
from attire_verification.models import PERFECT_SCORE, VerifyResult


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ATTIRE_WARMUP", "0")
    monkeypatch.delenv("SERVICE_API_KEY", raising=False)

    def _fake_verify(image, **kwargs):
        return VerifyResult(
            score=PERFECT_SCORE,
            failReasons=[],
            imagePath=str(image),
            role=kwargs.get("role"),
        )

    monkeypatch.setattr("attire_verification.api.run_verify", _fake_verify)
    with TestClient(app) as test_client:
        yield test_client


def _photo() -> dict:
    return {"image": ("photo.jpg", io.BytesIO(b"not-a-real-jpeg"), "image/jpeg")}


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_verify_returns_cli_payload(client: TestClient):
    response = client.post("/verify", files=_photo(), data={"role": "fc"})
    assert response.status_code == 200
    body = response.json()
    assert body["score"] == PERFECT_SCORE
    assert body["role"] == "FC"
    assert body["imagePath"] == "photo.jpg"
    assert "expected" not in body
    assert "match" not in body


def test_verify_defaults_role_to_br(client: TestClient):
    response = client.post("/verify", files=_photo())
    assert response.status_code == 200
    assert response.json()["role"] == "BR"


def test_verify_rejects_unknown_role(client: TestClient):
    response = client.post("/verify", files=_photo(), data={"role": "INTERN"})
    assert response.status_code == 422
    assert "unknown role" in response.json()["detail"]


def test_verify_rejects_non_image(client: TestClient):
    files = {"image": ("notes.txt", io.BytesIO(b"hello"), "text/plain")}
    response = client.post("/verify", files=files)
    assert response.status_code == 422


def test_verify_rejects_empty_image(client: TestClient):
    files = {"image": ("photo.jpg", io.BytesIO(b""), "image/jpeg")}
    response = client.post("/verify", files=files)
    assert response.status_code == 422


def test_verify_requires_api_key_when_configured(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SERVICE_API_KEY", "secret")
    denied = client.post("/verify", files=_photo())
    assert denied.status_code == 401

    allowed = client.post("/verify", files=_photo(), headers={"X-API-Key": "secret"})
    assert allowed.status_code == 200
