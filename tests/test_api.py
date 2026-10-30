import pytest
from fastapi.testclient import TestClient

from support_ticket.api.main import create_app


def test_available_api(artifact):
    with TestClient(create_app(str(artifact))) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200
        info = client.get("/model-info").json()
        assert info["synthetic"]
        assert "config" not in info
        response = client.post("/predict", json={"ticket_id": "a", "text": "invoice charge"})
        assert response.status_code == 200
        assert response.headers["x-request-id"]
        assert response.json()["ticket_id"] == "a"
        batch = client.post(
            "/predict/batch",
            json={
                "tickets": [
                    {"ticket_id": "b", "text": "login password"},
                    {"ticket_id": "a", "text": "invoice charge"},
                ]
            },
        )
        assert batch.status_code == 200
        assert [r["ticket_id"] for r in batch.json()["predictions"]] == ["b", "a"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"text": ""},
        {"text": " \n"},
        {"text": None},
        {"text": 123},
        {"text": "x" * 10001},
        {"text": "hi", "unexpected": True},
    ],
)
def test_invalid_inputs(artifact, payload):
    with TestClient(create_app(str(artifact))) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        assert all("input" not in e for e in response.json()["detail"])


def test_malformed_and_batch(artifact):
    with TestClient(create_app(str(artifact))) as client:
        assert (
            client.post(
                "/predict", content="{", headers={"Content-Type": "application/json"}
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/predict", content="text", headers={"Content-Type": "text/plain"}
            ).status_code
            == 422
        )
        for tickets in [[], [{"text": "hi"}] * 101]:
            assert client.post("/predict/batch", json={"tickets": tickets}).status_code == 422


def test_body_limit(artifact, monkeypatch):
    monkeypatch.setenv("MAX_BODY_BYTES", "100")
    with TestClient(create_app(str(artifact))) as client:
        assert client.post("/predict", content="x" * 101).status_code == 413


def test_no_model(tmp_path):
    with TestClient(create_app(str(tmp_path))) as client:
        assert client.get("/health").status_code == 200
        for url in ["/ready", "/model-info"]:
            assert client.get(url).status_code == 503
        assert client.post("/predict", json={"text": "help"}).status_code == 503


def test_corrupt_model(tmp_path):
    (tmp_path / "metadata.json").write_text("not json")
    with TestClient(create_app(str(tmp_path))) as client:
        assert client.get("/ready").status_code == 503
