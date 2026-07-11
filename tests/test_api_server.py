from fastapi.testclient import TestClient

from src.generator.prompt_curation import PROMPT_CURATION_VERSION, PromptCurationManifest
from src.server.api.app import create_app
from src.server.api.manager import SessionManager


def _manifest(prompt: str = "cat") -> PromptCurationManifest:
    return PromptCurationManifest(
        version=PROMPT_CURATION_VERSION,
        user_prompt=prompt,
        anchor_count=7,
        scaffold="fixed centered subject",
        prompt_template="centered {anchor} cat",
        anchor_labels=[f"axis_{i}" for i in range(7)],
        realized_prompts=[f"centered axis_{i} cat" for i in range(7)],
        notes="",
        model={"provider": "openai", "name": "fake"},
    )


class FakeCurationService:
    def __init__(self):
        self.writes = []

    def curate(self, user_prompt):
        return _manifest(user_prompt)

    def write(self, manifest, output_path):
        self.writes.append((manifest, output_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}", encoding="utf-8")


class FakeDiffusionClient:
    def __init__(self, manifest=None):
        self._manifest = manifest or _manifest().to_dict()

    def manifest(self):
        return self._manifest

    def render(self, _z, _frame_size):
        return b"\x89PNG\r\n\x1a\n"


class FakeEEGManager:
    def __init__(self, ready=False):
        self.ready = ready
        self.retry_count = 0
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def close(self):
        self.closed = True

    def retry_now(self):
        self.retry_count += 1
        return self.status()

    def status(self):
        return {
            "state": "ready" if self.ready else "error",
            "connected": self.ready,
            "calibrated": self.ready,
            "calibration_seconds": 30.0,
            "last_error": None if self.ready else "missing headset",
            "last_connected_at": None,
            "last_calibrated_at": None,
            "next_retry_at": None,
        }

    def require_ready_reward_source(self):
        if not self.ready:
            raise RuntimeError("EEG is not ready")
        raise AssertionError("real session reward source should not be needed in this test")


def _client(tmp_path, *, eeg_ready=False, remote_manifest=None):
    curation = FakeCurationService()
    eeg = FakeEEGManager(ready=eeg_ready)
    manager = SessionManager(
        repo_root=tmp_path,
        eeg_manager=eeg,
        curation_service=curation,
        diffusion_client_factory=lambda _url, _timeout: FakeDiffusionClient(remote_manifest),
    )
    return TestClient(create_app(session_manager=manager, eeg_manager=eeg)), manager, eeg, curation


def test_health(tmp_path):
    client, _, _, _ = _client(tmp_path)
    assert client.get("/health").json() == {"ok": True}


def test_eeg_status_and_retry(tmp_path):
    client, _, eeg, _ = _client(tmp_path)

    assert client.get("/eeg/status").json()["state"] == "error"
    response = client.post("/eeg/retry")

    assert response.status_code == 200
    assert eeg.retry_count == 1


def test_start_mock_session_curates_manifest_and_starts_thread(tmp_path):
    client, manager, _, curation = _client(tmp_path)

    response = client.post(
        "/session/start",
        json={
            "prompt": " cat ",
            "mock": True,
            "server_url": "http://localhost:8766",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["running"] is True
    assert body["prompt"] == "cat"
    assert body["pid"] is None
    assert body["manifest_path"].endswith("-cat.json")
    assert curation.writes[0][0].user_prompt == "cat"
    manager.stop()


def test_start_real_session_requires_ready_eeg(tmp_path):
    client, _, _, _ = _client(tmp_path, eeg_ready=False)

    response = client.post(
        "/session/start",
        json={"prompt": "cat", "mock": False, "server_url": "http://localhost:8766"},
    )

    assert response.status_code == 409
    assert response.json()["detail"]["error"] == "EEG is not ready"


def test_duplicate_start_returns_conflict(tmp_path):
    client, manager, _, _ = _client(tmp_path)

    first = client.post("/session/start", json={"prompt": "cat", "mock": True})
    second = client.post("/session/start", json={"prompt": "dog", "mock": True})

    assert first.status_code == 200
    assert second.status_code == 409
    manager.stop()


def test_stop_without_session_is_harmless(tmp_path):
    client, _, _, _ = _client(tmp_path)

    response = client.post("/session/stop")

    assert response.status_code == 200
    assert response.json()["running"] is False


def test_stop_terminates_running_session(tmp_path):
    client, _, _, _ = _client(tmp_path)
    client.post("/session/start", json={"prompt": "cat", "mock": True})

    response = client.post("/session/stop")

    assert response.status_code == 200
    assert response.json()["running"] is False


def test_logs_clamps_line_count(tmp_path):
    client, _, _, _ = _client(tmp_path)
    manager = client.app.state.session_manager
    manager._logs.extend(str(i) for i in range(1200))

    response = client.get("/session/logs?lines=1000")

    assert response.status_code == 200
    assert len(response.json()["lines"]) == 1000
    assert response.json()["lines"][0] == "200"


def test_rejects_invalid_server_url(tmp_path):
    client, _, _, _ = _client(tmp_path)

    response = client.post("/session/start", json={"prompt": "cat", "server_url": "localhost:8766"})

    assert response.status_code == 422


def test_manifest_mismatch_returns_conflict(tmp_path):
    remote = _manifest("dog").to_dict()
    client, _, _, _ = _client(tmp_path, remote_manifest=remote)

    response = client.post("/session/start", json={"prompt": "cat", "mock": True})

    assert response.status_code == 409
    assert response.json()["detail"]["error"] == "diffusion server manifest does not match curated prompt"
