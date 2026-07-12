import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.common.config import Config
from src.generator.prompt_curation import PROMPT_CURATION_VERSION, PromptCurationManifest
from src.server.api.app import create_app
from src.server.api.diffusion_supervisor_client import RemoteDiffusionSupervisorClient
from src.server.api.manager import SessionManager
from src.session.frame_store import FrameStore


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


class FakeDiffusionProcessManager:
    def __init__(self, base_url="http://127.0.0.1:8766", raise_exc=None):
        self.base_url = base_url
        self.raise_exc = raise_exc
        self.restarts = []
        self.stopped = False

    def restart(self, manifest_path, manifest=None):
        self.restarts.append(manifest_path)
        self.manifest = manifest
        if self.raise_exc is not None:
            raise self.raise_exc
        return {}

    def stop(self):
        self.stopped = True


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


def _client(tmp_path, *, eeg_ready=False, remote_manifest=None, diffusion_process_manager=None):
    curation = FakeCurationService()
    eeg = FakeEEGManager(ready=eeg_ready)
    manager = SessionManager(
        repo_root=tmp_path,
        eeg_manager=eeg,
        curation_service=curation,
        diffusion_client_factory=lambda _url, _timeout: FakeDiffusionClient(remote_manifest),
        diffusion_process_manager=diffusion_process_manager,
        frame_store_factory=lambda: FrameStore(tmp_path),
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
    assert body["reward_estimate"] == 0.0
    assert body["optimizer_state"] == "calibrate"
    assert body["step_index"] == 0
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


class ReadyEEGManager(FakeEEGManager):
    def __init__(self, reward_source):
        super().__init__(ready=True)
        self._reward_source = reward_source

    def require_ready_reward_source(self):
        return self._reward_source


class FakeRewardSource:
    def read_reward(self):
        return SimpleNamespace(r=0.0, raw_faa=0.0)


class RecordingFinalizer:
    def __init__(self):
        self.calls = []

    def finalize(self, png, subject):
        self.calls.append((png, subject))
        return b"FINAL:" + png


def test_real_session_runs_finalize_and_writes_target_frame(tmp_path):
    # A real (non-mock) session captures snapshots, so the last frame flows
    # through the finalizer and lands as target_frame.png for the frontend.
    finalizer = RecordingFinalizer()
    eeg = ReadyEEGManager(FakeRewardSource())
    manager = SessionManager(
        repo_root=tmp_path,
        eeg_manager=eeg,
        curation_service=FakeCurationService(),
        diffusion_client_factory=lambda _url, _timeout: FakeDiffusionClient(),
        finalizer_factory=lambda: finalizer,
        frame_store_factory=lambda: FrameStore(tmp_path),
    )
    client = TestClient(create_app(session_manager=manager, eeg_manager=eeg))

    response = client.post("/session/start", json={"prompt": "cat", "mock": False})
    assert response.status_code == 200

    # The optimizer stops on its own at max_steps; wait for the thread to exit.
    manager.stop()
    for _ in range(50):
        if not manager.status()["running"]:
            break
        time.sleep(0.05)

    assert manager.status()["running"] is False
    raw = FakeDiffusionClient().render(None, None)
    assert (tmp_path / "session_end.png").read_bytes() == raw
    assert (tmp_path / "target_frame.png").read_bytes() == b"FINAL:" + raw
    assert finalizer.calls and finalizer.calls[0][1] == "cat"
    status = manager.status()
    assert status["phase"] == "completed"
    assert status["result_ready"] is True
    assert status["result_refined"] is True
    assert status["finalize_error"] is None


def test_retry_finalization_refines_saved_raw_frame(tmp_path):
    finalizer = RecordingFinalizer()
    manager = SessionManager(
        repo_root=tmp_path,
        curation_service=FakeCurationService(),
        finalizer_factory=lambda: finalizer,
        frame_store_factory=lambda: FrameStore(tmp_path),
    )
    FrameStore(tmp_path).save_end(b"raw-png")
    manager._prompt = "cat"
    client = TestClient(create_app(session_manager=manager, eeg_manager=FakeEEGManager()))

    response = client.post("/session/finalize/retry")
    assert response.status_code == 200
    for _ in range(50):
        if not manager.status()["running"]:
            break
        time.sleep(0.01)

    assert (tmp_path / "target_frame.png").read_bytes() == b"FINAL:raw-png"
    assert manager.status()["result_refined"] is True
    assert finalizer.calls == [(b"raw-png", "cat")]


def test_retry_finalization_requires_completed_raw_frame(tmp_path):
    manager = SessionManager(
        repo_root=tmp_path,
        curation_service=FakeCurationService(),
        finalizer_factory=RecordingFinalizer,
        frame_store_factory=lambda: FrameStore(tmp_path),
    )
    manager._prompt = "cat"
    client = TestClient(create_app(session_manager=manager, eeg_manager=FakeEEGManager()))

    response = client.post("/session/finalize/retry")

    assert response.status_code == 404
    assert response.json()["detail"] == "session_end.png not found"


def test_make_finalizer_uses_injected_factory(tmp_path):
    sentinel = object()
    manager = SessionManager(
        repo_root=tmp_path,
        curation_service=FakeCurationService(),
        finalizer_factory=lambda: sentinel,
    )

    assert manager._make_finalizer() is sentinel


def test_make_finalizer_returns_none_when_disabled(tmp_path):
    config = Config()
    config.generator.finalize_enabled = False
    manager = SessionManager(
        repo_root=tmp_path,
        curation_service=FakeCurationService(),
        config=config,
    )

    assert manager._make_finalizer() is None


def test_manifest_mismatch_returns_conflict(tmp_path):
    remote = _manifest("dog").to_dict()
    client, _, _, _ = _client(tmp_path, remote_manifest=remote)

    response = client.post("/session/start", json={"prompt": "cat", "mock": True})

    assert response.status_code == 409
    assert response.json()["detail"]["error"] == "diffusion server manifest does not match curated prompt"


def test_managed_diffusion_restarts_with_curated_manifest_before_session(tmp_path):
    process = FakeDiffusionProcessManager(base_url="http://managed:8766")
    client, manager, _, curation = _client(
        tmp_path,
        remote_manifest=_manifest("cat").to_dict(),
        diffusion_process_manager=process,
    )

    response = client.post(
        "/session/start",
        json={"prompt": "cat", "mock": True, "server_url": "http://ignored:9999"},
    )

    assert response.status_code == 200
    assert process.restarts
    assert process.restarts[0].name.endswith("-cat.json")
    assert curation.writes[0][1] == process.restarts[0]
    manager.stop()


def test_managed_diffusion_startup_failure_returns_bad_gateway(tmp_path):
    process = FakeDiffusionProcessManager(raise_exc=RuntimeError("boom"))
    client, _, _, _ = _client(tmp_path, diffusion_process_manager=process)

    response = client.post("/session/start", json={"prompt": "cat", "mock": True})

    assert response.status_code == 502
    assert "managed diffusion startup failed" in response.json()["detail"]


def test_remote_diffusion_supervisor_client_posts_manifest(monkeypatch, tmp_path):
    manifest = _manifest("cat")
    calls = []

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"render_url": "http://gpu:8766"}

    def fake_post(url, json, timeout):
        calls.append((url, json, timeout))
        return FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)
    client = RemoteDiffusionSupervisorClient("http://gpu:8010/", timeout_s=12)

    client.restart(tmp_path / "cat.json", manifest)

    assert client.base_url == "http://gpu:8766"
    assert calls[0][0] == "http://gpu:8010/diffusion/restart"
    assert calls[0][1]["filename"] == "cat.json"
    assert calls[0][1]["manifest"]["user_prompt"] == "cat"
    assert calls[0][2] == 12
