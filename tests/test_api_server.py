import io

from fastapi.testclient import TestClient

from scripts import api_server


class FakeProcess:
    _next_pid = 1000

    def __init__(self, cmd, **kwargs):
        FakeProcess._next_pid += 1
        self.pid = FakeProcess._next_pid
        self.cmd = cmd
        self.kwargs = kwargs
        self.stdout = io.StringIO("ready\n")
        self._exit_code = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self._exit_code

    def terminate(self):
        self.terminated = True
        self._exit_code = -15

    def wait(self, timeout=None):
        if self._exit_code is None:
            self._exit_code = 0
        return self._exit_code

    def kill(self):
        self.killed = True
        self._exit_code = -9

    def send_signal(self, _signal):
        self.killed = True
        self._exit_code = -9


def _client(monkeypatch):
    processes = []

    def fake_popen(cmd, **kwargs):
        process = FakeProcess(cmd, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(api_server.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(api_server, "manager", api_server.SessionManager())
    monkeypatch.setattr(api_server.manager, "_start_reader", lambda _process: None)
    return TestClient(api_server.app), processes


def test_health(monkeypatch):
    client, _ = _client(monkeypatch)
    assert client.get("/health").json() == {"ok": True}


def test_start_mock_session_builds_command(monkeypatch):
    client, processes = _client(monkeypatch)

    response = client.post(
        "/session/start",
        json={
            "prompt": " cat ",
            "mock": True,
            "baseline_seconds": 5,
            "server_url": "http://localhost:8766",
        },
    )

    assert response.status_code == 200
    assert response.json()["running"] is True
    cmd = processes[0].cmd
    assert cmd[:3] == [api_server.sys.executable, "scripts/run_real_eeg_optimizer.py", "--mock"]
    assert "--baseline" in cmd
    assert "5.0" in cmd
    assert "--server-url" in cmd
    assert "http://localhost:8766" in cmd


def test_start_real_session_omits_mock(monkeypatch):
    client, processes = _client(monkeypatch)

    response = client.post(
        "/session/start",
        json={"mock": False, "baseline_seconds": 0, "server_url": "https://example.test"},
    )

    assert response.status_code == 200
    assert "--mock" not in processes[0].cmd


def test_duplicate_start_returns_conflict(monkeypatch):
    client, _ = _client(monkeypatch)

    first = client.post("/session/start", json={"mock": True})
    second = client.post("/session/start", json={"mock": True})

    assert first.status_code == 200
    assert second.status_code == 409


def test_stop_without_session_is_harmless(monkeypatch):
    client, _ = _client(monkeypatch)

    response = client.post("/session/stop")

    assert response.status_code == 200
    assert response.json()["running"] is False


def test_stop_terminates_running_session(monkeypatch):
    client, processes = _client(monkeypatch)
    client.post("/session/start", json={"mock": True})

    response = client.post("/session/stop")

    assert response.status_code == 200
    assert response.json()["running"] is False
    assert processes[0].terminated is True


def test_logs_clamps_line_count(monkeypatch):
    client, _ = _client(monkeypatch)
    api_server.manager._logs.extend(str(i) for i in range(1200))

    response = client.get("/session/logs?lines=1000")

    assert response.status_code == 200
    assert len(response.json()["lines"]) == 1000
    assert response.json()["lines"][0] == "200"


def test_rejects_invalid_server_url(monkeypatch):
    client, _ = _client(monkeypatch)

    response = client.post("/session/start", json={"server_url": "localhost:8766"})

    assert response.status_code == 422
