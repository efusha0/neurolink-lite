"""The one-command demo: UI build decisions, static serving, and the command end to end."""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import neurolink.cli as cli
import neurolink.realtime.demo as demo
from neurolink.realtime.demo import (
    DemoSetupError,
    build_ui,
    built_ui,
    ensure_ui,
    node_is_supported,
    parse_node_version,
    port_is_free,
    run_demo,
    ui_needs_build,
)
from neurolink.realtime.demo_model import DEMO_MODEL_PATH, DemoBundle, load_bundle, save_bundle
from neurolink.realtime.pseudo_online import REST, Recording
from neurolink.realtime.server import Hub, create_app
from tests.test_demo_live import tiny_eegnet
from tests.test_realtime_core import PowerModel

SFREQ = 160.0
CHANNELS = ("C3", "Cz", "C4", "Pz")
PAGE = "<!doctype html><title>NeuroLink-lite</title>"
runner = CliRunner()


def make_frontend(root: Path) -> Path:
    frontend = root / "frontend"
    (frontend / "src").mkdir(parents=True)
    for name in ("index.html", "package.json", "package-lock.json", "src/main.ts"):
        (frontend / name).write_text("{}", encoding="utf-8")
    return frontend


def set_mtime(path: Path, seconds_from_now: float) -> None:
    stamp = time.time() + seconds_from_now
    os.utime(path, (stamp, stamp))


class FakeTools:
    """Stands in for node and npm: records the commands and writes what a build would."""

    def __init__(self, node_version: str = "v22.12.0", fail: str | None = None) -> None:
        self.node_version = node_version
        self.fail = fail
        self.commands: list[list[str]] = []

    def which(self, name: str) -> str | None:
        return f"/tools/{name}"

    def run(self, command: list[str], cwd: Path) -> str:
        self.commands.append([Path(command[0]).name, *command[1:]])
        if command[1:] == ["--version"]:
            return self.node_version + "\n"
        if self.fail == command[-1]:
            raise DemoSetupError(f"`npm {command[-1]}` failed")
        if command[1:] == ["ci"]:
            (cwd / "node_modules").mkdir(exist_ok=True)
            (cwd / "node_modules" / ".package-lock.json").write_text("{}", encoding="utf-8")
        if command[1:] == ["run", "build"]:
            (cwd / "dist").mkdir(exist_ok=True)
            (cwd / "dist" / "index.html").write_text(PAGE, encoding="utf-8")
        return ""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def synthetic_recording(seconds: float = 6.0) -> Recording:
    n = int(seconds * SFREQ)
    rng = np.random.default_rng(3)
    data = rng.normal(size=(n, len(CHANNELS))).astype(np.float32)
    data[:, 2] += 20 * np.sin(2 * np.pi * 12 * np.arange(n) / SFREQ)  # "right": power on C4
    labels = np.array([REST] * 160 + [1] * (n - 160), dtype=np.int8)
    return Recording(7, 12, data, CHANNELS, SFREQ, labels)


# ------------------------------------------------------------------------------- Node.js
def test_node_versions_are_parsed_and_compared_with_the_minimum() -> None:
    assert parse_node_version("v22.12.0\n") == (22, 12, 0)
    assert parse_node_version("24.11.1") == (24, 11, 1)
    assert parse_node_version("v23.0.0-nightly2024") == (23, 0, 0)
    assert parse_node_version("node: command not found") is None
    assert node_is_supported((22, 12, 0)) and node_is_supported((24, 0, 0))
    assert not node_is_supported((22, 11, 9)) and not node_is_supported((20, 19, 0))


# ------------------------------------------------------------------------------ UI build
def test_ui_is_built_when_missing_or_older_than_its_sources(tmp_path: Path) -> None:
    frontend = make_frontend(tmp_path)
    assert ui_needs_build(frontend) and built_ui(frontend) is None

    (frontend / "dist").mkdir()
    page = frontend / "dist" / "index.html"
    page.write_text(PAGE, encoding="utf-8")
    set_mtime(page, +60)
    assert not ui_needs_build(frontend)
    assert built_ui(frontend) == frontend / "dist"

    set_mtime(frontend / "src" / "main.ts", +120)  # a source file changed after the build
    assert ui_needs_build(frontend)


def test_first_run_installs_dependencies_and_builds_then_reuses_the_result(tmp_path: Path) -> None:
    frontend = make_frontend(tmp_path)
    tools, lines = FakeTools(), []
    dist = ensure_ui(frontend, echo=lines.append, which=tools.which, run=tools.run)
    assert dist == frontend / "dist" and (dist / "index.html").is_file()
    assert tools.commands == [["node", "--version"], ["npm", "ci"], ["npm", "run", "build"]]
    assert any("npm ci" in line for line in lines) and any("build" in line for line in lines)

    # Second run: nothing to do, Node.js is not even looked for.
    set_mtime(dist / "index.html", +60)
    assert ensure_ui(frontend, which=lambda _name: None, run=tools.run) == dist
    assert len(tools.commands) == 3

    # --rebuild-ui: build again, but the installed dependencies are still current.
    set_mtime(frontend / "node_modules" / ".package-lock.json", +60)
    ensure_ui(frontend, rebuild=True, which=tools.which, run=tools.run)
    assert tools.commands[3:] == [["node", "--version"], ["npm", "run", "build"]]

    # A newer lock file (after `git pull`) means the dependencies are installed again.
    set_mtime(frontend / "package-lock.json", +120)
    ensure_ui(frontend, which=tools.which, run=tools.run)
    assert tools.commands[5:] == [["node", "--version"], ["npm", "ci"], ["npm", "run", "build"]]


def test_missing_or_old_node_is_explained(tmp_path: Path) -> None:
    frontend = make_frontend(tmp_path)
    with pytest.raises(DemoSetupError) as missing:
        build_ui(frontend, which=lambda _name: None)
    message = str(missing.value)
    assert "Node.js was not found" in message
    assert "Node.js 22.12 or newer" in message and "https://nodejs.org" in message

    def only_node(name: str) -> str | None:
        return "/tools/node" if name == "node" else None

    with pytest.raises(DemoSetupError, match="npm was not found"):
        build_ui(frontend, which=only_node)

    old = FakeTools(node_version="v18.19.1")
    with pytest.raises(DemoSetupError) as too_old:
        build_ui(frontend, which=old.which, run=old.run)
    assert "Node.js 18.19.1" in str(too_old.value) and "22.12 or newer" in str(too_old.value)
    assert old.commands == [["node", "--version"]]  # nothing was installed or built


def test_build_failures_and_a_missing_frontend_are_reported(tmp_path: Path) -> None:
    with pytest.raises(DemoSetupError, match="UI sources were not found"):
        build_ui(tmp_path / "nowhere")

    frontend = make_frontend(tmp_path)
    broken = FakeTools(fail="build")
    with pytest.raises(DemoSetupError, match="npm build"):
        build_ui(frontend, which=broken.which, run=broken.run)

    # The real runner: a failing command becomes a DemoSetupError with the tool's output.
    import sys

    with pytest.raises(DemoSetupError) as failed:
        demo._run([sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"], tmp_path)
    assert "boom" in str(failed.value)
    with pytest.raises(DemoSetupError, match="could not run"):
        demo._run([str(tmp_path / "no-such-tool")], tmp_path)
    assert demo._run([sys.executable, "-c", "print('v22.12.0')"], tmp_path).strip() == "v22.12.0"


# -------------------------------------------------------------------------- static serving
def test_the_built_ui_is_served_next_to_the_api(tmp_path: Path) -> None:
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text(PAGE, encoding="utf-8")
    (tmp_path / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")

    with TestClient(create_app(Hub(), info={"threshold": 0.7}, static_dir=tmp_path)) as client:
        page = client.get("/")
        assert page.status_code == 200 and page.text == PAGE
        assert "text/html" in page.headers["content-type"]
        assert client.get("/assets/app.js").text == "console.log(1)"
        # The API routes are not shadowed by the page.
        assert client.get("/info").json()["threshold"] == 0.7
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/missing.js").status_code == 404
        with client.websocket_connect("/ws"):
            pass

    with TestClient(create_app(Hub())) as client:  # no UI: API only, as before
        assert client.get("/").status_code == 404
        assert client.get("/health").status_code == 200


# -------------------------------------------------------------------------- shipped model
def test_the_shipped_demo_model_loads_and_never_saw_the_demo_subject() -> None:
    """data/models/demo_model.pt is committed so that the demo runs without training."""
    assert DEMO_MODEL_PATH.is_file(), "the demo model must ship with the repository"
    assert DEMO_MODEL_PATH.stat().st_size < 1_000_000
    bundle = load_bundle(DEMO_MODEL_PATH)
    meta = bundle.meta
    assert len(bundle.ch_names) == 21 and bundle.band == (8.0, 30.0) and bundle.car
    assert 0.5 < bundle.threshold < 1.0
    assert meta["candidate"] == "cross/eegnet/causal"
    assert meta["demo_subject"] not in meta["train_subjects"]  # no leakage into the demo
    assert len(meta["train_subjects"]) == meta["n_train_subjects"]


# ------------------------------------------------------------------------------------ CLI
def test_demo_without_a_model_points_to_train_demo(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["demo", "--model-path", str(tmp_path / "missing.pt")])
    assert result.exit_code == 1
    assert "was not found" in result.output and "neurolink train-demo" in result.output


def demo_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, port: int) -> list[str]:
    """A demo run on a tiny model and a synthetic recording instead of the real ones."""
    bundle = DemoBundle(
        model=tiny_eegnet(), ch_names=CHANNELS, band=(8.0, 30.0), car=True, threshold=0.6,
        meta={"demo_subject": 7, "candidate": "test", "n_train_subjects": 3},
    )  # fmt: skip
    model = save_bundle(bundle, tmp_path / "demo.pt")
    monkeypatch.setattr(cli, "_load_recording", lambda *_args: synthetic_recording())
    return ["demo", "--model-path", str(model), "--raw-dir", str(tmp_path), "--no-open",
            "--results-dir", str(tmp_path), "--port", str(port)]  # fmt: skip


def test_demo_says_what_to_install_when_the_ui_cannot_be_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_node(**_kwargs: Any) -> Path:
        return build_ui(make_frontend(tmp_path), which=lambda _name: None)

    monkeypatch.setattr(demo, "ensure_ui", no_node)
    result = runner.invoke(cli.app, demo_args(tmp_path, monkeypatch, free_port()))
    assert result.exit_code == 1, result.output
    assert "Cannot start the demo" in result.output
    assert "Node.js 22.12 or newer" in result.output and "nodejs.org" in result.output
    assert "downloading S007 run 12 from PhysioNet" in result.output  # nothing in --raw-dir


def test_demo_reports_a_failed_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args = demo_args(tmp_path, monkeypatch, free_port())

    def offline(*_args: Any) -> Any:
        raise OSError("no route to host")

    monkeypatch.setattr(cli, "_load_recording", offline)
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 1
    assert "no route to host" in result.output and "physionet.org" in result.output


@pytest.mark.lsl
def test_demo_command_serves_the_page_and_live_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("pylsl")
    from websockets.sync.client import connect

    static = tmp_path / "dist"
    static.mkdir()
    (static / "index.html").write_text(PAGE, encoding="utf-8")
    seen: dict[str, Any] = {}

    def ready(url: str) -> None:
        seen["page"] = urllib.request.urlopen(url, timeout=5).read().decode()
        seen["info"] = json.loads(urllib.request.urlopen(url + "info", timeout=5).read())
        with connect(url.replace("http", "ws") + "ws") as ws:
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline and "prediction" not in seen:
                message = json.loads(ws.recv(timeout=15.0))
                if message["type"] == "prediction":
                    seen["prediction"] = message

    bundle = DemoBundle(PowerModel(), ("C3", "C4"), (8.0, 30.0), False, 0.5)
    port = free_port()
    try:
        stats = run_demo(
            bundle, synthetic_recording(), static, port=port, on_ready=ready, duration_s=0.2,
            extra_info={"cursor_travel_time_s": 1.0},
        )  # fmt: skip
    except TimeoutError as exc:
        pytest.skip(f"LSL unavailable: {exc}")
    assert seen["page"] == PAGE
    assert seen["info"]["cursor_travel_time_s"] == 1.0 and seen["info"]["step_s"] == 0.25
    assert seen["prediction"]["replay_pass"] == 1 and seen["prediction"]["true"] in (
        "rest",
        "right",
    )
    assert stats["samples_sent"] > 0 and stats["passes"] >= 1
    assert port_is_free("127.0.0.1", port)  # the server is gone

    # A failing `on_ready` stops the demo and is raised to the caller.
    def broken(_url: str) -> None:
        raise RuntimeError("browser exploded")

    with pytest.raises(RuntimeError, match="browser exploded"):
        run_demo(bundle, synthetic_recording(), static, port=free_port(), on_ready=broken)

    # The whole command, with the UI build replaced by the page above.
    monkeypatch.setattr(demo, "ensure_ui", lambda **_kwargs: static)
    args = [*demo_args(tmp_path, monkeypatch, free_port()), "--duration", "1"]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert "NeuroLink-lite is running: http://127.0.0.1:" in result.output
    assert "has never seen S007" in result.output and "stopped after" in result.output

    # Without --no-open the page is opened in the browser once the server is up.
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url) or True)
    port = free_port()
    args = [a for a in demo_args(tmp_path, monkeypatch, port) if a != "--no-open"]
    result = runner.invoke(cli.app, [*args, "--duration", "0.5"])
    assert result.exit_code == 0, result.output
    assert opened == [f"http://127.0.0.1:{port}/"]


def test_demo_refuses_a_port_that_is_taken() -> None:
    bundle = DemoBundle(PowerModel(), ("C3", "C4"), (8.0, 30.0), False, 0.5)
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = int(busy.getsockname()[1])
        assert not port_is_free("127.0.0.1", port)
        with pytest.raises(DemoSetupError, match="--port"):
            run_demo(bundle, synthetic_recording(), None, port=port)
