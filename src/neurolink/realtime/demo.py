"""The one-command demo: replay streamer, online classifier, server and the built UI.

``neurolink demo`` runs everything in one process and serves the browser UI from the same
port as the WebSocket, so nothing but a browser is needed once the UI has been built. The
UI is built on the first run (``frontend/dist`` is not committed), which needs Node.js;
everything that can go wrong before the demo starts is reported as a ``DemoSetupError``
with a message that says what to do.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import socket
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from neurolink.config import PROJECT_ROOT
from neurolink.realtime.demo_model import DemoBundle
from neurolink.realtime.live import LiveSystem, build_live
from neurolink.realtime.pseudo_online import Recording
from neurolink.realtime.streamer import DEFAULT_CHUNK, EEG_STREAM, MARKER_STREAM, ReplayStreamer

log = logging.getLogger(__name__)

FRONTEND_DIR = PROJECT_ROOT / "frontend"
UI_DIST_NAME = "dist"
MIN_NODE = (22, 12)
"""Oldest Node.js the UI tool chain supports (vite 8 and vitest 5; see frontend/package.json)."""
NODE_HINT = (
    f"Install Node.js {MIN_NODE[0]}.{MIN_NODE[1]} or newer (the LTS installer from "
    "https://nodejs.org includes npm), open a new terminal and run the command again. "
    "Only the first run needs it: the built page is kept in frontend/dist."
)
_UI_SOURCES = ("index.html", "package.json", "package-lock.json", "vite.config.ts", "tsconfig.json")
SERVER_START_TIMEOUT_S = 30.0

Echo = Callable[[str], None]
Which = Callable[[str], str | None]


class DemoSetupError(RuntimeError):
    """The demo cannot start; the message tells the user what to do about it."""


# ------------------------------------------------------------------------------ UI build
def parse_node_version(text: str) -> tuple[int, int, int] | None:
    """``"v22.12.0\\n"`` -> ``(22, 12, 0)``; ``None`` if it is not a version."""
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)\S*", text.strip())
    return (int(match[1]), int(match[2]), int(match[3])) if match else None


def node_is_supported(version: tuple[int, int, int]) -> bool:
    return version[:2] >= MIN_NODE


def _newest(paths: list[Path]) -> float:
    return max((p.stat().st_mtime for p in paths if p.is_file()), default=0.0)


def ui_needs_build(frontend: Path = FRONTEND_DIR) -> bool:
    """True when ``dist`` is missing or older than the UI sources."""
    built = frontend / UI_DIST_NAME / "index.html"
    if not built.is_file():
        return True
    sources = [frontend / name for name in _UI_SOURCES]
    sources += [p for p in (frontend / "src").rglob("*") if p.is_file()]
    return _newest(sources) > built.stat().st_mtime


def _dependencies_are_stale(frontend: Path) -> bool:
    """True when ``node_modules`` is missing or was installed from another lock file."""
    installed = frontend / "node_modules" / ".package-lock.json"
    lock = frontend / "package-lock.json"
    if not installed.is_file():
        return True
    return lock.is_file() and lock.stat().st_mtime > installed.stat().st_mtime


def _run(command: list[str], cwd: Path) -> str:
    """Run a build step; return its output or raise ``DemoSetupError`` with the tail of it."""
    try:
        done = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
    except OSError as exc:
        raise DemoSetupError(f"could not run {command[0]!r}: {exc}") from exc
    output = (done.stdout or "") + (done.stderr or "")
    if done.returncode != 0:
        tail = "\n".join(output.strip().splitlines()[-15:])
        shown = " ".join([Path(command[0]).stem, *command[1:]])
        raise DemoSetupError(f"`{shown}` failed in {cwd}:\n{tail}")
    return output


def build_ui(
    frontend: Path = FRONTEND_DIR,
    echo: Echo = lambda _line: None,
    which: Which = shutil.which,
    run: Callable[[list[str], Path], str] = _run,
) -> Path:
    """Install the UI dependencies if needed and build ``frontend/dist``; returns its path."""
    if not (frontend / "package.json").is_file():
        raise DemoSetupError(
            f"the UI sources were not found in {frontend}. Run the demo from a clone of the "
            "repository."
        )
    node, npm = which("node"), which("npm")
    if node is None or npm is None:
        missing = "Node.js" if node is None else "npm"
        raise DemoSetupError(
            f"the browser UI has to be built once and {missing} was not found. {NODE_HINT}"
        )
    version = parse_node_version(run([node, "--version"], frontend))
    if version is None or not node_is_supported(version):
        found = ".".join(map(str, version)) if version else "an unknown version"
        raise DemoSetupError(
            f"the browser UI has to be built once, but Node.js {found} ({node}) is too old "
            f"for its build tools. {NODE_HINT}"
        )
    if _dependencies_are_stale(frontend):
        echo("installing the UI dependencies (npm ci) ...")
        run([npm, "ci"], frontend)
    echo("building the UI (npm run build) ...")
    run([npm, "run", "build"], frontend)
    dist = frontend / UI_DIST_NAME
    if not (dist / "index.html").is_file():
        raise DemoSetupError(f"the UI build finished but {dist / 'index.html'} is missing")
    return dist


def ensure_ui(
    frontend: Path = FRONTEND_DIR,
    rebuild: bool = False,
    echo: Echo = lambda _line: None,
    which: Which = shutil.which,
    run: Callable[[list[str], Path], str] = _run,
) -> Path:
    """Path of the built UI, building it first when it is missing, outdated or ``rebuild``."""
    if rebuild or ui_needs_build(frontend):
        return build_ui(frontend, echo, which, run)
    return frontend / UI_DIST_NAME


def built_ui(frontend: Path = FRONTEND_DIR) -> Path | None:
    """The built UI if there is one (``neurolink serve`` offers it, but never builds it)."""
    dist = frontend / UI_DIST_NAME
    return dist if (dist / "index.html").is_file() else None


# -------------------------------------------------------------------------------- running
def port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def run_demo(
    bundle: DemoBundle,
    recording: Recording,
    static_dir: Path | None,
    host: str = "127.0.0.1",
    port: int = 8000,
    extra_info: dict[str, Any] | None = None,
    on_ready: Callable[[str], None] | None = None,
    duration_s: float | None = None,
    chunk: int = DEFAULT_CHUNK,
) -> dict[str, Any]:
    """Replay ``recording`` in a loop and serve predictions and the UI until interrupted.

    Blocks until Ctrl+C, or for ``duration_s`` seconds after the server is up. ``on_ready``
    is called with the page URL once the server accepts connections (the CLI opens the
    browser there). The LSL streams get process-specific names, so the demo does not mix
    with a ``neurolink stream`` that may be running.
    """
    import uvicorn

    if not port_is_free(host, port):
        raise DemoSetupError(
            f"port {port} on {host} is already in use (another `neurolink demo` or "
            "`neurolink serve`?). Stop that program or choose another port with --port."
        )
    tag = os.getpid()
    eeg_name, marker_name = f"{EEG_STREAM}-demo-{tag}", f"{MARKER_STREAM}-demo-{tag}"
    streamer = ReplayStreamer(recording, chunk=chunk, eeg_name=eeg_name, marker_name=marker_name)
    stop = threading.Event()
    stream_thread = threading.Thread(
        target=streamer.run, kwargs={"stop": stop, "loop": True}, daemon=True
    )
    stream_thread.start()
    live: LiveSystem | None = None
    ready_error: list[BaseException] = []
    try:
        live = build_live(
            bundle, eeg_name, marker_name, extra_info=extra_info, static_dir=static_dir
        )
        server = uvicorn.Server(uvicorn.Config(live.app, host=host, port=port, log_level="warning"))
        url = f"http://{host}:{port}/"

        def watch() -> None:
            waited = time.monotonic()
            while not server.started:
                if server.should_exit or time.monotonic() - waited > SERVER_START_TIMEOUT_S:
                    return
                time.sleep(0.05)
            try:
                if on_ready is not None:
                    on_ready(url)
            except BaseException as exc:  # reported by the caller's thread
                ready_error.append(exc)
                server.should_exit = True
                return
            if duration_s is not None:
                stop.wait(duration_s)
                server.should_exit = True

        threading.Thread(target=watch, daemon=True).start()
        server.run()  # returns on Ctrl+C or when `should_exit` is set
        worker_error = getattr(getattr(live.app.state, "worker", None), "error", None)
    finally:
        if live is not None:
            live.close()  # inlets first, so liblsl does not report a broken stream
        stop.set()
        stream_thread.join(5.0)
    if ready_error:
        raise ready_error[0]
    if worker_error is not None:
        raise RuntimeError(f"the decoder stopped with an error: {worker_error!r}")
    return {"samples_sent": streamer.n_sent, "passes": streamer.n_passes}
