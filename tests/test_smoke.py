"""Smoke tests: the package imports and the CLI responds."""

from __future__ import annotations

import numpy as np
from typer.testing import CliRunner

from neurolink import __version__
from neurolink.cli import app
from neurolink.config import MOTOR_CORTEX_CHANNELS, SFREQ, PreprocessConfig
from neurolink.utils.seed import set_seed

runner = CliRunner()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_stub_commands_exit_with_code_2() -> None:
    for cmd in ["stream", "serve", "demo"]:
        result = runner.invoke(app, [cmd])
        assert result.exit_code == 2, cmd


def test_config_defaults() -> None:
    cfg = PreprocessConfig()
    assert cfg.band == (8.0, 30.0)
    assert cfg.tmax - cfg.tmin == 2.0
    assert SFREQ == 160.0
    assert len(MOTOR_CORTEX_CHANNELS) == 21
    assert len(set(MOTOR_CORTEX_CHANNELS)) == len(MOTOR_CORTEX_CHANNELS)


def test_set_seed_is_deterministic() -> None:
    set_seed(0)
    a = np.random.rand(3)
    set_seed(0)
    b = np.random.rand(3)
    np.testing.assert_array_equal(a, b)
