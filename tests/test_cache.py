from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neurolink.config import PreprocessConfig
from neurolink.data.cache import (
    Dataset,
    cache_key,
    cache_path,
    config_from_dict,
    config_to_dict,
    load_dataset,
    save_dataset,
)


def _dataset(config: PreprocessConfig | None = None) -> Dataset:
    rng = np.random.default_rng(0)
    n = 12
    return Dataset(
        X=rng.normal(size=(n, 3, 320)).astype(np.float32),
        y=(np.arange(n) % 2).astype(np.int64),
        subjects=np.repeat([1, 2], n // 2).astype(np.int64),
        runs=np.tile([4, 8, 12], n // 3).astype(np.int64),
        ch_names=("C3", "Cz", "C4"),
        sfreq=160.0,
        config=config or PreprocessConfig(channels=("C3", "Cz", "C4"), subjects=(1, 2)),
    )


def test_roundtrip(tmp_path: Path) -> None:
    ds = _dataset()
    path = save_dataset(ds, tmp_path / "sub" / "ds.npz")
    loaded = load_dataset(path)
    np.testing.assert_array_equal(loaded.X, ds.X)
    np.testing.assert_array_equal(loaded.y, ds.y)
    np.testing.assert_array_equal(loaded.subjects, ds.subjects)
    np.testing.assert_array_equal(loaded.runs, ds.runs)
    assert loaded.ch_names == ds.ch_names
    assert loaded.sfreq == ds.sfreq
    assert loaded.config == ds.config
    assert loaded.X.dtype == np.float32 and loaded.y.dtype == np.int64


def test_cache_key_is_deterministic_and_config_sensitive() -> None:
    a = PreprocessConfig()
    assert cache_key(a) == cache_key(PreprocessConfig())
    assert cache_key(a) != cache_key(PreprocessConfig(car=False))
    assert cache_key(a) != cache_key(PreprocessConfig(band=(8.0, 32.0)))
    assert cache_key(a) != cache_key(PreprocessConfig(channels=("C3",)))
    assert cache_key(a) != cache_key(PreprocessConfig(subjects=(1,)))
    assert cache_path(a, Path("x")).name == f"dataset_{cache_key(a)}.npz"


def test_config_dict_roundtrip() -> None:
    cfg = PreprocessConfig(channels=("C3", "C4"), subjects=(3, 5), car=False)
    assert config_from_dict(config_to_dict(cfg)) == cfg
    assert config_from_dict(config_to_dict(PreprocessConfig())) == PreprocessConfig()


def test_dataset_validates_shapes() -> None:
    ds = _dataset()
    with pytest.raises(ValueError, match="number of epochs"):
        Dataset(ds.X, ds.y[:-1], ds.subjects, ds.runs, ds.ch_names, ds.sfreq, ds.config)
    with pytest.raises(ValueError, match="ch_names"):
        Dataset(ds.X, ds.y, ds.subjects, ds.runs, ("C3",), ds.sfreq, ds.config)


def test_unique_subjects() -> None:
    np.testing.assert_array_equal(_dataset().unique_subjects, [1, 2])
