from __future__ import annotations

import pytest

from neurolink.config import ALL_SUBJECTS, DEV_SUBJECTS
from neurolink.utils.subjects import parse_subjects


def test_keywords() -> None:
    assert parse_subjects("all") == ALL_SUBJECTS
    assert parse_subjects("dev") == DEV_SUBJECTS
    assert parse_subjects("DEV") == DEV_SUBJECTS


def test_ranges_and_lists() -> None:
    assert parse_subjects("1-3") == (1, 2, 3)
    assert parse_subjects("5, 1, 3-4") == (1, 3, 4, 5)
    assert parse_subjects("2,2,2") == (2,)


@pytest.mark.parametrize("spec", ["0", "110", "5-2", "", "abc"])
def test_invalid(spec: str) -> None:
    with pytest.raises(ValueError):
        parse_subjects(spec)
