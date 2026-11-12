"""Parsing of subject specifications given on the command line."""

from __future__ import annotations

from neurolink.config import ALL_SUBJECTS, DEV_SUBJECTS


def parse_subjects(spec: str) -> tuple[int, ...]:
    """Parse ``"1-10"``, ``"1,2,5-7"``, ``"all"`` or ``"dev"`` into sorted unique subject ids."""
    spec = spec.strip().lower()
    if spec == "all":
        return ALL_SUBJECTS
    if spec == "dev":
        return DEV_SUBJECTS
    ids: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                raise ValueError(f"invalid range {part!r}")
            ids.update(range(lo, hi + 1))
        else:
            ids.add(int(part))
    bad = sorted(i for i in ids if i not in ALL_SUBJECTS)
    if bad:
        raise ValueError(f"subject ids out of range 1-109: {bad}")
    if not ids:
        raise ValueError("empty subject specification")
    return tuple(sorted(ids))
