"""Load key=value pairs from a ``.env`` file into ``os.environ`` (no extra dependency)."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path | str | None = None) -> None:
    """Set environment variables from ``path`` if the file exists.

    Existing non-empty values are not overwritten (same spirit as python-dotenv
    with ``override=False``): export in shell wins over ``.env``.
    """
    if path is None:
        return
    p = Path(path)
    if not p.is_file():
        return
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError:
        return
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        key, val = s.split("=", 1)
        key = key.strip()
        if not key:
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        cur = os.environ.get(key)
        if cur is not None and cur != "":
            continue
        os.environ[key] = val
