"""Append-only JSONL audit log for desktop MCP."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

MAX_LOG_BYTES = 10 * 1024 * 1024


class AuditLog:
    """Rotating JSONL audit log.

    When the file exceeds ``MAX_LOG_BYTES`` it is rotated to ``.1``.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        self._rotate_if_needed()
        record["_ts"] = time.time()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str, ensure_ascii=False))
            f.write("\n")

    def _rotate_if_needed(self) -> None:
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return
        if size > MAX_LOG_BYTES:
            backup = self.path.with_suffix(".jsonl.1")
            if backup.exists():
                backup.unlink()
            self.path.rename(backup)
