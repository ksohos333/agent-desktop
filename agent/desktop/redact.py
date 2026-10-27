"""Redaction helpers for desktop MCP."""

from __future__ import annotations

import re


def redact_text(text: str | None, patterns: list[str]) -> tuple[str, list[dict]]:
    """Apply regex patterns to a string, replacing matches with ``[REDACTED]``.

    Returns the redacted text and a list of match records (no captured secret
    values; only match positions and pattern identifiers).
    """
    if not text:
        return text or "", []

    matches: list[dict] = []
    seen: set[tuple[int, int]] = set()

    for raw in patterns:
        pat = re.compile(raw)
        for m in pat.finditer(text):
            key = (m.start(), m.end())
            if key in seen:
                continue
            seen.add(key)
            matches.append({"start": m.start(), "end": m.end(), "pattern": raw})

    if not matches:
        return text, []

    # Apply replacements in reverse order so earlier offsets remain valid.
    chars = list(text)
    for m in sorted(matches, key=lambda x: x["start"], reverse=True):
        chars[m["start"] : m["end"]] = list("[REDACTED]")
    return "".join(chars), matches
