"""Confirmation-token store for sensitive desktop actions."""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field


@dataclass
class ConfirmToken:
    token: str
    action: str
    args: dict
    created_at: float = field(default_factory=time.time)
    ttl_seconds: int = 120

    def expired(self) -> bool:
        return time.time() - self.created_at > self.ttl_seconds


class ConfirmStore:
    """In-memory store of pending confirmations.

    Tokens are one-time use. If the policy TTL is shorter than the store TTL, the
    policy wins during evaluation, but stale entries expire lazily here as well.
    """

    def __init__(self, default_ttl: int = 120) -> None:
        self.default_ttl = default_ttl
        self._pending: dict[str, ConfirmToken] = {}

    def create(self, action: str, args: dict, ttl: int | None = None) -> str:
        self._expire()
        token = "desktop-confirm-" + secrets.token_urlsafe(16)
        self._pending[token] = ConfirmToken(
            token=token,
            action=action,
            args=args,
            ttl_seconds=ttl or self.default_ttl,
        )
        return token

    def consume(self, token: str) -> ConfirmToken | None:
        self._expire()
        entry = self._pending.pop(token, None)
        if entry is None or entry.expired():
            return None
        return entry

    def get(self, token: str) -> ConfirmToken | None:
        self._expire()
        entry = self._pending.get(token)
        if entry is None or entry.expired():
            return None
        return entry

    def _expire(self) -> None:
        now = time.time()
        expired = [t for t, e in self._pending.items() if e.created_at + e.ttl_seconds < now]
        for t in expired:
            del self._pending[t]

    def list_pending(self) -> list[dict]:
        self._expire()
        return [{"token": e.token, "action": e.action, "age_seconds": round(time.time() - e.created_at, 1)} for e in self._pending.values()]
