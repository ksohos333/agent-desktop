"""Scoped desktop session store for native-dialog control.

A desktop session is a bounded, explicitly-approved period during which
otherwise-blind coordinate/keyboard actions are allowed on the virtual display.
It uses a two-gate model:

  1. ``desktop_session_open`` returns a confirm token; the human approves the
     declared purpose, window hint, budget, and TTL.
  2. The first *acting* action inside the session (click/key/type/etc.) returns
     another confirm token with the specific action context; the human approves
     that first physical action.

After both gates, the session is ``active`` and subsequent gated actions run
inside the budget without per-action confirmation. The session auto-closes when
it expires or exhausts its action budget.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DesktopSession:
    """Runtime record of an open desktop session."""

    id: str
    purpose: str
    window_hint: str
    opened_at: float
    expires_at: float
    max_actions: int
    actions_used: int = 0
    state: str = "armed"  # armed -> active -> closed

    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    def ttl_remaining(self) -> float:
        return max(0.0, self.expires_at - time.time())

    def describe_context(self, action: str, args: dict[str, Any]) -> str:
        """Human-readable context for the first-acting-action confirm."""
        parts = [f"action={action}"]
        if action == "desktop_mouse_click":
            parts.append(f"button={args.get('button', 'left')} count={args.get('count', 1)}")
        elif action == "desktop_key":
            parts.append(f"combo={args.get('combo', '')!r}")
        elif action == "desktop_type_text":
            text = args.get("text", "")
            parts.append(f"text_length={len(text)}")
        elif action == "desktop_drag":
            parts.append(
                f"from=({args.get('x1')},{args.get('y1')}) "
                f"to=({args.get('x2')},{args.get('y2')})"
            )
        elif action == "desktop_window_focus":
            parts.append(f"window_id={args.get('window_id')!r}")
        elif action == "desktop_clipboard_get":
            parts.append("read clipboard")
        parts.append(f"purpose={self.purpose}")
        if self.window_hint:
            parts.append(f"window_hint={self.window_hint}")
        return "; ".join(parts)


class DesktopSessionStore:
    """In-memory store of desktop sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, DesktopSession] = {}
        self._active_id: str | None = None

    def create(
        self,
        purpose: str,
        window_hint: str,
        max_actions: int,
        ttl_seconds: int,
    ) -> DesktopSession:
        """Open a new armed session and make it active."""
        self._cleanup()
        sid = "desktop-session-" + secrets.token_urlsafe(12)
        now = time.time()
        session = DesktopSession(
            id=sid,
            purpose=purpose,
            window_hint=window_hint,
            opened_at=now,
            expires_at=now + ttl_seconds,
            max_actions=max_actions,
            state="armed",
        )
        self._sessions[sid] = session
        self._active_id = sid
        return session

    def active_session(self) -> DesktopSession | None:
        """Return the current active session, or None if closed/expired."""
        self._cleanup()
        if self._active_id is None:
            return None
        session = self._sessions.get(self._active_id)
        if session is None or session.state == "closed" or session.is_expired():
            self._active_id = None
            return None
        return session

    def get(self, sid: str) -> DesktopSession | None:
        """Return a session by id, or None if gone/expired."""
        self._cleanup()
        return self._sessions.get(sid)

    def close(self, sid: str) -> bool:
        """Close a session. Returns True if it existed."""
        session = self._sessions.get(sid)
        if session:
            session.state = "closed"
        if self._active_id == sid:
            self._active_id = None
        return session is not None

    def close_active(self) -> DesktopSession | None:
        """Close the active session and return it."""
        session = self.active_session()
        if session:
            self.close(session.id)
        return session

    def arm(self, sid: str) -> bool:
        """Transition a session from armed to active."""
        session = self._sessions.get(sid)
        if session and session.state == "armed":
            session.state = "active"
            return True
        return False

    def record_action(self, sid: str) -> None:
        """Increment the action budget for a session."""
        session = self._sessions.get(sid)
        if session:
            session.actions_used += 1

    def _cleanup(self) -> None:
        """Drop closed/expired sessions and clear the active pointer if stale."""
        now = time.time()
        gone = [
            sid
            for sid, s in self._sessions.items()
            if s.state == "closed" or s.is_expired()
        ]
        for sid in gone:
            if self._active_id == sid:
                self._active_id = None
            self._sessions.pop(sid, None)

    def status(self) -> dict[str, Any]:
        """Summary of the active session."""
        self._cleanup()
        session = self.active_session()
        if session is None:
            return {"active": False, "session": None}
        return {
            "active": True,
            "session": {
                "id": session.id,
                "purpose": session.purpose,
                "window_hint": session.window_hint,
                "state": session.state,
                "actions_used": session.actions_used,
                "max_actions": session.max_actions,
                "ttl_remaining_seconds": round(session.ttl_remaining(), 1),
            },
        }
