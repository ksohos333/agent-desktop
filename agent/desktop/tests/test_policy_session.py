"""Tests for the desktop session gating subsystem.

These tests run without a browser or X server. They exercise the policy
engine's session gate and the session store lifecycle directly.
"""

from __future__ import annotations

import json
import time
import unittest
from pathlib import Path

from agent.desktop.policy import PolicyEngine
from agent.desktop.session import DesktopSessionStore


MINIMAL_POLICY = {
    "version": 1,
    "enabled": True,
    "kill_switch_file": "~/nonexistent-kill-switch",
    "browser": {
        "allowed_origins": ["https://*"],
        "blocked_origins": [],
        "max_actions_per_session": 1000,
    },
    "desktop": {
        "display": ":2",
        "require_virtual_display": True,
        "rate_limit_per_minute": 120,
        "session": {
            "default_ttl_seconds": 120,
            "default_max_actions": 5,
            "gated_tools": [
                "desktop_mouse_click",
                "desktop_drag",
                "desktop_type_text",
                "desktop_key",
                "desktop_window_focus",
                "desktop_clipboard_get",
            ],
            "key_whitelist": [
                "Return",
                "Tab",
                "Escape",
                "Left",
                "Right",
                "Up",
                "Down",
                "BackSpace",
                "Delete",
                "Control+l",
                "Control+a",
            ],
            "mouse_button_whitelist": ["left"],
            "max_mouse_click_count": 1,
        },
    },
    "deny": {
        "actions": ["desktop_clipboard_set"],
        "url_patterns": [],
    },
    "confirm": {
        "require_for": ["desktop_session_open"],
        "ttl_seconds": 120,
    },
    "redact": {"snapshot_patterns": [], "screenshot_ocr": False},
    "audit": {"path": "~/local-agent/logs/desktop-audit.jsonl", "screenshots": False},
    "openrouter": {
        "api_key_env": "OPENROUTER_API_KEY",
        "vision_model": "openrouter/google/gemini-2.5-flash-preview",
        "max_budget_cents_per_session": 100,
    },
}


class DesktopSessionTestCase(unittest.TestCase):
    """Base test case that builds a PolicyEngine with a temp policy."""

    def setUp(self) -> None:
        self.tmp = Path(__file__).with_name("_test_policy.json")
        self.tmp.write_text(json.dumps(MINIMAL_POLICY), encoding="utf-8")
        self.store = DesktopSessionStore()
        self.engine = PolicyEngine(path=self.tmp, session_store=self.store)

    def tearDown(self) -> None:
        try:
            self.tmp.unlink()
        except FileNotFoundError:
            pass

    def _open_session(self, purpose: str = "cv_upload", window_hint: str = "chromium") -> str:
        d = self.engine.evaluate("desktop_session_open", {"purpose": purpose, "window_hint": window_hint})
        self.assertEqual(d.verdict, "confirm")
        d2 = self.engine.evaluate(
            "desktop_session_open",
            {"purpose": purpose, "window_hint": window_hint},
            confirmed=True,
        )
        self.assertEqual(d2.verdict, "allow")
        # The executor would create the session; simulate it directly in unit tests.
        session = self.store.create(purpose, window_hint, 5, 120)
        self.assertEqual(session.state, "armed")
        return session.id


class StoreTests(DesktopSessionTestCase):
    def test_create_active_session(self):
        s = self.store.create("test", "window", 10, 60)
        self.assertEqual(s.state, "armed")
        self.assertEqual(self.store.active_session(), s)

    def test_expire_closes_active(self):
        self.store.create("test", "window", 10, 1)
        self.assertIsNotNone(self.store.active_session())
        time.sleep(1.1)
        self.assertIsNone(self.store.active_session())

    def test_close_active(self):
        s = self.store.create("test", "window", 10, 60)
        self.assertEqual(self.store.close_active(), s)
        self.assertIsNone(self.store.active_session())


class NoSessionDenyTests(DesktopSessionTestCase):
    def test_gated_tools_require_session(self):
        cases = [
            ("desktop_mouse_click", {"button": "left", "count": 1}),
            ("desktop_drag", {"x1": 0, "y1": 0, "x2": 10, "y2": 10}),
            ("desktop_type_text", {"text": "hello"}),
            ("desktop_key", {"combo": "Return"}),
            ("desktop_window_focus", {"window_id": "123"}),
            ("desktop_clipboard_get", {}),
        ]
        for action, args in cases:
            with self.subTest(action=action):
                d = self.engine.evaluate(action, args)
                self.assertEqual(d.verdict, "deny")
                self.assertIn("no desktop session open", d.reason)

    def test_confirmed_true_cannot_bypass_no_session(self):
        d = self.engine.evaluate(
            "desktop_mouse_click",
            {"button": "left", "count": 1},
            confirmed=True,
        )
        self.assertEqual(d.verdict, "deny")
        self.assertIn("no desktop session open", d.reason)


class UngatedToolsTests(DesktopSessionTestCase):
    def test_allowed_without_session(self):
        cases = [
            ("desktop_status", {}),
            ("desktop_screenshot", {}),
            ("desktop_mouse_move", {"x": 100, "y": 100}),
            ("desktop_window_list", {}),
        ]
        for action, args in cases:
            with self.subTest(action=action):
                d = self.engine.evaluate(action, args)
                self.assertEqual(d.verdict, "allow")


class OpenSessionTests(DesktopSessionTestCase):
    def test_session_open_returns_confirm(self):
        d = self.engine.evaluate("desktop_session_open", {"purpose": "cv_upload"})
        self.assertEqual(d.verdict, "confirm")

    def test_session_open_invalid_purpose_denied(self):
        d = self.engine.evaluate("desktop_session_open", {"purpose": ""})
        self.assertEqual(d.verdict, "deny")

    def test_session_open_confirmed_creates_armed_session(self):
        self._open_session()


class TwoGateTests(DesktopSessionTestCase):
    def test_first_acting_action_requires_confirm(self):
        sid = self._open_session()
        d = self.engine.evaluate("desktop_mouse_click", {"button": "left", "count": 1})
        self.assertEqual(d.verdict, "confirm")
        self.assertIsNotNone(d.metadata)
        self.assertEqual(d.metadata.get("session_id"), sid)
        self.assertIn("first acting action", d.reason)

    def test_first_acting_action_confirmed_arms_session(self):
        self._open_session()
        d = self.engine.evaluate("desktop_mouse_click", {"button": "left", "count": 1})
        self.assertEqual(d.verdict, "confirm")
        d2 = self.engine.evaluate(
            "desktop_mouse_click",
            {"button": "left", "count": 1, **d.metadata},
            confirmed=True,
        )
        self.assertEqual(d2.verdict, "allow")
        session = self.store.active_session()
        self.assertIsNotNone(session)
        self.assertEqual(session.state, "active")
        self.assertEqual(session.actions_used, 1)

    def test_first_action_confirm_includes_context(self):
        self._open_session()
        d = self.engine.evaluate("desktop_key", {"combo": "Return"})
        self.assertEqual(d.verdict, "confirm")
        self.assertIsNotNone(d.metadata)
        self.assertIn("cv_upload", d.metadata.get("session_purpose", ""))
        self.assertIn("Return", d.metadata.get("context", ""))


class WhitelistTests(DesktopSessionTestCase):
    def test_disallowed_mouse_button_denied(self):
        sid = self._open_session()
        self.store.arm(sid)
        d = self.engine.evaluate("desktop_mouse_click", {"button": "right", "count": 1})
        self.assertEqual(d.verdict, "deny")
        self.assertIn("not allowed in desktop session", d.reason)

    def test_disallowed_key_combo_denied(self):
        sid = self._open_session()
        self.store.arm(sid)
        d = self.engine.evaluate("desktop_key", {"combo": "Control+c"})
        self.assertEqual(d.verdict, "deny")
        self.assertIn("not allowed in desktop session", d.reason)

    def test_disallowed_multi_click_denied(self):
        sid = self._open_session()
        self.store.arm(sid)
        d = self.engine.evaluate("desktop_mouse_click", {"button": "left", "count": 2})
        self.assertEqual(d.verdict, "deny")


class BudgetAndExpiryTests(DesktopSessionTestCase):
    def test_session_action_budget_enforced(self):
        sid = self._open_session()
        self.store.arm(sid)
        # max_actions is 5 in the fixture policy.
        for i in range(5):
            d = self.engine.evaluate("desktop_mouse_click", {"button": "left", "count": 1})
            self.assertEqual(d.verdict, "allow", f"action {i} should be allowed")
        d = self.engine.evaluate("desktop_mouse_click", {"button": "left", "count": 1})
        self.assertEqual(d.verdict, "deny")
        self.assertIn("budget exceeded", d.reason)
        self.assertIsNone(self.store.active_session())

    def test_session_expiry_enforced(self):
        sid = self._open_session()
        self.store.arm(sid)
        session = self.store.get(sid)
        self.assertIsNotNone(session)
        session.expires_at = time.time() - 1
        # Once expired, the store cleans it up and evaluate sees no session.
        d = self.engine.evaluate("desktop_mouse_click", {"button": "left", "count": 1})
        self.assertEqual(d.verdict, "deny")
        self.assertIsNone(self.store.active_session())


class MiscSessionTests(DesktopSessionTestCase):
    def test_clipboard_set_hard_denied(self):
        sid = self._open_session()
        self.store.arm(sid)
        d = self.engine.evaluate("desktop_clipboard_set", {"text": "secret"})
        self.assertEqual(d.verdict, "deny")

    def test_session_close_ends_session(self):
        sid = self._open_session()
        self.store.arm(sid)
        d = self.engine.evaluate("desktop_session_close", {})
        self.assertEqual(d.verdict, "allow")
        # The executor would close the session; simulate it directly.
        self.store.close_active()
        self.assertIsNone(self.store.active_session())


if __name__ == "__main__":
    unittest.main()
