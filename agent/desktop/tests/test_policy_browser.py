"""Tests for browser policy rules (current-page URL check, upload policy)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from agent.desktop.policy import PolicyEngine
from agent.desktop.session import DesktopSessionStore


MINIMAL_POLICY = {
    "version": 1,
    "enabled": True,
    "kill_switch_file": "~/nonexistent-kill-switch",
    "browser": {
        "headless": True,
        "display": ":2",
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
            "gated_tools": [],
            "key_whitelist": [],
            "mouse_button_whitelist": ["left"],
            "max_mouse_click_count": 1,
        },
    },
    "deny": {"actions": [], "url_patterns": ["*/checkout*"]},
    "confirm": {"require_for": [], "ttl_seconds": 120},
    "redact": {"snapshot_patterns": [], "screenshot_ocr": False},
    "audit": {"path": "~/local-agent/logs/desktop-audit.jsonl", "screenshots": False},
    "openrouter": {
        "api_key_env": "OPENROUTER_API_KEY",
        "vision_model": "openrouter/google/gemini-2.5-flash-preview",
        "max_budget_cents_per_session": 100,
    },
}


class BrowserPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(__file__).with_name("_test_browser_policy.json")
        self.tmp.write_text(json.dumps(MINIMAL_POLICY), encoding="utf-8")
        self.engine = PolicyEngine(path=self.tmp, session_store=DesktopSessionStore())

    def tearDown(self) -> None:
        try:
            self.tmp.unlink()
        except FileNotFoundError:
            pass

    def test_current_url_blocks_non_https(self):
        d = self.engine.evaluate(
            "browser_upload",
            {"element_ref": "input[type=file]", "file_paths": ["/tmp/cv.pdf"], "current_url": "http://evil.com"},
        )
        self.assertEqual(d.verdict, "deny")
        self.assertIn("origin blocked", d.reason)

    def test_current_url_allows_https(self):
        d = self.engine.evaluate(
            "browser_upload",
            {"element_ref": "input[type=file]", "file_paths": ["/tmp/cv.pdf"], "current_url": "https://boards.greenhouse.io"},
        )
        self.assertEqual(d.verdict, "allow")

    def test_current_url_blocks_deny_pattern(self):
        d = self.engine.evaluate(
            "browser_fill",
            {"element_ref": "input", "text": "123", "current_url": "https://example.com/checkout/pay"},
        )
        self.assertEqual(d.verdict, "deny")
        self.assertIn("URL matches deny pattern", d.reason)

    def test_browser_upload_not_in_confirm_list_by_default(self):
        # The fatigue fix: upload should not require a per-action confirm.
        d = self.engine.evaluate(
            "browser_upload",
            {"element_ref": "input[type=file]", "file_paths": ["/tmp/cv.pdf"], "current_url": "https://example.com"},
        )
        self.assertEqual(d.verdict, "allow")

    def test_browser_navigate_uses_requested_url_not_current_url(self):
        # Navigate should be judged by the target URL, not current_url.
        d = self.engine.evaluate(
            "browser_navigate",
            {"url": "https://example.com", "current_url": "http://evil.com"},
        )
        self.assertEqual(d.verdict, "allow")

    def test_default_policy_has_browser_display(self):
        default = PolicyEngine._default_policy()
        self.assertIn("display", default["browser"])
        self.assertEqual(default["browser"]["display"], ":2")


if __name__ == "__main__":
    unittest.main()
