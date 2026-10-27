"""Tests for the confirm_every browser policy mode."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from agent.desktop.policy import PolicyEngine


MINIMAL_POLICY = {
    "version": 1,
    "enabled": True,
    "kill_switch_file": "~/nonexistent-kill-switch",
    "browser": {
        "headless": True,
        "display": ":2",
        "confirm_every": True,
        "verbose": True,
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
    "deny": {"actions": [], "url_patterns": []},
    "confirm": {"require_for": [], "ttl_seconds": 120},
    "redact": {"snapshot_patterns": [], "screenshot_ocr": False},
    "audit": {"path": "~/local-agent/logs/desktop-audit.jsonl", "screenshots": False},
    "openrouter": {
        "api_key_env": "OPENROUTER_API_KEY",
        "vision_model": "openrouter/google/gemini-2.5-flash-preview",
        "max_budget_cents_per_session": 100,
    },
}


class ConfirmEveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(__file__).with_name("_test_confirm_every.json")
        self.tmp.write_text(json.dumps(MINIMAL_POLICY), encoding="utf-8")
        self.engine = PolicyEngine(path=self.tmp)

    def tearDown(self) -> None:
        self.tmp.unlink(missing_ok=True)

    def test_click_requires_confirm_when_confirm_every_enabled(self) -> None:
        d = self.engine.evaluate(
            "browser_click",
            {"element_ref": "text=Pricing", "current_url": "https://example.com"},
        )
        self.assertEqual(d.verdict, "confirm")
        self.assertIn("confirm_every", d.reason)

    def test_click_allowed_after_explicit_confirmation(self) -> None:
        d = self.engine.evaluate(
            "browser_click",
            {"element_ref": "text=Pricing", "current_url": "https://example.com"},
            confirmed=True,
        )
        self.assertEqual(d.verdict, "allow")

    def test_click_allowed_when_confirm_every_disabled(self) -> None:
        policy = dict(MINIMAL_POLICY)
        policy["browser"] = dict(MINIMAL_POLICY["browser"])
        policy["browser"]["confirm_every"] = False
        tmp = Path(__file__).with_name("_test_confirm_every_off.json")
        tmp.write_text(json.dumps(policy), encoding="utf-8")
        try:
            engine = PolicyEngine(path=tmp)
            d = engine.evaluate(
                "browser_click",
                {"element_ref": "text=Pricing", "current_url": "https://example.com"},
            )
            self.assertEqual(d.verdict, "allow")
        finally:
            tmp.unlink(missing_ok=True)

    def test_non_click_actions_not_affected_by_confirm_every(self) -> None:
        d = self.engine.evaluate(
            "browser_fill",
            {"element_ref": "input[name=email]", "text": "test@example.com", "current_url": "https://example.com"},
        )
        self.assertEqual(d.verdict, "allow")


if __name__ == "__main__":
    unittest.main()
