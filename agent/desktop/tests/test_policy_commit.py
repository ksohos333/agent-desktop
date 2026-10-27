"""Tests for the commit-boundary (never_click) classifier and scope decision."""

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
    "deny": {"actions": [], "url_patterns": []},
    "confirm": {"require_for": [], "ttl_seconds": 120},
    "never_click": {
        "patterns": [
            r"(?i)\bsubmit\b",
            r"(?i)\bapply\b",
            r"(?i)\bsend\b",
            r"(?i)\bconfirm\b",
            r"(?i)\bagree\b",
            r"(?i)\bsign\b",
            r"(?i)\baccept\b",
            r"(?i)\bpay\b",
            r"(?i)\bplace\b",
            r"(?i)\benroll\b",
        ],
        "session_scope": [],
    },
    "redact": {"snapshot_patterns": [], "screenshot_ocr": False},
    "audit": {"path": "~/local-agent/logs/desktop-audit.jsonl", "screenshots": False},
    "openrouter": {
        "api_key_env": "OPENROUTER_API_KEY",
        "vision_model": "openrouter/google/gemini-2.5-flash-preview",
        "max_budget_cents_per_session": 100,
    },
}


class CommitClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(__file__).with_name("_test_commit_policy.json")
        self.tmp.write_text(json.dumps(MINIMAL_POLICY), encoding="utf-8")
        self.engine = PolicyEngine(path=self.tmp, session_store=DesktopSessionStore())

    def tearDown(self) -> None:
        try:
            self.tmp.unlink()
        except FileNotFoundError:
            pass

    COMMIT_LABELS = [
        "Submit Application",
        "Apply Now",
        "Send Application",
        "Submit",
        "Confirm & Submit",
        "Agree & Submit",
        "Accept Offer",
        "Sign and Submit",
        "Place Order",
        "Enroll Now",
        "PAY NOW",
        "submit",
    ]

    SAFE_LABELS = [
        "Save & Continue",
        "Continue",
        "Next",
        "Back",
        "Save Draft",
        "Add Resume",
        "Review Application",
        "Submit feedback",  # commit verb but non-commit context
        "Send message",     # commit verb but chat context
        "Post comment",
        "Cancel",
        "Close",
    ]

    def test_commit_labels_denied_globally(self):
        for label in self.COMMIT_LABELS:
            with self.subTest(label=label):
                d = self.engine.check_commit("browser_click", label)
                self.assertEqual(d.verdict, "deny", f"{label!r} should be denied")

    def test_safe_labels_allowed_globally(self):
        # Labels with commit verbs in non-commit contexts are known false
        # positives in global mode; the rest should be allowed.
        expected_denies = {"Submit feedback", "Send message"}
        for label in self.SAFE_LABELS:
            with self.subTest(label=label):
                d = self.engine.check_commit("browser_click", label)
                if label in expected_denies:
                    self.assertEqual(d.verdict, "deny", f"{label!r} is a known global FP")
                else:
                    self.assertEqual(d.verdict, "allow", f"{label!r} should be allowed")

    def test_unlabeled_element_confirms(self):
        d = self.engine.check_commit("browser_click", "")
        self.assertEqual(d.verdict, "confirm")
        self.assertIn("unlabeled", d.reason)

    def test_whitespace_only_confirms(self):
        d = self.engine.check_commit("browser_click", "   ")
        self.assertEqual(d.verdict, "confirm")


class ScopedCommitTests(unittest.TestCase):
    def setUp(self) -> None:
        policy = {**MINIMAL_POLICY}
        policy["never_click"] = {
            **policy["never_click"],
            "session_scope": ["jobapply"],
        }
        self.tmp = Path(__file__).with_name("_test_commit_scoped_policy.json")
        self.tmp.write_text(json.dumps(policy), encoding="utf-8")
        self.engine = PolicyEngine(path=self.tmp, session_store=DesktopSessionStore())

    def tearDown(self) -> None:
        try:
            self.tmp.unlink()
        except FileNotFoundError:
            pass

    def test_commit_allowed_outside_scope(self):
        d = self.engine.check_commit("browser_click", "Submit Application")
        self.assertEqual(d.verdict, "allow")

    def test_commit_denied_inside_scope(self):
        self.engine.enter_scope("jobapply")
        d = self.engine.check_commit("browser_click", "Submit Application")
        self.assertEqual(d.verdict, "deny")
        self.engine.exit_scope("jobapply")

    def test_false_positive_allowed_outside_scope(self):
        # "Submit feedback" and similar non-commit contexts are the main
        # argument for scoping never_click to jobapply sessions.
        d = self.engine.check_commit("browser_click", "Submit feedback")
        self.assertEqual(d.verdict, "allow")

    def test_unlabeled_still_confirms_outside_scope(self):
        d = self.engine.check_commit("browser_click", "")
        self.assertEqual(d.verdict, "confirm")


class EnterKeyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(__file__).with_name("_test_enter_policy.json")
        self.tmp.write_text(json.dumps(MINIMAL_POLICY), encoding="utf-8")
        self.engine = PolicyEngine(path=self.tmp, session_store=DesktopSessionStore())

    def tearDown(self) -> None:
        try:
            self.tmp.unlink()
        except FileNotFoundError:
            pass

    def test_enter_requires_confirm(self):
        for key in ("Enter", "Return"):
            with self.subTest(key=key):
                d = self.engine.evaluate("browser_press", {"key": key})
                self.assertEqual(d.verdict, "confirm")
                self.assertIn("submit", d.reason.lower())

    def test_tab_allowed(self):
        d = self.engine.evaluate("browser_press", {"key": "Tab"})
        self.assertEqual(d.verdict, "allow")

    def test_enter_confirmed_then_allowed(self):
        d = self.engine.evaluate("browser_press", {"key": "Enter"}, confirmed=True)
        self.assertEqual(d.verdict, "allow")


class DefaultPolicyTests(unittest.TestCase):
    def test_default_policy_has_never_click(self):
        default = PolicyEngine._default_policy()
        self.assertIn("never_click", default)
        self.assertIn("patterns", default["never_click"])
        self.assertIn("session_scope", default["never_click"])
        self.assertEqual(default["never_click"]["session_scope"], ["jobapply"])


if __name__ == "__main__":
    unittest.main()
