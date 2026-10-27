"""Policy engine for desktop MCP.

Evaluates every action against an editable JSON policy and returns
``allow``, ``deny`` or ``confirm``. Also enforces rate limits and per-session
action budgets.
"""

from __future__ import annotations

import fnmatch
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.desktop.session import DesktopSessionStore

DEFAULT_POLICY_PATH = Path.home() / "local-agent" / "config" / "desktop-policy.json"


@dataclass(frozen=True)
class Decision:
    action: str
    verdict: str  # "allow", "deny", or "confirm"
    reason: str
    confirm_token: str | None = None
    metadata: dict[str, Any] | None = None


def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(fnmatch.translate(pattern), re.IGNORECASE)


class PolicyEngine:
    """Guardrail policy evaluator."""

    def __init__(
        self,
        path: Path | str | None = None,
        session_store: DesktopSessionStore | None = None,
    ) -> None:
        self.path = Path(path) if path else DEFAULT_POLICY_PATH
        self._policy: dict[str, Any] = {}
        self._last_load = 0.0
        self._session_actions = 0
        self._action_log: list[float] = []
        self._vision_spend_cents = 0.0
        self._vision_tokens = 0
        self._vision_calls = 0
        self.session_store = session_store
        self._scopes: set[str] = set()
        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load(self) -> None:
        try:
            data = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self._policy = self._default_policy()
            return
        try:
            self._policy = json.loads(data)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid desktop policy JSON: {exc}") from exc
        self._last_load = time.time()

    def reload(self) -> None:
        self._load()

    def policy(self) -> dict[str, Any]:
        """Return a fresh view of the current policy."""
        if time.time() - self._last_load > 5:
            self._load()
        return self._policy

    @staticmethod
    def _default_policy() -> dict[str, Any]:
        return {
            "version": 1,
            "enabled": True,
            "kill_switch_file": str(Path.home() / "local-agent" / "config" / "DESKTOP_KILL"),
            "browser": {
                "profile_mode": "isolated",
                "logged_in_profiles": [],
                "headless": True,
                "display": ":2",
                "confirm_every": False,
                "verbose": False,
                "allowed_origins": ["https://*"],
                "blocked_origins": [],
                "max_actions_per_session": 1000,
                "max_screenshot_bytes": 5 * 1024 * 1024,
            },
            "desktop": {
                "display": ":2",
                "require_virtual_display": True,
                "rate_limit_per_minute": 120,
                "session": {
                    "default_ttl_seconds": 120,
                    "default_max_actions": 25,
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
                "actions": [],
                "url_patterns": [],
            },
            "confirm": {
                "require_for": [],
                "ttl_seconds": 120,
            },
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
                "session_scope": ["jobapply"],
            },
            "redact": {
                "snapshot_patterns": [
                    r"(?i)password",
                    r"sk-[A-Za-z0-9]+",
                    r"\b\d{13,19}\b",
                ],
                "screenshot_ocr": False,
            },
            "audit": {
                "path": str(Path.home() / "local-agent" / "logs" / "desktop-audit.jsonl"),
                "screenshots": False,
            },
            "openrouter": {
                "api_key_env": "OPENROUTER_API_KEY",
                "vision_model": "openrouter/google/gemini-2.5-flash-preview",
                "max_budget_cents_per_session": 100,
                "max_image_width": 1280,
                "max_image_height": 1280,
                "max_image_bytes": 1500000,
                "allow_text_vision": True,
                "system_prompt": "You are a concise desktop observer. Describe what you see in plain text. Do not include passwords, credit card numbers, or other secrets.",
            },
        }

    # ------------------------------------------------------------------
    # OpenRouter budget helpers
    # ------------------------------------------------------------------
    def record_vision_usage(self, usage: Any) -> None:
        """Track vision model usage against the session budget."""
        if not usage:
            return
        self._vision_calls += 1
        self._vision_tokens += getattr(usage, "total_tokens", 0)
        # Generic price fallback: ~$0.50 / 1M tokens for flash-class models.
        # Real costs should be added when prices are fetched.
        price_per_mtok = self._get_model_price(getattr(usage, "model", ""))
        est_cents = getattr(usage, "total_tokens", 0) * price_per_mtok / 1_000_000 * 100
        self._vision_spend_cents += est_cents

    def _get_model_price(self, model: str) -> float:
        # Static fallback price per million tokens (prompt+completion average).
        # Update this map as better data is available.
        prices: dict[str, float] = {
            "openrouter/google/gemini-2.5-flash-preview": 0.2,
            "openrouter/qwen/qwen2.5-vl-72b-instruct": 1.0,
            "openrouter/anthropic/claude-3.5-sonnet": 3.0,
        }
        for prefix, price in prices.items():
            if model.startswith(prefix.rsplit("/", 1)[0]):
                return price
        return 0.5

    def check_vision_budget(self, new_request_cost_estimate_cents: float = 0.0) -> tuple[bool, str, dict[str, Any]]:
        """Return (allowed, reason, budget_status)."""
        section = self.policy().get("openrouter", {})
        max_budget = section.get("max_budget_cents_per_session", 100)
        if self._vision_spend_cents + new_request_cost_estimate_cents > max_budget:
            return False, f"vision budget exceeded: {self._vision_spend_cents:.2f}c / {max_budget}c", self.vision_budget_status()
        return True, "budget OK", self.vision_budget_status()

    def vision_budget_status(self) -> dict[str, Any]:
        section = self.policy().get("openrouter", {})
        return {
            "vision_calls": self._vision_calls,
            "vision_tokens": self._vision_tokens,
            "vision_spend_cents_estimated": round(self._vision_spend_cents, 4),
            "max_budget_cents_per_session": section.get("max_budget_cents_per_session", 100),
            "vision_model": section.get("vision_model", "not configured"),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _killed(self) -> bool:
        default_ks = str(DEFAULT_POLICY_PATH.parent / "DESKTOP_KILL")
        ks_path = self.policy().get("kill_switch_file", default_ks)
        ks = Path(ks_path).expanduser()
        return ks.is_file()

    def _block_url(self, url: str | None) -> bool:
        if not url:
            return False
        for pat in self.policy().get("deny", {}).get("url_patterns", []):
            if _compile(pat).search(url):
                return True
        return False

    def _match_action(self, action: str, patterns: list[str]) -> bool:
        for pat in patterns:
            # Support glob-style matching; "browser_*" matches browser_navigate etc.
            if fnmatch.fnmatchcase(action.lower(), pat.lower()):
                return True
        return False

    def _origin_allowed(self, url: str | None) -> bool:
        if not url:
            return True
        allowed = self.policy().get("browser", {}).get("allowed_origins", ["*"])
        blocked = self.policy().get("browser", {}).get("blocked_origins", [])
        # blocklist first
        for pat in blocked:
            if _compile(pat).match(url):
                return False
        # allowlist
        for pat in allowed:
            if _compile(pat).match(url):
                return True
        return False

    def _rate_ok(self) -> bool:
        window = 60.0
        now = time.time()
        cutoff = now - window
        self._action_log = [t for t in self._action_log if t > cutoff]
        limit = self.policy().get("desktop", {}).get("rate_limit_per_minute", 120)
        return len(self._action_log) < limit

    # ------------------------------------------------------------------
    # Desktop session gating
    # ------------------------------------------------------------------
    def _session_gate(
        self,
        action: str,
        args: dict[str, Any],
        confirmed: bool,
    ) -> Decision | None:
        """Enforce scoped session rules for blind desktop actions.

        Returns a Decision only for actions that are gated by a desktop
        session. Non-gated actions return None and proceed through normal
        policy checks.
        """
        if self.session_store is None:
            return None

        policy = self.policy()
        session_cfg = policy.get("desktop", {}).get("session", {})
        gated = session_cfg.get("gated_tools", [])
        if not self._match_action(action, gated):
            return None

        session = self.session_store.active_session()
        if session is None:
            return Decision(
                action,
                "deny",
                "no desktop session open; use desktop_session_open first",
            )

        if session.is_expired():
            self.session_store.close(session.id)
            return Decision(action, "deny", "desktop session expired")

        if session.actions_used >= session.max_actions:
            self.session_store.close(session.id)
            return Decision(action, "deny", "desktop session action budget exceeded")

        # Tool-specific restrictions.
        if action == "desktop_mouse_click":
            button = args.get("button", "left")
            count = args.get("count", 1)
            allowed_buttons = session_cfg.get("mouse_button_whitelist", ["left"])
            max_count = session_cfg.get("max_mouse_click_count", 1)
            if button not in allowed_buttons or count > max_count:
                return Decision(
                    action,
                    "deny",
                    f"click button={button} count={count} not allowed in desktop session",
                )
        elif action == "desktop_key":
            combo = args.get("combo", "")
            whitelist = session_cfg.get("key_whitelist", [])
            if combo not in whitelist:
                return Decision(
                    action,
                    "deny",
                    f"key combo '{combo}' not allowed in desktop session",
                )

        # Two-gate model: the first acting action inside an armed session
        # requires a second confirm that carries the specific action context.
        if session.state == "armed":
            if not confirmed:
                ctx = session.describe_context(action, args)
                return Decision(
                    action,
                    "confirm",
                    f"first acting action in desktop session '{session.purpose}': {ctx}",
                    metadata={
                        "session_id": session.id,
                        "session_purpose": session.purpose,
                        "context": ctx,
                    },
                )
            self.session_store.arm(session.id)

        # Active session (or just armed and confirmed): allow and consume budget.
        self.session_store.record_action(session.id)
        return Decision(
            action,
            "allow",
            f"desktop session action allowed ({session.actions_used}/{session.max_actions})",
        )

    def _validate_session_open(self, args: dict[str, Any]) -> str | None:
        """Validate desktop_session_open args. Returns error string or None."""
        purpose = args.get("purpose", "")
        if not purpose or not isinstance(purpose, str):
            return "session purpose must be a non-empty string"
        max_actions = args.get("max_actions")
        if max_actions is not None and (not isinstance(max_actions, int) or max_actions <= 0):
            return "max_actions must be a positive integer"
        ttl = args.get("ttl_seconds")
        if ttl is not None and (not isinstance(ttl, int) or ttl <= 0):
            return "ttl_seconds must be a positive integer"
        return None

    # ------------------------------------------------------------------
    # Scope / commit-boundary helpers
    # ------------------------------------------------------------------
    def enter_scope(self, scope: str) -> None:
        """Activate a named task scope (e.g. 'jobapply')."""
        self._scopes.add(scope)

    def exit_scope(self, scope: str) -> None:
        """Deactivate a named task scope."""
        self._scopes.discard(scope)

    def in_scope(self, scope: str) -> bool:
        return scope in self._scopes

    @property
    def active_scopes(self) -> list[str]:
        return sorted(self._scopes)

    def _commit_match(self, label_text: str) -> bool:
        """Return True if the label matches a never_click pattern."""
        cfg = self.policy().get("never_click", {})
        patterns = cfg.get("patterns", [])
        for pat in patterns:
            try:
                if re.search(pat, label_text, re.IGNORECASE):
                    return True
            except re.error:
                continue
        return False

    def check_commit(self, action: str, label_text: str) -> Decision:
        """Classify a resolved clickable label as allow/confirm/deny.

        deny   -> label matches a never_click pattern (and scope is active).
        confirm -> label is empty/unknown (can't verify safety).
        allow  -> label looks safe.
        """
        cfg = self.policy().get("never_click", {})
        scopes = cfg.get("session_scope", [])
        scoped_active = not scopes or any(s in self._scopes for s in scopes)

        if scoped_active and self._commit_match(label_text):
            return Decision(
                action,
                "deny",
                f"commit-like label blocked: {label_text[:80]}",
            )

        if not label_text or not label_text.strip():
            return Decision(
                action,
                "confirm",
                "unlabeled commit-like element; confirm",
                metadata={"label": label_text},
            )

        return Decision(action, "allow", "label ok")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        action: str,
        args: dict[str, Any] | None = None,
        confirmed: bool = False,
    ) -> Decision:
        args = args or {}

        if self._killed():
            return Decision(action, "deny", "kill switch is active")

        policy = self.policy()
        if not policy.get("enabled", True):
            return Decision(action, "deny", "policy is disabled")

        # Hard action deny list.
        denied_actions = policy.get("deny", {}).get("actions", [])
        if self._match_action(action, denied_actions):
            return Decision(action, "deny", f"action '{action}' is in deny list")

        # URL / origin checks for browser actions.
        if action.startswith("browser_"):
            url = (
                args.get("url")
                or args.get("href")
                or args.get("location")
                or args.get("current_url")
            )
            if not self._origin_allowed(url):
                return Decision(action, "deny", f"origin blocked by policy: {url}")
            if self._block_url(url):
                return Decision(action, "deny", f"URL matches deny pattern: {url}")

        # Desktop session gating (before global budget/confirm so a confirmed
        # flag can never bypass a missing or expired session).
        if action.startswith("desktop_"):
            gate = self._session_gate(action, args, confirmed)
            if gate is not None:
                return gate

        # desktop_session_open argument validation.
        if action == "desktop_session_open":
            error = self._validate_session_open(args)
            if error:
                return Decision(action, "deny", error)

        # Session budget.
        max_actions = policy.get("browser", {}).get("max_actions_per_session", 1000)
        if self._session_actions >= max_actions:
            return Decision(action, "deny", f"session action budget exceeded ({max_actions})")

        # Rate limit.
        if not self._rate_ok():
            return Decision(action, "deny", "rate limit exceeded")

        # Enter/Return in a browser form is a submit vector: require confirm.
        if action == "browser_press" and not confirmed:
            key = args.get("key", "")
            if key in ("Enter", "Return"):
                return Decision(
                    action,
                    "confirm",
                    "Enter may submit a form; confirm",
                )

        # Confirm-every mode: explicit user approval required for every click.
        if not confirmed:
            browser_cfg = policy.get("browser", {})
            if browser_cfg.get("confirm_every", False) and action == "browser_click":
                return Decision(
                    action,
                    "confirm",
                    "confirm_every enabled: all browser clicks require explicit confirmation",
                )

        # Confirm requirement (skipped when action has already been confirmed).
        if not confirmed:
            confirm_actions = policy.get("confirm", {}).get("require_for", [])
            if self._match_action(action, confirm_actions):
                return Decision(action, "confirm", f"action '{action}' requires confirmation")

        return Decision(action, "allow", "policy check passed")

    def record_action(self) -> None:
        """Call this after an action is actually executed."""
        self._session_actions += 1
        self._action_log.append(time.time())

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.policy().get("enabled", True),
            "killed": self._killed(),
            "session_actions": self._session_actions,
            "policy_path": str(self.path),
            "rate_log_entries_last_60s": len(self._action_log),
            "active_scopes": self.active_scopes,
        }
