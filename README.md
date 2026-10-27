# agent-desktop — guarded desktop & browser control MCP

An MCP server that lets an AI agent drive a Linux desktop and browser — behind
a policy engine. Every action passes through rules, optional human confirmation,
secret redaction, and an append-only audit trail.

The premise: agents that click, type, and screenshot need guardrails more than
they need more capabilities. This server makes the guardrails the architecture.

## Architecture

```
agent/desktop/
├── server.py      # MCP server — 37 tools, wiring only
├── policy.py      # PolicyEngine: rules, verdicts (allow/confirm/deny), confirm tokens
├── confirm.py     # ConfirmStore: TTL'd human-approval tokens
├── audit.py       # AuditLog: append-only JSONL record of every action
├── redact.py      # Secret redaction before anything leaves the machine
├── browser.py     # Playwright controller with commit/rollback scopes
├── desktop.py     # X11 desktop control (mouse, keyboard, windows, clipboard)
├── openrouter.py  # Optional vision via OpenRouter (API key stays in env)
├── session.py     # Desktop session store — gating per session
└── tests/         # pytest suite; policy tests run without a browser or X server
```

Request flow: `tool call → PolicyEngine verdict → (confirm token if required) →
executor → redaction → audit entry`.

## Tool families

- **Browser (16)** — `browser_navigate`, `browser_click`, `browser_fill`,
  `browser_snapshot`, `browser_screenshot`, `browser_tabs`,
  `browser_enter_scope`/`browser_exit_scope` (commit/rollback),
  `browser_visual_query`, …
- **Desktop (16)** — `desktop_mouse_move`, `desktop_mouse_click`,
  `desktop_type_text`, `desktop_key`, `desktop_window_list`,
  `desktop_clipboard_get`/`_set`, `desktop_screenshot`,
  `desktop_visual_query`, …
- **Governance** — `policy_status`, `policy_reload`, `desktop_confirm`
  (approve a pending action), `desktop_session_open`/`_close`/`_status`,
  `vision_status`

## Install

```bash
git clone <this repo> && cd agent-desktop
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/playwright install chromium   # for browser tools
pytest agent/desktop/tests              # policy tests, no X server needed
```

Run: `python -m agent.desktop.server` from the repo root.

## Policy configuration

Rules load from a JSON policy file (default
`~/local-agent/config/desktop-policy.json`, overridable). The policy defines
which actions are allowed, which require an explicit confirm token, protected
paths, and the audit log location. API keys are read from environment
variables — never from the policy file.

## License

[MIT](LICENSE)
