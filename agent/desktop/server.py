"""MCP server exposing guarded browser/desktop primitives."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from mcp.server import MCPServer

from agent.desktop.policy import PolicyEngine, Decision
from agent.desktop.audit import AuditLog
from agent.desktop.confirm import ConfirmStore
from agent.desktop.session import DesktopSessionStore
from agent.desktop.redact import redact_text
from agent.desktop.browser import BrowserController, CommitBlockedError
from agent.desktop.desktop import DesktopController
from agent.desktop.openrouter import OpenRouterClient, OpenRouterError, resize_image

mcp = MCPServer("desktop")
session_store = DesktopSessionStore()
engine = PolicyEngine(session_store=session_store)
audit = AuditLog(engine.policy()["audit"]["path"])
confirm_store = ConfirmStore(default_ttl=engine.policy().get("confirm", {}).get("ttl_seconds", 120))
browser_controller: BrowserController | None = None
desktop_controller: DesktopController | None = None


def _vision_client() -> OpenRouterClient | None:
    """Create an OpenRouter client using policy configuration, or None if not configured."""
    section = engine.policy().get("openrouter", {})
    api_key_env = section.get("api_key_env", "OPENROUTER_API_KEY")
    api_key = os.environ.get(api_key_env) or section.get("api_key")
    if not api_key:
        return None
    return OpenRouterClient(
        api_key=api_key,
        vision_model=section.get("vision_model", "openrouter/google/gemini-2.5-flash-preview"),
    )

ArgDict = dict[str, Any]
BrowserExecutor = Callable[[ArgDict], Awaitable[dict[str, Any]]]
DesktopExecutor = Callable[[ArgDict], dict[str, Any]]


# ---------------------------------------------------------------------------
# Policy helpers
# ---------------------------------------------------------------------------

def _decision_response(action: str, decision: Decision) -> str:
    summary = {
        "ok": decision.verdict == "allow",
        "verdict": decision.verdict,
        "action": action,
        "reason": decision.reason,
    }
    if decision.confirm_token:
        summary["confirm_token"] = decision.confirm_token
    if decision.metadata:
        summary["context"] = decision.metadata
        if decision.metadata.get("narration"):
            summary["narration"] = decision.metadata["narration"]
    return json.dumps(summary, indent=2)


def _confirm_token_args(action: str, args: ArgDict, decision: Decision) -> ArgDict:
    """Merge policy metadata (e.g. session context) into confirm-token args."""
    if not decision.metadata:
        return args
    return {**args, **decision.metadata}


def _current_url() -> str:
    """Return the current page URL if the browser is running, else ''."""
    bc = browser_controller
    if bc is not None and not bc._state.closed and bc._state.page is not None:
        return bc._state.page.url
    return ""


def _narrate(action: str, args: ArgDict, label: str = "", url: str = "") -> str | None:
    """Return a human-readable sentence describing a planned browser action."""
    if not engine.policy().get("browser", {}).get("verbose", False):
        return None
    if action == "browser_click":
        return f"🖱️ CLICK planned — label='{label}', selector='{args.get('element_ref')}', current_url='{url}'"
    if action == "browser_fill":
        text = args.get("text", "")
        snippet = text[:40].replace("'", "\\'")
        suffix = "..." if len(text) > 40 else ""
        return f"⌨️ FILL planned — selector='{args.get('element_ref')}', text='{snippet}{suffix}', url='{url}'"
    if action == "browser_upload":
        return f"📎 UPLOAD planned — selector='{args.get('element_ref')}', files={args.get('file_paths')}, url='{url}'"
    if action == "browser_navigate":
        return f"🌐 NAVIGATE planned — target='{args.get('url')}'"
    if action == "browser_press":
        return f"⌨️ KEY planned — key='{args.get('key')}', url='{url}'"
    return f"⚙️ {action.upper()} planned — current_url='{url}'"


def _confirm_response(action: str, args: ArgDict, decision: Decision, label: str = "", url: str = "") -> str:
    """Create a confirm-token response, optionally with a verbose narration."""
    narration = _narrate(action, args, label, url)
    token_args = _confirm_token_args(action, args, decision)
    if narration:
        token_args["narration"] = narration
    token = confirm_store.create(action, token_args)
    metadata = dict(decision.metadata) if decision.metadata else {}
    if label:
        metadata["label"] = label
    if narration:
        metadata["narration"] = narration
    return _decision_response(
        action,
        Decision(action, "confirm", decision.reason, token, metadata if metadata else None),
    )


async def _resolve_commit_label(element_ref: str) -> str:
    """Resolve an element and return its human-readable label text."""
    try:
        return await _init_browser().resolve_label(element_ref)
    except Exception:
        return ""


def _check(action: str, args: ArgDict) -> Decision:
    # Enforce current-page origin checks for browser actions (except navigate,
    # which is checked against its target URL).
    if action.startswith("browser_") and action != "browser_navigate":
        bc = browser_controller
        if bc is not None and not bc._state.closed and bc._state.page is not None:
            args = {**args, "current_url": bc._state.page.url}
    decision = engine.evaluate(action, args)
    if decision.verdict == "allow":
        engine.record_action()
    return decision


def _audit(action: str, args: ArgDict, decision: Decision, result: Any, duration_ms: float) -> None:
    try:
        audit.write(
            {
                "action": action,
                "args": args,
                "verdict": decision.verdict,
                "reason": decision.reason,
                "result_summary": _summarize(result),
                "duration_ms": duration_ms,
            }
        )
    except Exception:
        pass


def _summarize(result: Any) -> Any:
    """Strip large payloads from audit records."""
    if isinstance(result, str):
        try:
            obj = json.loads(result)
        except Exception:
            obj = {"text": result[:500]}
    else:
        obj = result
    if not isinstance(obj, dict):
        return obj
    cleaned: dict = {}
    for k, v in obj.items():
        if k == "base64":
            cleaned[k] = f"<redacted:{len(v) if isinstance(v, str) else 0} chars>"
        elif k == "text":
            cleaned[k] = v[:500] if isinstance(v, str) else v
        elif isinstance(v, str) and len(v) > 2000:
            cleaned[k] = v[:2000] + "...<truncated>"
        else:
            cleaned[k] = v
    return cleaned


def _init_browser() -> BrowserController:
    global browser_controller
    if browser_controller is None:
        browser_policy = engine.policy().get("browser", {})
        browser_controller = BrowserController(
            headless=browser_policy.get("headless", True),
            profile_mode=browser_policy.get("profile_mode", "isolated"),
            display=browser_policy.get("display") if not browser_policy.get("headless", True) else None,
        )
    return browser_controller


def _init_desktop() -> DesktopController:
    global desktop_controller
    if desktop_controller is None:
        desktop_controller = DesktopController(engine.policy())
    return desktop_controller


# ---------------------------------------------------------------------------
# Meta / guardrail tools
# ---------------------------------------------------------------------------

@mcp.tool()
def vision_status() -> str:
    """Return OpenRouter vision budget and configuration status."""
    status = {
        "configured": _vision_client() is not None,
        "budget": engine.vision_budget_status(),
    }
    return json.dumps(status, indent=2)


@mcp.tool()
def policy_status() -> str:
    """Return the current guardrail policy status and pending confirmations."""
    status = engine.status()
    status["pending_confirmations"] = confirm_store.list_pending()
    status["openrouter_budget"] = engine.vision_budget_status()
    return json.dumps(status, indent=2)


@mcp.tool()
def policy_reload() -> str:
    """Reload the desktop-policy.json file from disk."""
    try:
        engine.reload()
        return json.dumps({"ok": True, "path": str(engine.path)}, indent=2)
    except Exception as exc:
        return json.dumps({"ok": False, "error": str(exc)}, indent=2)


@mcp.tool()
def desktop_kill() -> str:
    """Activate the kill switch. All desktop/browser tools will refuse until the sentinel file is removed."""
    ks = Path(engine.policy().get("kill_switch_file", "~")).expanduser()
    ks.parent.mkdir(parents=True, exist_ok=True)
    ks.write_text("killed\n", encoding="utf-8")
    return json.dumps({"ok": True, "kill_switch_file": str(ks)}, indent=2)


@mcp.tool()
async def desktop_confirm(token: str) -> str:
    """Confirm a pending sensitive action by token and execute it."""
    entry = confirm_store.consume(token)
    if entry is None:
        return json.dumps({"ok": False, "error": "token not found or expired"}, indent=2)

    decision = engine.evaluate(entry.action, entry.args, confirmed=True)
    if decision.verdict != "allow":
        return json.dumps({"ok": False, "error": decision.reason, "verdict": decision.verdict}, indent=2)

    executor = _EXECUTORS.get(entry.action)
    if executor is None:
        return json.dumps({"ok": False, "error": f"no executor for action: {entry.action}"}, indent=2)

    # Desktop executors are synchronous; browser executors are async.
    t0 = time.time()
    try:
        if entry.action.startswith("desktop_"):
            result = executor(entry.args)
        else:
            result = await executor(entry.args)
    except Exception as exc:
        duration_ms = round((time.time() - t0) * 1000, 2)
        _audit(entry.action, entry.args, decision, {"error": str(exc)}, duration_ms)
        return json.dumps({"ok": False, "error": str(exc)}, indent=2)
    duration_ms = round((time.time() - t0) * 1000, 2)
    engine.record_action()
    _audit(entry.action, entry.args, decision, result, duration_ms)
    return json.dumps({"ok": True, "executed": entry.action, **result}, indent=2)


# ---------------------------------------------------------------------------
# Browser executors
# ---------------------------------------------------------------------------

async def _exec_browser_navigate(args: ArgDict) -> dict[str, Any]:
    return await _init_browser().navigate(args["url"])


async def _exec_browser_snapshot(args: ArgDict) -> dict[str, Any]:
    result = await _init_browser().snapshot(full_page=args.get("full_page", False))
    patterns = engine.policy().get("redact", {}).get("snapshot_patterns", [])
    if "text" in result:
        text_redacted, _ = redact_text(result["text"], patterns)
        result["text"] = text_redacted
    return result


async def _exec_browser_click(args: ArgDict) -> dict[str, Any]:
    checker = lambda label: engine.check_commit("browser_click", label)
    return await _init_browser().click(
        args["element_ref"],
        args.get("button", "left"),
        commit_checker=checker,
    )


async def _exec_browser_fill(args: ArgDict) -> dict[str, Any]:
    return await _init_browser().fill(args["element_ref"], args["text"])


async def _exec_browser_upload(args: ArgDict) -> dict[str, Any]:
    return await _init_browser().upload(args["element_ref"], args["file_paths"])


async def _exec_browser_select(args: ArgDict) -> dict[str, Any]:
    return await _init_browser().select(args["element_ref"], args["value"])


async def _exec_browser_press(args: ArgDict) -> dict[str, Any]:
    return await _init_browser().press(args["key"])


async def _exec_browser_scroll(args: ArgDict) -> dict[str, Any]:
    return await _init_browser().scroll(args.get("direction", "down"), args.get("amount", 500))


async def _exec_browser_screenshot(args: ArgDict) -> dict[str, Any]:
    return await _init_browser().screenshot(full_page=args.get("full_page", False))


async def _exec_browser_tabs(args: ArgDict) -> dict[str, Any]:
    return await _init_browser().tabs(args.get("action", "list"), args.get("index"))


async def _exec_browser_wait_for(args: ArgDict) -> dict[str, Any]:
    return await _init_browser().wait_for(args.get("seconds", 1.0))


async def _exec_browser_close(args: ArgDict) -> dict[str, Any]:
    return await _init_browser().close_tab()


async def _execute_visual_query(b64: str, args: ArgDict) -> dict[str, Any]:
    """Shared vision-query path for browser and desktop screenshots."""
    section = engine.policy().get("openrouter", {})
    client = _vision_client()
    if client is None:
        raise RuntimeError("OpenRouter API key not configured")

    allowed, reason, budget = engine.check_vision_budget()
    if not allowed:
        raise RuntimeError(reason)

    resized, meta = resize_image(
        b64,
        max_width=section.get("max_image_width", 1280),
        max_height=section.get("max_image_height", 1280),
        max_bytes=section.get("max_image_bytes", 1_500_000),
    )

    answer, usage = client.visual_query(
        resized,
        args.get("prompt", ""),
        model=section.get("vision_model"),
        system=section.get("system_prompt"),
    )
    engine.record_vision_usage(usage)
    return {
        "answer": answer,
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "model": usage.model,
        },
        "image_meta": meta,
        "budget": engine.vision_budget_status(),
    }


def _sync_visual_query(ss: dict[str, Any], args: ArgDict) -> dict[str, Any]:
    """Run an async visual query from a synchronous desktop executor.

    Uvicorn is running an event loop in the main thread. We use a small helper
    thread that spins up its own asyncio loop so the HTTP client call does not
    interfere with the server's loop.
    """
    import asyncio
    import concurrent.futures
    import threading

    def _run_in_thread() -> dict[str, Any]:
        new_loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(new_loop)
            return new_loop.run_until_complete(_execute_visual_query(ss["base64"], args))
        finally:
            new_loop.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_run_in_thread).result(timeout=180)


async def _exec_browser_visual_query(args: ArgDict) -> dict[str, Any]:
    b64 = await _init_browser().screenshot(full_page=args.get("full_page", False))
    return await _execute_visual_query(b64["base64"], args)


def _exec_desktop_visual_query(args: ArgDict) -> dict[str, Any]:
    ss = _init_desktop().screenshot()
    return _sync_visual_query(ss, args)


def _exec_desktop_status(args: ArgDict) -> dict[str, Any]:
    return _init_desktop().status()


def _exec_desktop_start_display(args: ArgDict) -> dict[str, Any]:
    return _init_desktop().ensure_xephyr()


def _exec_desktop_screenshot(args: ArgDict) -> dict[str, Any]:
    return _init_desktop().screenshot()


def _exec_desktop_mouse_move(args: ArgDict) -> dict[str, Any]:
    return _init_desktop().mouse_move(args["x"], args["y"])


def _exec_desktop_mouse_click(args: ArgDict) -> dict[str, Any]:
    return _init_desktop().mouse_click(args.get("button", "left"), args.get("count", 1))


def _exec_desktop_drag(args: ArgDict) -> dict[str, Any]:
    return _init_desktop().drag(args["x1"], args["y1"], args["x2"], args["y2"])


def _exec_desktop_type_text(args: ArgDict) -> dict[str, Any]:
    return _init_desktop().type_text(args["text"])


def _exec_desktop_key(args: ArgDict) -> dict[str, Any]:
    return _init_desktop().key(args["combo"])


def _exec_desktop_window_list(args: ArgDict) -> dict[str, Any]:
    return _init_desktop().window_list()


def _exec_desktop_window_focus(args: ArgDict) -> dict[str, Any]:
    return _init_desktop().window_focus(args["window_id"])


def _exec_desktop_clipboard_get(args: ArgDict) -> dict[str, Any]:
    return _init_desktop().clipboard_get()


def _exec_desktop_clipboard_set(args: ArgDict) -> dict[str, Any]:
    return _init_desktop().clipboard_set(args["text"])


def _exec_desktop_session_open(args: ArgDict) -> dict[str, Any]:
    cfg = engine.policy().get("desktop", {}).get("session", {})
    ttl = args.get("ttl_seconds") or cfg.get("default_ttl_seconds", 120)
    max_actions = args.get("max_actions") or cfg.get("default_max_actions", 25)
    session = session_store.create(
        purpose=args["purpose"],
        window_hint=args.get("window_hint", ""),
        max_actions=int(max_actions),
        ttl_seconds=int(ttl),
    )
    return {
        "session_id": session.id,
        "purpose": session.purpose,
        "window_hint": session.window_hint,
        "state": session.state,
        "max_actions": session.max_actions,
        "ttl_seconds": ttl,
    }


def _exec_desktop_session_close(args: ArgDict) -> dict[str, Any]:
    session = session_store.close_active()
    return {
        "closed": session is not None,
        "session_id": session.id if session else None,
        "actions_used": session.actions_used if session else 0,
    }


def _exec_desktop_session_status(args: ArgDict) -> dict[str, Any]:
    return session_store.status()


_EXECUTORS: dict[str, BrowserExecutor | DesktopExecutor] = {
    "browser_navigate": _exec_browser_navigate,
    "browser_snapshot": _exec_browser_snapshot,
    "browser_click": _exec_browser_click,
    "browser_fill": _exec_browser_fill,
    "browser_upload": _exec_browser_upload,
    "browser_select": _exec_browser_select,
    "browser_press": _exec_browser_press,
    "browser_scroll": _exec_browser_scroll,
    "browser_screenshot": _exec_browser_screenshot,
    "browser_tabs": _exec_browser_tabs,
    "browser_wait_for": _exec_browser_wait_for,
    "browser_close": _exec_browser_close,
    "browser_visual_query": _exec_browser_visual_query,
    "desktop_status": _exec_desktop_status,
    "desktop_start_display": _exec_desktop_start_display,
    "desktop_screenshot": _exec_desktop_screenshot,
    "desktop_mouse_move": _exec_desktop_mouse_move,
    "desktop_mouse_click": _exec_desktop_mouse_click,
    "desktop_drag": _exec_desktop_drag,
    "desktop_type_text": _exec_desktop_type_text,
    "desktop_key": _exec_desktop_key,
    "desktop_window_list": _exec_desktop_window_list,
    "desktop_window_focus": _exec_desktop_window_focus,
    "desktop_clipboard_get": _exec_desktop_clipboard_get,
    "desktop_clipboard_set": _exec_desktop_clipboard_set,
    "desktop_visual_query": _exec_desktop_visual_query,
    "desktop_session_open": _exec_desktop_session_open,
    "desktop_session_close": _exec_desktop_session_close,
    "desktop_session_status": _exec_desktop_session_status,
}



# ---------------------------------------------------------------------------
# Browser tools
# ---------------------------------------------------------------------------

async def _run_executor(
    action: str,
    args: ArgDict,
    executor: BrowserExecutor | DesktopExecutor,
    decision: Decision,
    confirmed: bool = False,
) -> str:
    """Execute a browser or desktop action, catch errors, audit, and return JSON."""
    t0 = time.time()
    try:
        if action.startswith("desktop_"):
            result = executor(args)
        else:
            result = await executor(args)
    except CommitBlockedError as exc:
        duration_ms = round((time.time() - t0) * 1000, 2)
        block_decision = Decision(action, "deny", str(exc))
        _audit(action, args, block_decision, {"error": str(exc)}, duration_ms)
        return _decision_response(action, block_decision)
    except Exception as exc:
        duration_ms = round((time.time() - t0) * 1000, 2)
        _audit(action, args, decision, {"error": str(exc)}, duration_ms)
        return json.dumps({"ok": False, "action": action, "error": str(exc)}, indent=2)
    duration_ms = round((time.time() - t0) * 1000, 2)
    _audit(action, args, decision, result, duration_ms)
    if confirmed:
        engine.record_action()
    return json.dumps(result, indent=2)


@mcp.tool()
async def browser_status() -> str:
    """Check whether the browser automation context is running."""
    decision = _check("browser_status", {})
    if decision.verdict != "allow":
        return _decision_response("browser_status", decision)
    bc = _init_browser()
    is_closed = bc._state.closed
    return json.dumps(
        {
            "ok": True,
            "running": not is_closed,
            "headless": bc.headless,
            "profile_mode": bc.profile_mode,
        },
        indent=2,
    )


@mcp.tool()
async def browser_navigate(url: str) -> str:
    """Navigate the browser to a URL."""
    decision = _check("browser_navigate", {"url": url})
    if decision.verdict == "deny":
        return _decision_response("browser_navigate", decision)
    if decision.verdict == "confirm":
        return _confirm_response("browser_navigate", {"url": url}, decision, url=_current_url())
    return await _run_executor("browser_navigate", {"url": url}, _exec_browser_navigate, decision)


@mcp.tool()
async def browser_snapshot(full_page: bool = False) -> str:
    """Return text + DOM snapshot of the current page (cheap, no vision)."""
    args = {"full_page": full_page}
    decision = _check("browser_snapshot", args)
    if decision.verdict != "allow":
        return _decision_response("browser_snapshot", decision)
    return await _run_executor("browser_snapshot", args, _exec_browser_snapshot, decision)


@mcp.tool()
async def browser_click(element_ref: str, button: str = "left") -> str:
    """Click an element by CSS selector, xpath (//...), role (aria=button), or text."""
    args = {"element_ref": element_ref, "button": button}
    decision = _check("browser_click", args)
    if decision.verdict == "deny":
        return _decision_response("browser_click", decision)

    # Second-stage commit-boundary check: resolve the actual label before any
    # token is issued or click is executed.
    label = await _resolve_commit_label(element_ref)
    commit_decision = engine.check_commit("browser_click", label)
    if commit_decision.verdict == "deny":
        return _decision_response("browser_click", commit_decision)

    enriched_args = {**args, "label": label}
    url = _current_url()

    if commit_decision.verdict == "confirm":
        return _confirm_response("browser_click", enriched_args, commit_decision, label=label, url=url)

    if decision.verdict == "confirm":
        return _confirm_response("browser_click", enriched_args, decision, label=label, url=url)

    return await _run_executor("browser_click", enriched_args, _exec_browser_click, decision)


@mcp.tool()
async def browser_fill(element_ref: str, text: str) -> str:
    """Fill a form field. Any text matching redaction patterns is hidden from the result."""
    args = {"element_ref": element_ref, "text": text}
    decision = _check("browser_fill", args)
    if decision.verdict == "deny":
        return _decision_response("browser_fill", decision)
    if decision.verdict == "confirm":
        return _confirm_response("browser_fill", args, decision, url=_current_url())
    return await _run_executor("browser_fill", args, _exec_browser_fill, decision)


@mcp.tool()
async def browser_upload(element_ref: str, file_paths: list[str]) -> str:
    """Attach one or more files to a file input element deterministically.

    This bypasses native dialogs using Playwright's set_input_files and is the
    preferred upload path. Falls back to desktop native-dialog sessions only when
    the site blocks this automation.
    """
    args = {"element_ref": element_ref, "file_paths": file_paths}
    decision = _check("browser_upload", args)
    if decision.verdict == "deny":
        return _decision_response("browser_upload", decision)
    if decision.verdict == "confirm":
        return _confirm_response("browser_upload", args, decision, url=_current_url())
    return await _run_executor("browser_upload", args, _exec_browser_upload, decision)


@mcp.tool()
async def browser_select(element_ref: str, value: str) -> str:
    """Select an option in a <select> element."""
    args = {"element_ref": element_ref, "value": value}
    decision = _check("browser_select", args)
    if decision.verdict != "allow":
        return _decision_response("browser_select", decision)
    return await _run_executor("browser_select", args, _exec_browser_select, decision)


@mcp.tool()
async def browser_press(key: str) -> str:
    """Press a key on the keyboard, e.g. 'Enter', 'Tab', 'Control+a'."""
    args = {"key": key}
    decision = _check("browser_press", args)
    if decision.verdict == "deny":
        return _decision_response("browser_press", decision)
    if decision.verdict == "confirm":
        return _confirm_response("browser_press", args, decision, url=_current_url())
    return await _run_executor("browser_press", args, _exec_browser_press, decision)


@mcp.tool()
async def browser_scroll(direction: str = "down", amount: int = 500) -> str:
    """Scroll the page: direction up/down/left/right."""
    args = {"direction": direction, "amount": amount}
    decision = _check("browser_scroll", args)
    if decision.verdict != "allow":
        return _decision_response("browser_scroll", decision)
    return await _run_executor("browser_scroll", args, _exec_browser_scroll, decision)


@mcp.tool()
async def browser_screenshot(full_page: bool = False) -> str:
    """Take a screenshot. Subject to size limits and confirmation policy."""
    args = {"full_page": full_page}
    decision = _check("browser_screenshot", args)
    if decision.verdict == "deny":
        return _decision_response("browser_screenshot", decision)
    if decision.verdict == "confirm":
        return _confirm_response("browser_screenshot", args, decision, url=_current_url())

    t0 = time.time()
    try:
        result = await _exec_browser_screenshot(args)
    except Exception as exc:
        duration_ms = round((time.time() - t0) * 1000, 2)
        _audit("browser_screenshot", args, decision, {"error": str(exc)}, duration_ms)
        return json.dumps({"ok": False, "action": "browser_screenshot", "error": str(exc)}, indent=2)
    max_bytes = engine.policy().get("browser", {}).get("max_screenshot_bytes", 5 * 1024 * 1024)
    if result["bytes"] > max_bytes:
        return json.dumps(
            {"ok": False, "error": f"screenshot {result['bytes']} bytes exceeds limit {max_bytes}"},
            indent=2,
        )
    duration_ms = round((time.time() - t0) * 1000, 2)
    _audit("browser_screenshot", {**args, "bytes": result["bytes"]}, decision, result, duration_ms)
    return json.dumps(result, indent=2)


@mcp.tool()
async def browser_tabs(action: str = "list", index: int | None = None) -> str:
    """List, create, switch or close tabs."""
    args = {"action": action, "index": index}
    decision = _check("browser_tabs", args)
    if decision.verdict != "allow":
        return _decision_response("browser_tabs", decision)
    return await _run_executor("browser_tabs", args, _exec_browser_tabs, decision)


@mcp.tool()
async def browser_wait_for(seconds: float = 1.0) -> str:
    """Wait for a specified number of seconds on the current page."""
    args = {"seconds": seconds}
    decision = _check("browser_wait_for", args)
    if decision.verdict != "allow":
        return _decision_response("browser_wait_for", decision)
    return await _run_executor("browser_wait_for", args, _exec_browser_wait_for, decision)


@mcp.tool()
async def browser_close() -> str:
    """Close the active tab."""
    decision = _check("browser_close", {})
    if decision.verdict != "allow":
        return _decision_response("browser_close", decision)
    return await _run_executor("browser_close", {}, _exec_browser_close, decision)


@mcp.tool()
async def browser_enter_scope(scope: str) -> str:
    """Enter a named task scope (e.g. 'jobapply') to activate scoped rules."""
    engine.enter_scope(scope)
    return json.dumps({"ok": True, "scope": scope, "active_scopes": engine.active_scopes}, indent=2)


@mcp.tool()
async def browser_exit_scope(scope: str) -> str:
    """Exit a named task scope."""
    engine.exit_scope(scope)
    return json.dumps({"ok": True, "scope": scope, "active_scopes": engine.active_scopes}, indent=2)


@mcp.tool()
async def browser_visual_query(prompt: str, full_page: bool = False) -> str:
    """Capture the browser and ask a cheap OpenRouter vision model about it.

    This tool is the expensive fallback. Prefer browser_snapshot + a text model.
    """
    args = {"prompt": prompt, "full_page": full_page}
    decision = _check("browser_visual_query", args)
    if decision.verdict == "deny":
        return _decision_response("browser_visual_query", decision)
    if decision.verdict == "confirm":
        token = confirm_store.create("browser_visual_query", _confirm_token_args("browser_visual_query", args, decision))
        return _decision_response("browser_visual_query", Decision("browser_visual_query", "confirm", decision.reason, token))
    return await _run_executor("browser_visual_query", args, _exec_browser_visual_query, decision, confirmed=True)


# ---------------------------------------------------------------------------
# Desktop tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def desktop_status() -> str:
    """Check the virtual desktop display status."""
    decision = _check("desktop_status", {})
    if decision.verdict != "allow":
        return _decision_response("desktop_status", decision)
    return await _run_executor("desktop_status", {}, _exec_desktop_status, decision)


@mcp.tool()
async def desktop_start_display() -> str:
    """Start the Xephyr virtual display if it is not already running."""
    decision = _check("desktop_start_display", {})
    if decision.verdict != "allow":
        return _decision_response("desktop_start_display", decision)
    return await _run_executor("desktop_start_display", {}, _exec_desktop_start_display, decision)


@mcp.tool()
async def desktop_screenshot() -> str:
    """Take a screenshot of the virtual desktop."""
    decision = _check("desktop_screenshot", {})
    if decision.verdict == "deny":
        return _decision_response("desktop_screenshot", decision)
    if decision.verdict == "confirm":
        token = confirm_store.create("desktop_screenshot", _confirm_token_args("desktop_screenshot", {}, decision))
        return _decision_response("desktop_screenshot", Decision("desktop_screenshot", "confirm", decision.reason, token))
    return await _run_executor("desktop_screenshot", {}, _exec_desktop_screenshot, decision)


@mcp.tool()
async def desktop_mouse_move(x: int, y: int) -> str:
    """Move the mouse cursor to (x, y) on the virtual desktop."""
    args = {"x": x, "y": y}
    decision = _check("desktop_mouse_move", args)
    if decision.verdict == "deny":
        return _decision_response("desktop_mouse_move", decision)
    if decision.verdict == "confirm":
        token = confirm_store.create("desktop_mouse_move", _confirm_token_args("desktop_mouse_move", args, decision))
        return _decision_response("desktop_mouse_move", Decision("desktop_mouse_move", "confirm", decision.reason, token))
    return await _run_executor("desktop_mouse_move", args, _exec_desktop_mouse_move, decision)


@mcp.tool()
async def desktop_mouse_click(button: str = "left", count: int = 1) -> str:
    """Click a mouse button on the virtual desktop."""
    args = {"button": button, "count": count}
    decision = _check("desktop_mouse_click", args)
    if decision.verdict == "deny":
        return _decision_response("desktop_mouse_click", decision)
    if decision.verdict == "confirm":
        token = confirm_store.create("desktop_mouse_click", _confirm_token_args("desktop_mouse_click", args, decision))
        return _decision_response("desktop_mouse_click", Decision("desktop_mouse_click", "confirm", decision.reason, token))
    return await _run_executor("desktop_mouse_click", args, _exec_desktop_mouse_click, decision)


@mcp.tool()
async def desktop_drag(x1: int, y1: int, x2: int, y2: int) -> str:
    """Drag from (x1, y1) to (x2, y2) on the virtual desktop."""
    args = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    decision = _check("desktop_drag", args)
    if decision.verdict == "deny":
        return _decision_response("desktop_drag", decision)
    if decision.verdict == "confirm":
        token = confirm_store.create("desktop_drag", _confirm_token_args("desktop_drag", args, decision))
        return _decision_response("desktop_drag", Decision("desktop_drag", "confirm", decision.reason, token))
    return await _run_executor("desktop_drag", args, _exec_desktop_drag, decision)


@mcp.tool()
async def desktop_type_text(text: str) -> str:
    """Type text into the focused window on the virtual desktop."""
    args = {"text": text}
    decision = _check("desktop_type_text", args)
    if decision.verdict == "deny":
        return _decision_response("desktop_type_text", decision)
    if decision.verdict == "confirm":
        token = confirm_store.create("desktop_type_text", _confirm_token_args("desktop_type_text", args, decision))
        return _decision_response("desktop_type_text", Decision("desktop_type_text", "confirm", decision.reason, token))
    return await _run_executor("desktop_type_text", args, _exec_desktop_type_text, decision)


@mcp.tool()
async def desktop_key(combo: str) -> str:
    """Press a key or key combo (e.g. 'Control+a', 'Return') on the virtual desktop."""
    args = {"combo": combo}
    decision = _check("desktop_key", args)
    if decision.verdict == "deny":
        return _decision_response("desktop_key", decision)
    if decision.verdict == "confirm":
        token = confirm_store.create("desktop_key", _confirm_token_args("desktop_key", args, decision))
        return _decision_response("desktop_key", Decision("desktop_key", "confirm", decision.reason, token))
    return await _run_executor("desktop_key", args, _exec_desktop_key, decision)


@mcp.tool()
async def desktop_window_list() -> str:
    """List windows on the virtual desktop."""
    decision = _check("desktop_window_list", {})
    if decision.verdict == "deny":
        return _decision_response("desktop_window_list", decision)
    if decision.verdict == "confirm":
        token = confirm_store.create("desktop_window_list", _confirm_token_args("desktop_window_list", {}, decision))
        return _decision_response("desktop_window_list", Decision("desktop_window_list", "confirm", decision.reason, token))
    return await _run_executor("desktop_window_list", {}, _exec_desktop_window_list, decision)


@mcp.tool()
async def desktop_window_focus(window_id: str) -> str:
    """Focus a window on the virtual desktop by id."""
    args = {"window_id": window_id}
    decision = _check("desktop_window_focus", args)
    if decision.verdict == "deny":
        return _decision_response("desktop_window_focus", decision)
    if decision.verdict == "confirm":
        token = confirm_store.create("desktop_window_focus", _confirm_token_args("desktop_window_focus", args, decision))
        return _decision_response("desktop_window_focus", Decision("desktop_window_focus", "confirm", decision.reason, token))
    return await _run_executor("desktop_window_focus", args, _exec_desktop_window_focus, decision)


@mcp.tool()
async def desktop_clipboard_get() -> str:
    """Read the virtual desktop clipboard (requires xclip)."""
    decision = _check("desktop_clipboard_get", {})
    if decision.verdict == "deny":
        return _decision_response("desktop_clipboard_get", decision)
    if decision.verdict == "confirm":
        token = confirm_store.create("desktop_clipboard_get", _confirm_token_args("desktop_clipboard_get", {}, decision))
        return _decision_response("desktop_clipboard_get", Decision("desktop_clipboard_get", "confirm", decision.reason, token))
    return await _run_executor("desktop_clipboard_get", {}, _exec_desktop_clipboard_get, decision)


@mcp.tool()
async def desktop_clipboard_set(text: str) -> str:
    """Write text to the virtual desktop clipboard (requires xclip)."""
    args = {"text": text}
    decision = _check("desktop_clipboard_set", args)
    if decision.verdict == "deny":
        return _decision_response("desktop_clipboard_set", decision)
    if decision.verdict == "confirm":
        token = confirm_store.create("desktop_clipboard_set", _confirm_token_args("desktop_clipboard_set", args, decision))
        return _decision_response("desktop_clipboard_set", Decision("desktop_clipboard_set", "confirm", decision.reason, token))
    return await _run_executor("desktop_clipboard_set", args, _exec_desktop_clipboard_set, decision)


@mcp.tool()
async def desktop_visual_query(prompt: str) -> str:
    """Capture the virtual desktop and ask a cheap OpenRouter vision model about it.

    This tool is the expensive fallback. Prefer desktop_screenshot + a text model,
    or a text snapshot when possible.
    """
    args = {"prompt": prompt}
    decision = _check("desktop_visual_query", args)
    if decision.verdict == "deny":
        return _decision_response("desktop_visual_query", decision)
    if decision.verdict == "confirm":
        token = confirm_store.create("desktop_visual_query", _confirm_token_args("desktop_visual_query", args, decision))
        return _decision_response("desktop_visual_query", Decision("desktop_visual_query", "confirm", decision.reason, token))
    return await _run_executor("desktop_visual_query", args, _exec_desktop_visual_query, decision, confirmed=True)


@mcp.tool()
async def desktop_session_open(
    purpose: str,
    window_hint: str = "",
    max_actions: int | None = None,
    ttl_seconds: int | None = None,
) -> str:
    """Open a scoped desktop session for native-dialog control.

    The session requires human confirmation before it becomes active. Once
    active, the first acting action inside the session requires a second
    confirmation; subsequent actions run within the budget/whitelist without
    per-action confirmation until the session closes or expires.
    """
    args = {
        "purpose": purpose,
        "window_hint": window_hint,
        "max_actions": max_actions,
        "ttl_seconds": ttl_seconds,
    }
    decision = _check("desktop_session_open", args)
    if decision.verdict == "deny":
        return _decision_response("desktop_session_open", decision)
    if decision.verdict == "confirm":
        token = confirm_store.create(
            "desktop_session_open",
            _confirm_token_args("desktop_session_open", args, decision),
        )
        return _decision_response(
            "desktop_session_open",
            Decision("desktop_session_open", "confirm", decision.reason, token),
        )
    return await _run_executor("desktop_session_open", args, _exec_desktop_session_open, decision)


@mcp.tool()
async def desktop_session_close() -> str:
    """Close the active desktop session."""
    decision = _check("desktop_session_close", {})
    if decision.verdict != "allow":
        return _decision_response("desktop_session_close", decision)
    return await _run_executor("desktop_session_close", {}, _exec_desktop_session_close, decision)


@mcp.tool()
async def desktop_session_status() -> str:
    """Return the active desktop session status."""
    decision = _check("desktop_session_status", {})
    if decision.verdict != "allow":
        return _decision_response("desktop_session_status", decision)
    return await _run_executor("desktop_session_status", {}, _exec_desktop_session_status, decision)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    mcp.run(transport="streamable-http", host="127.0.0.1", port=8781)


if __name__ == "__main__":
    main()
