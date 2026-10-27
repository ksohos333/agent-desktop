"""Playwright browser wrapper for desktop MCP.

Runs a single persistent Playwright browser context per server process. The
context is launched with an isolated profile by default and can optionally be
headed for debugging.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from playwright.async_api import async_playwright, BrowserContext, Page, Locator


class CommitBlockedError(Exception):
    """Raised when a click target is blocked by the commit-boundary policy."""


@dataclass
class BrowserState:
    context: BrowserContext | None = None
    page: Page | None = None
    pages: list[Page] | None = None
    closed: bool = True


class BrowserController:
    """Thin async wrapper around a Playwright context."""

    def __init__(
        self,
        headless: bool = True,
        profile_mode: str = "isolated",
        display: str | None = None,
    ) -> None:
        self.headless = headless
        self.profile_mode = profile_mode
        self.display = display
        self._playwright = None
        self._browser = None
        self._state = BrowserState(closed=True)

    async def start(self) -> dict[str, Any]:
        if not self._state.closed:
            return {"status": "already running"}

        self._playwright = await async_playwright().start()
        args = []
        if self.headless:
            args.append("--headless=new")
        # Stability args only. We do not pass anti-automation flags because
        # this agent is not designed to evade site protections.
        args.extend(
            [
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ]
        )
        launch_kwargs: dict[str, Any] = {
            "headless": self.headless,
            "args": args,
        }
        if self.display:
            launch_kwargs["env"] = {"DISPLAY": self.display}
        self._browser = await self._playwright.chromium.launch(**launch_kwargs)
        context_args: dict[str, Any] = {
            "viewport": {"width": 1920, "height": 1080},
            "device_scale_factor": 1,
        }
        if self.profile_mode == "isolated":
            context_args["geolocation"] = None
        self._state.context = await self._browser.new_context(**context_args)
        self._state.page = await self._state.context.new_page()
        self._state.pages = [self._state.page]
        self._state.closed = False
        return {"status": "started", "headless": self.headless, "profile_mode": self.profile_mode}

    async def ensure_started(self) -> None:
        if self._state.closed:
            await self.start()

    async def stop(self) -> dict[str, Any]:
        try:
            if self._browser:
                await self._browser.close()
        except Exception as exc:
            return {"status": "error", "message": str(exc)}
        finally:
            if self._playwright:
                await self._playwright.stop()
            self._state = BrowserState(closed=True)
        return {"status": "stopped"}

    async def navigate(self, url: str) -> dict[str, Any]:
        await self.ensure_started()
        page = self._state.page
        assert page is not None
        start = time.time()
        response = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        elapsed = round(time.time() - start, 2)
        return {
            "url": page.url,
            "status": response.status if response else None,
            "title": await page.title(),
            "load_time_seconds": elapsed,
        }

    async def snapshot(self, full_page: bool = False) -> dict[str, Any]:
        """Return text + DOM snapshot of the current page (cheap, no vision)."""
        await self.ensure_started()
        page = self._state.page
        assert page is not None
        title = await page.title()
        url = page.url
        body_text = await page.inner_text("body")
        html = await page.content()
        # Lightweight list of interactive elements for clicking/filling.
        interactive: list[dict[str, Any]] = []
        for tag in ["a", "button", "input", "textarea", "select"]:
            try:
                locators = await page.locator(tag).all()
            except Exception:
                continue
            for i, el in enumerate(locators):
                try:
                    visible = await el.is_visible()
                    if not visible:
                        continue
                    txt = ""
                    if tag != "input":
                        txt = (await el.inner_text()).strip()[:80]
                    else:
                        txt = await el.input_value() or ""
                    placeholder = await el.get_attribute("placeholder") or ""
                    name_attr = await el.get_attribute("name") or ""
                    aria = await el.get_attribute("aria-label") or ""
                    el_type = await el.get_attribute("type") or ""
                    box = await el.bounding_box() if tag in ("button", "a", "input") else None
                    escaped = txt[:40].replace("'", "\\'")
                    entry = {
                        "tag": tag,
                        "index": i,
                        "selector": f"{tag}:has-text('{escaped}')" if txt else f"{tag} >> nth={i}",
                        "text": txt,
                        "placeholder": placeholder,
                        "name": name_attr,
                        "aria_label": aria,
                        "type": el_type,
                        "box": box,
                    }
                    interactive.append(entry)
                except Exception:
                    pass
        return {
            "url": url,
            "title": title,
            "full_page": full_page,
            "text": body_text,
            "html_excerpt": html[:16384],
            "interactive_elements": interactive,
            "type": "snapshot",
        }

    async def _element_label(self, el: Locator) -> str:
        """Collect human-readable label text from an element and its attrs."""
        parts: list[str] = []
        txt = (await el.inner_text()).strip()
        if txt:
            parts.append(txt)
        for attr in ("aria-label", "value", "title"):
            val = (await el.get_attribute(attr) or "").strip()
            if val and val not in parts:
                parts.append(val)
        return " ".join(parts)

    async def resolve_label(self, element_ref: str) -> str:
        """Resolve an element reference and return its label text.

        If the selector matches multiple elements, the first visible one is used.
        """
        await self.ensure_started()
        page = self._state.page
        assert page is not None
        el = await _resolve(page, element_ref)
        if await el.count() > 1:
            el = el.first
        return await self._element_label(el)

    async def click(
        self,
        element_ref: str,
        button: str = "left",
        commit_checker: Any | None = None,
    ) -> dict[str, Any]:
        await self.ensure_started()
        page = self._state.page
        assert page is not None
        el = await _resolve(page, element_ref)
        if await el.count() > 1:
            el = el.first
        if commit_checker is not None:
            label = await self._element_label(el)
            decision = commit_checker(label)
            if decision.verdict == "deny":
                raise CommitBlockedError(decision.reason)
        await el.click(button=button)
        return {"clicked": element_ref, "url": page.url, "title": await page.title()}

    async def fill(self, element_ref: str, text: str, redact: bool = False) -> dict[str, Any]:
        await self.ensure_started()
        page = self._state.page
        assert page is not None
        el = await _resolve(page, element_ref)
        await el.fill(text)
        return {
            "filled": element_ref,
            "length": len(text),
            "redacted": redact,
            "url": page.url,
        }

    async def upload(self, element_ref: str, file_paths: list[str]) -> dict[str, Any]:
        """Attach files to a file input element deterministically.

        Uses Playwright's set_input_files, which bypasses native dialogs and
        works with standard <input type=file> elements. Missing files are
        rejected before any page mutation.
        """
        await self.ensure_started()
        page = self._state.page
        assert page is not None

        resolved_paths: list[str] = []
        for path in file_paths:
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"upload file not found: {path}")
            resolved_paths.append(str(p.resolve()))

        el = await _resolve(page, element_ref)
        await el.set_input_files(resolved_paths)
        return {
            "uploaded": element_ref,
            "files": resolved_paths,
            "count": len(resolved_paths),
            "url": page.url,
            "title": await page.title(),
        }

    async def select(self, element_ref: str, value: str) -> dict[str, Any]:
        await self.ensure_started()
        page = self._state.page
        assert page is not None
        el = await _resolve(page, element_ref)
        await el.select_option(value)
        return {"selected": element_ref, "value": value, "url": page.url}

    async def press(self, key: str) -> dict[str, Any]:
        await self.ensure_started()
        page = self._state.page
        assert page is not None
        await page.keyboard.press(key)
        return {"pressed": key, "url": page.url, "title": await page.title()}

    async def scroll(self, direction: str = "down", amount: int = 500) -> dict[str, Any]:
        await self.ensure_started()
        page = self._state.page
        assert page is not None
        if direction == "up":
            await page.mouse.wheel(0, -amount)
        elif direction == "down":
            await page.mouse.wheel(0, amount)
        elif direction == "left":
            await page.mouse.wheel(-amount, 0)
        elif direction == "right":
            await page.mouse.wheel(amount, 0)
        else:
            raise ValueError(f"unsupported scroll direction: {direction}")
        return {"scrolled": direction, "amount": amount, "url": page.url}

    async def screenshot(self, full_page: bool = False) -> dict[str, Any]:
        await self.ensure_started()
        page = self._state.page
        assert page is not None
        png_bytes = await page.screenshot(full_page=full_page, type="png")
        b64 = base64.b64encode(png_bytes).decode("ascii")
        return {
            "type": "image/png",
            "base64": b64,
            "bytes": len(png_bytes),
            "url": page.url,
            "full_page": full_page,
        }

    async def tabs(self, action: str = "list", index: int | None = None) -> dict[str, Any]:
        await self.ensure_started()
        context = self._state.context
        assert context is not None
        pages = context.pages
        if action == "list":
            return {
                "tabs": [
                    {"index": i, "url": p.url, "title": await p.title()} for i, p in enumerate(pages)
                ],
                "count": len(pages),
            }
        if action == "new":
            new_page = await context.new_page()
            self._state.pages = context.pages
            self._state.page = new_page
            return {"tab": len(context.pages) - 1, "url": new_page.url, "action": "new"}
        if action == "switch":
            if index is None or index < 0 or index >= len(pages):
                raise ValueError(f"invalid tab index: {index}")
            self._state.page = pages[index]
            await pages[index].bring_to_front()
            return {"switched_to": index, "url": pages[index].url, "title": await pages[index].title()}
        if action == "close":
            if index is None or index < 0 or index >= len(pages):
                raise ValueError(f"invalid tab index: {index}")
            await pages[index].close()
            self._state.pages = context.pages
            return {"closed": index, "remaining": len(context.pages)}
        raise ValueError(f"unsupported tab action: {action}")

    async def wait_for(self, seconds: float = 1.0) -> dict[str, Any]:
        await self.ensure_started()
        page = self._state.page
        assert page is not None
        await page.wait_for_timeout(seconds * 1000)
        return {"waited_seconds": seconds, "url": page.url}

    async def close_tab(self) -> dict[str, Any]:
        await self.ensure_started()
        page = self._state.page
        assert page is not None
        await page.close()
        context = self._state.context
        assert context is not None
        self._state.pages = context.pages
        self._state.page = context.pages[0] if context.pages else None
        return {"tabs_remaining": len(context.pages)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _resolve(page: Page, ref: str) -> Any:
    """Resolve an element reference: CSS selector, nth selector, or text-based."""
    # Xpath.
    if ref.startswith("//") or ref.startswith("("):
        return page.locator(f"xpath={ref}")
    # Attr/name selectors exact: name=foo, aria=label
    if ref.startswith("name="):
        return page.locator(f"[name='{ref.split('=', 1)[1]}']")
    if ref.startswith("aria="):
        return page.get_by_role(ref.split("=", 1)[1])
    # `tag >> nth=N`
    if ">> nth=" in ref:
        tag, _, n = ref.partition(" >> nth=")
        return page.locator(tag).nth(int(n))
    if ref.startswith("text="):
        return page.get_by_text(ref.split("=", 1)[1])
    # Default to CSS selector.
    return page.locator(ref)
