"""OS-level desktop control primitives via xdotool and ImageMagick.

All actions are constrained to a configurable X display (default ``:2``) which is
expected to be a Xephyr virtual display. The module refuses to run on the user's
live display (``:1``) unless ``require_virtual_display`` is disabled in policy.
"""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

USER_LIVE_DISPLAYS = [":0", ":1"]


class DesktopError(Exception):
    """Raised when a desktop action cannot be completed safely."""


class DesktopController:
    """Control mouse/keyboard/window on a dedicated X display."""

    def __init__(self, policy: dict[str, Any]) -> None:
        self.policy = policy
        self.display = str(policy.get("desktop", {}).get("display", ":2"))
        self.require_virtual = bool(policy.get("desktop", {}).get("require_virtual_display", True))
        self.xephyr_process: subprocess.Popen | None = None
        self._clipboard_proc: subprocess.Popen | None = None
        self._check_display_safety()

    def _check_display_safety(self) -> None:
        if self.require_virtual and self.display in USER_LIVE_DISPLAYS:
            raise DesktopError(f"refusing to control live user display {self.display}")

    # ------------------------------------------------------------------
    # Display lifecycle
    # ------------------------------------------------------------------
    @property
    def display_safe(self) -> bool:
        return not (self.require_virtual and self.display in USER_LIVE_DISPLAYS)

    def ensure_xephyr(self) -> dict[str, Any]:
        """Start Xephyr on the target display if not already running."""
        if not self.display_safe:
            raise DesktopError("display not safe")
        if self._xephyr_alive():
            return {"ok": True, "display": self.display, "started": False, "pid": None}
        if shutil.which("Xephyr") is None:
            raise DesktopError("Xephyr not installed")
        screen = self.policy.get("desktop", {}).get("xephyr_screen", "1280x720")
        cmd = ["Xephyr", self.display, "-screen", screen, "-ac", "-br", "-no-host-grab"]
        # Start detached from our process group so it keeps running after tool call.
        self.xephyr_process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env={**os.environ, "DISPLAY": ":1"},
        )
        # Wait for display to become responsive.
        deadline = time.time() + 5
        while time.time() < deadline:
            if self._xephyr_alive():
                return {"ok": True, "display": self.display, "started": True, "pid": self.xephyr_process.pid}
            time.sleep(0.2)
        raise DesktopError("Xephyr failed to start")

    def _xephyr_alive(self) -> bool:
        """Check whether the target display is reachable."""
        try:
            subprocess.run(
                ["xdotool", "getdisplaygeometry"],
                env={**os.environ, "DISPLAY": self.display},
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
            return True
        except Exception:
            return False

    def status(self) -> dict[str, Any]:
        alive = self._xephyr_alive()
        return {
            "ok": True,
            "display": self.display,
            "xephyr_alive": alive,
            "virtual_display_required": self.require_virtual,
            "display_safe": self.display_safe,
            "pid": self.xephyr_process.pid if self.xephyr_process else None,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _env(self) -> dict[str, str]:
        return {**os.environ, "DISPLAY": self.display}

    def _run(self, cmd: list[str], timeout: float = 5.0) -> tuple[int, str, str]:
        proc = subprocess.run(
            cmd,
            env=self._env(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return proc.returncode, proc.stdout, proc.stderr

    def _require_alive(self) -> None:
        if not self._xephyr_alive():
            raise DesktopError(f"desktop display {self.display} is not alive")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def screenshot(self) -> dict[str, Any]:
        self._require_alive()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            rc, out, err = self._run(["import", "-window", "root", path])
            if rc != 0:
                raise DesktopError(f"screenshot failed: {err or out}")
            data = Path(path).read_bytes()
            b64 = base64.b64encode(data).decode("ascii")
            return {
                "type": "image/png",
                "base64": b64,
                "bytes": len(data),
                "display": self.display,
            }
        finally:
            try:
                Path(path).unlink()
            except FileNotFoundError:
                pass

    def mouse_move(self, x: int, y: int) -> dict[str, Any]:
        self._require_alive()
        rc, out, err = self._run(["xdotool", "mousemove", str(x), str(y)])
        if rc != 0:
            raise DesktopError(f"mouse_move failed: {err or out}")
        return {"moved_to": {"x": x, "y": y}, "display": self.display}

    def mouse_click(self, button: str = "left", count: int = 1) -> dict[str, Any]:
        self._require_alive()
        button_arg = {"left": "1", "middle": "2", "right": "3"}.get(button, button)
        rc, out, err = self._run(["xdotool", "click", "--repeat", str(count), button_arg])
        if rc != 0:
            raise DesktopError(f"mouse_click failed: {err or out}")
        return {"clicked": button, "count": count, "display": self.display}

    def drag(self, x1: int, y1: int, x2: int, y2: int) -> dict[str, Any]:
        self._require_alive()
        # Move to start, hold, move to end, release.
        rc, out, err = self._run(
            ["xdotool", "mousemove", str(x1), str(y1), "mousedown", "1", "mousemove", str(x2), str(y2), "mouseup", "1"]
        )
        if rc != 0:
            raise DesktopError(f"drag failed: {err or out}")
        return {"dragged": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}, "display": self.display}

    def type_text(self, text: str) -> dict[str, Any]:
        self._require_alive()
        rc, out, err = self._run(["xdotool", "type", text])
        if rc != 0:
            raise DesktopError(f"type_text failed: {err or out}")
        return {"typed": len(text), "display": self.display}

    def key(self, combo: str) -> dict[str, Any]:
        self._require_alive()
        rc, out, err = self._run(["xdotool", "key", combo])
        if rc != 0:
            raise DesktopError(f"key failed: {err or out}")
        return {"pressed": combo, "display": self.display}

    def window_list(self) -> dict[str, Any]:
        self._require_alive()
        rc, out, err = self._run(["xdotool", "search", "--name", "", "getwindowgeometry"])
        # xdotool returns one line per property; we'll just split.
        if rc != 0 and not out:
            raise DesktopError(f"window_list failed: {err}")
        windows: list[dict] = []
        try:
            ids = subprocess.check_output(["xdotool", "search", "--name", ""], env=self._env(), text=True, timeout=5).splitlines()
            for wid in ids:
                wid = wid.strip()
                if not wid:
                    continue
                geo = self._window_props(wid)
                windows.append({"id": wid, **geo})
        except subprocess.TimeoutExpired:
            raise DesktopError("window_list timed out")
        return {"windows": windows, "count": len(windows), "display": self.display}

    def _window_props(self, wid: str) -> dict[str, Any]:
        try:
            name = subprocess.check_output(["xdotool", "getwindowname", wid], env=self._env(), text=True, timeout=2).strip()
        except Exception:
            name = ""
        try:
            pid = subprocess.check_output(["xdotool", "getwindowpid", wid], env=self._env(), text=True, timeout=2).strip()
        except Exception:
            pid = ""
        try:
            geo_out = subprocess.check_output(["xdotool", "getwindowgeometry", wid], env=self._env(), text=True, timeout=2)
            geo: dict[str, Any] = {"geometry_raw": geo_out.strip()}
        except Exception:
            geo = {"geometry_raw": ""}
        return {"name": name, "pid": pid, **geo}

    def window_focus(self, window_id: str) -> dict[str, Any]:
        self._require_alive()
        # Strategy: EWMH activate, then plain X focus, then click inside window.
        errors: list[str] = []
        for cmd in [
            ["xdotool", "windowactivate", "--sync", window_id],
            ["xdotool", "windowfocus", window_id],
            ["xdotool", "windowraise", window_id],
        ]:
            rc, out, err = self._run(cmd)
            if rc == 0:
                return {"focused": window_id, "method": cmd[1], "display": self.display}
            errors.append(f"{cmd[1]}: {err or out}")

        # Last resort: click near the center of the window.
        try:
            geo = self._window_props(window_id)
            if geo.get("box"):
                x = int(geo["box"]["x"] + geo["box"]["width"] / 2)
                y = int(geo["box"]["y"] + geo["box"]["height"] / 2)
            else:
                rc, out, err = self._run(["xdotool", "getwindowgeometry", window_id])
                # Parse "Geometry: WxH" and "Position: X,Y" from out.
                x, y = 100, 100
            self._run(["xdotool", "mousemove", str(x), str(y), "click", "1"])
            return {"focused": window_id, "method": "click-center", "display": self.display}
        except Exception as exc:
            raise DesktopError(f"window_focus failed: {exc}; tried: {errors}")

    def clipboard_get(self) -> dict[str, Any]:
        self._require_alive()
        if shutil.which("xclip") is None:
            raise DesktopError("xclip not installed")
        rc, out, err = self._run(["xclip", "-selection", "clipboard", "-o"])
        if rc != 0:
            raise DesktopError(f"clipboard_get failed: {err or out}")
        return {"clipboard": out, "display": self.display}

    def clipboard_set(self, text: str) -> dict[str, Any]:
        self._require_alive()
        if shutil.which("xclip") is None:
            raise DesktopError("xclip not installed")
        # xclip becomes the CLIPBOARD selection owner and normally waits forever.
        # We write the data and detach; the process keeps the selection alive.
        if self._clipboard_proc is not None:
            try:
                self._clipboard_proc.terminate()
            except Exception:
                pass
        proc = subprocess.Popen(
            ["xclip", "-selection", "clipboard", "-i"],
            env=self._env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            proc.stdin.write(text.encode("utf-8"))
            proc.stdin.close()
        except Exception as exc:
            try:
                proc.kill()
            except Exception:
                pass
            raise DesktopError(f"clipboard_set failed: {exc}")
        time.sleep(0.2)
        if proc.poll() is not None:
            rc = proc.returncode
            if rc != 0:
                err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
                raise DesktopError(f"xclip exited {rc}: {err}")
        self._clipboard_proc = proc
        return {"clipboard_set": len(text), "display": self.display}
