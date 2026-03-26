"""Private console/progress rendering helpers."""

from __future__ import annotations

import html
import sys
import time
from dataclasses import dataclass
from typing import TextIO

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    tqdm = None

from octosense._internal.env import detect_display_mode

__all__ = ["ProgressBar", "format_bytes"]


def format_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown"
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PiB"


def _format_bytes_compact(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "?"
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(size)}{unit}"
            if size >= 100:
                return f"{size:.0f}{unit}"
            if size >= 10:
                return f"{size:.1f}{unit}"
            return f"{size:.2f}{unit}"
        size /= 1024.0
    return f"{size:.2f}PiB"


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds == float("inf"):
        return "?"
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


@dataclass
class ProgressBar:
    """Compact progress bar suitable for CLI and notebook logs."""

    label: str
    total: int | None = None
    width: int = 10
    stream: TextIO | None = None
    min_interval_sec: float = 0.2

    def __post_init__(self) -> None:
        self.stream = self.stream or sys.stderr
        self._current = 0
        self._started_at = time.monotonic()
        self._finished = False
        self._mode = detect_display_mode(self.stream)
        self._last_render = 0.0
        self._display_handle = None
        self._progress = self._create_progress()

    def _create_progress(self):
        if tqdm is None or self._mode in {"plain", "notebook"}:
            return None
        return tqdm(
            total=self.total,
            desc=self.label,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
            mininterval=self.min_interval_sec,
            dynamic_ncols=True,
            leave=True,
            file=self.stream,
            bar_format=(
                "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}]"
            )
            if self.total is not None
            else "{desc}: {n_fmt} [{elapsed}, {rate_fmt}]",
        )

    def update(self, current: int) -> None:
        self._current = max(0, int(current))
        if self._finished:
            return
        if self._mode == "notebook":
            now = time.monotonic()
            if now - self._last_render < self.min_interval_sec:
                return
            self._render_notebook(now=now)
            self._last_render = now
            return
        if self._progress is None:
            return
        delta = self._current - int(self._progress.n)
        if delta > 0:
            self._progress.update(delta)

    def finish(self) -> None:
        if self._finished:
            return
        if self.total is not None:
            self._current = max(self._current, int(self.total))
        if self._mode == "notebook":
            self._render_notebook(now=time.monotonic())
        elif self._progress is None:
            self._render_plain(now=time.monotonic())
        else:
            remaining = self._current - int(self._progress.n)
            if remaining > 0:
                self._progress.update(remaining)
            self._progress.refresh()
            self._progress.close()
        self._finished = True

    def _render_plain(self, *, now: float) -> None:
        line = self._line(now=now)
        self.stream.write(line + "\n")
        self.stream.flush()

    def _render_notebook(self, *, now: float) -> None:
        line = self._line(now=now)
        try:
            from IPython.display import display
        except Exception:
            self._render_plain(now=now)
            return
        payload = {
            "text/plain": line,
            "text/html": (
                "<pre style='margin:0; font-family: var(--jp-code-font-family, monospace);'>"
                f"{html.escape(line)}"
                "</pre>"
            ),
        }
        if self._display_handle is None:
            self._display_handle = display(payload, raw=True, display_id=True)
            return
        self._display_handle.update(payload, raw=True)

    def _line(self, *, now: float) -> str:
        elapsed = max(0.001, now - self._started_at)
        rate = self._current / elapsed
        rate_text = f"{_format_bytes_compact(int(rate))}/s" if rate > 0 else "?/s"
        if self.total and self.total > 0:
            ratio = min(1.0, self._current / self.total)
            filled = int(self.width * ratio)
            bar = "█" * filled + " " * (self.width - filled)
            percent = f"{ratio * 100:3.0f}%"
            remaining = max(0, self.total - self._current)
            eta = (remaining / rate) if rate > 0 else None
            return (
                f"{self.label}: {percent}|{bar}| "
                f"{_format_bytes_compact(self._current)}/{_format_bytes_compact(self.total)} "
                f"[{_format_duration(elapsed)}<{_format_duration(eta)}, {rate_text}]"
            )
        return (
            f"{self.label}: {_format_bytes_compact(self._current)} "
            f"[{_format_duration(elapsed)}, {rate_text}]"
        )
