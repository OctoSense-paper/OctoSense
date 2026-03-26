"""Private environment probing helpers."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from typing import Any, Literal, Protocol

DisplayMode = Literal["terminal", "notebook", "plain"]

__all__ = ["collect_runtime_env", "detect_display_mode"]


class _SupportsIsatty(Protocol):
    def isatty(self) -> bool:
        ...


def _safe_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "unknown"


def _torch_mps_available(torch_module: Any) -> bool:
    backends = getattr(torch_module, "backends", None)
    mps_backend = getattr(backends, "mps", None)
    is_available = getattr(mps_backend, "is_available", None)
    return bool(is_available()) if callable(is_available) else False


def detect_display_mode(stream: _SupportsIsatty | None = None) -> DisplayMode:
    """Classify the current display environment for internal console rendering."""

    if stream is not None and bool(getattr(stream, "isatty", lambda: False)()):
        return "terminal"
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is not None and getattr(shell, "kernel", None) is not None:
            return "notebook"
    except Exception:
        pass
    return "plain"


def collect_runtime_env(*, extra: dict[str, Any] | None = None) -> dict[str, str]:
    payload: dict[str, Any] = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.platform(),
        "numpy": _safe_version("numpy"),
        "torch": _safe_version("torch"),
    }
    try:
        import torch

        payload["torch_cuda"] = str(torch.version.cuda or "none")
        cudnn = torch.backends.cudnn.version()  # type: ignore[no-untyped-call]
        payload["torch_cudnn"] = str(cudnn if cudnn is not None else "none")
        payload["cuda"] = str(torch.cuda.is_available())
        payload["mps"] = str(_torch_mps_available(torch))
    except Exception:
        payload["torch_cuda"] = "unknown"
        payload["torch_cudnn"] = "unknown"
        payload["cuda"] = "unknown"
        payload["mps"] = "unknown"

    if extra:
        payload.update(dict(extra))
    return {str(key): str(value) for key, value in payload.items()}
