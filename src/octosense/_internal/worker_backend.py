"""Private multiprocessing backend helpers."""

from __future__ import annotations

import multiprocessing as mp
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

__all__ = [
    "WorkerBackendStatus",
    "inspect_worker_backend",
    "recommended_start_method",
    "resolve_num_workers",
]


@dataclass(frozen=True)
class WorkerBackendStatus:
    """Snapshot of DataLoader multiprocessing availability."""

    available: bool
    reason: str | None
    sharing_strategy: str | None
    torch_shm_manager: str | None
    repaired_permissions: bool = False


def _get_sharing_strategy() -> str:
    return cast(str, torch.multiprocessing.get_sharing_strategy())


def _spawn_main_is_importable() -> bool:
    main_mod = sys.modules.get("__main__")
    if main_mod is None:
        return False
    main_file = getattr(main_mod, "__file__", None)
    if not isinstance(main_file, str) or main_file.startswith("<"):
        return False
    return Path(main_file).exists()


def _torch_shm_manager_path() -> Path | None:
    torch_bin = Path(torch.__file__).resolve().parent / "bin" / "torch_shm_manager"
    return torch_bin if torch_bin.exists() else None


def _ensure_executable(path: Path) -> tuple[bool, bool, str | None]:
    if os.access(path, os.X_OK):
        return True, False, None
    try:
        mode = path.stat().st_mode
        target_mode = mode | stat.S_IXUSR
        if mode & stat.S_IRGRP:
            target_mode |= stat.S_IXGRP
        if mode & stat.S_IROTH:
            target_mode |= stat.S_IXOTH
        if target_mode != mode:
            os.chmod(path, target_mode)
    except OSError as exc:
        return False, False, f"unable to chmod torch_shm_manager ({path}): {exc}"
    if os.access(path, os.X_OK):
        return True, True, None
    return False, False, f"torch_shm_manager is not executable: {path}"


def recommended_start_method() -> str:
    if sys.platform == "darwin" or os.name == "nt":
        return "spawn"
    return "fork"


def inspect_worker_backend(*, try_fix_permissions: bool = True) -> WorkerBackendStatus:
    try:
        strategy = _get_sharing_strategy()
    except Exception as exc:  # pragma: no cover - defensive
        return WorkerBackendStatus(False, f"failed to read torch multiprocessing sharing strategy: {exc}", None, None)

    try:
        start_method = mp.get_start_method(allow_none=True)
    except RuntimeError:
        start_method = None
    if start_method is None:
        start_method = recommended_start_method()

    if start_method == "spawn" and not _spawn_main_is_importable():
        return WorkerBackendStatus(
            False,
            "spawn multiprocessing requires an importable __main__ module; use a python file entrypoint or set num_workers=0",
            strategy,
            None,
        )

    shm_path = _torch_shm_manager_path()
    shm_path_str = str(shm_path) if shm_path else None

    if strategy != "file_system":
        return WorkerBackendStatus(True, None, strategy, shm_path_str)

    if shm_path is None:
        return WorkerBackendStatus(
            False,
            "file_system sharing strategy requires torch_shm_manager, but it was not found in this torch installation",
            strategy,
            None,
        )

    if os.access(shm_path, os.X_OK):
        return WorkerBackendStatus(True, None, strategy, shm_path_str)

    if not try_fix_permissions:
        return WorkerBackendStatus(
            False,
            "torch_shm_manager is not executable under file_system sharing; set num_workers=0 or chmod +x the binary",
            strategy,
            shm_path_str,
        )

    executable, repaired, failure = _ensure_executable(shm_path)
    if not executable:
        return WorkerBackendStatus(
            False,
            f"file_system sharing unavailable because torch_shm_manager cannot execute: {failure}",
            strategy,
            shm_path_str,
        )

    reason = (
        "Repaired torch_shm_manager executable permission for file_system multiprocessing backend"
        if repaired
        else None
    )
    return WorkerBackendStatus(True, reason, strategy, shm_path_str, repaired)


def resolve_num_workers(
    requested_num_workers: int,
    *,
    strict: bool = False,
    try_fix_permissions: bool = True,
) -> tuple[int, WorkerBackendStatus]:
    """Resolve an effective worker count for private runtime call sites."""

    if requested_num_workers < 0:
        raise ValueError("requested_num_workers must be >= 0")

    if requested_num_workers == 0:
        shm_path = _torch_shm_manager_path()
        status = WorkerBackendStatus(True, None, _get_sharing_strategy(), str(shm_path) if shm_path is not None else None)
        return 0, status

    status = inspect_worker_backend(try_fix_permissions=try_fix_permissions)
    if status.available:
        return requested_num_workers, status

    if strict:
        raise RuntimeError(status.reason or "DataLoader multiprocessing backend unavailable")

    return 0, status
