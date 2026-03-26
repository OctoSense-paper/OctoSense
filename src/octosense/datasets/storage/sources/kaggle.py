"""Kaggle storage source helper."""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


def download_from_kaggle(
    dataset: str,
    target: str | Path,
    *,
    force: bool = False,
    unzip: bool = True,
) -> Path:
    destination = Path(target)
    cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(destination)]
    if unzip:
        cmd.append("--unzip")
    if force:
        cmd.append("--force")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise ValueError(
            "Kaggle CLI not found. Install `kaggle`, configure credentials, then retry."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() or exc.stdout.strip() or "unknown Kaggle download error"
        raise ValueError(
            f"Kaggle download failed for {dataset} via {shlex.join(cmd)}: {stderr}"
        ) from exc
    return destination


__all__ = ["download_from_kaggle"]
