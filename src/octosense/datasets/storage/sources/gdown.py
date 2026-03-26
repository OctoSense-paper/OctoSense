"""gdown storage source helper."""

from __future__ import annotations

import subprocess
from pathlib import Path


def download_from_gdown(
    url: str,
    target: str | Path,
    *,
    is_folder: bool = False,
) -> Path:
    destination = Path(target)
    cmd = ["gdown"]
    if is_folder:
        cmd.extend(["--folder", url, "-O", str(destination)])
    else:
        cmd.extend([url, "-O", str(destination)])
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise ValueError(
            "gdown not found. Install `gdown` to download Google Drive datasets."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() or exc.stdout.strip() or "unknown Google Drive error"
        raise ValueError(f"Google Drive download failed for {url}: {stderr}") from exc
    return destination


__all__ = ["download_from_gdown"]
