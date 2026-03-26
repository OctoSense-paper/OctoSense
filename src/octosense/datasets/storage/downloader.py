"""Dataset download helpers and CLI."""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import tarfile
import urllib.parse
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from octosense.datasets.base import resolve_dataset_root
from octosense.datasets.catalog import get_dataset_card, get_dataset_download_config
from octosense.datasets.storage.cache import resolve_cache_layout
from octosense.datasets.storage.checksums import build_checksum_record
from octosense.datasets.storage.receipts import (
    DATASET_RECEIPT_NAME,
    DATASET_RECEIPT_SCHEMA_VERSION,
    DatasetDownloadReceipt,
    load_receipt,
    receipt_path,
    write_dataset_receipt,
)
from octosense.datasets.storage.sources.gdown import download_from_gdown
from octosense.datasets.storage.sources.http import download_from_http
from octosense.datasets.storage.sources.kaggle import download_from_kaggle
from octosense.datasets.storage.sources.manual import mark_manual_source
from octosense._internal.console import ProgressBar


def _announce_download(message: str) -> None:
    print(message, flush=True)


def _download_via_http(url: str, destination: Path) -> Path:
    progress = ProgressBar(label=f"Downloading {destination.name}", total=None)

    def _update(downloaded_bytes: int, total_bytes: int | None) -> None:
        if progress.total is None and total_bytes is not None:
            progress.total = total_bytes
        progress.update(downloaded_bytes)

    try:
        return download_from_http(url, destination, progress_callback=_update)
    finally:
        progress.finish()


def _download_with_kaggle(url: str, dataset_root: Path, force: bool) -> None:
    parsed = urllib.parse.urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    try:
        idx = parts.index("datasets")
        slug = "/".join(parts[idx + 1 : idx + 3])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Unsupported Kaggle dataset URL: {url}") from exc
    _announce_download(f"kaggle: kaggle datasets download -d {slug} -p {dataset_root} --unzip")
    download_from_kaggle(slug, dataset_root, force=force, unzip=True)


def _download_with_gdown(url: str, dataset_root: Path, downloads_root: Path) -> None:
    is_folder = "/folders/" in url
    target = dataset_root if is_folder else downloads_root / "downloaded_file"
    _announce_download(
        f"gdown: {'gdown --folder ' if is_folder else 'gdown '}{url} -O {target}"
    )
    download_from_gdown(url, target, is_folder=is_folder)


def _extract_archive(path: Path, dataset_root: Path) -> bool:
    lower_name = path.name.lower()
    if lower_name.endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            archive.extractall(dataset_root)
        return True
    if lower_name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz")):
        with tarfile.open(path) as archive:
            archive.extractall(dataset_root)
        return True
    return False


def _current_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _receipt_path(dataset_root: Path) -> Path:
    return receipt_path(dataset_root)


def _load_dataset_receipt(dataset_root: Path) -> dict[str, Any] | None:
    receipt = load_receipt(dataset_root)
    if receipt is None:
        return None
    return receipt.to_dict()


def _dataset_payload_available(dataset_root: Path, *, ignored_names: set[str]) -> bool:
    for entry in dataset_root.iterdir():
        if entry.name in ignored_names or entry.name.startswith("."):
            continue
        return True
    return False


def _clear_dataset_payload(dataset_root: Path, *, ignored_names: set[str]) -> None:
    for entry in dataset_root.iterdir():
        if entry.name in ignored_names or entry.name.startswith("."):
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def _resolve_requested_part(
    dataset_id: str,
    supported_parts: tuple[str, ...],
    requested_part: str | None,
) -> str | None:
    if requested_part is None:
        return None
    if not supported_parts:
        raise ValueError(
            f"Dataset '{dataset_id}' does not support download parts; received part='{requested_part}'."
        )
    if requested_part not in supported_parts:
        supported = ", ".join(supported_parts)
        raise ValueError(
            f"Dataset '{dataset_id}' does not support download part '{requested_part}'. "
            f"Supported parts: {supported}"
        )
    return requested_part


def _requested_part_is_satisfied(existing_part: str | None, requested_part: str | None) -> bool:
    if requested_part is None:
        return True
    if existing_part is None:
        return False
    return existing_part == requested_part


def _write_dataset_receipt(
    *,
    dataset_root: Path,
    card_id: str,
    display_name: str,
    license_payload: dict[str, object],
    access_payload: dict[str, object],
    download_payload: dict[str, object],
    selected_part: str | None,
    selected_source_url: str | None,
    transport: str,
    force_download: bool,
    downloads_dir: Path,
    note: str,
    source_root: Path | None = None,
) -> Path:
    checksum_records = [
        build_checksum_record(path).to_dict()
        for path in sorted(downloads_dir.iterdir())
        if path.is_file()
    ]
    source_record = {
        "transport": transport,
        "selected_source_url": selected_source_url,
        "source_root": os.fspath(source_root) if source_root is not None else None,
    }
    if transport == "manual":
        source_record.update(mark_manual_source(card_id, note, source_root=source_root))
    receipt = DatasetDownloadReceipt(
        schema_version=DATASET_RECEIPT_SCHEMA_VERSION,
        dataset_id=card_id,
        display_name=display_name,
        resolved_root=os.fspath(dataset_root),
        license=dict(license_payload),
        access=dict(access_payload),
        download_policy=dict(download_payload),
        selected_part=selected_part,
        selected_source_url=selected_source_url,
        transport=transport,
        downloaded_at=_current_timestamp(),
        downloads_dir=os.fspath(downloads_dir),
        source_root=os.fspath(source_root) if source_root is not None else None,
        force_download=force_download,
        note=note,
        checksums=checksum_records,
        sources=[source_record],
    )
    return write_dataset_receipt(dataset_root, receipt)


def _download_one(url: str, dataset_root: Path, downloads_root: Path, force: bool) -> str:
    if "kaggle.com/datasets/" in url:
        _download_with_kaggle(url, dataset_root, force)
        return "kaggle"
    if "drive.google.com/" in url:
        _download_with_gdown(url, dataset_root, downloads_root)
        return "gdown"

    parsed = urllib.parse.urlparse(url)
    filename = Path(parsed.path).name
    if not filename:
        raise ValueError(f"Automated download is not supported for non-file URL: {url}")

    target = downloads_root / filename
    if target.exists() and not force:
        extracted = _extract_archive(target, dataset_root)
        if not extracted:
            shutil.copy2(target, dataset_root / target.name)
        return "http"
    _announce_download(f"download: {filename} <= {url}")
    _download_via_http(url, target)
    extracted = _extract_archive(target, dataset_root)
    if not extracted:
        shutil.copy2(target, dataset_root / target.name)
    return "http"


def _download_huggingface_chunked_tar(
    *,
    base_url: str,
    chunk_names: tuple[str, ...],
    dataset_root: Path,
    downloads_root: Path,
    force: bool,
) -> None:
    chunk_paths: list[Path] = []
    for chunk_name in chunk_names:
        target = downloads_root / chunk_name
        if force or not target.exists():
            _download_via_http(f"{base_url.rstrip('/')}/{chunk_name}", target)
        chunk_paths.append(target)

    tar_path = downloads_root / "downloaded_dataset.tar"
    if force or not tar_path.exists():
        with tar_path.open("wb") as merged:
            for chunk_path in chunk_paths:
                with chunk_path.open("rb") as chunk_file:
                    shutil.copyfileobj(chunk_file, merged)

    for chunk_path in chunk_paths:
        if chunk_path.exists():
            chunk_path.unlink()

    if not _extract_archive(tar_path, dataset_root):
        raise ValueError(f"Chunked archive is not a supported tar payload: {tar_path}")


def _resolve_source_url(source_payload: Mapping[str, Any]) -> str:
    env_name = str(source_payload.get("url_env") or "").strip()
    if env_name:
        override = os.environ.get(env_name)
        if override is not None:
            normalized = override.strip()
            if normalized:
                return normalized
    url = str(source_payload.get("url") or "").strip()
    if not url:
        raise ValueError(f"Download source is missing a non-empty url/url_env: {source_payload!r}")
    return url


def _download_from_source_spec(
    source_payload: Mapping[str, Any],
    *,
    dataset_root: Path,
    downloads_root: Path,
    force: bool,
) -> tuple[str, str | None, str]:
    kind = str(source_payload.get("kind") or "").strip()
    note = str(source_payload.get("note") or "").strip()
    if kind in {"http", "http_archive"}:
        url = _resolve_source_url(source_payload)
        transport = _download_one(url, dataset_root, downloads_root, force)
        return transport, url, note
    if kind == "kaggle":
        url = _resolve_source_url(source_payload)
        transport = _download_one(url, dataset_root, downloads_root, force)
        return transport, url, note
    if kind == "gdown":
        url = _resolve_source_url(source_payload)
        transport = _download_one(url, dataset_root, downloads_root, force)
        return transport, url, note
    if kind == "huggingface_chunked_tar":
        base_url = str(source_payload.get("base_url") or "").strip()
        chunk_names = tuple(str(item) for item in source_payload.get("chunk_names", ()))
        if not base_url or not chunk_names:
            raise ValueError(
                "huggingface_chunked_tar source requires non-empty base_url and chunk_names."
            )
        _download_huggingface_chunked_tar(
            base_url=base_url,
            chunk_names=chunk_names,
            dataset_root=dataset_root,
            downloads_root=downloads_root,
            force=force,
        )
        source_url = str(source_payload.get("script_url") or source_payload.get("source_url") or "")
        return "huggingface_chunked_tar", source_url or None, note
    raise ValueError(f"Unsupported download source kind: {kind!r}")


def _normalize_handler_callable(handler_ref: str) -> tuple[Any, Callable[..., tuple[str, str | None, str, Path | None]]]:
    module_name, separator, function_name = handler_ref.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError(
            "Dataset download handler must use the 'package.module:function_name' format."
        )
    module = importlib.import_module(module_name)
    handler = getattr(module, function_name, None)
    if not callable(handler):
        raise ValueError(f"Dataset download handler '{handler_ref}' is not callable.")
    return module, handler


def _default_infer_existing_part(receipt_payload: dict[str, Any] | None) -> str | None:
    if receipt_payload is None:
        return None
    selected_part = receipt_payload.get("selected_part")
    if isinstance(selected_part, str) and selected_part:
        return selected_part
    return None


def _load_download_handler(
    download_config: Any,
) -> tuple[Any | None, Callable[..., tuple[str, str | None, str, Path | None]] | None]:
    handler_ref = str(getattr(download_config, "handler", None) or "").strip()
    if not handler_ref:
        return None, None
    return _normalize_handler_callable(handler_ref)


def _default_selected_source_url(download_config: Any) -> str | None:
    raw_sources = tuple(getattr(download_config, "sources", ()))
    if not raw_sources:
        return None
    first_source = raw_sources[0]
    candidate = str(first_source.get("script_url") or first_source.get("source_url") or "").strip()
    if candidate:
        return candidate
    try:
        return _resolve_source_url(first_source)
    except ValueError:
        return None


def _format_download_blocker(card: Any) -> str:
    reasons: list[str] = []
    if card.access.kind == "application_required":
        reasons.append(f"Dataset '{card.dataset_id}' requires an application-approved download flow.")
    elif card.access.kind == "manual_only":
        reasons.append(f"Dataset '{card.dataset_id}' must be prepared manually.")
    elif card.access.kind != "public_download":
        reasons.append(
            f"Dataset '{card.dataset_id}' declares unsupported access kind {card.access.kind!r}."
        )

    if not card.download.allow_automated_download:
        reasons.append(
            "OctoSense download policy for this dataset does not allow automated downloads."
        )

    if card.access.application_url:
        reasons.append(f"Apply via: {card.access.application_url}")
    elif card.access.public_urls:
        reasons.append(f"Public entry points: {', '.join(card.access.public_urls)}")

    for note in (card.access.note, card.download.note, card.license.note):
        normalized = str(note).strip()
        if normalized:
            reasons.append(normalized)

    return " ".join(dict.fromkeys(reasons))


def _assert_download_supported(card: Any, download_config: Any) -> None:
    raw_sources = tuple(getattr(download_config, "sources", ()))
    has_configured_sources = bool(raw_sources)
    has_handler = bool(str(getattr(download_config, "handler", None) or "").strip())
    if not card.download.allow_automated_download:
        raise ValueError(_format_download_blocker(card))
    if card.access.kind == "public_download" and (has_configured_sources or has_handler):
        return
    raise ValueError(_format_download_blocker(card))


def download_dataset(
    dataset_id: str,
    *,
    root: str | Path | None = None,
    force: bool = False,
    part: str | None = None,
    source_root: str | Path | None = None,
) -> Path:
    """Download one dataset into its configured OctoSense storage root."""

    card = get_dataset_card(dataset_id)
    download_config = get_dataset_download_config(dataset_id)
    raw_download_config = download_config.to_dict()
    _assert_download_supported(card, download_config)

    dataset_root = resolve_dataset_root(dataset_id, override=root)
    layout = resolve_cache_layout(
        dataset_id,
        dataset_root_override=dataset_root,
        downloads_dirname=card.storage.downloads_dirname,
        cache_dirname=card.storage.cache_dirname,
        receipt_name=DATASET_RECEIPT_NAME,
    ).ensure()
    dataset_root = layout.dataset_root
    downloads_root = layout.downloads_root

    handler_module, download_handler = _load_download_handler(download_config)

    ignored_names = {
        card.storage.downloads_dirname,
        card.storage.cache_dirname,
        DATASET_RECEIPT_NAME,
    }
    selected_part = _resolve_requested_part(
        dataset_id,
        card.download.supported_parts,
        part,
    )
    selected_source_url = _default_selected_source_url(download_config)
    display_name = card.storage.canonical_display_name or card.display_name
    card_id = card.dataset_id
    note = card.download.note or card.access.note or card.license.note
    resolved_source_root: Path | None = None
    existing_receipt = _load_dataset_receipt(dataset_root)
    infer_existing_part = getattr(handler_module, "infer_existing_part", None)
    if callable(infer_existing_part):
        existing_part = infer_existing_part(
            dataset_root,
            existing_receipt,
            download_config=raw_download_config,
        )
    else:
        existing_part = _default_infer_existing_part(existing_receipt)
    if _dataset_payload_available(dataset_root, ignored_names=ignored_names) and not force:
        if _requested_part_is_satisfied(existing_part, selected_part):
            if existing_receipt is None or existing_receipt.get("selected_part") != existing_part:
                normalize_reuse_selected_source_url = getattr(
                    handler_module,
                    "normalize_reuse_selected_source_url",
                    None,
                )
                if callable(normalize_reuse_selected_source_url):
                    reuse_selected_source_url = normalize_reuse_selected_source_url(
                        existing_part,
                        selected_source_url,
                        download_config=raw_download_config,
                    )
                else:
                    reuse_selected_source_url = selected_source_url
                _write_dataset_receipt(
                    dataset_root=dataset_root,
                    card_id=card_id,
                    display_name=display_name,
                    license_payload=card.license.to_dict(),
                    access_payload=card.access.to_dict(),
                    download_payload=card.download.to_dict(),
                    selected_part=existing_part or selected_part,
                    selected_source_url=reuse_selected_source_url,
                    transport="reuse_existing",
                    force_download=False,
                    downloads_dir=downloads_root,
                    note=note,
                    source_root=None,
                )
            return dataset_root
        prepare_existing_payload = getattr(
            handler_module,
            "prepare_existing_payload_for_requested_part",
            None,
        )
        if callable(prepare_existing_payload):
            prepare_existing_payload(
                dataset_root,
                existing_part=existing_part,
                selected_part=selected_part,
                ignored_names=ignored_names,
                download_config=raw_download_config,
            )

    last_error: ValueError | None = None
    transport = "unknown"
    if download_handler is not None:
        transport, selected_source_url, note, resolved_source_root = download_handler(
            dataset_root,
            downloads_root,
            card=card,
            download_config=raw_download_config,
            part=selected_part,
            force=force,
            ignored_names=ignored_names,
            source_root=source_root,
            announce_download=_announce_download,
            download_source=lambda source_payload, *, dataset_root, downloads_root, force: _download_from_source_spec(
                source_payload,
                dataset_root=dataset_root,
                downloads_root=downloads_root,
                force=force,
            ),
            reset_dataset_payload=lambda: _clear_dataset_payload(
                dataset_root,
                ignored_names=ignored_names,
            ),
        )
        last_error = None
    else:
        raw_sources = download_config.sources
        if not raw_sources:
            raise ValueError(
                f"Dataset '{dataset_id}' allows automated download but card.yaml download.sources is empty."
            )
        for source_payload in raw_sources:
            try:
                transport, selected_source_url, source_note = _download_from_source_spec(
                    source_payload,
                    dataset_root=dataset_root,
                    downloads_root=downloads_root,
                    force=force,
                )
                if source_note:
                    note = source_note
                last_error = None
                break
            except ValueError as exc:
                last_error = exc

    if last_error is not None:
        raise last_error
    _write_dataset_receipt(
        dataset_root=dataset_root,
        card_id=card_id,
        display_name=display_name,
        license_payload=card.license.to_dict(),
        access_payload=card.access.to_dict(),
        download_payload=card.download.to_dict(),
        selected_part=selected_part,
        selected_source_url=selected_source_url,
        transport=transport,
        force_download=force,
        downloads_dir=downloads_root,
        note=note,
        source_root=resolved_source_root,
    )
    return dataset_root


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a OctoSense dataset into its storage root"
    )
    parser.add_argument(
        "dataset_id",
        help="Canonical dataset id declared by octosense.datasets builtin sidecars",
    )
    parser.add_argument("--root", default=None, help="Override target dataset root")
    parser.add_argument(
        "--part",
        default=None,
        help="Optional dataset-specific download part such as 'full' or 'sample'",
    )
    parser.add_argument(
        "--source-root",
        default=None,
        help="Optional local source root used by dataset-specific subset builders",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    path = download_dataset(
        args.dataset_id,
        root=args.root,
        force=args.force,
        part=args.part,
        source_root=args.source_root,
    )
    print(os.fspath(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
