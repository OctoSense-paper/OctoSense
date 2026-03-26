"""Widar3-specific download hooks for the generic datasets downloader."""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlparse


DownloadSourceFn = Callable[..., tuple[str, str | None, str]]
_METADATA_TEMPLATE_PATH = Path(__file__).with_name("metas_template.csv")
_EXPECTED_SAMPLE_PART = "sample"
_EXPECTED_FULL_PART = "full"


def _string_list(values: Any, *, field_name: str) -> tuple[str, ...]:
    if values is None or values == "":
        return ()
    if not isinstance(values, list):
        raise ValueError(f"Widar3 card field '{field_name}' must be a list.")
    return tuple(str(item).strip() for item in values if str(item).strip())


def _resolve_part_config(download_config: Mapping[str, Any], part: str | None) -> dict[str, Any]:
    parts = download_config.get("parts", {})
    if not isinstance(parts, Mapping):
        raise ValueError("Widar3 card download.parts must be a mapping.")
    if part is None:
        raise ValueError(
            "Widar3 download requires an explicit part. Use part='sample' for the "
            "20181211 subset or part='full' for the complete FTP mirror."
        )
    payload = parts.get(part)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Widar3 card download.parts.{part} must be a mapping.")
    return {str(key): value for key, value in payload.items()}


def _resolve_source_url(source_payload: Mapping[str, Any]) -> str:
    url = str(source_payload.get("url") or "").strip()
    if not url:
        raise ValueError(f"Widar3 download source is missing url: {source_payload!r}")
    return url


def _ftp_credentials(download_config: Mapping[str, Any]) -> dict[str, Any]:
    credentials = download_config.get("ftp_credentials")
    if credentials is None:
        return {}
    if not isinstance(credentials, Mapping):
        raise ValueError("Widar3 card download.ftp_credentials must be a mapping when provided.")
    return {str(key): value for key, value in credentials.items()}


def _resolve_credential(
    *payloads: Mapping[str, Any],
    env_field: str,
    inline_field: str,
    label: str,
) -> str:
    for payload in payloads:
        env_name = str(payload.get(env_field) or "").strip()
        if env_name:
            env_value = os.environ.get(env_name)
            if env_value is not None and env_value.strip():
                return env_value.strip()
    for payload in payloads:
        inline_value = str(payload.get(inline_field) or "").strip()
        if inline_value:
            return inline_value
    raise ValueError(f"Widar3 FTP source is missing {label}.")


def _download_ftp_archive(
    source_payload: Mapping[str, Any],
    *,
    ftp_credentials: Mapping[str, Any],
    downloads_root: Path,
    force: bool,
    announce_download: Callable[[str], None],
) -> Path:
    url = _resolve_source_url(source_payload)
    filename = Path(urlparse(url).path).name
    if not filename:
        raise ValueError(f"Widar3 FTP source URL does not point to a file: {url}")
    target = downloads_root / filename
    if target.exists() and not force:
        return target
    if force and target.exists():
        target.unlink()

    username = _resolve_credential(
        source_payload,
        ftp_credentials,
        env_field="username_env",
        inline_field="username",
        label="username",
    )
    password = _resolve_credential(
        source_payload,
        ftp_credentials,
        env_field="password_env",
        inline_field="password",
        label="password",
    )
    curl_bin = shutil.which("curl")
    if curl_bin is None:
        raise ValueError("Widar3 FTP download requires curl to be installed.")
    command = [
        curl_bin,
        "--fail",
        "--location",
        "--disable-epsv",
        "--connect-timeout",
        "30",
        "--retry",
        "3",
        "--retry-delay",
        "2",
        "--continue-at",
        "-",
        "--user",
        f"{username}:{password}",
        "--output",
        os.fspath(target),
        url,
    ]
    announce_download(f"widar3 ftp: {filename} <= {url}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"Widar3 FTP download failed for {url}") from exc
    return target


def _source_date(source_payload: Mapping[str, Any]) -> str:
    explicit = str(source_payload.get("date") or "").strip()
    if explicit:
        return explicit
    stem = Path(urlparse(_resolve_source_url(source_payload)).path).stem
    if stem.startswith("CSI_"):
        return stem.removeprefix("CSI_")
    raise ValueError(f"Cannot infer Widar3 archive date from source payload: {source_payload!r}")


def _find_date_root(dataset_root: Path, *, date: str) -> Path:
    canonical = dataset_root / "CSI" / date
    if canonical.is_dir():
        return canonical
    candidate = dataset_root / date
    if candidate.is_dir():
        return candidate

    matches = [
        path
        for path in dataset_root.rglob(date)
        if path.is_dir() and path.name == date and path != canonical
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"Widar3 archive extraction did not create a date directory for {date}.")
    raise ValueError(f"Widar3 archive extraction created multiple candidate date roots for {date}.")


def _normalize_date_payload(dataset_root: Path, *, date: str) -> None:
    canonical_root = dataset_root / "CSI" / date
    resolved_root = _find_date_root(dataset_root, date=date)
    canonical_root.parent.mkdir(parents=True, exist_ok=True)
    if resolved_root != canonical_root:
        if canonical_root.exists():
            shutil.rmtree(canonical_root)
        shutil.move(str(resolved_root), str(canonical_root))
        parent = resolved_root.parent
        while parent != dataset_root and parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
            parent = parent.parent
    if not any(canonical_root.rglob("*.dat")):
        raise ValueError(f"Widar3 normalized date root {canonical_root} does not contain any .dat files.")


def _extract_archive(archive_path: Path, *, dataset_root: Path, date: str) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(dataset_root)
    _normalize_date_payload(dataset_root, date=date)


def _load_template_rows() -> list[dict[str, str]]:
    with _METADATA_TEMPLATE_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        return [
            {str(key).strip(): str(value).strip() for key, value in row.items()}
            for row in reader
        ]


def _write_metadata_csv(dataset_root: Path, *, dates: tuple[str, ...]) -> None:
    selected_dates = set(dates)
    rows = [row for row in _load_template_rows() if row.get("file") in selected_dates]
    if not rows:
        raise ValueError(f"No Widar3 metadata rows matched requested dates: {sorted(selected_dates)}")
    fieldnames = ["file", "room", "gesture_list", "user", "sample_num"]
    with (dataset_root / "metas.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="|")
        writer.writeheader()
        writer.writerows(rows)


def _dataset_dates_ready(dataset_root: Path, *, dates: tuple[str, ...]) -> bool:
    if not dates:
        return False
    return all((dataset_root / "CSI" / date).is_dir() for date in dates) and (dataset_root / "metas.csv").is_file()


def infer_existing_part(
    dataset_root: Path,
    receipt_payload: dict[str, Any] | None,
    *,
    download_config: Mapping[str, Any],
) -> str | None:
    if receipt_payload is not None:
        selected_part = receipt_payload.get("selected_part")
        if isinstance(selected_part, str) and selected_part:
            try:
                part_config = _resolve_part_config(download_config, selected_part)
            except ValueError:
                part_config = None
            if part_config is not None:
                payload = part_config.get("payload", {})
                if not isinstance(payload, Mapping):
                    raise ValueError(
                        f"Widar3 card download.parts.{selected_part}.payload must be a mapping."
                    )
                dates = _string_list(
                    payload.get("dates"),
                    field_name=f"download.parts.{selected_part}.payload.dates",
                )
                if _dataset_dates_ready(dataset_root, dates=dates):
                    return selected_part
    for part_name in (_EXPECTED_FULL_PART, _EXPECTED_SAMPLE_PART):
        try:
            part_config = _resolve_part_config(download_config, part_name)
        except ValueError:
            continue
        payload = part_config.get("payload", {})
        if not isinstance(payload, Mapping):
            raise ValueError(f"Widar3 card download.parts.{part_name}.payload must be a mapping.")
        dates = _string_list(
            payload.get("dates"),
            field_name=f"download.parts.{part_name}.payload.dates",
        )
        if _dataset_dates_ready(dataset_root, dates=dates):
            return part_name
    return None


def prepare_existing_payload_for_requested_part(
    dataset_root: Path,
    *,
    existing_part: str | None,
    selected_part: str | None,
    ignored_names: set[str],
    download_config: Mapping[str, Any],
) -> None:
    del download_config
    if existing_part == selected_part or selected_part is None:
        return
    if existing_part == _EXPECTED_FULL_PART and selected_part == _EXPECTED_SAMPLE_PART:
        raise ValueError(
            "Target root already contains Widar3 full. Use a different root for the smaller "
            "sample subset instead of replacing the full dataset in place."
        )
    if existing_part == _EXPECTED_SAMPLE_PART and selected_part == _EXPECTED_FULL_PART:
        for entry in dataset_root.iterdir():
            if entry.name in ignored_names or entry.name.startswith("."):
                continue
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()


def download_widar3_dataset(
    dataset_root: Path,
    downloads_root: Path,
    *,
    card: Any,
    download_config: Mapping[str, Any],
    part: str | None,
    force: bool,
    ignored_names: set[str],
    source_root: str | Path | None,
    announce_download: Callable[[str], None],
    download_source: DownloadSourceFn,
    reset_dataset_payload: Callable[[], None],
) -> tuple[str, str | None, str, Path | None]:
    del card, ignored_names, source_root, download_source
    part_config = _resolve_part_config(download_config, part)
    payload = part_config.get("payload", {})
    if not isinstance(payload, Mapping):
        raise ValueError(f"Widar3 card download.parts.{part}.payload must be a mapping.")
    ftp_credentials = _ftp_credentials(download_config)
    part_dates = _string_list(payload.get("dates"), field_name=f"download.parts.{part}.payload.dates")
    part_variant = str(payload.get("variant") or "").strip() or str(part)
    raw_sources = part_config.get("sources", ())
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError(f"Widar3 card download.parts.{part}.sources must be a non-empty list.")
    sources = [{str(key): value for key, value in source.items()} for source in raw_sources if isinstance(source, Mapping)]
    if len(sources) != len(raw_sources):
        raise ValueError(f"Widar3 card download.parts.{part}.sources entries must be mappings.")

    if force:
        reset_dataset_payload()
    elif _dataset_dates_ready(dataset_root, dates=part_dates):
        return (
            "reuse_existing",
            _resolve_source_url(sources[0]),
            f"Widar3 {part_variant} payload already available.",
            None,
        )

    for source in sources:
        archive_path = _download_ftp_archive(
            source,
            ftp_credentials=ftp_credentials,
            downloads_root=downloads_root,
            force=force,
            announce_download=announce_download,
        )
        _extract_archive(
            archive_path,
            dataset_root=dataset_root,
            date=_source_date(source),
        )
    _write_metadata_csv(dataset_root, dates=part_dates)
    note = f"Widar3 {part_variant} prepared from the FTP archive set ({len(sources)} archive(s))."
    return "ftp_archive", _resolve_source_url(sources[0]), note, None


__all__ = [
    "download_widar3_dataset",
    "infer_existing_part",
    "prepare_existing_payload_for_requested_part",
]
