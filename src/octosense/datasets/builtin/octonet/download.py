"""OctoNet-specific download hooks for the generic datasets downloader."""

from __future__ import annotations

import csv
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .ingest import parse_optional_int

DownloadSourceFn = Callable[..., tuple[str, str | None, str]]
OCTONET_WIFI_SAMPLE_PROFILE = "course_v1"
OCTONET_WIFI_SAMPLE_ACTIVITIES = ("sit", "walk", "sleep", "falldown", "jump")
OCTONET_WIFI_SAMPLE_SUBJECT_IDS = (1, 2, 3, 4, 5)
OCTONET_WIFI_SAMPLE_EXP_TO_SUBJECT = {
    1: 1,
    2: 2,
    101: 3,
    102: 4,
    201: 5,
}


def _is_non_sample_resource_path_field(field_name: str) -> bool:
    normalized = field_name.strip()
    return normalized.endswith("_data_path") and normalized != "node_1_wifi_data_path"


def _ordered_fieldnames(
    existing_fieldnames: list[str] | None,
    *,
    extras: tuple[str, ...],
) -> list[str]:
    fieldnames = list(existing_fieldnames or [])
    for name in extras:
        if name not in fieldnames:
            fieldnames.append(name)
    return fieldnames


@dataclass(frozen=True)
class OctonetWiFiCourseSampleSpec:
    profile_id: str = OCTONET_WIFI_SAMPLE_PROFILE
    activities: tuple[str, ...] = OCTONET_WIFI_SAMPLE_ACTIVITIES
    subject_ids: tuple[int, ...] = OCTONET_WIFI_SAMPLE_SUBJECT_IDS
    scene_ids: tuple[int, ...] = (1, 2, 3)
    exp_id_to_subject: tuple[tuple[int, int], ...] = tuple(
        sorted(OCTONET_WIFI_SAMPLE_EXP_TO_SUBJECT.items())
    )
    max_clips_per_exp_activity: int = 1

    @property
    def exp_to_subject_map(self) -> dict[int, int]:
        return dict(self.exp_id_to_subject)

    @property
    def allowed_exp_ids(self) -> tuple[int, ...]:
        return tuple(exp_id for exp_id, _ in self.exp_id_to_subject)


@dataclass(frozen=True)
class OctonetWiFiSamplePreparationSummary:
    profile_id: str
    sample_count: int
    subject_ids: tuple[int, ...]
    scene_ids: tuple[int, ...]
    activities: tuple[str, ...]
    exp_ids: tuple[int, ...]


def prepare_octonet_wifi_sample_download(
    source_root: Path,
    dataset_root: Path,
    *,
    spec: OctonetWiFiCourseSampleSpec | None = None,
) -> OctonetWiFiSamplePreparationSummary:
    sample_spec = spec or OctonetWiFiCourseSampleSpec()
    source_root = source_root.expanduser().resolve()
    dataset_root = dataset_root.expanduser().resolve()
    if source_root == dataset_root:
        raise ValueError(
            "OctoNet sample must be built into a different target root than the full source root."
        )

    metadata_path = source_root / "cut_manual.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"OctoNet source metadata not found at {metadata_path}")

    selected_rows: list[dict[str, str]] = []
    selected_subject_ids: set[int] = set()
    selected_scene_ids: set[int] = set()
    selected_activity_names: set[str] = set()
    selected_exp_ids: set[int] = set()
    selected_files: dict[str, Path] = {}
    counts_by_exp_activity: dict[tuple[int, str], int] = defaultdict(int)
    exp_to_subject = sample_spec.exp_to_subject_map

    with metadata_path.open(newline="", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle)
        source_fieldnames = list(reader.fieldnames or [])
        ordered_rows = sorted(
            list(reader),
            key=lambda row: (
                parse_optional_int(row.get("exp_id")) or 0,
                (row.get("activity") or "").strip(),
                (row.get("started_at") or "").strip(),
            ),
        )
        fieldnames = _ordered_fieldnames(
            source_fieldnames,
            extras=(
                "subject_id",
                "exp_id",
                "scene_id",
                "scene_name",
                "sample_kind",
                "sample_profile",
            ),
        )
        for row in ordered_rows:
            wifi_path = (row.get("node_1_wifi_data_path") or "").strip()
            if not wifi_path:
                continue
            exp_id = parse_optional_int(row.get("exp_id"))
            if exp_id is None or exp_id not in exp_to_subject:
                continue
            subject_id = parse_optional_int(row.get("subject_id"))
            if subject_id is None:
                continue
            if subject_id not in sample_spec.subject_ids:
                continue
            scene_id = parse_optional_int(row.get("scene_id"))
            if scene_id is None or scene_id not in sample_spec.scene_ids:
                continue
            activity = (row.get("activity") or "").strip()
            if activity not in sample_spec.activities:
                continue
            key = (exp_id, activity)
            if counts_by_exp_activity[key] >= sample_spec.max_clips_per_exp_activity:
                continue
            source_file = source_root / wifi_path
            if not source_file.is_file():
                continue

            output_row = dict(row)
            output_row["user_id"] = str(subject_id)
            output_row["subject_id"] = str(subject_id)
            output_row["exp_id"] = str(exp_id)
            output_row["scene_id"] = str(scene_id)
            output_row["scene_name"] = f"scene_{scene_id}"
            output_row["activity"] = activity
            output_row["sample_kind"] = "sample"
            output_row["sample_profile"] = sample_spec.profile_id
            for name in tuple(output_row):
                if _is_non_sample_resource_path_field(name):
                    output_row[name] = ""

            selected_rows.append({name: output_row.get(name, "") for name in fieldnames})
            selected_subject_ids.add(subject_id)
            selected_scene_ids.add(scene_id)
            selected_activity_names.add(activity)
            selected_exp_ids.add(exp_id)
            selected_files[wifi_path] = source_file
            counts_by_exp_activity[key] += 1

    if not selected_rows:
        raise ValueError(
            "No OctoNet WiFi rows matched the fixed course sample spec in the provided source root."
        )

    missing_subjects = sorted(set(sample_spec.subject_ids) - selected_subject_ids)
    missing_scenes = sorted(set(sample_spec.scene_ids) - selected_scene_ids)
    missing_activities = sorted(set(sample_spec.activities) - selected_activity_names)
    if missing_subjects or missing_scenes or missing_activities:
        problems: list[str] = []
        if missing_subjects:
            problems.append(f"missing subjects {missing_subjects}")
        if missing_scenes:
            problems.append(f"missing scenes {missing_scenes}")
        if missing_activities:
            problems.append(f"missing activities {missing_activities}")
        raise ValueError(
            "OctoNet source root does not satisfy the fixed course sample spec: "
            + ", ".join(problems)
        )

    for wifi_path, source_file in selected_files.items():
        target_file = dataset_root / wifi_path
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target_file)

    fieldnames = list(selected_rows[0])
    dataset_root.mkdir(parents=True, exist_ok=True)
    with (dataset_root / "cut_manual.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)

    return OctonetWiFiSamplePreparationSummary(
        profile_id=sample_spec.profile_id,
        sample_count=len(selected_rows),
        subject_ids=tuple(sorted(selected_subject_ids)),
        scene_ids=tuple(sorted(selected_scene_ids)),
        activities=tuple(sorted(selected_activity_names)),
        exp_ids=tuple(sorted(selected_exp_ids)),
    )


def _string_tuple(values: Any, *, field_name: str) -> tuple[str, ...]:
    if values is None or values == "":
        return ()
    if not isinstance(values, list):
        raise ValueError(f"OctoNet card field '{field_name}' must be a list.")
    return tuple(str(item).strip() for item in values if str(item).strip())


def _resolve_part_config(download_config: Mapping[str, Any], part: str | None) -> dict[str, Any]:
    parts = download_config.get("parts", {})
    if not isinstance(parts, Mapping):
        raise ValueError("OctoNet card download.parts must be a mapping.")
    if part is None:
        raise ValueError("OctoNet download requires an explicit part configured in card.yaml.")
    payload = parts.get(part)
    if not isinstance(payload, Mapping):
        raise ValueError(f"OctoNet card download.parts.{part} must be a mapping.")
    return {str(key): value for key, value in payload.items()}


def _iter_detectable_parts(download_config: Mapping[str, Any]) -> list[tuple[int, str, dict[str, Any]]]:
    parts = download_config.get("parts", {})
    if not isinstance(parts, Mapping):
        raise ValueError("OctoNet card download.parts must be a mapping.")

    detectable: list[tuple[int, str, dict[str, Any]]] = []
    for part_name, raw_part_config in parts.items():
        if not isinstance(raw_part_config, Mapping):
            raise ValueError(f"OctoNet card download.parts.{part_name} must be a mapping.")
        payload = raw_part_config.get("payload", {})
        if payload is None or payload == "":
            payload = {}
        if not isinstance(payload, Mapping):
            raise ValueError(f"OctoNet card download.parts.{part_name}.payload must be a mapping.")
        marker_paths = _string_tuple(
            payload.get("marker_paths"),
            field_name=f"download.parts.{part_name}.payload.marker_paths",
        )
        if not marker_paths:
            continue
        detect_priority = int(payload.get("detect_priority") or 0)
        detectable.append(
            (detect_priority, str(part_name), {str(key): value for key, value in payload.items()})
        )
    detectable.sort(key=lambda item: (-item[0], item[1]))
    return detectable


def _payload_marker_paths(payload_config: Mapping[str, Any], *, field_name: str) -> tuple[str, ...]:
    marker_paths = _string_tuple(payload_config.get("marker_paths"), field_name=field_name)
    if not marker_paths:
        raise ValueError(f"OctoNet card field '{field_name}' must contain at least one path.")
    return marker_paths


def _required_metadata_columns(
    payload_config: Mapping[str, Any],
    *,
    field_name: str,
) -> tuple[str, ...]:
    return _string_tuple(
        payload_config.get("required_metadata_columns"),
        field_name=f"{field_name}.required_metadata_columns",
    )


def _metadata_columns_ready(
    dataset_root: Path,
    payload_config: Mapping[str, Any],
    *,
    field_name: str,
) -> bool:
    required_columns = _required_metadata_columns(payload_config, field_name=field_name)
    if not required_columns:
        return True
    metadata_path = dataset_root / "cut_manual.csv"
    if not metadata_path.is_file():
        return False
    try:
        with metadata_path.open(newline="", encoding="utf-8", errors="ignore") as handle:
            reader = csv.DictReader(handle)
            fieldnames = {str(name).strip() for name in (reader.fieldnames or []) if str(name).strip()}
    except OSError:
        return False
    return all(column in fieldnames for column in required_columns)


def _payload_ready(dataset_root: Path, payload_config: Mapping[str, Any], *, field_name: str) -> bool:
    return all(
        (
            all(
                (dataset_root / marker_path).exists()
                for marker_path in _payload_marker_paths(payload_config, field_name=field_name)
            ),
            _metadata_columns_ready(dataset_root, payload_config, field_name=field_name),
        )
    )


def _describe_marker_paths(payload_config: Mapping[str, Any], *, field_name: str) -> str:
    parts = [", ".join(_payload_marker_paths(payload_config, field_name=field_name))]
    required_columns = _required_metadata_columns(payload_config, field_name=field_name)
    if required_columns:
        parts.append(
            "cut_manual.csv columns: " + ", ".join(required_columns)
        )
    return "; ".join(parts)


def _normalize_payload_layout(
    dataset_root: Path,
    payload_config: Mapping[str, Any],
    *,
    ignored_names: set[str],
    field_name: str,
) -> None:
    if _payload_ready(dataset_root, payload_config, field_name=field_name):
        return

    normalize = payload_config.get("normalize", {})
    if normalize is None or normalize == "":
        normalize = {}
    if not isinstance(normalize, Mapping):
        raise ValueError(f"OctoNet card field '{field_name}.normalize' must be a mapping.")
    mode = str(normalize.get("mode") or "").strip()
    if mode == "single_nested_root":
        candidates = [
            entry
            for entry in dataset_root.iterdir()
            if entry.name not in ignored_names and not entry.name.startswith(".")
        ]
        if len(candidates) == 1 and candidates[0].is_dir():
            nested_root = candidates[0]
            if _payload_ready(nested_root, payload_config, field_name=field_name):
                for child in list(nested_root.iterdir()):
                    shutil.move(str(child), str(dataset_root / child.name))
                nested_root.rmdir()
                return
    elif mode:
        raise ValueError(f"Unsupported OctoNet payload normalize mode: {mode!r}")

    raise ValueError(
        "Downloaded OctoNet payload did not match the declared card layout. Required paths at "
        f"dataset root: {_describe_marker_paths(payload_config, field_name=field_name)}."
    )


def infer_existing_part(
    dataset_root: Path,
    receipt_payload: dict[str, Any] | None,
    *,
    download_config: Mapping[str, Any],
) -> str | None:
    receipt_selected_part: str | None = None
    if receipt_payload is not None:
        selected_part = receipt_payload.get("selected_part")
        if isinstance(selected_part, str) and selected_part:
            receipt_selected_part = selected_part
    if receipt_selected_part in {"sample", "full"}:
        try:
            receipt_part_config = _resolve_part_config(download_config, receipt_selected_part)
        except ValueError:
            receipt_part_config = None
        if receipt_part_config is not None:
            payload_config = receipt_part_config.get("payload", {})
            if payload_config is None or payload_config == "":
                payload_config = {}
            if not isinstance(payload_config, Mapping):
                raise ValueError(
                    f"OctoNet card download.parts.{receipt_selected_part}.payload must be a mapping."
                )
            payload_config = {str(key): value for key, value in payload_config.items()}
            if _payload_ready(
                dataset_root,
                payload_config,
                field_name=f"download.parts.{receipt_selected_part}.payload",
            ):
                return receipt_selected_part

    for _priority, part_name, payload_config in _iter_detectable_parts(download_config):
        if _payload_ready(
            dataset_root,
            payload_config,
            field_name=f"download.parts.{part_name}.payload",
        ):
            return part_name
    return None


def normalize_reuse_selected_source_url(
    existing_part: str | None,
    selected_source_url: str | None,
    *,
    download_config: Mapping[str, Any],
) -> str | None:
    if not existing_part:
        return selected_source_url

    part_config = _resolve_part_config(download_config, existing_part)
    payload_config = part_config.get("payload", {})
    if payload_config is None or payload_config == "":
        payload_config = {}
    if not isinstance(payload_config, Mapping):
        raise ValueError(
            f"OctoNet card download.parts.{existing_part}.payload must be a mapping."
        )
    if payload_config.get("reuse_selected_source_url") is False:
        return None
    return selected_source_url


def prepare_existing_payload_for_requested_part(
    dataset_root: Path,
    *,
    existing_part: str | None,
    selected_part: str | None,
    ignored_names: set[str],
    download_config: Mapping[str, Any],
) -> None:
    del download_config
    if existing_part == "full" and selected_part == "sample":
        raise ValueError(
            "Target root already contains OctoNet full. Use a different root for the smaller "
            "sample subset instead of replacing the full dataset in place."
        )
    if selected_part == "full" and existing_part == "sample":
        for entry in dataset_root.iterdir():
            if entry.name in ignored_names or entry.name.startswith("."):
                continue
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()


def _resolve_sample_source_root(
    dataset_root: Path,
    *,
    source_root: str | Path | None,
) -> Path:
    if source_root is None:
        raise ValueError(
            "OctoNet sample requires an explicit full-dataset source root after archive download "
            "failure. Pass source_root=... pointing to the canonical full OctoNet root."
        )

    resolved = Path(source_root).expanduser().resolve()
    if resolved == dataset_root:
        raise ValueError(
            "OctoNet sample must be built into a different target root than the full source root."
        )
    if not (resolved / "cut_manual.csv").is_file():
        raise ValueError(f"OctoNet source metadata not found at {resolved / 'cut_manual.csv'}")
    return resolved


def download_octonet_dataset(
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
    del card  # card metadata is already projected into download_config for this handler.
    part_config = _resolve_part_config(download_config, part)
    payload_config = part_config.get("payload", {})
    if payload_config is None or payload_config == "":
        payload_config = {}
    if not isinstance(payload_config, Mapping):
        raise ValueError(f"OctoNet card download.parts.{part}.payload must be a mapping.")
    payload_config = {str(key): value for key, value in payload_config.items()}

    sources = part_config.get("sources", ())
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"OctoNet download.parts.{part}.sources must be a non-empty list.")

    reset_dataset_payload()
    last_source_error: ValueError | None = None
    for source_payload in sources:
        if not isinstance(source_payload, Mapping):
            raise ValueError(f"OctoNet download.parts.{part}.sources entries must be mappings.")
        try:
            transport, selected_source_url, source_note = download_source(
                source_payload,
                dataset_root=dataset_root,
                downloads_root=downloads_root,
                force=force,
            )
            _normalize_payload_layout(
                dataset_root,
                payload_config,
                ignored_names=ignored_names,
                field_name=f"download.parts.{part}.payload.marker_paths",
            )
            return transport, selected_source_url, source_note, None
        except ValueError as exc:
            last_source_error = exc
            reset_dataset_payload()

    if part == "sample":
        resolved_source_root = _resolve_sample_source_root(
            dataset_root,
            source_root=source_root,
        )
        announce_download(f"prepare: octonet sample <= {resolved_source_root}")
        summary = prepare_octonet_wifi_sample_download(resolved_source_root, dataset_root)
        _normalize_payload_layout(
            dataset_root,
            payload_config,
            ignored_names=ignored_names,
            field_name=f"download.parts.{part}.payload.marker_paths",
        )
        detail = (
            "OctoNet sample is the fixed WiFi course subset built from the canonical full "
            f"OctoNet source root. Built profile={summary.profile_id} with "
            f"{summary.sample_count} clips across scenes={summary.scene_ids} and "
            f"subjects={summary.subject_ids}."
        )
        if last_source_error is not None:
            detail = (
                "OctoNet sample archive download failed; rebuilt from source_root instead. "
                f"Source error: {last_source_error}. {detail}"
            )
        return "generated_sample", None, detail, resolved_source_root

    del announce_download
    if last_source_error is not None:
        raise last_source_error
    raise ValueError(f"OctoNet download part '{part}' did not complete from any configured source.")


__all__ = [
    "OCTONET_WIFI_SAMPLE_PROFILE",
    "OctonetWiFiCourseSampleSpec",
    "OctonetWiFiSamplePreparationSummary",
    "download_octonet_dataset",
    "infer_existing_part",
    "normalize_reuse_selected_source_url",
    "prepare_octonet_wifi_sample_download",
    "prepare_existing_payload_for_requested_part",
]
