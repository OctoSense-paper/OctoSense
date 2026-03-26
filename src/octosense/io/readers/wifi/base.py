"""Canonical WiFi reader contract."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch

from octosense.io.readers._base import CanonicalReader, ReaderError
from octosense.io.profiles.wifi import get_wifi_subcarrier_indices
from octosense.io.semantics.loader import ReaderDefinitionBundle, load_reader_definition_bundle
from octosense.io.semantics.normalizer import apply_binding, resolve_semantic_entry
from octosense.io.semantics.schema import AxisSchema, build_axis_schema
from octosense.io.tensor import RadioTensor, SignalMetadata, apply_reader_runtime_contract

logger = logging.getLogger(__name__)


def _optional_float(value: object | None) -> float | None:
    return None if value is None else float(value)


def _compact_metadata_kwargs(**kwargs: object) -> dict[str, object]:
    return {key: value for key, value in kwargs.items() if value is not None}


class BaseWiFiReader(CanonicalReader, ABC):
    """Canonical sample-level contract for WiFi device-family readers."""

    modality: str = "wifi"
    device_family: str = "generic"
    device_name: str = "UnknownDevice"
    reader_version: str = "1.0"

    def __init__(self) -> None:
        super().__init__()
        self._reader_definition_bundle = load_reader_definition_bundle(
            self.modality,
            self.device_family,
        )

    @abstractmethod
    def read_file(self, file_path: str | Path) -> list[RadioTensor]:
        """Read CSI data from file and return a list of RadioTensor frames.

        Args:
            file_path: Path to CSI data file

        Returns:
            List of RadioTensor objects (one per CSI frame)

        Raises:
            ReaderError: If file format is invalid or parsing fails
        """
        pass

    @abstractmethod
    def validate_format(self, file_path: str | Path) -> tuple[bool, str]:
        """Validate that file format matches this reader.

        Args:
            file_path: Path to CSI data file

        Returns:
            Tuple of (is_valid, error_message). error_message is empty when valid.
        """
        pass

    def _merge_signals_to_tensor(
        self,
        signals: list[RadioTensor],
        *,
        file_path: str | Path,
    ) -> RadioTensor:
        """Merge frame-level RadioTensors into one canonical time-series tensor."""
        if len(signals) < 2:
            raise ReaderError(
                f"Need at least 2 packets to compute sample_rate, got {len(signals)}. "
                f"File: {file_path}"
            )

        tensors = [signal.to_tensor(contiguous=True) for signal in signals]
        merged_data = torch.cat(tensors, dim=0).contiguous()

        timestamps = self._merge_timestamps_seconds(signals)
        sample_rate = self._infer_stream_sample_rate(signals)

        first_meta = signals[0].metadata
        first_tensor = signals[0]
        first_tensor_ndim = len(first_tensor.shape)
        num_subc = first_tensor.shape[1] if first_tensor_ndim >= 2 else 0

        metadata = SignalMetadata(
            **_compact_metadata_kwargs(
                modality=first_meta.modality,
                center_freq=first_meta.center_freq,
                bandwidth=first_meta.bandwidth,
                sample_rate=sample_rate,
                subcarrier_spacing=first_meta.subcarrier_spacing,
                timestamp_start=float(timestamps[0].item()),
                subcarrier_indices=first_meta.subcarrier_indices,
                reader_id=self.reader_id,
                capture_device=self.device_name,
                data_version=first_meta.data_version,
            )
        )

        time_values = (timestamps - timestamps[0]).numpy()
        metadata.set_coord("time", time_values, unit="s")
        if num_subc > 0:
            subcarrier_indices = get_wifi_subcarrier_indices(first_meta)
            if subcarrier_indices.size > 0:
                metadata.set_coord("subc", subcarrier_indices, unit="index")
        if first_tensor_ndim >= 3:
            metadata.set_coord("tx", np.arange(int(first_tensor.shape[2])), unit="index")
        if first_tensor_ndim >= 4:
            metadata.set_coord("rx", np.arange(int(first_tensor.shape[3])), unit="index")

        self._populate_merged_metadata(metadata, first_meta)

        schema = self._canonical_wifi_axis_schema()

        merged_tensor = RadioTensor.from_reader(merged_data, schema, metadata=metadata)
        logger.info(
            "Merged %d packets into time series: shape=%s, sample_rate=%s Hz",
            len(signals),
            tuple(merged_tensor.shape),
            "unknown" if sample_rate is None else f"{sample_rate:.2f}",
        )
        return merged_tensor

    def _infer_stream_sample_rate(
        self,
        signals: Sequence[RadioTensor],
    ) -> float | None:
        if len(signals) < 2:
            return None
        timestamps = self._merge_timestamps_seconds(list(signals))
        diffs = torch.diff(timestamps)
        valid_diffs = diffs[diffs > 0]
        if len(valid_diffs) == 0:
            return None
        return float(1.0 / valid_diffs.median().item())

    def _assign_stream_sample_rate(
        self,
        signals: Sequence[RadioTensor],
    ) -> float | None:
        sample_rate = self._infer_stream_sample_rate(signals)
        if sample_rate is None:
            return None
        for signal in signals:
            signal.metadata.sample_rate = sample_rate
        return sample_rate

    def _merge_timestamps_seconds(self, signals: list[RadioTensor]) -> torch.Tensor:
        return torch.tensor(
            [signal.metadata.timestamp_start for signal in signals],
            dtype=torch.float64,
        )

    def read(self, file_path: str | Path) -> RadioTensor:
        """Canonical reader path that materializes one semantic sample tensor."""
        return self._merge_signals_to_tensor(self.read_file(file_path), file_path=file_path)

    @property
    def reader_definition_bundle(self) -> ReaderDefinitionBundle:
        return self._reader_definition_bundle

    def _canonicalize_payload(
        self,
        raw_payload: Mapping[str, Any],
        *,
        keep_unmapped: bool,
    ) -> dict[str, Any]:
        bundle = self.reader_definition_bundle
        return apply_binding(
            raw_payload,
            bundle.binding_plan.fields,
            binding_entries=bundle.binding.fields,
            binding_converters=bundle.binding_converters,
            converter_context=bundle.converter_context,
            known_canonical_names=bundle.canonical_export_names,
            keep_unmapped=keep_unmapped,
            raw_value_payload=raw_payload,
        )

    def _finalize_runtime_contract(
        self,
        metadata: SignalMetadata,
        *,
        raw_payload: Mapping[str, Any],
        keep_unmapped: bool = False,
    ) -> dict[str, Any]:
        return apply_reader_runtime_contract(
            metadata,
            self.reader_definition_bundle,
            raw_payload=raw_payload,
            keep_unmapped=keep_unmapped,
        )

    def _canonical_wifi_axis_schema(self) -> AxisSchema:
        bundle = self.reader_definition_bundle
        return build_axis_schema(
            tuple(bundle.binding_plan.axes.values()) or bundle.canonical_axes,
            semantic_registry=bundle.semantic_registry,
            aliases=bundle.aliases,
        )

    def _apply_runtime_bridge(self, metadata: SignalMetadata) -> None:
        """Populate runtime-facing semantic bindings from the canonical reader bundle."""
        bundle = self.reader_definition_bundle
        field_sources: dict[str, list[str]] = {}
        field_statuses: dict[str, list[str]] = {}
        field_namespaces: dict[str, list[str]] = {}
        signal_semantic_id: str | None = None
        signal_statuses: list[str] = []
        signal_representation_id: str | None = None
        for raw_name, target in bundle.binding_plan.fields.items():
            source_entry = bundle.binding.fields.get(raw_name)
            exported_name = target.exported_name
            field_sources.setdefault(exported_name, []).append(raw_name)
            if source_entry is not None and source_entry.status:
                field_statuses.setdefault(exported_name, []).append(source_entry.status)
            if source_entry is not None and source_entry.source_namespace:
                field_namespaces.setdefault(exported_name, []).append(source_entry.source_namespace)
            if (
                target.kind == "representation"
                and signal_representation_id is None
                and exported_name == "data_format"
            ):
                signal_representation_id = target.semantic_id

        signal_sources: list[str] = []
        signal_namespaces: list[str] = []
        for raw_name, target in bundle.binding_plan.fields.items():
            if target.kind != "tensor":
                continue
            signal_sources.append(raw_name)
            source_entry = bundle.binding.fields.get(raw_name)
            if source_entry is not None and source_entry.status:
                signal_statuses.append(source_entry.status)
            if source_entry is not None and source_entry.source_namespace:
                signal_namespaces.append(source_entry.source_namespace)
            if signal_semantic_id is None:
                signal_semantic_id = target.semantic_id

        if signal_semantic_id is None:
            signal_source = bundle.binding.signal_source
            if signal_source:
                signal_sources.append(signal_source)
                source_entry = bundle.binding.fields.get(signal_source)
                if source_entry is not None and source_entry.status:
                    signal_statuses.append(source_entry.status)
                if source_entry is not None and source_entry.source_namespace:
                    signal_namespaces.append(source_entry.source_namespace)
                signal_entry = resolve_semantic_entry(
                    signal_source,
                    semantic_registry=bundle.semantic_registry,
                    kind="tensor",
                )
                if signal_entry is not None:
                    signal_semantic_id = signal_entry.semantic_id

        if signal_representation_id is None:
            for target in bundle.binding_plan.fields.values():
                if target.kind == "representation":
                    signal_representation_id = target.semantic_id
                    break

        metadata.apply_runtime_bridge(
            bundle.semantic_registry,
            binding_sources=field_sources,
            binding_statuses={
                exported_name: runtime_status
                for exported_name, runtime_status in (
                    (
                        exported_name,
                        _runtime_status(field_statuses.get(exported_name, ())),
                    )
                    for exported_name in field_sources
                )
                if runtime_status is not None
            },
            binding_provenance={
                exported_name: runtime_provenance
                for exported_name, runtime_provenance in (
                    (
                        exported_name,
                        _runtime_provenance(source_namespaces=field_namespaces.get(exported_name, ())),
                    )
                    for exported_name in field_sources
                )
                if runtime_provenance is not None
            },
            signal_semantic_id=signal_semantic_id,
            signal_status=_runtime_status(signal_statuses),
            signal_representation_id=signal_representation_id,
            signal_provenance=_runtime_provenance(
                source_fields=signal_sources,
                source_namespaces=signal_namespaces,
            ),
        )

    def _structured_merged_extra_fields(self) -> tuple[str, ...]:
        raw_fields = self.reader_definition_bundle.config.get("merged_extra_fields", ())
        if raw_fields in (None, ""):
            return ()
        if not isinstance(raw_fields, Sequence) or isinstance(raw_fields, (str, bytes)):
            raise ReaderError(
                "reader device.yaml merged_extra_fields must be a list of field names",
                context={"reader_id": self.reader_id},
            )
        return tuple(str(field_name) for field_name in raw_fields)

    def _populate_merged_metadata(
        self,
        merged_metadata: SignalMetadata,
        packet_metadata: SignalMetadata,
    ) -> None:
        """Carry only sample-level metadata into the merged semantic sample.

        Packet-local runtime payloads stay on frame-level tensors. The merged
        reader surface should keep only explicitly declared sample-level extras
        plus the canonical runtime bridge for the merged semantic sample.
        """

        for field_name in self._structured_merged_extra_fields():
            if field_name in packet_metadata.extra:
                merged_metadata.extra[field_name] = deepcopy(packet_metadata.extra[field_name])
        self._apply_runtime_bridge(merged_metadata)


def _runtime_provenance(
    *,
    source_fields: Sequence[str] = (),
    source_namespaces: Sequence[str] = (),
) -> dict[str, Any] | None:
    provenance: dict[str, Any] = {}
    unique_fields = list(dict.fromkeys(str(source) for source in source_fields))
    if unique_fields:
        provenance["source_fields"] = unique_fields
    unique_namespaces = list(dict.fromkeys(str(namespace) for namespace in source_namespaces))
    if unique_namespaces:
        provenance["source_namespace"] = (
            unique_namespaces[0] if len(unique_namespaces) == 1 else unique_namespaces
        )
    return provenance or None


def _runtime_status(statuses: Sequence[str]) -> str | list[str] | None:
    unique_statuses = list(dict.fromkeys(str(status) for status in statuses if status not in (None, "")))
    if not unique_statuses:
        return None
    if len(unique_statuses) == 1:
        return unique_statuses[0]
    return unique_statuses
