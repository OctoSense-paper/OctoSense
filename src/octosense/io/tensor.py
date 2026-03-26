"""Canonical IO sample tensor owner and modality schema builders."""

import contextvars
from collections.abc import Callable, Iterable, Mapping
from functools import lru_cache
import math
import os
from types import TracebackType
from typing import Any, Literal

import numpy as np
import torch

from octosense.core.describe import Describable, DescribeNode
from octosense.core.errors import DimensionError, MetadataError
from octosense.io.semantics.loader import ReaderDefinitionBundle, load_semantic_bundle
from octosense.io.semantics.metadata import CoordinateAxis, SignalMetadata
from octosense.io.semantics.normalizer import apply_binding, resolve_semantic_entry
from octosense.io.semantics.schema import AxisMetadata, AxisSchema, build_axis_schema

_MODE: contextvars.ContextVar[Literal["safety", "performance"]] = contextvars.ContextVar(
    "_MODE",
    default="safety",
)
_TRACK_META: contextvars.ContextVar[bool] = contextvars.ContextVar("_TRACK_META", default=True)
_PERF_SAMPLE_EVERY: contextvars.ContextVar[int] = contextvars.ContextVar(
    "_PERF_SAMPLE_EVERY",
    default=0,
)
_PERF_SAMPLE_COUNTER: contextvars.ContextVar[int] = contextvars.ContextVar(
    "_PERF_SAMPLE_COUNTER",
    default=0,
)
_METADATA_UNSET = object()


class SafetyMode:
    """Context manager for full semantic-boundary validation."""

    def __init__(self) -> None:
        self._token: contextvars.Token[Literal["safety", "performance"]] | None = None

    def __enter__(self) -> "SafetyMode":
        self._token = _MODE.set("safety")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._token is not None:
            _MODE.reset(self._token)


class PerformanceMode:
    """Context manager for semantic-boundary fast paths."""

    def __init__(self, sample_every: int | None = None) -> None:
        if sample_every is None:
            env_value = os.getenv("OCTOSENSE_PERF_SAMPLE_EVERY", "0").strip() or "0"
            sample_every = int(env_value)
        if sample_every < 0:
            raise ValueError("sample_every must be >= 0")
        self._sample_every = sample_every
        self._token: contextvars.Token[Literal["safety", "performance"]] | None = None
        self._sample_every_token: contextvars.Token[int] | None = None
        self._sample_counter_token: contextvars.Token[int] | None = None

    def __enter__(self) -> "PerformanceMode":
        self._token = _MODE.set("performance")
        self._sample_every_token = _PERF_SAMPLE_EVERY.set(self._sample_every)
        self._sample_counter_token = _PERF_SAMPLE_COUNTER.set(0)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._token is not None:
            _MODE.reset(self._token)
        if self._sample_every_token is not None:
            _PERF_SAMPLE_EVERY.reset(self._sample_every_token)
        if self._sample_counter_token is not None:
            _PERF_SAMPLE_COUNTER.reset(self._sample_counter_token)


class TrackMeta:
    """Context manager controlling metadata propagation."""

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._token: contextvars.Token[bool] | None = None

    def __enter__(self) -> "TrackMeta":
        self._token = _TRACK_META.set(self._enabled)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._token is not None:
            _TRACK_META.reset(self._token)


def get_mode() -> Literal["safety", "performance"]:
    return _MODE.get()


def is_safety_mode() -> bool:
    return _MODE.get() == "safety"


def is_performance_mode() -> bool:
    return _MODE.get() == "performance"


def get_perf_validation_sample_every() -> int:
    return _PERF_SAMPLE_EVERY.get()


def should_sample_full_validation() -> bool:
    if not is_performance_mode():
        return False
    every = _PERF_SAMPLE_EVERY.get()
    if every <= 0:
        return False
    next_count = _PERF_SAMPLE_COUNTER.get() + 1
    _PERF_SAMPLE_COUNTER.set(next_count)
    return (next_count % every) == 0


def infer_mode() -> Literal["safety", "performance"]:
    try:
        import torch

        if not torch.is_grad_enabled():
            return "performance"
    except ImportError:
        pass
    return "safety"


def is_tracking_meta() -> bool:
    return _TRACK_META.get()


def get_track_meta() -> bool:
    return _TRACK_META.get()


class RadioTensor(Describable):
    """Canonical sample-level tensor wrapper that carries axis semantics and metadata.

    RadioTensor wraps a PyTorch tensor with:
    - AxisSchema: Dimension semantics (axis names, units)
    - SignalMetadata: Three-layer metadata (physical, coordinates, provenance)

    Key features:
    - Zero-copy unwrap via as_tensor() for performance path
    - Automatic metadata updates during slicing/permutation
    - Human-readable introspection via describe()
    - Full pickle support for DataLoader multiprocessing

    Example:
        >>> data = torch.randn(10, 128, 3, complex64)
        >>> schema = AxisSchema(axes=('time', 'subc', 'rx'))
        >>> metadata = SignalMetadata(center_freq=2.4e9, bandwidth=20e6)
        >>> rt = RadioTensor(data, schema, metadata)
        >>> print(rt.describe())
    """

    def __init__(
        self,
        data: torch.Tensor,
        axis_schema: AxisSchema,
        metadata: SignalMetadata | None = None,
    ) -> None:
        """Initialize RadioTensor.

        Args:
            data: PyTorch tensor (shape must match axis_schema)
            axis_schema: Dimension semantics descriptor
            metadata: Signal metadata (default: empty SignalMetadata)
        Raises:
            DimensionError: If data.ndim != len(axis_schema)
        """
        if data.ndim != len(axis_schema):
            raise DimensionError(
                f"Tensor dimensions ({data.ndim}) don't match schema length ({len(axis_schema)}). "
                f"Tensor shape: {data.shape}, Schema axes: {axis_schema.axes}"
            )

        # Enforce named tensor contract: names must be None
        names = data.names
        if names is not None and any(name is not None for name in names):
            data = data.rename(None)

        self._data = data
        self._axis_schema = axis_schema
        self._metadata = metadata or SignalMetadata()

        # Validate in safety mode
        if is_safety_mode():
            self._validate_coords_consistency()

    def _validate_coords_consistency(self) -> None:
        """Validate coordinate lengths match tensor dimensions (SafetyMode only)."""
        for axis_name, coord in self._metadata.coords.items():
            if coord.axis_name != axis_name:
                raise MetadataError(
                    f"Coordinate key '{axis_name}' must match CoordinateAxis.axis_name "
                    f"('{coord.axis_name}')"
                )
            if coord.values is not None:
                values = np.asarray(coord.values)
                if values.ndim != 1:
                    raise MetadataError(
                        f"Coordinate '{axis_name}' values must be 1-D, got ndim={values.ndim}"
                    )
                axis_idx = self._axis_schema.index(axis_name)
                expected_len = self._data.shape[axis_idx]
                actual_len = int(values.shape[0])
                if actual_len != expected_len:
                    raise MetadataError(
                        f"Coordinate '{axis_name}' length ({actual_len}) "
                        f"doesn't match tensor dimension ({expected_len})"
                    )

    def as_tensor(self) -> torch.Tensor:
        """Unwrap to plain PyTorch tensor, making a contiguous copy if needed.

        When the underlying tensor is contiguous this is a zero-copy view.
        When the tensor is non-contiguous a contiguous copy is returned silently
        so that validation code (e.g. _validate_input in SafetyMode) does not
        crash with RuntimeError.

        Returns:
            torch.Tensor that is guaranteed to be contiguous

        Example:
            >>> rt = RadioTensor(data, schema, metadata)
            >>> tensor = rt.as_tensor()  # Zero-copy when already contiguous
            >>> logits = model(tensor)

        Note:
            For explicit device transfer or dtype casting use to_tensor() instead.
        """
        if not self._data.is_contiguous():
            return self._data.contiguous()
        return self._data

    def validate(self) -> bool:
        """Validate tensor/schema/metadata consistency at semantic boundaries.

        Readers and other semantic boundary adapters should prefer
        ``RadioTensor.from_reader(...)`` so construction and validation stay on
        the same canonical path. Direct ``validate()`` calls remain available
        for explicit self-checks and test assertions.
        """
        if self._data.ndim != len(self._axis_schema):
            raise DimensionError(
                f"Tensor dimensions ({self._data.ndim}) don't match schema length ({len(self._axis_schema)}). "
                f"Tensor shape: {self._data.shape}, Schema axes: {self._axis_schema.axes}"
            )
        unknown_coords = sorted(set(self._metadata.coords) - set(self._axis_schema.axes))
        if unknown_coords:
            raise MetadataError(
                "Coordinate axes are not present in the tensor schema: "
                f"{unknown_coords}. Schema axes: {self._axis_schema.axes}"
            )
        self._validate_coords_consistency()
        if self._metadata.subcarrier_indices and self._axis_schema.has_axis("subc"):
            expected_len = int(self._data.shape[self._axis_schema.index("subc")])
            actual_len = len(self._metadata.subcarrier_indices)
            if actual_len != expected_len:
                raise MetadataError(
                    "metadata.subcarrier_indices length "
                    f"({actual_len}) doesn't match 'subc' dimension ({expected_len})"
                )
        return True

    @classmethod
    def from_reader(
        cls,
        data: torch.Tensor,
        axis_schema: AxisSchema,
        metadata: SignalMetadata | None = None,
    ) -> "RadioTensor":
        """Construct a reader-boundary ``RadioTensor`` and validate it once.

        This is the canonical path for hardware readers that materialize a new
        semantic tensor at the system boundary.
        """
        tensor = cls(data, axis_schema, metadata)
        tensor.validate()
        return tensor

    def to_tensor(
        self,
        contiguous: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Convert to plain PyTorch tensor with explicit copy/cast/move options.

        This method provides explicit control over memory operations,
        following constitution principle 8.2.

        Args:
            contiguous: If True, ensure output is contiguous (may copy)
            dtype: Target dtype (None = no cast)
            device: Target device (None = no move)

        Returns:
            PyTorch tensor (potentially copied/cast/moved)

        Example:
            >>> rt = RadioTensor(data, schema, metadata)
            >>> # Explicit contiguous copy
            >>> tensor = rt.to_tensor(contiguous=True)
            >>> # Move to GPU with dtype cast
            >>> tensor = rt.to_tensor(dtype=torch.float32, device='cuda')
        """
        result = self._data

        if dtype is not None and result.dtype != dtype:
            result = result.to(dtype=dtype)

        if device is not None:
            result = result.to(device=device)

        if contiguous and not result.is_contiguous():
            result = result.contiguous()

        return result

    def describe_tree(self) -> DescribeNode:
        """Structured description of schema and metadata."""
        children: list[DescribeNode] = [
            self._axis_schema.describe_tree(),
            self._metadata.describe_tree(include_coords=False),
            DescribeNode(
                kind="coords",
                name="coords",
                children=tuple(
                    coord.describe_tree()
                    for _, coord in sorted(self._metadata.coords.items(), key=lambda item: item[0])
                ),
            ),
        ]
        return DescribeNode(
            kind="radiotensor",
            name="RadioTensor",
            fields={
                "shape": tuple(int(dim) for dim in self._data.shape),
                "dtype": str(self._data.dtype),
                "device": str(self._data.device),
                "modality": self._metadata.modality or None,
            },
            children=tuple(children),
        )

    def runtime_payload(self) -> dict[str, Any]:
        """Runtime-facing semantic bridge for downstream consumers."""
        axis_payload: dict[str, dict[str, Any]] = {}
        axis_runtime_bindings = self._metadata.extra.get(_AXIS_RUNTIME_EXTRA_KEY, {})
        if not isinstance(axis_runtime_bindings, Mapping):
            axis_runtime_bindings = {}
        shared_provenance = self._metadata.runtime_provenance_payload()
        signal_runtime = self._metadata.signal_runtime
        signal_status = signal_runtime.status if signal_runtime is not None else None
        signal_representation_id = (
            signal_runtime.representation_id if signal_runtime is not None else None
        )
        signal_provenance = _merge_runtime_provenance(
            shared_provenance,
            signal_runtime.provenance if signal_runtime is not None else {},
        )
        for axis_name in self._axis_schema.axes:
            axis_meta = self._axis_schema.get_metadata(axis_name)
            coord = self._metadata.get_coord(axis_name)
            axis_runtime = axis_runtime_bindings.get(axis_name, {})
            if not isinstance(axis_runtime, Mapping):
                axis_runtime = {}
            axis_payload[axis_name] = {
                "semantic_id": axis_meta.semantic_id if axis_meta is not None else None,
                "status": axis_runtime.get("status", signal_status),
                "representation_id": axis_runtime.get(
                    "representation_id",
                    signal_representation_id,
                ),
                "provenance": dict(
                    _merge_runtime_provenance(
                        signal_provenance,
                        axis_runtime.get("provenance", {}),
                    )
                ),
                "unit": coord.unit if coord is not None else (axis_meta.unit if axis_meta is not None else None),
                "length": int(self._data.shape[self._axis_schema.index(axis_name)]),
            }

        payload = self._metadata.semantic_runtime_payload()
        payload["axes"] = axis_payload
        payload["shape"] = tuple(int(dim) for dim in self._data.shape)
        payload["dtype"] = str(self._data.dtype)
        return payload

    def with_metadata(
        self,
        metadata: SignalMetadata,
    ) -> "RadioTensor":
        """Return a metadata-updated view that keeps reader-owned tensor storage."""

        return RadioTensor(
            self._data,
            self._axis_schema,
            metadata,
        )

    def with_metadata_updates(
        self,
        *,
        modifier: Callable[[SignalMetadata], None] | None = None,
        modality: str | object = _METADATA_UNSET,
        capture_device: str | object = _METADATA_UNSET,
        extra_updates: Mapping[str, Any] | None = None,
        extra_defaults: Mapping[str, Any] | None = None,
    ) -> "RadioTensor":
        """Clone metadata, apply dataset-side augmentation, and preserve tensor ownership."""

        metadata = self._metadata.copy()
        if modality is not _METADATA_UNSET:
            metadata.modality = str(modality)
        if capture_device is not _METADATA_UNSET:
            metadata.capture_device = str(capture_device)
        if extra_defaults or extra_updates:
            extra = dict(metadata.extra)
            if extra_defaults:
                for key, value in extra_defaults.items():
                    extra.setdefault(key, value)
            if extra_updates:
                extra.update(extra_updates)
            metadata.extra = extra
        if modifier is not None:
            modifier(metadata)
        return self.with_metadata(metadata)

    def to(self, device: torch.device | str) -> "RadioTensor":
        """Transfer to device (preserves metadata).

        Args:
            device: Target device (e.g., 'cuda', 'cpu')

        Returns:
            New RadioTensor on target device
        """
        new_data = self._data.to(device)
        return RadioTensor(
            new_data,
            self._axis_schema,
            self._metadata.copy(),
        )

    def __getitem__(self, key: Any) -> Any:
        """Slicing with automatic metadata update or semantic axis lookup.

        Args:
            key: Slice key (int, slice, tuple of slices) or axis name

        Returns:
            New RadioTensor with updated metadata, or a ``CoordinateAxis`` when
            ``key`` is a semantic axis name.

        Example:
            >>> rt = RadioTensor(data, schema, metadata)
            >>> rt_slice = rt[:5, :, 0]  # First 5 time steps, first RX antenna
            >>> time_axis = rt["time"]
            >>> time_axis.unit
            's'
        """
        if isinstance(key, str):
            if not self._axis_schema.has_axis(key):
                suggestion = self._axis_schema.suggest_axis_name(key)
                available = ", ".join(self._axis_schema.axes)
                if suggestion is not None:
                    raise KeyError(
                        f"Axis '{key}' not found. Available axes: {available}. "
                        f"Did you mean '{suggestion}'?"
                    )
                raise KeyError(f"Axis '{key}' not found. Available axes: {available}.")

            coord = self._metadata.get_coord(key)
            if coord is not None:
                return coord

            axis_idx = self._axis_schema.index(key)
            axis_size = int(self._data.shape[axis_idx])
            axis_meta = self._axis_schema.axis_metadata.get(key)
            unit = axis_meta.unit if axis_meta is not None else ""
            return CoordinateAxis(
                axis_name=key,
                values=np.arange(axis_size, dtype=np.int64),
                unit=unit,
            )

        # Normalize key to full tuple
        if not isinstance(key, tuple):
            key = (key,)

        if Ellipsis in key:
            ellipsis_index = key.index(Ellipsis)
            num_missing = self._data.ndim - (len(key) - 1)
            key = (
                key[:ellipsis_index]
                + (slice(None),) * max(0, num_missing)
                + key[ellipsis_index + 1 :]
            )

        if any(k is None for k in key):
            raise ValueError("None/newaxis indexing is not supported for RadioTensor")

        if len(key) < self._data.ndim:
            key = key + (slice(None),) * (self._data.ndim - len(key))
        if len(key) > self._data.ndim:
            raise IndexError(
                f"Too many indices for RadioTensor: {len(key)} > {self._data.ndim}"
            )

        new_data = self._data[key]
        new_metadata = self._metadata.copy()

        # Build new schema (drop axes for integer indexing)
        new_axes: list[str] = []
        for axis_name, slice_key in zip(self._axis_schema.axes, key, strict=False):
            if isinstance(slice_key, int):
                continue
            if isinstance(slice_key, slice):
                new_axes.append(axis_name)
            else:
                raise TypeError(
                    f"Unsupported index type for axis '{axis_name}': {type(slice_key)}"
                )

        new_axis_metadata = {
            k: v for k, v in self._axis_schema.axis_metadata.items() if k in new_axes
        }
        new_schema = AxisSchema(axes=tuple(new_axes), axis_metadata=new_axis_metadata)

        # Remove coordinates for eliminated axes
        eliminated_axes = set(self._axis_schema.axes) - set(new_axes)
        for axis_name in eliminated_axes:
            if axis_name in new_metadata.coords:
                del new_metadata.coords[axis_name]

        # Update coordinates for sliced axes (when tracking metadata)
        if is_tracking_meta() and is_safety_mode():
            for axis_name, slice_key in zip(
                self._axis_schema.axes, key, strict=False
            ):
                if isinstance(slice_key, slice):
                    coord = new_metadata.get_coord(axis_name)
                    if coord and coord.values is not None:
                        coord_values = coord.values[slice_key]
                        new_metadata.set_coord(axis_name, coord_values, coord.unit)

                    # Update timestamp_start if slicing time axis
                    if axis_name == "time" and slice_key.start is not None:
                        if new_metadata.sample_rate:
                            start = slice_key.start
                            if start is not None and start < 0:
                                axis_size = self._data.shape[self._axis_schema.index(axis_name)]
                                start = axis_size + start
                            if start and start > 0:
                                time_offset = start / new_metadata.sample_rate
                                new_metadata.timestamp_start += time_offset

                elif isinstance(slice_key, int) and axis_name == "time":
                    if new_metadata.sample_rate:
                        axis_size = self._data.shape[self._axis_schema.index(axis_name)]
                        idx = slice_key + axis_size if slice_key < 0 else slice_key
                        if idx > 0:
                            new_metadata.timestamp_start += idx / new_metadata.sample_rate

        return RadioTensor(
            new_data,
            new_schema,
            new_metadata,
        )

    @staticmethod
    def concat(tensors: list["RadioTensor"], axis: str) -> "RadioTensor":
        """Concatenate RadioTensors along specified axis with compatibility validation.

        This implements constitution principle 3: metadata propagation invariants
        with compatibility validation (units, monotonicity, mapping consistency).

        Args:
            tensors: List of RadioTensors to concatenate
            axis: Axis name to concatenate along

        Returns:
            Concatenated RadioTensor

        Raises:
            DimensionError: If schemas don't match
            MetadataError: If metadata is incompatible

        Example:
            >>> rt1 = RadioTensor(data1, schema, meta1)
            >>> rt2 = RadioTensor(data2, schema, meta2)
            >>> rt_concat = RadioTensor.concat([rt1, rt2], axis='time')
        """
        if not tensors:
            raise ValueError("Cannot concatenate empty list of tensors")

        first = tensors[0]

        # Validate all have same schema
        for i, rt in enumerate(tensors[1:], 1):
            if rt.axis_schema.axes != first.axis_schema.axes:
                raise DimensionError(
                    f"Cannot concatenate: tensor {i} has different axes.\n"
                    f"Expected: {list(first.axis_schema.axes)}\n"
                    f"Got: {list(rt.axis_schema.axes)}\n"
                    f"Fix: Ensure all tensors have the same axis schema."
                )

        # Validate axis exists
        if not first.axis_schema.has_axis(axis):
            suggestion = first.axis_schema.suggest_axis_name(axis)
            raise DimensionError(
                f"Concatenation axis '{axis}' not in schema.",
                available_axes=first.axis_schema.axes,
                suggestion=suggestion,
            )

        # Validate metadata compatibility (SafetyMode)
        if is_safety_mode() and is_tracking_meta():
            # Layer A: Physical constants must match
            for rt in tensors[1:]:
                if (
                    abs(rt.metadata.center_freq - first.metadata.center_freq) > 1e6
                ):  # 1 MHz tolerance
                    raise MetadataError(
                        f"Cannot concatenate: incompatible center_freq.\n"
                        f"Expected: {first.metadata.center_freq / 1e9:.3f} GHz\n"
                        f"Got: {rt.metadata.center_freq / 1e9:.3f} GHz\n"
                        f"Fix: Ensure all signals are from same frequency band."
                    )

                if abs(rt.metadata.bandwidth - first.metadata.bandwidth) > 1e6:  # 1 MHz tolerance
                    raise MetadataError(
                        f"Cannot concatenate: incompatible bandwidth.\n"
                        f"Expected: {first.metadata.bandwidth / 1e6:.1f} MHz\n"
                        f"Got: {rt.metadata.bandwidth / 1e6:.1f} MHz\n"
                        f"Fix: Ensure all signals have same bandwidth."
                    )

            # Layer B: Coordinate units must match
            for axis_name in first.axis_schema.axes:
                first_coord = first.metadata.get_coord(axis_name)
                if first_coord:
                    for rt in tensors[1:]:
                        rt_coord = rt.metadata.get_coord(axis_name)
                        if rt_coord and rt_coord.unit != first_coord.unit:
                            raise MetadataError(
                                f"Cannot concatenate: incompatible units for axis '{axis_name}'.\n"
                                f"Expected unit: {first_coord.unit}\n"
                                f"Got unit: {rt_coord.unit}\n"
                                f"Fix: Convert coordinates to same unit before concatenation."
                            )

        # Concatenate tensors
        axis_idx = first.axis_schema.index(axis)
        concat_data = torch.cat([rt._data for rt in tensors], dim=axis_idx)

        # Merge metadata
        merged_metadata = first.metadata.copy()

        # Update coordinates for concatenation axis (if tracked)
        if is_tracking_meta():
            coord = merged_metadata.get_coord(axis)
            if coord and coord.values is not None:
                # Concatenate coordinate values
                import numpy as np

                all_coord_values = [
                    rt.metadata.get_coord(axis).values
                    for rt in tensors
                    if rt.metadata.get_coord(axis)
                    and rt.metadata.get_coord(axis).values is not None
                ]
                if all_coord_values:
                    merged_coord_values = np.concatenate(all_coord_values)
                    merged_metadata.set_coord(axis, merged_coord_values, coord.unit)

        return RadioTensor(
            concat_data,
            first.axis_schema,
            merged_metadata,
        )

    def permute_axes(self, *axes: str) -> "RadioTensor":
        """Permute dimensions by axis names.

        Args:
            *axes: Target axis order (e.g., 'batch', 'subc', 'time', 'rx')

        Returns:
            New RadioTensor with permuted dimensions

        Raises:
            DimensionError: If axes don't match schema

        Example:
            >>> rt = RadioTensor(data, AxisSchema(('time', 'subc', 'rx')), metadata)
            >>> rt_perm = rt.permute_axes('subc', 'time', 'rx')  # Swap time and subc
        """
        # Validate axes
        if set(axes) != set(self._axis_schema.axes):
            missing = set(self._axis_schema.axes) - set(axes)
            extra = set(axes) - set(self._axis_schema.axes)
            msg = f"Permute axes {axes} don't match schema."
            if missing:
                msg += f"\n  Missing axes: {list(missing)}"
            if extra:
                msg += f"\n  Extra axes: {list(extra)}"
            raise DimensionError(msg, available_axes=self._axis_schema.axes)

        # Compute permutation indices
        perm_indices = [self._axis_schema.index(ax) for ax in axes]

        # Permute data
        new_data = self._data.permute(perm_indices)
        new_schema = AxisSchema(axes=axes, axis_metadata=self._axis_schema.axis_metadata)

        return RadioTensor(
            new_data,
            new_schema,
            self._metadata.copy(),
        )

    def get_axis_index(self, name: str) -> int:
        """Get dimension index for named axis.

        Args:
            name: Axis name

        Returns:
            Zero-based dimension index

        Raises:
            DimensionError: If axis not found
        """
        if not self._axis_schema.has_axis(name):
            suggestion = self._axis_schema.suggest_axis_name(name)
            raise DimensionError(
                f"Axis '{name}' not found in schema.",
                available_axes=self._axis_schema.axes,
                suggestion=suggestion,
            )
        return self._axis_schema.index(name)

    def sel(self, **kwargs: int | slice) -> "RadioTensor":
        """Select data by axis names (semantic slicing).

        This method provides intuitive slicing using axis names instead of
        positional indices, following constitution principle 2 (semantic axis contract).

        Args:
            **kwargs: Axis name to index/slice mappings
                     e.g., time=slice(0, 100), subc=0, tx=0

        Returns:
            New RadioTensor with selected data

        Raises:
            DimensionError: If axis name not found in schema

        Example:
            >>> rt = RadioTensor(data, schema, metadata)
            >>> # Old way: manual index lookup
            >>> time_idx = rt.get_axis_index("time")
            >>> subc_idx = rt.get_axis_index("subc")
            >>> rt_slice = rt.as_tensor().select(subc_idx, 0).select(time_idx, slice(0, 100))
            >>>
            >>> # New way: semantic slicing
            >>> rt_slice = rt.sel(time=slice(0, 100), subc=0)
            >>>
            >>> # Multi-dimensional slicing
            >>> heatmap = rt_fft.sel(tx=0, rx=0)  # Returns (freq, subc)

        Note:
            - Supports both int indices and slice objects
            - Automatically updates metadata (coordinates, timestamps)
            - Unspecified axes are kept intact (equivalent to ':')
        """
        # Validate all axis names exist
        for axis_name in kwargs.keys():
            if not self._axis_schema.has_axis(axis_name):
                suggestion = self._axis_schema.suggest_axis_name(axis_name)
                raise DimensionError(
                    f"Axis '{axis_name}' not found in schema.",
                    available_axes=self._axis_schema.axes,
                    suggestion=suggestion,
                )

        # Build full slice tuple and track which axes are kept
        slice_tuple = []
        new_axes = []

        for axis_name in self._axis_schema.axes:
            if axis_name in kwargs:
                idx_or_slice = kwargs[axis_name]
                # Validate index bounds for int indices
                if isinstance(idx_or_slice, int):
                    axis_size = self._data.shape[self._axis_schema.index(axis_name)]
                    if idx_or_slice < 0:
                        idx_or_slice = axis_size + idx_or_slice
                    if idx_or_slice < 0 or idx_or_slice >= axis_size:
                        raise DimensionError(
                            (
                                f"Index {kwargs[axis_name]} out of bounds for axis "
                                f"'{axis_name}' with size {axis_size}."
                            ),
                            available_axes=self._axis_schema.axes,
                            suggestion=f"Valid range: 0-{axis_size - 1}",
                        )
                    # Integer index removes the dimension
                    slice_tuple.append(idx_or_slice)
                else:
                    # Slice keeps the dimension
                    slice_tuple.append(idx_or_slice)
                    new_axes.append(axis_name)
            else:
                slice_tuple.append(slice(None))  # Keep all
                new_axes.append(axis_name)

        # Slice the data
        new_data = self._data[tuple(slice_tuple)]

        # Create new schema without eliminated axes
        new_schema = AxisSchema(
            axes=tuple(new_axes),
            axis_metadata={
                k: v for k, v in self._axis_schema.axis_metadata.items() if k in new_axes
            },
        )

        # Copy metadata
        new_metadata = self._metadata.copy()

        # Remove coordinates for eliminated axes (axes not in new_axes)
        eliminated_axes = set(self._axis_schema.axes) - set(new_axes)
        for axis_name in eliminated_axes:
            if axis_name in new_metadata.coords:
                del new_metadata.coords[axis_name]

        # Update coordinates for kept axes (in SafetyMode)
        if is_safety_mode():
            if is_tracking_meta():
                # Build a mapping of old axis index to slice key
                axis_slices = {}
                for i, axis_name in enumerate(self._axis_schema.axes):
                    axis_slices[axis_name] = slice_tuple[i]

                # Update coords for sliced axes (not eliminated)
                for axis_name in new_axes:
                    slice_key = axis_slices[axis_name]
                    if isinstance(slice_key, slice):
                        coord = new_metadata.get_coord(axis_name)
                        if coord and coord.values is not None:
                            coord_values = coord.values[slice_key]
                            new_metadata.set_coord(axis_name, coord_values, coord.unit)

                        # Update timestamp_start if slicing time axis
                        if axis_name == "time" and slice_key.start is not None:
                            if new_metadata.sample_rate and slice_key.start > 0:
                                time_offset = slice_key.start / new_metadata.sample_rate
                                new_metadata.timestamp_start += time_offset

        return RadioTensor(
            new_data,
            new_schema,
            new_metadata,
        )

    def isel(self, **kwargs: int | slice) -> "RadioTensor":
        """Select data by axis names and positional indices.

        This is an alias for sel() since current implementation uses
        positional indices. Provided for API consistency with xarray.

        Args:
            **kwargs: Axis name to index/slice mappings

        Returns:
            New RadioTensor with selected data

        Raises:
            DimensionError: If axis name not found in schema

        Example:
            >>> rt = RadioTensor(data, schema, metadata)
            >>> rt_slice = rt.isel(time=slice(0, 100), tx=0)

        Note:
            In future versions, sel() may support coordinate-based selection
            (e.g., sel(time=0.5) to find nearest time value), while isel()
            will always use positional indices.
        """
        return self.sel(**kwargs)

    # Pickle support
    def __getstate__(self) -> dict[str, Any]:
        """Get state for pickling."""
        return {
            "_data": self._data,
            "_axis_schema": self._axis_schema,
            "_metadata": self._metadata,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Set state from unpickling."""
        self._data = state["_data"]
        self._axis_schema = state["_axis_schema"]
        self._metadata = state["_metadata"]

    # Property delegations
    @property
    def shape(self) -> torch.Size:
        """Tensor shape."""
        return self._data.shape

    @property
    def dtype(self) -> torch.dtype:
        """Tensor dtype."""
        return self._data.dtype

    @property
    def device(self) -> torch.device:
        """Tensor device."""
        return self._data.device

    @property
    def axis_schema(self) -> AxisSchema:
        """Access axis schema."""
        return self._axis_schema

    @property
    def metadata(self) -> SignalMetadata:
        """Access metadata."""
        return self._metadata

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RadioTensor(shape={tuple(self.shape)}, dtype={self.dtype}, "
            f"axes={self._axis_schema.axes}, device={self.device})"
        )


def resolved_sample_rate(sample: RadioTensor) -> float:
    """Resolve a positive sample rate from metadata or time coordinates."""

    sample_rate = getattr(sample.metadata, "sample_rate", None)
    if sample_rate is not None:
        sample_rate = float(sample_rate)
        if np.isfinite(sample_rate) and sample_rate > 0:
            return float(sample_rate)

    time_coord = sample.metadata.get_coord("time")
    if time_coord is not None and time_coord.values is not None and len(time_coord.values) >= 2:
        deltas = np.diff(np.asarray(time_coord.values, dtype=np.float64))
        finite = deltas[np.isfinite(deltas) & (deltas > 0)]
        if finite.size:
            return float(1.0 / np.median(finite))

    raise MetadataError(
        "RadioTensor requires metadata.sample_rate or at least two positive time coordinates "
        "to resolve a sample rate."
    )


def resize_axis(
    sample: RadioTensor,
    *,
    axis_name: str,
    target_length: int,
    crop: Literal["left", "center"] = "left",
) -> RadioTensor:
    if axis_name not in sample.axis_schema.axes:
        return sample
    if int(target_length) <= 0:
        raise ValueError(f"target_length must be positive, got {target_length}")
    if crop not in {"left", "center"}:
        raise ValueError(f"Unsupported crop mode: {crop}")

    axis_idx = sample.get_axis_index(axis_name)
    tensor = sample.to_tensor(contiguous=True)
    current_length = int(tensor.shape[axis_idx])
    target_length = int(target_length)
    if current_length == target_length:
        return sample

    metadata = sample.metadata.copy()
    coord = metadata.get_coord(axis_name)
    if current_length > target_length:
        start_idx = (current_length - target_length) // 2 if crop == "center" else 0
        indices = [slice(None)] * tensor.ndim
        indices[axis_idx] = slice(start_idx, start_idx + target_length)
        resized = tensor[tuple(indices)].contiguous()
        if coord is not None and coord.values is not None:
            new_axis = np.asarray(coord.values, dtype=np.float64)[start_idx : start_idx + target_length]
            metadata.set_coord(axis_name, new_axis, unit=coord.unit or "index")
        return RadioTensor(resized, sample.axis_schema, metadata)

    pad_length = target_length - current_length
    pad_shape = list(tensor.shape)
    pad_shape[axis_idx] = pad_length
    padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    resized = torch.cat([tensor, padding], dim=axis_idx).contiguous()

    sample_rate = resolved_sample_rate(sample)
    if coord is not None and coord.values is not None:
        original = np.asarray(coord.values, dtype=np.float64)
        if len(original):
            last_value = float(original[-1])
            extra = last_value + (1.0 / sample_rate) * np.arange(1, pad_length + 1, dtype=np.float64)
            new_axis = np.concatenate([original, extra], axis=0)
        else:
            new_axis = np.arange(target_length, dtype=np.float64) / sample_rate
        metadata.set_coord(axis_name, new_axis, unit=coord.unit or "s")
    elif axis_name == "time":
        metadata.set_coord(
            axis_name,
            np.arange(target_length, dtype=np.float64) / sample_rate,
            unit="s",
        )
        metadata.sample_rate = float(sample_rate)

    return RadioTensor(resized, sample.axis_schema, metadata)


def resize_time_axis(
    sample: RadioTensor,
    target_length: int,
    *,
    crop: Literal["left", "center"] = "left",
) -> RadioTensor:
    """Crop or zero-pad the ``time`` axis to a fixed length."""

    return resize_axis(
        sample,
        axis_name="time",
        target_length=target_length,
        crop=crop,
    )


def build_reader_axis_schema(bundle: ReaderDefinitionBundle) -> AxisSchema:
    """Build an axis schema directly from one reader definition bundle."""

    return build_axis_schema(
        tuple(bundle.binding_plan.axes.values()) or bundle.canonical_axes,
        semantic_registry=bundle.semantic_registry,
        aliases=bundle.aliases,
    )


def _build_semantic_axis_schema(
    modality: str,
    axis_semantic_ids: tuple[str, ...],
) -> AxisSchema:
    semantic_bundle = load_semantic_bundle(modality)
    return build_axis_schema(
        axis_semantic_ids,
        semantic_registry=semantic_bundle["semantic_registry"],
        aliases=semantic_bundle["resolved_aliases"],
    )


def apply_reader_runtime_contract(
    metadata: SignalMetadata,
    bundle: ReaderDefinitionBundle,
    *,
    raw_payload: Mapping[str, Any] | None = None,
    canonical_payload: Mapping[str, Any] | None = None,
    keep_unmapped: bool = False,
    signal_semantic_id: str | None = None,
) -> dict[str, Any]:
    """Canonicalize reader metadata and attach runtime semantic bindings."""

    if canonical_payload is not None and raw_payload is not None:
        raise ValueError("apply_reader_runtime_contract accepts either raw_payload or canonical_payload")
    raw_value_payload = raw_payload or {}
    if canonical_payload is None:
        resolved_canonical_payload = apply_binding(
            raw_value_payload,
            bundle.binding_plan.fields,
            binding_entries=bundle.binding.fields,
            binding_converters=bundle.binding_converters,
            converter_context=bundle.converter_context,
            known_canonical_names=bundle.canonical_export_names,
            keep_unmapped=keep_unmapped,
            raw_value_payload=raw_value_payload,
        )
    else:
        resolved_canonical_payload = dict(canonical_payload)

    metadata.apply_canonical_scalar_payload(resolved_canonical_payload)
    metadata.modality = bundle.metadata_spec.modality
    metadata.reader_id = bundle.metadata_spec.reader_id
    metadata.extra.update(_runtime_extra_payload(resolved_canonical_payload))

    field_sources = _binding_sources_for_payload(bundle, resolved_canonical_payload)
    field_statuses = _binding_statuses_for_payload(bundle, resolved_canonical_payload)
    field_provenance = _binding_provenance_for_payload(bundle, resolved_canonical_payload)
    resolved_signal_semantic_id = signal_semantic_id or _resolve_reader_signal_semantic_id(bundle)
    signal_representation_id = _resolve_reader_signal_representation_id(bundle)
    signal_sources = _resolve_reader_signal_sources(bundle, resolved_signal_semantic_id)
    signal_status = _resolve_reader_signal_status(bundle, resolved_signal_semantic_id)
    signal_provenance = _resolve_reader_signal_provenance(bundle, resolved_signal_semantic_id)
    if signal_sources:
        signal_provenance = _merge_runtime_provenance(
            signal_provenance or {},
            {"source_fields": signal_sources},
        )

    metadata.apply_runtime_bridge(
        bundle.semantic_registry,
        binding_sources=field_sources,
        binding_statuses=field_statuses,
        binding_provenance=field_provenance,
        signal_semantic_id=resolved_signal_semantic_id,
        signal_status=signal_status,
        signal_representation_id=signal_representation_id,
        signal_provenance=signal_provenance,
    )
    metadata.extra[_AXIS_RUNTIME_EXTRA_KEY] = _resolve_axis_runtime_bindings(
        bundle,
        signal_provenance=signal_provenance,
        signal_representation_id=signal_representation_id,
    )
    return resolved_canonical_payload


@lru_cache(maxsize=1)
def build_wifi_csi_axis_schema() -> AxisSchema:
    return _build_semantic_axis_schema(
        "wifi",
        (
            "octo.common.axis.time",
            "octo.wifi.axis.subc",
            "octo.common.axis.tx",
            "octo.common.axis.rx",
        ),
    )


@lru_cache(maxsize=1)
def build_radar_axis_schema() -> AxisSchema:
    return _build_semantic_axis_schema(
        "mmwave",
        (
            "octo.common.axis.frame",
            "octo.common.axis.chirp",
            "octo.common.axis.adc",
            "octo.common.axis.ant",
        ),
    )


@lru_cache(maxsize=1)
def build_ble_rssi_axis_schema() -> AxisSchema:
    return _build_semantic_axis_schema(
        "ble",
        (
            "octo.common.axis.time",
            "octo.ble.axis.channel",
        ),
    )


@lru_cache(maxsize=1)
def build_uwb_cir_axis_schema() -> AxisSchema:
    return _build_semantic_axis_schema(
        "uwb",
        (
            "octo.common.axis.time",
            "octo.common.axis.tap",
            "octo.common.axis.ant",
        ),
    )


@lru_cache(maxsize=1)
def build_acoustic_waveform_axis_schema() -> AxisSchema:
    return _build_semantic_axis_schema(
        "acoustic",
        (
            "octo.common.axis.time",
            "octo.common.axis.sample",
            "octo.common.axis.mic",
        ),
    )


_CANONICAL_AXIS_SCHEMA_BUILDERS = {
    "wifi": build_wifi_csi_axis_schema,
    "mmwave": build_radar_axis_schema,
    "ble": build_ble_rssi_axis_schema,
    "uwb": build_uwb_cir_axis_schema,
    "acoustic": build_acoustic_waveform_axis_schema,
}


_CANONICAL_SCALAR_RUNTIME_NAMES = {
    "modality",
    "center_freq",
    "bandwidth",
    "sample_rate",
    "subcarrier_spacing",
    "timestamp",
    "reader_id",
    "capture_device",
}

_AXIS_RUNTIME_EXTRA_KEY = "__axis_runtime_bindings__"


def _runtime_extra_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: value
        for name, value in payload.items()
        if name not in _CANONICAL_SCALAR_RUNTIME_NAMES
    }


def _binding_sources_for_payload(
    bundle: ReaderDefinitionBundle,
    canonical_payload: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    field_sources: dict[str, list[str]] = {}
    payload_names = set(canonical_payload)
    for raw_name, target in bundle.binding_plan.fields.items():
        if target.exported_name not in payload_names:
            continue
        field_sources.setdefault(target.exported_name, []).append(raw_name)
    return {
        exported_name: tuple(dict.fromkeys(raw_names))
        for exported_name, raw_names in field_sources.items()
    }


def _binding_statuses_for_payload(
    bundle: ReaderDefinitionBundle,
    canonical_payload: Mapping[str, Any],
) -> dict[str, str | list[str]]:
    payload_names = set(canonical_payload)
    statuses: dict[str, list[str]] = {}
    for raw_name, target in bundle.binding_plan.fields.items():
        if target.exported_name not in payload_names:
            continue
        source_entry = bundle.binding.fields.get(raw_name)
        if source_entry is None or not source_entry.status:
            continue
        statuses.setdefault(target.exported_name, []).append(source_entry.status)
    return {
        exported_name: normalized
        for exported_name, normalized in (
            (exported_name, _runtime_status(values))
            for exported_name, values in statuses.items()
        )
        if normalized is not None
    }


def _binding_provenance_for_payload(
    bundle: ReaderDefinitionBundle,
    canonical_payload: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    payload_names = set(canonical_payload)
    namespaces: dict[str, list[str]] = {}
    for raw_name, target in bundle.binding_plan.fields.items():
        if target.exported_name not in payload_names:
            continue
        source_entry = bundle.binding.fields.get(raw_name)
        if source_entry is None or not source_entry.source_namespace:
            continue
        namespaces.setdefault(target.exported_name, []).append(source_entry.source_namespace)
    return {
        exported_name: runtime_provenance
        for exported_name, runtime_provenance in (
            (
                exported_name,
                _runtime_provenance(source_namespaces=values),
            )
            for exported_name, values in namespaces.items()
        )
        if runtime_provenance is not None
    }


def _resolve_reader_signal_semantic_id(bundle: ReaderDefinitionBundle) -> str | None:
    signal_source = bundle.binding.signal_source
    if signal_source:
        signal_entry = resolve_semantic_entry(
            signal_source,
            semantic_registry=bundle.semantic_registry,
            kind="tensor",
        )
        if signal_entry is not None:
            return signal_entry.semantic_id

    tensor_entries = [
        entry for entry in bundle.semantic_registry.values() if entry.kind == "tensor"
    ]
    if len(tensor_entries) == 1:
        return tensor_entries[0].semantic_id
    return None


def _resolve_reader_signal_representation_id(
    bundle: ReaderDefinitionBundle,
) -> str | None:
    representation_targets = [
        target for target in bundle.binding_plan.fields.values() if target.kind == "representation"
    ]
    if not representation_targets:
        return None
    ranked_targets = sorted(
        representation_targets,
        key=_representation_target_priority,
    )
    return ranked_targets[0].semantic_id


def _resolve_reader_signal_sources(
    bundle: ReaderDefinitionBundle,
    signal_semantic_id: str | None,
) -> list[str]:
    signal_sources: list[str] = []
    signal_source = bundle.binding.signal_source
    if signal_source:
        signal_sources.append(signal_source)
    for raw_name, target in bundle.binding_plan.fields.items():
        if target.kind == "tensor":
            signal_sources.append(raw_name)
        elif signal_semantic_id is not None and target.semantic_id == signal_semantic_id:
            signal_sources.append(raw_name)
    return list(dict.fromkeys(signal_sources))


def _resolve_reader_signal_status(
    bundle: ReaderDefinitionBundle,
    signal_semantic_id: str | None,
) -> str | list[str] | None:
    statuses: list[str] = []
    for raw_name, target in bundle.binding_plan.fields.items():
        if target.kind != "tensor" and (
            signal_semantic_id is None or target.semantic_id != signal_semantic_id
        ):
            continue
        source_entry = bundle.binding.fields.get(raw_name)
        if source_entry is not None and source_entry.status:
            statuses.append(source_entry.status)
    signal_source = bundle.binding.signal_source
    signal_entry = bundle.binding.fields.get(signal_source)
    if signal_entry is not None and signal_entry.status:
        statuses.append(signal_entry.status)
    return _runtime_status(statuses)


def _resolve_reader_signal_provenance(
    bundle: ReaderDefinitionBundle,
    signal_semantic_id: str | None,
) -> dict[str, Any] | None:
    namespaces: list[str] = []
    for raw_name, target in bundle.binding_plan.fields.items():
        if target.kind != "tensor" and (
            signal_semantic_id is None or target.semantic_id != signal_semantic_id
        ):
            continue
        source_entry = bundle.binding.fields.get(raw_name)
        if source_entry is not None and source_entry.source_namespace:
            namespaces.append(source_entry.source_namespace)
    signal_source = bundle.binding.signal_source
    signal_entry = bundle.binding.fields.get(signal_source)
    if signal_entry is not None and signal_entry.source_namespace:
        namespaces.append(signal_entry.source_namespace)
    return _runtime_provenance(source_namespaces=namespaces)


def _resolve_axis_runtime_bindings(
    bundle: ReaderDefinitionBundle,
    *,
    signal_provenance: Mapping[str, Any] | None,
    signal_representation_id: str | None,
) -> dict[str, dict[str, Any]]:
    axis_runtime: dict[str, dict[str, Any]] = {}
    for raw_name, target in bundle.binding_plan.axes.items():
        source_entry = bundle.binding.axes.get(raw_name)
        provenance = _merge_runtime_provenance(
            signal_provenance or {},
            _runtime_provenance(source_fields=(raw_name,)) or {},
        )
        if source_entry is not None and source_entry.source_namespace:
            provenance = _merge_runtime_provenance(
                provenance,
                _runtime_provenance(source_namespaces=(source_entry.source_namespace,)) or {},
            )
        axis_runtime[target.exported_name] = {
            "status": source_entry.status if source_entry is not None else None,
            "representation_id": signal_representation_id,
            "provenance": provenance,
        }
    return axis_runtime


def _runtime_provenance(
    *,
    source_fields: Iterable[str] = (),
    source_namespaces: Iterable[str] = (),
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


def _runtime_status(statuses: Iterable[str]) -> str | list[str] | None:
    unique_statuses = list(dict.fromkeys(str(status) for status in statuses if status not in (None, "")))
    if not unique_statuses:
        return None
    if len(unique_statuses) == 1:
        return unique_statuses[0]
    return unique_statuses


def _merge_runtime_provenance(
    base: Mapping[str, Any],
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if value in (None, "", [], ()):
            continue
        existing = merged.get(key)
        if existing is None:
            merged[key] = value
            continue
        if existing == value:
            continue
        existing_values = _runtime_value_list(existing)
        new_values = _runtime_value_list(value)
        combined = list(dict.fromkeys(existing_values + new_values))
        merged[key] = combined[0] if len(combined) == 1 else combined
    return merged


def _runtime_value_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _representation_target_priority(target: Any) -> tuple[int, str, str]:
    exported_name = str(getattr(target, "exported_name", "") or "")
    semantic_id = str(getattr(target, "semantic_id", "") or "")
    tokens = {part for part in (*semantic_id.split("."), *exported_name.split("_")) if part}
    if {"format", "layout", "encoding"} & tokens:
        rank = 0
    elif {"dimension", "dimensions", "axis", "axes", "names", "name"} & tokens:
        rank = 2
    else:
        rank = 1
    return (rank, semantic_id, exported_name)


def build_axis_schema_for_modality(
    modality: Literal["wifi", "mmwave", "ble", "uwb", "acoustic"],
) -> AxisSchema:
    """Resolve the canonical sample schema builder for one IO modality."""

    return _CANONICAL_AXIS_SCHEMA_BUILDERS[modality]()


__all__ = [
    "AxisMetadata",
    "AxisSchema",
    "CoordinateAxis",
    "RadioTensor",
    "apply_reader_runtime_contract",
    "build_axis_schema_for_modality",
    "build_reader_axis_schema",
    "resolved_sample_rate",
    "resize_time_axis",
    "SignalMetadata",
    "build_acoustic_waveform_axis_schema",
    "build_ble_rssi_axis_schema",
    "build_radar_axis_schema",
    "build_uwb_cir_axis_schema",
    "build_wifi_csi_axis_schema",
]
