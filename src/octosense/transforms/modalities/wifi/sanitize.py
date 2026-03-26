"""WiFi sanitation operators."""

import time

import torch
import torch.nn.functional as F

from octosense.core.contracts import AxisContract
from octosense.core.errors import DimensionError
from octosense.io.semantics.metadata import TransformRecord
from octosense.io.semantics.schema import AxisSchema
from octosense.io.tensor import RadioTensor, is_tracking_meta
from octosense.transforms.backends.cuda.tensor import (
    butterworth_iir,
    filter_along_axis_batched,
    lfilter_initial_state,
)
from octosense.transforms.core.base import BaseTransform
from octosense.transforms.core.registry import registered_operator


def _resolve_repair_indices(
    *,
    metadata,
    subc_count: int,
    explicit_indices: tuple[int, ...] | None,
    metadata_key: str | None,
) -> tuple[int, ...]:
    """Resolve pilot/null repair positions from explicit spec-owned inputs only."""
    if explicit_indices is not None:
        raw_indices: tuple[int, ...] = explicit_indices
    elif metadata_key:
        raw_indices = tuple(int(index) for index in metadata.extra.get(metadata_key, ()))
    else:
        raw_indices = ()
    return tuple(index for index in raw_indices if 0 <= index < subc_count)


def _linear_interp_rows(
    source_x: torch.Tensor,
    source_y: torch.Tensor,
    target_x: torch.Tensor,
) -> torch.Tensor:
    if int(target_x.numel()) == 0:
        return source_y.new_empty((*source_y.shape[:-1], 0))
    if int(source_x.numel()) == 1:
        return source_y[..., :1].expand(*source_y.shape[:-1], int(target_x.numel()))

    right = torch.searchsorted(source_x, target_x, right=True).clamp(1, int(source_x.numel()) - 1)
    left = right - 1
    x0 = source_x[left]
    x1 = source_x[right]
    y0 = source_y[..., left]
    y1 = source_y[..., right]
    weight = ((target_x - x0) / (x1 - x0).clamp_min(torch.finfo(source_x.dtype).eps)).to(source_y.dtype)
    return torch.lerp(y0, y1, weight.unsqueeze(0))


def _cubic_interp_rows(
    source_x: torch.Tensor,
    source_y: torch.Tensor,
    target_x: torch.Tensor,
) -> torch.Tensor:
    if int(target_x.numel()) == 0:
        return source_y.new_empty((*source_y.shape[:-1], 0))
    if int(source_x.numel()) < 4:
        return _linear_interp_rows(source_x, source_y, target_x)

    center = torch.searchsorted(source_x, target_x).clamp(1, int(source_x.numel()) - 2)
    start = (center - 2).clamp(0, int(source_x.numel()) - 4)
    idx = start.unsqueeze(-1) + torch.arange(4, device=source_x.device)
    xs = source_x[idx]
    ys = source_y[:, idx]
    weights = []
    for basis_idx in range(4):
        basis = torch.ones_like(target_x, dtype=source_y.dtype)
        for other_idx in range(4):
            if other_idx == basis_idx:
                continue
            numerator = target_x - xs[:, other_idx]
            denominator = _safe_denominator(
                xs[:, basis_idx] - xs[:, other_idx],
                torch.finfo(source_x.dtype).eps,
            )
            basis = basis * (numerator / denominator).to(source_y.dtype)
        weights.append(basis)
    stacked_weights = torch.stack(weights, dim=-1)
    return (ys * stacked_weights.unsqueeze(0)).sum(dim=-1)


def _interp_rows(
    source_x: torch.Tensor,
    source_y: torch.Tensor,
    target_x: torch.Tensor,
    method: str,
) -> torch.Tensor:
    if method == "linear":
        return _linear_interp_rows(source_x, source_y, target_x)
    return _cubic_interp_rows(source_x, source_y, target_x)


def _safe_denominator(values: torch.Tensor, eps: float) -> torch.Tensor:
    signs = torch.where(values < 0, values.new_tensor(-1.0), values.new_tensor(1.0))
    return torch.where(values.abs() < eps, signs * eps, values)


def _unwrap_phase_rows(phase: torch.Tensor) -> torch.Tensor:
    if phase.shape[-1] <= 1:
        return phase
    diffs = phase[..., 1:] - phase[..., :-1]
    wrapped = (diffs + torch.pi) % (2 * torch.pi) - torch.pi
    wrapped = torch.where((wrapped == -torch.pi) & (diffs > 0), wrapped + 2 * torch.pi, wrapped)
    unwrapped = phase.clone()
    unwrapped[..., 1:] = phase[..., :1] + torch.cumsum(wrapped, dim=-1)
    return unwrapped


def _temporal_filter_coefficients(
    *,
    sample_rate: int,
    lowpass_cutoff_hz: float,
    highpass_cutoff_hz: float,
    lowpass_order: int,
    highpass_order: int,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build WiFi temporal filter coefficients from owner-provided policy values."""
    half_rate = float(sample_rate) / 2.0
    upper_cutoff = min(float(lowpass_cutoff_hz), half_rate - 1.0)
    if upper_cutoff <= 0.0:
        raise ValueError(
            "lowpass_cutoff_hz must remain below Nyquist after clipping, "
            f"got sample_rate={sample_rate}, lowpass_cutoff_hz={lowpass_cutoff_hz}"
        )
    low_num, low_den = butterworth_iir(
        order=int(lowpass_order),
        cutoff_hz=upper_cutoff,
        sample_rate=float(sample_rate),
        btype="low",
        dtype=dtype,
        device=device,
    )
    high_num, high_den = butterworth_iir(
        order=int(highpass_order),
        cutoff_hz=float(highpass_cutoff_hz),
        sample_rate=float(sample_rate),
        btype="high",
        dtype=dtype,
        device=device,
    )
    return low_num, low_den, high_num, high_den


def wifi_preprocess_batch(
    batch: torch.Tensor,
    *,
    time_axis: int,
    subc_axis: int,
    rx_axis: int,
    sample_rate: int,
    lowpass_cutoff_hz: float,
    highpass_cutoff_hz: float,
    lowpass_order: int,
    highpass_order: int,
) -> torch.Tensor:
    """Apply the canonical WiFi sanitize chain to a batched complex tensor."""
    if batch.ndim < 3:
        raise ValueError(f"wifi_preprocess_batch expects at least 3 dims, got shape {tuple(batch.shape)}")
    if not batch.is_complex():
        raise ValueError("wifi_preprocess_batch expects complex CSI input")

    normalized = F.normalize(batch, p=1.0, dim=int(subc_axis), eps=torch.finfo(batch.real.dtype).eps)
    rx_major = normalized.movedim(int(rx_axis), -1).contiguous()
    num_rx = int(rx_major.shape[-1])
    if num_rx == 1:
        cleaned = rx_major
    elif num_rx == 2:
        cleaned = (rx_major[..., 0] * rx_major[..., 1].conj()).unsqueeze(-1)
    else:
        cleaned = rx_major * torch.roll(rx_major.conj(), shifts=-1, dims=-1)
    cleaned = cleaned.movedim(-1, int(rx_axis)).contiguous()

    low_num, low_den, high_num, high_den = _temporal_filter_coefficients(
        sample_rate=sample_rate,
        lowpass_cutoff_hz=lowpass_cutoff_hz,
        highpass_cutoff_hz=highpass_cutoff_hz,
        lowpass_order=lowpass_order,
        highpass_order=highpass_order,
        dtype=torch.float64,
        device=batch.device,
    )
    high_zi = lfilter_initial_state(high_num, high_den)
    low_zi = lfilter_initial_state(low_num, low_den)

    restored = filter_along_axis_batched(
        cleaned,
        axis=int(time_axis),
        stages=((low_num, low_den, low_zi), (high_num, high_den, high_zi)),
    )
    return restored.contiguous()


@registered_operator(required_axes=[])
class SubcarrierL1Normalize(BaseTransform):
    """Normalize complex CSI over the subcarrier axis."""

    def __init__(
        self,
        *,
        subc_axis: str = "subc",
        skip_reader_ids: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self.subc_axis = subc_axis

    @property
    def input_contract(self) -> AxisContract:
        return AxisContract(required_axes=[self.subc_axis], dtype_constraint="complex")

    @property
    def output_contract(self) -> AxisContract:
        return AxisContract(required_axes=[self.subc_axis], dtype_constraint="complex")

    def forward(self, x: RadioTensor) -> RadioTensor:
        self._validate_input(x)
        metadata = x.metadata.copy()
        subc_dim = x.get_axis_index(self.subc_axis)
        data = x.as_tensor()
        metadata.transforms.append(
            TransformRecord(
                name="SubcarrierL1Normalize",
                params={"subc_axis": self.subc_axis},
                timestamp=time.time(),
            )
        )
        normalized = F.normalize(data, p=1.0, dim=subc_dim, eps=torch.finfo(data.real.dtype).eps)
        return RadioTensor(normalized.contiguous(), x.axis_schema, metadata)


@registered_operator(required_axes=[])
class AdjacentRxConjugate(BaseTransform):
    """Multiply each RX stream with its adjacent conjugate counterpart."""

    def __init__(
        self,
        *,
        rx_axis: str = "rx",
        skip_reader_ids: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self.rx_axis = rx_axis

    @property
    def input_contract(self) -> AxisContract:
        return AxisContract(required_axes=[self.rx_axis], dtype_constraint="complex")

    @property
    def output_contract(self) -> AxisContract:
        return AxisContract(dtype_constraint="complex")

    def forward(self, x: RadioTensor) -> RadioTensor:
        self._validate_input(x)
        metadata = x.metadata.copy()
        rx_dim = x.get_axis_index(self.rx_axis)
        data = x.as_tensor().movedim(rx_dim, -1)
        num_rx = int(data.shape[-1])
        if num_rx == 1:
            cleaned = data
        elif num_rx == 2:
            cleaned = (data[..., 0] * data[..., 1].conj()).unsqueeze(-1)
        else:
            cleaned = data * torch.roll(data.conj(), shifts=-1, dims=-1)
        cleaned = cleaned.movedim(-1, rx_dim).contiguous()
        metadata.transforms.append(
            TransformRecord(
                name="AdjacentRxConjugate",
                params={"rx_axis": self.rx_axis},
                timestamp=time.time(),
            )
        )
        return RadioTensor(cleaned, x.axis_schema, metadata)


@registered_operator(required_axes=[])
class TemporalFilter(BaseTransform):
    """Apply one owner-configured low-pass and high-pass cascade along time."""

    def __init__(
        self,
        *,
        time_axis: str = "time",
        sample_rate: int,
        lowpass_cutoff_hz: float,
        highpass_cutoff_hz: float,
        lowpass_order: int,
        highpass_order: int,
        skip_reader_ids: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self.time_axis = time_axis
        self.sample_rate = int(sample_rate)
        self.lowpass_cutoff_hz = float(lowpass_cutoff_hz)
        self.highpass_cutoff_hz = float(highpass_cutoff_hz)
        self.lowpass_order = int(lowpass_order)
        self.highpass_order = int(highpass_order)

    @property
    def input_contract(self) -> AxisContract:
        return AxisContract(required_axes=[self.time_axis], dtype_constraint="complex")

    @property
    def output_contract(self) -> AxisContract:
        return AxisContract(required_axes=[self.time_axis], dtype_constraint="complex")

    def forward(self, x: RadioTensor) -> RadioTensor:
        self._validate_input(x)
        metadata = x.metadata.copy()
        low_num_t, low_den_t, high_num_t, high_den_t = _temporal_filter_coefficients(
            sample_rate=self.sample_rate,
            lowpass_cutoff_hz=self.lowpass_cutoff_hz,
            highpass_cutoff_hz=self.highpass_cutoff_hz,
            lowpass_order=self.lowpass_order,
            highpass_order=self.highpass_order,
            dtype=torch.float64,
            device=x.device,
        )

        time_dim = x.get_axis_index(self.time_axis)
        low_zi = lfilter_initial_state(low_num_t, low_den_t)
        high_zi = lfilter_initial_state(high_num_t, high_den_t)
        data = filter_along_axis_batched(
            x.as_tensor(),
            axis=time_dim,
            stages=((low_num_t, low_den_t, low_zi), (high_num_t, high_den_t, high_zi)),
        )

        metadata.transforms.append(
            TransformRecord(
                name="TemporalFilter",
                params={
                    "time_axis": self.time_axis,
                    "sample_rate": self.sample_rate,
                    "lowpass_cutoff_hz": self.lowpass_cutoff_hz,
                    "highpass_cutoff_hz": self.highpass_cutoff_hz,
                    "lowpass_order": self.lowpass_order,
                    "highpass_order": self.highpass_order,
                },
                timestamp=time.time(),
            )
        )
        return RadioTensor(data.contiguous(), x.axis_schema, metadata)


class WiFiPreprocess(BaseTransform):
    """Stable WiFi sanitize chain shared by multiple preset owners."""

    def __init__(
        self,
        *,
        sample_rate: int,
        lowpass_cutoff_hz: float,
        highpass_cutoff_hz: float,
        lowpass_order: int,
        highpass_order: int,
        skip_reader_ids: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.lowpass_cutoff_hz = float(lowpass_cutoff_hz)
        self.highpass_cutoff_hz = float(highpass_cutoff_hz)
        self.lowpass_order = int(lowpass_order)
        self.highpass_order = int(highpass_order)

    @property
    def input_contract(self) -> AxisContract:
        return AxisContract(required_axes=["time", "subc", "tx", "rx"], dtype_constraint="complex")

    @property
    def output_contract(self) -> AxisContract:
        return AxisContract(required_axes=["time", "subc", "tx", "rx"], dtype_constraint="complex")

    def forward(self, x: RadioTensor) -> RadioTensor:
        self._validate_input(x)
        normalized = SubcarrierL1Normalize()(x)
        cleaned = AdjacentRxConjugate()(normalized)
        filtered = TemporalFilter(
            sample_rate=self.sample_rate,
            lowpass_cutoff_hz=self.lowpass_cutoff_hz,
            highpass_cutoff_hz=self.highpass_cutoff_hz,
            lowpass_order=self.lowpass_order,
            highpass_order=self.highpass_order,
        )(cleaned)
        filtered.metadata.transforms.append(
            TransformRecord(
                name="WiFiPreprocess",
                params={
                    "sample_rate": self.sample_rate,
                    "lowpass_cutoff_hz": self.lowpass_cutoff_hz,
                    "highpass_cutoff_hz": self.highpass_cutoff_hz,
                    "lowpass_order": self.lowpass_order,
                    "highpass_order": self.highpass_order,
                },
                timestamp=time.time(),
            )
        )
        return filtered


@registered_operator(
    required_axes=[],
)
class HampelFilter(BaseTransform):
    """Hampel filter for outlier detection and removal in CSI time series.

    Uses a sliding window approach with Median Absolute Deviation (MAD)
    to detect outliers along the specified axis and replace them with
    the local median value.

    For each point x[i], if |x[i] - median(window)| > threshold * MAD(window),
    it is classified as an outlier and replaced with median(window).

    Operates on the magnitude of complex data.

    .. warning::
        **NOT differentiable.** HampelFilter uses a Python ``for``-loop over
        time steps and non-continuous operations (``median``, ``argsort``,
        boolean indexing via ``torch.where``). Gradients will **not** flow
        through this transform. Do **not** place it inside a training graph
        or after a learnable layer if you need end-to-end backpropagation.
        Use it only as a pre-processing step applied before gradient-based
        training begins.

    Reference:
        Hampel, F.R., "The influence curve and its role in robust estimation",
        JASA, 1974.
    """

    def __init__(
        self,
        axis: str = "time",
        window_size: int = 11,
        threshold: float = 3.0,
    ) -> None:
        """Initialize HampelFilter.

        Args:
            axis: Axis along which to apply the filter.
            window_size: Size of the sliding window (must be odd).
            threshold: Number of MAD deviations to classify as outlier.
        """
        super().__init__()
        if window_size % 2 == 0:
            raise ValueError(
                f"HampelFilter window_size must be odd, got {window_size}. "
                f"Fix: Use an odd window size (e.g., {window_size + 1})."
            )
        self.axis = axis
        self.window_size = window_size
        self.threshold = threshold
        self._mad_scale = 1.4826

    @property
    def input_contract(self) -> AxisContract:
        return AxisContract(
            required_axes=[self.axis],
            dtype_constraint="complex",
        )

    @property
    def output_contract(self) -> AxisContract:
        return AxisContract(
            required_axes=[self.axis],
            dtype_constraint="complex",
        )

    def forward(self, x: RadioTensor) -> RadioTensor:
        self._validate_input(x)
        data = x.as_tensor()

        if not x.axis_schema.has_axis(self.axis):
            raise DimensionError(
                f"HampelFilter requires axis '{self.axis}'.",
                available_axes=list(x.axis_schema.axes),
                suggestion=x.axis_schema.suggest_axis_name(self.axis),
            )
        ax_dim = x.axis_schema.index(self.axis)
        n = data.shape[ax_dim]

        half_win = self.window_size // 2
        data_t = data.movedim(ax_dim, 0)
        magnitude = data_t.abs()
        result = data_t.clone()

        for i in range(n):
            lo = max(0, i - half_win)
            hi = min(n, i + half_win + 1)
            window_mag = magnitude[lo:hi]
            window_csi = data_t[lo:hi]

            med_mag = window_mag.median(dim=0).values
            mad = (window_mag - med_mag.unsqueeze(0)).abs().median(dim=0).values
            sigma = self._mad_scale * mad

            deviation = (magnitude[i] - med_mag).abs()
            is_outlier = deviation > self.threshold * sigma

            win_size = window_mag.shape[0]
            med_idx = win_size // 2
            sorted_idx = window_mag.argsort(dim=0)
            gather_idx = sorted_idx[med_idx : med_idx + 1]
            med_complex = torch.gather(window_csi, 0, gather_idx.expand_as(window_csi[:1])).squeeze(0)

            result[i] = torch.where(is_outlier, med_complex, result[i])

        output = result.movedim(0, ax_dim)

        track_meta = is_tracking_meta()
        new_metadata = x.metadata.copy() if track_meta else x.metadata
        if track_meta:
            new_metadata.transforms.append(
                TransformRecord(
                    name="HampelFilter",
                    params={
                        "axis": self.axis,
                        "window_size": self.window_size,
                        "threshold": self.threshold,
                    },
                    timestamp=time.time(),
                )
            )

        return RadioTensor(
            data=output,
            axis_schema=x.axis_schema,
            metadata=new_metadata,
        )


@registered_operator(
    required_axes=[],
    required_meta=[],
    description=(
        "Interpolate sparse pilot-grouped WiFi CSI onto the reader's canonical dense subcarrier grid."
    ),
)
class PilotSubcarrierInterpolate(BaseTransform):
    """Repair pilot/null subcarriers in-place using explicit interpolation policy.

    This operator does not densify the subcarrier axis. Instead, it:

    - reads a small set of subcarrier indices that should be repaired from
      explicit params or metadata;
    - interpolates only those positions in-place;
    - keeps the existing tensor shape and axis metadata unchanged.
    """

    def __init__(
        self,
        axis_name: str = "subc",
        method: str = "linear",
        repair_indices: tuple[int, ...] | list[int] | None = None,
        metadata_key: str | None = "pilot_indices",
    ) -> None:
        super().__init__()
        if method not in {"cubic", "linear"}:
            raise ValueError("PilotSubcarrierInterpolate only supports method='cubic' or 'linear'")
        self.axis_name = axis_name
        self.method = method
        self.repair_indices = None if repair_indices is None else tuple(int(index) for index in repair_indices)
        self.metadata_key = None if metadata_key is None else str(metadata_key)

    @property
    def input_contract(self) -> AxisContract:
        return AxisContract(
            required_axes=[self.axis_name],
        )

    @property
    def output_contract(self) -> AxisContract:
        return AxisContract(
            required_axes=[self.axis_name],
        )

    def forward(self, x: RadioTensor) -> RadioTensor:
        self._validate_input(x)
        subc_dim = x.get_axis_index(self.axis_name)
        values = x.as_tensor().movedim(subc_dim, -1).contiguous()
        flat = values.reshape(-1, values.shape[-1])
        repair_indices = _resolve_repair_indices(
            metadata=x.metadata,
            subc_count=values.shape[-1],
            explicit_indices=self.repair_indices,
            metadata_key=self.metadata_key,
        )

        if not repair_indices:
            data = x.as_tensor().clone()
        else:
            source_x = torch.arange(values.shape[-1], device=values.device, dtype=torch.float64)
            repair_idx_tensor = torch.tensor(repair_indices, device=values.device, dtype=torch.long)
            repair_x = repair_idx_tensor.to(dtype=torch.float64)
            known_mask = torch.ones(values.shape[-1], device=values.device, dtype=torch.bool)
            known_mask[repair_idx_tensor] = False
            known_x = source_x[known_mask]
            repaired = flat.clone()
            if repaired.is_complex():
                magnitude = repaired.abs().to(torch.float64)
                phase = _unwrap_phase_rows(torch.angle(repaired).to(torch.float64))
                repaired_magnitude = _interp_rows(
                    known_x,
                    magnitude[..., known_mask],
                    repair_x,
                    self.method,
                )
                repaired_phase = _interp_rows(
                    known_x,
                    phase[..., known_mask],
                    repair_x,
                    self.method,
                )
                complex_updates = torch.polar(
                    repaired_magnitude.to(repaired.real.dtype),
                    repaired_phase.to(repaired.real.dtype),
                ).to(repaired.dtype)
                repaired[..., repair_idx_tensor] = complex_updates
                strategy = "magnitude_phase_unwrap"
            else:
                repaired[..., repair_idx_tensor] = _interp_rows(
                    known_x,
                    repaired[..., known_mask].to(torch.float64),
                    repair_x,
                    self.method,
                ).to(repaired.dtype)
                strategy = "amplitude_only"

            data = repaired.reshape(values.shape).movedim(-1, subc_dim).contiguous()

        metadata = x.metadata.copy()
        metadata.extra["pilot_interpolated"] = bool(repair_indices) or bool(
            metadata.extra.get("pilot_interpolated", False)
        )
        metadata.extra["pilot_interpolation"] = {
            "pilot_indices": list(repair_indices),
            "method": self.method,
            "metadata_key": self.metadata_key,
            "strategy": strategy if repair_indices else ("magnitude_phase_unwrap" if x.as_tensor().is_complex() else "amplitude_only"),
            "interpolated": bool(repair_indices),
            "reader_id": metadata.reader_id,
            "capture_device": metadata.capture_device,
            "bandwidth": metadata.bandwidth,
            "subc_count": int(values.shape[-1]),
        }
        metadata.add_transform(
            "PilotSubcarrierInterpolate",
            {
                "axis_name": self.axis_name,
                "method": self.method,
                "metadata_key": self.metadata_key,
                "pilot_indices": list(repair_indices),
            },
            consumed_axes=[self.axis_name],
            produced_axes=[self.axis_name],
            derivation=(
                "Repair pilot/null subcarriers in-place using amplitude interpolation for "
                "real inputs and amplitude plus unwrapped-phase interpolation for complex inputs."
            ),
        )

        axis_metadata = dict(x.axis_schema.axis_metadata)
        schema = AxisSchema(x.axis_schema.axes, axis_metadata=axis_metadata) if axis_metadata else x.axis_schema

        return RadioTensor(data=data, axis_schema=schema, metadata=metadata)


__all__ = [
    "AdjacentRxConjugate",
    "HampelFilter",
    "PilotSubcarrierInterpolate",
    "SubcarrierL1Normalize",
    "TemporalFilter",
    "WiFiPreprocess",
    "wifi_preprocess_batch",
]
