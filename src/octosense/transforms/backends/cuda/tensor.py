"""Generic tensor-native CUDA backend primitives for transforms."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_axis(axis: int, ndim: int) -> int:
    normalized = int(axis)
    if normalized < 0:
        normalized += int(ndim)
    if not 0 <= normalized < int(ndim):
        raise ValueError(f"Axis out of range for ndim={ndim}: got {axis}")
    return normalized


def flatten_axes_for_kernel(
    batch: torch.Tensor,
    *,
    input_axes: Sequence[str],
    ordered_axes: Sequence[str],
    merge_from: int,
) -> torch.Tensor:
    """Permute a batch to ``ordered_axes`` and flatten trailing axes for generic kernels."""
    axes = tuple(str(axis) for axis in input_axes)
    ordered = tuple(str(axis) for axis in ordered_axes)
    if batch.ndim != len(axes) + 1:
        raise ValueError(
            "Batch rank mismatch for axis flattening: "
            f"expected {len(axes) + 1}, got shape {tuple(batch.shape)}"
        )
    if merge_from < 1:
        raise ValueError(f"merge_from must be >= 1, got {merge_from}")
    missing_axes = [axis for axis in ordered if axis not in axes]
    if missing_axes:
        raise ValueError(f"Ordered axes must exist in input_axes, missing={tuple(missing_axes)}")
    unresolved = [axis for axis in axes if axis not in ordered]
    if unresolved:
        raise ValueError(f"flatten_axes_for_kernel requires full axis ownership, unresolved={tuple(unresolved)}")

    axis_to_index = {axis: offset + 1 for offset, axis in enumerate(axes)}
    permutation = [0, *(axis_to_index[axis] for axis in ordered)]
    prepared = batch.permute(*permutation).contiguous()
    if prepared.ndim <= merge_from:
        return prepared
    return prepared.flatten(start_dim=merge_from)


def align_tensor_to_contract(
    tensor: torch.Tensor,
    *,
    source_axes: Sequence[str],
    output_contract: Mapping[str, object],
    axis_map: Mapping[str, str | tuple[str, ...]],
) -> torch.Tensor:
    """Regroup a batch-major tensor from ``source_axes`` into a contract-aligned layout."""
    source = tuple(str(axis) for axis in source_axes)
    target_axes = tuple(output_contract.get("axes", ()))
    if tensor.ndim != len(source) + 1:
        raise ValueError(
            "Tensor rank mismatch for contract alignment: "
            f"source_axes={source}, got shape {tuple(tensor.shape)}"
        )
    if not target_axes:
        raise ValueError("align_tensor_to_contract requires output_contract['axes']")

    batch_size = int(tensor.shape[0])
    permutation: list[int] = []
    grouped_sizes: list[int] = []
    used_axes: list[str] = []
    for axis_name in target_axes:
        if axis_name not in axis_map:
            raise ValueError(f"Missing axis_map entry for target axis {axis_name!r}")
        mapped = axis_map[axis_name]
        source_group = (mapped,) if isinstance(mapped, str) else tuple(mapped)
        for source_axis in source_group:
            if source_axis not in source:
                raise ValueError(
                    f"axis_map[{axis_name!r}] references unknown source axis {source_axis!r}"
                )
            permutation.append(source.index(source_axis) + 1)
            used_axes.append(source_axis)
        group_size = 1
        for source_axis in source_group:
            group_size *= int(tensor.shape[source.index(source_axis) + 1])
        grouped_sizes.append(group_size)

    unresolved = [axis for axis in source if axis not in used_axes]
    if unresolved:
        raise ValueError(f"Contract alignment left unresolved source axes {tuple(unresolved)}")

    aligned = tensor.permute(0, *permutation).reshape(batch_size, *grouped_sizes).contiguous()
    fixed_sizes = dict(output_contract.get("fixed_sizes", {}))
    for offset, axis_name in enumerate(target_axes, start=1):
        expected = fixed_sizes.get(axis_name)
        if expected is not None and int(aligned.shape[offset]) != int(expected):
            raise ValueError(
                f"Axis size mismatch for '{axis_name}': expected {int(expected)}, "
                f"got {int(aligned.shape[offset])}"
            )
    return aligned


def sample_zscore_batched(
    tensor: torch.Tensor,
    *,
    dims: Sequence[int] | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Z-score each batch sample over selected non-batch axes."""
    if tensor.ndim < 2:
        raise ValueError(f"sample_zscore_batched expects at least 2 dims, got {tuple(tensor.shape)}")
    normalize_dims = tuple(dims) if dims is not None else tuple(range(1, tensor.ndim - 1))
    if not normalize_dims:
        return tensor
    std, mean = torch.std_mean(tensor, dim=normalize_dims, keepdim=True)
    return (tensor - mean) / std.clamp_min(float(eps))


def _apply_optional_preprocess(
    batch: torch.Tensor,
    *,
    preprocess: Callable[[torch.Tensor], torch.Tensor] | None,
    skip_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if preprocess is None:
        return batch
    if skip_mask is None:
        return preprocess(batch)
    mask = skip_mask.to(device=batch.device, dtype=torch.bool).reshape(-1)
    if int(mask.numel()) != int(batch.shape[0]):
        raise ValueError(f"Skip mask batch mismatch: expected {int(batch.shape[0])}, got {int(mask.numel())}")
    active_mask = ~mask
    if not bool(active_mask.any()):
        return batch
    updated = batch.clone()
    updated[active_mask] = preprocess(batch[active_mask])
    return updated


def normalize_iir_coefficients(
    num: torch.Tensor,
    den: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize and pad IIR coefficients to a shared order."""
    if num.ndim != 1 or den.ndim != 1:
        raise ValueError("IIR coefficients must be 1-D tensors")
    if den.numel() < 1:
        raise ValueError("IIR denominator must contain at least one coefficient")
    den0 = den[0]
    if torch.abs(den0) == 0:
        raise ValueError("IIR denominator leading coefficient must be nonzero")
    if float(den0.item()) != 1.0:
        num = num / den0
        den = den / den0
    order = max(int(num.numel()), int(den.numel()))
    if int(num.numel()) < order:
        num = F.pad(num, (0, order - int(num.numel())))
    if int(den.numel()) < order:
        den = F.pad(den, (0, order - int(den.numel())))
    return num.contiguous(), den.contiguous()


def _poly_from_roots(roots: torch.Tensor) -> torch.Tensor:
    coeffs = torch.ones(1, dtype=roots.dtype, device=roots.device)
    for root in roots:
        updated = torch.zeros(int(coeffs.numel()) + 1, dtype=roots.dtype, device=roots.device)
        updated[:-1] += coeffs
        updated[1:] -= root * coeffs
        coeffs = updated
    return coeffs


def butterworth_iir(
    *,
    order: int,
    cutoff_hz: float,
    sample_rate: float,
    btype: str,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Design a digital Butterworth low/high-pass filter using torch primitives."""
    normalized_type = str(btype).lower()
    if order <= 0:
        raise ValueError(f"Butterworth order must be positive, got {order}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    nyquist = float(sample_rate) / 2.0
    if not 0.0 < float(cutoff_hz) < nyquist:
        raise ValueError(f"cutoff_hz must satisfy 0 < cutoff < Nyquist ({nyquist}), got {cutoff_hz}")
    if normalized_type not in {"low", "high"}:
        raise ValueError(f"Unsupported Butterworth filter type '{btype}'")

    coeff_dtype = torch.complex128
    index = torch.arange(order, dtype=torch.float64, device=device)
    angles = math.pi * (2.0 * index + 1.0 + float(order)) / (2.0 * float(order))
    prototype_poles = torch.exp(1j * angles.to(dtype=torch.complex128))
    prototype_zeros = torch.empty(0, dtype=coeff_dtype, device=device)
    prototype_gain = torch.tensor(1.0, dtype=coeff_dtype, device=device)

    warped = 2.0 * float(sample_rate) * math.tan(math.pi * float(cutoff_hz) / float(sample_rate))
    if normalized_type == "low":
        analog_zeros = prototype_zeros
        analog_poles = prototype_poles * warped
        analog_gain = prototype_gain * (warped ** order)
    else:
        analog_zeros = torch.zeros(order, dtype=coeff_dtype, device=device)
        analog_poles = warped / prototype_poles
        analog_gain = prototype_gain / torch.prod(-prototype_poles)

    fs2 = torch.tensor(2.0 * float(sample_rate), dtype=coeff_dtype, device=device)
    degree = int(analog_poles.numel()) - int(analog_zeros.numel())
    digital_zeros = (fs2 + analog_zeros) / (fs2 - analog_zeros) if int(analog_zeros.numel()) else analog_zeros
    if degree > 0:
        digital_zeros = torch.cat(
            [digital_zeros, -torch.ones(degree, dtype=coeff_dtype, device=device)],
            dim=0,
        )
    digital_poles = (fs2 + analog_poles) / (fs2 - analog_poles)
    digital_gain = analog_gain * torch.prod(fs2 - analog_zeros) / torch.prod(fs2 - analog_poles)

    num = (_poly_from_roots(digital_zeros) * digital_gain).real.to(dtype=dtype)
    den = _poly_from_roots(digital_poles).real.to(dtype=dtype)
    return normalize_iir_coefficients(num.contiguous(), den.contiguous())


def lfilter_initial_state(
    num: torch.Tensor,
    den: torch.Tensor,
) -> torch.Tensor:
    """Return the steady-state initial condition used by zero-phase IIR filtering."""
    num, den = normalize_iir_coefficients(num, den)
    order = int(num.numel()) - 1
    if order <= 0:
        return num.new_empty((0,))

    coeff_dtype = torch.promote_types(num.dtype, den.dtype)
    num = num.to(dtype=coeff_dtype)
    den = den.to(dtype=coeff_dtype)
    identity = torch.eye(order, dtype=coeff_dtype, device=num.device)
    companion = torch.zeros((order, order), dtype=coeff_dtype, device=num.device)
    if order > 1:
        companion[1:, :-1] = torch.eye(order - 1, dtype=coeff_dtype, device=num.device)
    companion[0, :] = -den[1:]
    transition = companion.transpose(0, 1)
    drive = num[1:] - den[1:] * num[0]
    return torch.linalg.solve(identity - transition, drive).contiguous()


def gaussian_window(
    window_length: int,
    *,
    std: float,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create a periodic Gaussian window aligned with SciPy STFT defaults."""
    if window_length <= 0:
        raise ValueError(f"window_length must be positive, got {window_length}")
    if std <= 0:
        raise ValueError(f"std must be positive, got {std}")
    signal_windows = getattr(getattr(torch, "signal", None), "windows", None)
    gaussian = getattr(signal_windows, "gaussian", None)
    if gaussian is not None:
        window = gaussian(
            window_length,
            std=float(std),
            sym=False,
            dtype=dtype,
            device=device,
        )
        return window.contiguous()
    center = 0.5 * float(window_length)
    positions = torch.arange(window_length, dtype=torch.float64, device=device) - center
    window = torch.exp(-0.5 * (positions / float(std)) ** 2)
    return window.to(dtype=dtype).contiguous()


def odd_extension_batched(values: torch.Tensor, edge: int) -> torch.Tensor:
    """Mirror-extend `[N, T]` signals using SciPy-style odd extension."""
    if values.ndim != 2:
        raise ValueError(f"Odd extension expects [N, T] input, got shape {tuple(values.shape)}")
    if edge < 1:
        return values
    if int(values.shape[1]) <= edge:
        raise ValueError(
            "Temporal filter input is shorter than filtfilt pad length: "
            f"time={int(values.shape[1])}, padlen={edge}"
        )
    left_end = values[:, :1]
    left_ext = values[:, 1 : edge + 1].flip(dims=(1,))
    right_end = values[:, -1:]
    right_ext = values[:, -(edge + 1) : -1].flip(dims=(1,))
    return torch.cat((2 * left_end - left_ext, values, 2 * right_end - right_ext), dim=1)


def lfilter_batched(
    values: torch.Tensor,
    *,
    num: torch.Tensor,
    den: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one causal IIR filter to a batch of flattened signals."""
    if values.ndim != 2:
        raise ValueError(f"Batched lfilter expects [N, T] input, got shape {tuple(values.shape)}")
    num, den = normalize_iir_coefficients(num, den)
    coeff_dtype = values.dtype if values.is_complex() else values.real.dtype
    num = num.to(device=values.device, dtype=coeff_dtype)
    den = den.to(device=values.device, dtype=coeff_dtype)
    if values.is_complex():
        num = num.to(dtype=values.dtype)
        den = den.to(dtype=values.dtype)

    order = int(num.numel()) - 1
    if order <= 0:
        output = values * num[0]
        return output, values.new_empty((int(values.shape[0]), 0))

    if initial_state is None:
        state = values.new_zeros((int(values.shape[0]), order))
    else:
        state = initial_state.to(device=values.device, dtype=values.dtype).clone()
        if state.ndim == 1:
            state = state.unsqueeze(0).expand(int(values.shape[0]), -1).clone()
        expected_shape = (int(values.shape[0]), order)
        if tuple(state.shape) != expected_shape:
            raise ValueError(
                f"Initial state shape mismatch: expected {expected_shape}, got {tuple(state.shape)}"
            )

    output = torch.empty_like(values)
    num_mid = num[1:-1]
    den_mid = den[1:-1]
    for index in range(int(values.shape[1])):
        sample = values[:, index]
        response = num[0] * sample + state[:, 0]
        output[:, index] = response
        if order > 1:
            tail = state[:, 1:].clone()
            state[:, :-1] = (
                tail
                + num_mid.unsqueeze(0) * sample.unsqueeze(1)
                - den_mid.unsqueeze(0) * response.unsqueeze(1)
            )
        state[:, -1] = num[-1] * sample - den[-1] * response
    return output, state


def filtfilt_batched(
    values: torch.Tensor,
    *,
    num: torch.Tensor,
    den: torch.Tensor,
    zi: torch.Tensor,
    padlen: int | None = None,
) -> torch.Tensor:
    """Apply zero-phase IIR filtering to `[N, T]` signals."""
    if values.ndim != 2:
        raise ValueError(f"Batched filtfilt expects [N, T] input, got shape {tuple(values.shape)}")
    ntaps = max(int(num.numel()), int(den.numel()))
    edge = ntaps * 3 if padlen is None else int(padlen)
    extended = odd_extension_batched(values, edge) if edge > 0 else values
    initial = zi.to(device=values.device, dtype=values.dtype)
    forward_state = initial.unsqueeze(0) * extended[:, :1]
    forward, _ = lfilter_batched(extended, num=num, den=den, initial_state=forward_state)
    reversed_forward = forward.flip(dims=(1,))
    backward_state = initial.unsqueeze(0) * reversed_forward[:, :1]
    backward, _ = lfilter_batched(
        reversed_forward,
        num=num,
        den=den,
        initial_state=backward_state,
    )
    filtered = backward.flip(dims=(1,))
    if edge > 0:
        filtered = filtered[:, edge:-edge]
    return filtered.contiguous()


def filtfilt_cascade_batched(
    values: torch.Tensor,
    *,
    stages: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    padlen: int | None = None,
) -> torch.Tensor:
    """Apply a sequence of zero-phase IIR stages to flattened `[N, T]` signals."""
    filtered = values
    for num, den, zi in stages:
        filtered = filtfilt_batched(filtered, num=num, den=den, zi=zi, padlen=padlen)
    return filtered.contiguous()


def filter_along_axis_batched(
    batch: torch.Tensor,
    *,
    axis: int,
    stages: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    padlen: int | None = None,
) -> torch.Tensor:
    """Apply a zero-phase IIR cascade along one axis of a batched tensor."""
    if batch.ndim < 1:
        raise ValueError("filter_along_axis_batched expects at least one dimension")
    axis = _normalize_axis(axis, batch.ndim)
    original_dtype = batch.dtype
    moved = batch.movedim(axis, 1).contiguous()
    batch_size = int(moved.shape[0])
    axis_length = int(moved.shape[1])
    flattened = moved.to(torch.complex128).reshape(batch_size, axis_length, -1)
    sequences = flattened.permute(0, 2, 1).reshape(-1, axis_length)
    filtered = filtfilt_cascade_batched(sequences, stages=stages, padlen=padlen)
    restored = filtered.reshape(batch_size, flattened.shape[2], axis_length).permute(0, 2, 1)
    return restored.reshape_as(moved).movedim(1, axis).to(original_dtype).contiguous()


def temporal_filter_batched(
    batch: torch.Tensor,
    *,
    low_num: torch.Tensor,
    low_den: torch.Tensor,
    filter_num: torch.Tensor,
    filter_den: torch.Tensor,
    low_zi: torch.Tensor,
    filter_zi: torch.Tensor,
) -> torch.Tensor:
    """Apply a two-stage zero-phase IIR cascade along axis 1 of `[B, T, ...]` tensors."""
    if batch.ndim < 2:
        raise ValueError(f"Temporal filter expects at least 2 dims, got shape {tuple(batch.shape)}")
    return filter_along_axis_batched(
        batch,
        axis=1,
        stages=((low_num, low_den, low_zi), (filter_num, filter_den, filter_zi)),
    )


def crop_or_pad_axis(
    values: torch.Tensor,
    *,
    axis: int,
    target_length: int,
) -> torch.Tensor:
    """Center-crop or zero-pad one axis to a fixed target length."""
    if target_length <= 0:
        raise ValueError(f"target_length must be positive, got {target_length}")
    axis = _normalize_axis(axis, values.ndim)
    current = int(values.shape[axis])
    if current == target_length:
        return values
    if current > target_length:
        start = (current - target_length) // 2
        stop = start + target_length
        slices = [slice(None)] * values.ndim
        slices[axis] = slice(start, stop)
        return values[tuple(slices)].contiguous()
    padding = [0] * (2 * values.ndim)
    padding[2 * (values.ndim - axis) - 1] = target_length - current
    return F.pad(values, tuple(padding)).contiguous()


def stft_magnitude_batched(
    sequences: torch.Tensor,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
    freq_indices: torch.Tensor | None = None,
    roll_shift: int = 0,
) -> torch.Tensor:
    """Compute magnitude STFT for flattened `[N, T]` signals."""
    if sequences.ndim != 2:
        raise ValueError(f"STFT kernel expects [N, T] input, got shape {tuple(sequences.shape)}")
    if n_fft <= 0 or hop_length <= 0 or win_length <= 0:
        raise ValueError(
            "STFT parameters must be positive: "
            f"n_fft={n_fft}, hop_length={hop_length}, win_length={win_length}"
        )
    if int(window.numel()) != int(win_length):
        raise ValueError(f"Window length mismatch: expected {win_length}, got {int(window.numel())}")
    spectra = torch.stft(
        sequences.to(torch.complex64),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window.to(device=sequences.device, dtype=sequences.real.dtype),
        center=True,
        pad_mode="constant",
        normalized=False,
        onesided=False,
        return_complex=True,
    )
    if freq_indices is not None:
        spectra = spectra[:, freq_indices.to(device=sequences.device), :]
    if roll_shift:
        spectra = torch.roll(spectra, shifts=int(roll_shift), dims=1)
    window_sum = window.sum().to(device=sequences.device, dtype=spectra.real.dtype).clamp_min(1e-8)
    return (spectra / window_sum).abs().to(torch.float32).contiguous()


def spectrogram_tensor_batched(
    batch: torch.Tensor,
    *,
    input_axes: Sequence[str],
    temporal_axis: str,
    spectral_axis: str,
    link_axes: Sequence[str],
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
    freq_indices: torch.Tensor | None = None,
    roll_shift: int = 0,
    target_time_bins: int | None = None,
) -> torch.Tensor:
    """Compute a generic batch-major spectrogram tensor without modality policy."""
    flattened = flatten_axes_for_kernel(
        batch.to(torch.complex64),
        input_axes=input_axes,
        ordered_axes=(temporal_axis, spectral_axis, *tuple(link_axes)),
        merge_from=3,
    )
    if flattened.ndim != 4:
        raise ValueError(
            "spectrogram_tensor_batched expects flattened layout [B, T, S, L], "
            f"got shape {tuple(flattened.shape)}"
        )
    batch_size, time_len, spectral_len, link_size = flattened.shape
    flattened = flattened - flattened.mean(dim=1, keepdim=True)
    sequences = flattened.permute(0, 2, 3, 1).reshape(-1, time_len)
    amplitude = stft_magnitude_batched(
        sequences,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        freq_indices=freq_indices,
        roll_shift=roll_shift,
    )
    amplitude = amplitude.reshape(
        batch_size,
        spectral_len,
        link_size,
        amplitude.shape[1],
        amplitude.shape[2],
    )
    amplitude = amplitude.permute(0, 4, 3, 1, 2).contiguous()
    if target_time_bins is not None:
        amplitude = crop_or_pad_axis(amplitude, axis=1, target_length=int(target_time_bins))
    return amplitude


class AxisMappedTensorRunner(nn.Module):
    """Generic runner that executes a tensor kernel and aligns its output to a target contract."""

    def __init__(
        self,
        *,
        input_axes: Sequence[str],
        source_axes: Sequence[str],
        kernel: Callable[[torch.Tensor], torch.Tensor],
        output_contract: Mapping[str, object] | None = None,
        axis_map: Mapping[str, str | tuple[str, ...]] | None = None,
        preprocess: Callable[[torch.Tensor], torch.Tensor] | None = None,
        normalize: Callable[[torch.Tensor], torch.Tensor] | None = None,
        project: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.input_axes = tuple(str(axis) for axis in input_axes)
        self.source_axes = tuple(str(axis) for axis in source_axes)
        self.kernel = kernel
        self.output_contract = dict(output_contract) if output_contract is not None else None
        self.axis_map = {str(key): value for key, value in (axis_map or {}).items()}
        self.preprocess = preprocess
        self.normalize = normalize
        self.project = project

        if self.output_contract is not None:
            target_axes = tuple(self.output_contract.get("axes", ()))
            if not target_axes:
                raise ValueError("AxisMappedTensorRunner requires output_contract['axes'] when output_contract is set")
            missing_axes = [axis for axis in target_axes if axis not in self.axis_map]
            if missing_axes:
                raise ValueError(f"Missing AxisMappedTensorRunner axis_map entries for {missing_axes}")

            used_source_axes: list[str] = []
            for mapped in self.axis_map.values():
                group = (mapped,) if isinstance(mapped, str) else tuple(mapped)
                used_source_axes.extend(group)
            unresolved = [axis for axis in self.source_axes if axis not in used_source_axes]
            if unresolved:
                raise ValueError(
                    "AxisMappedTensorRunner source axes are not fully mapped: "
                    f"source_axes={self.source_axes}, unresolved={tuple(unresolved)}"
                )

    def forward(self, batch: torch.Tensor, skip_mask: torch.Tensor | None = None) -> torch.Tensor:
        expected_ndim = len(self.input_axes) + 1
        if batch.ndim != expected_ndim:
            raise ValueError(
                "AxisMappedTensorRunner input rank mismatch: "
                f"expected batch + {self.input_axes}, got shape {tuple(batch.shape)}"
            )
        batch = _apply_optional_preprocess(batch, preprocess=self.preprocess, skip_mask=skip_mask)
        tensor = self.kernel(batch)
        if self.normalize is not None:
            tensor = self.normalize(tensor)
        if self.project is not None:
            tensor = self.project(tensor)
        if self.output_contract is None:
            return tensor.contiguous()
        return align_tensor_to_contract(
            tensor,
            source_axes=self.source_axes,
            output_contract=self.output_contract,
            axis_map=self.axis_map,
        )


__all__ = [
    "AxisMappedTensorRunner",
    "align_tensor_to_contract",
    "butterworth_iir",
    "crop_or_pad_axis",
    "flatten_axes_for_kernel",
    "filter_along_axis_batched",
    "filtfilt_batched",
    "filtfilt_cascade_batched",
    "gaussian_window",
    "lfilter_batched",
    "lfilter_initial_state",
    "normalize_iir_coefficients",
    "odd_extension_batched",
    "sample_zscore_batched",
    "spectrogram_tensor_batched",
    "stft_magnitude_batched",
    "temporal_filter_batched",
]
