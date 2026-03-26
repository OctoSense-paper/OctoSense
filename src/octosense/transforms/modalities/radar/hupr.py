"""HuPR raw heatmap and tensor adapters shared by case2 HuPR helpers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from octosense.io.semantics.metadata import SignalMetadata
from octosense.io.semantics.schema import AxisSchema
from octosense.io.tensor import RadioTensor

DEFAULT_NUM_RX = 4
DEFAULT_NUM_ADC_SAMPLES = 256
DEFAULT_NUM_CHIRPS_PER_FRAME = 64 * 3
DEFAULT_PROC_CHIRPS = 64
DEFAULT_GROUP_CHIRPS = 4
DEFAULT_ADC_RATIO = 4
DEFAULT_NUM_ANGLE_BINS = DEFAULT_NUM_ADC_SAMPLES // DEFAULT_ADC_RATIO
DEFAULT_NUM_ELEVATION_BINS = 8
DEFAULT_OUTPUT_CHIRPS = DEFAULT_PROC_CHIRPS // DEFAULT_GROUP_CHIRPS


def _copy_metadata_for_axes(metadata: SignalMetadata, axes: tuple[str, ...]) -> SignalMetadata:
    copied = metadata.copy()
    copied.coords = {name: coord for name, coord in copied.coords.items() if name in set(axes)}
    return copied


def remove_static_clutter(input_val: np.ndarray, *, axis: int = 0) -> np.ndarray:
    reordering = np.arange(len(input_val.shape))
    reordering[0] = axis
    reordering[axis] = 0
    input_val = input_val.transpose(reordering)
    mean = input_val.transpose(reordering).mean(0)
    output_val = input_val - np.expand_dims(mean, axis=0)
    return output_val.transpose(reordering)


def postprocess_fft3d(data_fft: np.ndarray) -> np.ndarray:
    data_fft = np.fft.fftshift(data_fft, axes=(0, 1))
    data_fft = np.transpose(data_fft, (2, 0, 1))
    data_fft = np.flip(data_fft, axis=(1, 2))
    return np.asarray(data_fft, dtype=np.complex64)


def generate_hupr_heatmap(
    frame: np.ndarray,
    *,
    num_rx: int = DEFAULT_NUM_RX,
    num_chirp: int = DEFAULT_NUM_CHIRPS_PER_FRAME,
    num_adc_samples: int = DEFAULT_NUM_ADC_SAMPLES,
    idx_proc_chirp: int = DEFAULT_PROC_CHIRPS,
    num_group_chirp: int = DEFAULT_GROUP_CHIRPS,
    adc_ratio: int = DEFAULT_ADC_RATIO,
    num_angle_bins: int = DEFAULT_NUM_ANGLE_BINS,
    num_ele_bins: int = DEFAULT_NUM_ELEVATION_BINS,
) -> np.ndarray:
    data_radar = np.zeros((num_rx * 2, idx_proc_chirp, num_adc_samples), dtype=np.complex64)
    data_radar2 = np.zeros((num_rx, idx_proc_chirp, num_adc_samples), dtype=np.complex64)

    for idx_rx in range(num_rx):
        for idx_chirp in range(num_chirp):
            if idx_chirp % 3 == 0:
                data_radar[idx_rx, idx_chirp // 3] = frame[idx_rx, idx_chirp]
            if idx_chirp % 3 == 1:
                data_radar2[idx_rx, idx_chirp // 3] = frame[idx_rx, idx_chirp]
            elif idx_chirp % 3 == 2:
                data_radar[idx_rx + num_rx, idx_chirp // 3] = frame[idx_rx, idx_chirp]

    data_radar = remove_static_clutter(np.transpose(data_radar, (1, 0, 2)), axis=0).transpose(1, 0, 2)
    data_radar2 = remove_static_clutter(np.transpose(data_radar2, (1, 0, 2)), axis=0).transpose(1, 0, 2)

    for idx_rx in range(data_radar.shape[0]):
        data_radar[idx_rx] = np.fft.fft2(data_radar[idx_rx])
    for idx_rx in range(data_radar2.shape[0]):
        data_radar2[idx_rx] = np.fft.fft2(data_radar2[idx_rx])

    data_radar = np.pad(
        data_radar,
        ((0, num_angle_bins - data_radar.shape[0]), (0, 0), (0, 0)),
        mode="constant",
    )
    data_radar2 = np.pad(
        data_radar2,
        ((2, num_angle_bins - data_radar2.shape[0] - 2), (0, 0), (0, 0)),
        mode="constant",
    )
    data_merge = np.stack((data_radar, data_radar2))
    data_merge = np.pad(
        data_merge,
        ((0, num_ele_bins - data_merge.shape[0]), (0, 0), (0, 0), (0, 0)),
        mode="constant",
    )

    for idx_chirp in range(idx_proc_chirp):
        for idx_adc in range(num_adc_samples):
            for idx_angle in (2, 3, 4, 5):
                data_merge[:, idx_angle, idx_chirp, idx_adc] = np.fft.fft(
                    data_merge[:, idx_angle, idx_chirp, idx_adc]
                )
            for idx_ele in range(num_ele_bins):
                data_merge[idx_ele, :, idx_chirp, idx_adc] = np.fft.fft(
                    data_merge[idx_ele, :, idx_chirp, idx_adc]
                )

    idx_adc_specific = list(range(94, 30, -1))
    reduced_adc = num_adc_samples // adc_ratio
    data_temp = np.zeros((idx_proc_chirp, reduced_adc, num_angle_bins, num_ele_bins), dtype=np.complex64)
    data_fft_group = np.zeros(
        (idx_proc_chirp // num_group_chirp, reduced_adc, num_angle_bins, num_ele_bins),
        dtype=np.complex64,
    )

    for idx_ele in range(num_ele_bins):
        for idx_rx in range(num_angle_bins):
            for idx_adc in range(reduced_adc):
                data_temp[:, idx_adc, idx_rx, idx_ele] = data_merge[
                    idx_ele, idx_rx, :, idx_adc_specific[idx_adc]
                ]
                data_temp[:, idx_adc, idx_rx, idx_ele] = np.fft.fftshift(
                    data_temp[:, idx_adc, idx_rx, idx_ele],
                    axes=(0,),
                )

    chirp_pad = idx_proc_chirp // num_group_chirp
    write_idx = 0
    for idx_chirp in range(idx_proc_chirp // 2 - chirp_pad // 2, idx_proc_chirp // 2 + chirp_pad // 2):
        data_fft_group[write_idx] = postprocess_fft3d(
            np.transpose(data_temp[idx_chirp], (1, 2, 0))
        )
        write_idx += 1

    return data_fft_group


def normalize_spatial_channels(channels: torch.Tensor, *, epsilon: float = 1e-8) -> torch.Tensor:
    if channels.ndim != 3:
        raise ValueError(f"Expected [channel, H, W], got {tuple(channels.shape)}")
    flattened = channels.reshape(channels.shape[0], -1)
    min_values = flattened.min(dim=1).values.view(-1, 1, 1)
    zeroed = channels - min_values
    max_values = zeroed.reshape(zeroed.shape[0], -1).max(dim=1).values.view(-1, 1, 1)
    normalized = zeroed / max_values.clamp_min(epsilon)
    flattened = normalized.reshape(normalized.shape[0], -1)
    std, mean = torch.std_mean(flattened, dim=1, unbiased=True)
    return (normalized - mean.view(-1, 1, 1)) / std.clamp_min(epsilon).view(-1, 1, 1)


class _HuPRMap(nn.Module):
    def __init__(self, fn) -> None:
        super().__init__()
        self._fn = fn

    def forward(self, sample: object) -> object:
        return self._fn(sample)


def tensorize_sample() -> nn.Module:
    return _HuPRMap(lambda sample: sample.as_tensor() if isinstance(sample, RadioTensor) else sample)


def select_view(view_name: str, *, view_to_index: dict[str, int], tensor_mode: bool = False) -> nn.Module:
    view_index = view_to_index[view_name]
    if tensor_mode:
        return _HuPRMap(lambda sample: sample[view_index])

    def _forward(sample: RadioTensor) -> RadioTensor:
        axis = sample.get_axis_index("view")
        data = sample.as_tensor().select(axis, view_index)
        axes = tuple(axis_name for axis_name in sample.axis_schema.axes if axis_name != "view")
        metadata = _copy_metadata_for_axes(sample.metadata, axes)
        metadata.extra["selected_view"] = view_name
        return RadioTensor(data=data, axis_schema=AxisSchema(axes), metadata=metadata)

    return _HuPRMap(_forward)


def select_center_group(*, tensor_mode: bool = False) -> nn.Module:
    if tensor_mode:
        return _HuPRMap(lambda sample: sample[sample.shape[0] // 2])

    def _forward(sample: RadioTensor) -> RadioTensor:
        axis = sample.get_axis_index("group")
        selected = int(sample.metadata.extra["center_group_index"])
        data = sample.as_tensor().select(axis, selected)
        axes = tuple(axis_name for axis_name in sample.axis_schema.axes if axis_name != "group")
        metadata = _copy_metadata_for_axes(sample.metadata, axes)
        metadata.extra["selected_group_index"] = selected
        return RadioTensor(data=data, axis_schema=AxisSchema(axes), metadata=metadata)

    return _HuPRMap(_forward)


def hupr_heatmap(*, tensor_mode: bool = False, clip_mode: bool = False) -> nn.Module:
    if tensor_mode and clip_mode:
        def _forward(sample: torch.Tensor) -> torch.Tensor:
            raw_clip = np.asarray(sample.detach().cpu().numpy(), dtype=np.complex64)
            leading_shape = raw_clip.shape[:-3]
            frames = raw_clip.reshape(-1, raw_clip.shape[-3], raw_clip.shape[-2], raw_clip.shape[-1])
            heatmaps = np.stack([generate_hupr_heatmap(frame) for frame in frames], axis=0)
            return torch.from_numpy(heatmaps.reshape(*leading_shape, *heatmaps.shape[-4:]))

        return _HuPRMap(_forward)

    if tensor_mode:
        def _forward(sample: torch.Tensor) -> torch.Tensor:
            frame = np.asarray(sample.detach().cpu().numpy(), dtype=np.complex64)
            return torch.from_numpy(generate_hupr_heatmap(frame))

        return _HuPRMap(_forward)

    if clip_mode:
        def _forward(sample: RadioTensor) -> RadioTensor:
            raw_clip = np.asarray(sample.as_tensor().detach().cpu().numpy(), dtype=np.complex64)
            leading_shape = raw_clip.shape[:-3]
            frames = raw_clip.reshape(-1, raw_clip.shape[-3], raw_clip.shape[-2], raw_clip.shape[-1])
            heatmaps = np.stack([generate_hupr_heatmap(frame) for frame in frames], axis=0)
            heatmaps = heatmaps.reshape(*leading_shape, DEFAULT_OUTPUT_CHIRPS, DEFAULT_NUM_ANGLE_BINS, DEFAULT_NUM_ANGLE_BINS, DEFAULT_NUM_ELEVATION_BINS)
            axes = sample.axis_schema.axes[:-3] + ("proc_chirp", "range", "azimuth", "elevation")
            return RadioTensor(
                data=torch.from_numpy(heatmaps),
                axis_schema=AxisSchema(axes),
                metadata=_copy_metadata_for_axes(sample.metadata, axes),
            )

        return _HuPRMap(_forward)

    def _forward(sample: RadioTensor) -> RadioTensor:
        frame = np.asarray(sample.as_tensor().detach().cpu().numpy(), dtype=np.complex64)
        heatmap = generate_hupr_heatmap(frame)
        axes = ("proc_chirp", "range", "azimuth", "elevation")
        return RadioTensor(
            data=torch.from_numpy(heatmap),
            axis_schema=AxisSchema(axes),
            metadata=_copy_metadata_for_axes(sample.metadata, axes),
        )

    return _HuPRMap(_forward)


def take_center_frames(
    *,
    num_frames: int,
    tensor_mode: bool = False,
    axis: int | None = None,
) -> nn.Module:
    if tensor_mode:
        if axis is None:
            raise ValueError("take_center_frames(..., tensor_mode=True) requires an explicit axis.")

        def _forward(sample: torch.Tensor) -> torch.Tensor:
            start = sample.shape[axis] // 2 - num_frames // 2
            return sample.narrow(axis, start, num_frames)

        return _HuPRMap(_forward)

    def _forward(sample: RadioTensor) -> RadioTensor:
        sample_axis = sample.get_axis_index("proc_chirp")
        start = sample.shape[sample_axis] // 2 - num_frames // 2
        indexes = torch.arange(start, start + num_frames, device=sample.as_tensor().device)
        data = sample.as_tensor().index_select(sample_axis, indexes)
        axes = list(sample.axis_schema.axes)
        axes[sample_axis] = "frame"
        axis_names = tuple(axes)
        return RadioTensor(
            data=data,
            axis_schema=AxisSchema(axis_names),
            metadata=_copy_metadata_for_axes(sample.metadata, axis_names),
        )

    return _HuPRMap(_forward)


def split_complex_to_ri(
    *,
    tensor_mode: bool = False,
    insert_after: str | None = None,
    dim: int | None = None,
) -> nn.Module:
    if tensor_mode:
        if dim is None:
            raise ValueError("split_complex_to_ri(..., tensor_mode=True) requires dim.")
        return _HuPRMap(lambda sample: torch.stack((sample.real, sample.imag), dim=dim))

    if insert_after is None:
        raise ValueError("split_complex_to_ri(..., tensor_mode=False) requires insert_after.")

    def _forward(sample: RadioTensor) -> RadioTensor:
        insert_index = sample.get_axis_index(insert_after) + 1
        data = torch.stack((sample.as_tensor().real, sample.as_tensor().imag), dim=insert_index)
        axes = list(sample.axis_schema.axes)
        axes.insert(insert_index, "ri")
        axis_names = tuple(axes)
        return RadioTensor(
            data=data,
            axis_schema=AxisSchema(axis_names),
            metadata=_copy_metadata_for_axes(sample.metadata, axis_names),
        )

    return _HuPRMap(_forward)


def normalize_spatial(
    *,
    tensor_mode: bool = False,
    channel_axis: str | None = None,
    height_axis: str | None = None,
    width_axis: str | None = None,
    permute: tuple[int, ...] | None = None,
    restore: tuple[int, ...] | None = None,
) -> nn.Module:
    if tensor_mode:
        if permute is None or restore is None:
            raise ValueError("normalize_spatial(..., tensor_mode=True) requires permute and restore.")

        def _forward(sample: torch.Tensor) -> torch.Tensor:
            permuted = sample.permute(*permute).contiguous()
            flattened = permuted.reshape(-1, permuted.shape[-3], permuted.shape[-2], permuted.shape[-1])
            normalized = torch.stack([normalize_spatial_channels(item) for item in flattened], dim=0)
            return normalized.reshape(permuted.shape).permute(*restore).contiguous()

        return _HuPRMap(_forward)

    if channel_axis is None or height_axis is None or width_axis is None:
        raise ValueError(
            "normalize_spatial(..., tensor_mode=False) requires channel_axis, height_axis, and width_axis."
        )

    def _forward(sample: RadioTensor) -> RadioTensor:
        data = sample.as_tensor().float()
        channel_dim = sample.get_axis_index(channel_axis)
        height_dim = sample.get_axis_index(height_axis)
        width_dim = sample.get_axis_index(width_axis)
        leading_dims = [
            dim_index
            for dim_index in range(data.ndim)
            if dim_index not in {channel_dim, height_dim, width_dim}
        ]
        permutation = leading_dims + [channel_dim, height_dim, width_dim]
        permuted = data.permute(permutation).contiguous()
        flattened = permuted.reshape(-1, permuted.shape[-3], permuted.shape[-2], permuted.shape[-1])
        normalized = torch.stack([normalize_spatial_channels(item) for item in flattened], dim=0)
        restored = normalized.reshape(permuted.shape)
        inverse = [0] * len(permutation)
        for new_index, old_index in enumerate(permutation):
            inverse[old_index] = new_index
        return RadioTensor(
            data=restored.permute(inverse).contiguous(),
            axis_schema=sample.axis_schema,
            metadata=_copy_metadata_for_axes(sample.metadata, sample.axis_schema.axes),
        )

    return _HuPRMap(_forward)


__all__ = [
    "generate_hupr_heatmap",
    "hupr_heatmap",
    "normalize_spatial_channels",
    "normalize_spatial",
    "select_center_group",
    "select_view",
    "split_complex_to_ri",
    "take_center_frames",
    "tensorize_sample",
]
