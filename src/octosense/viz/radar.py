"""Radar visualization helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import ArrayLike


def _plot_matrix(
    matrix: ArrayLike,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
) -> tuple[Figure, Axes]:
    array = np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError(f"Radar plotting expects a 2D matrix, got array with shape {array.shape}")
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(array, aspect="auto", origin="lower")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig, ax


def plot_range_doppler(array: ArrayLike, *, title: str = "") -> tuple[Figure, Axes]:
    return _plot_matrix(array, title=title, xlabel="Range", ylabel="Doppler")


def plot_range_angle(array: ArrayLike, *, title: str = "") -> tuple[Figure, Axes]:
    return _plot_matrix(array, title=title, xlabel="Range", ylabel="Angle")
