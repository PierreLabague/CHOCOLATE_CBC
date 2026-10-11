"""Reusable plotting helpers shared by the analysis scripts.

Everything here assumes :func:`chocolate.style.apply_style` has already
been called.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .results_io import mean_fluorescence
from .style import FLASH_COLOR, FLASH_LINE_COLOR, zernike_line_style
from .zernike import PRETTY_LABELS

__all__ = [
    "shade_fluorescence",
    "add_flash_background",
    "zernike_legend_handles",
    "add_bottom_legend",
]


def shade_fluorescence(
    axes, time_ms, fluorescence, peak_time_ms=None, max_alpha=0.35, bands=24000
):
    """Shade the background of an axes with the pump fluorescence.

    The shading follows the fluorescence signal in time, from transparent
    where the lamps are off to fully tinted at the peak of the flash. It
    gives an immediate visual reference of when the amplifier is pumped,
    without adding a curve that would compete with the data.

    It is drawn as a series of vertical bands whose vertical extent is
    given in axes coordinates, so it always spans the whole height of the
    frame. That matters in practice: an earlier version sized the shading
    to the limits in force when it was called, and any ``set_ylim`` issued
    afterwards left a strip of unshaded background. The order of the calls
    no longer has any effect.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        Target axes. Its limits may be set before or after this call.
    time_ms : numpy.ndarray
        Time base of the fluorescence signal, in milliseconds.
    fluorescence : numpy.ndarray
        Fluorescence signal, in arbitrary units. It is normalised
        internally, so the absolute scale is irrelevant.
    peak_time_ms : float, optional
        Instant of the flash maximum, marked by a dotted vertical line.
        Defaults to the maximum of ``fluorescence``.
    max_alpha : float, optional
        Opacity of the shading at the peak.
    bands : int, optional
        Number of vertical bands the signal is resampled onto. A few
        hundred is indistinguishable from a continuous gradient and keeps
        the figure light; drawing one band per sample would put thousands
        of patches in the file for no visible gain.

    Returns
    -------
    float
        The instant of the flash maximum that was used, in milliseconds.
    """
    time_ms = np.asarray(time_ms, dtype=float)
    fluorescence = np.asarray(fluorescence, dtype=float)

    finite = np.isfinite(fluorescence)
    if not finite.any():
        return np.nan

    baseline = np.nanpercentile(fluorescence[finite], 5)
    amplitude = np.nanmax(fluorescence[finite]) - baseline
    if amplitude <= 0:
        return np.nan
    normalised = np.clip((fluorescence - baseline) / amplitude, 0.0, 1.0)

    if peak_time_ms is None:
        peak_time_ms = float(time_ms[np.nanargmax(fluorescence)])

    # Resample onto a modest number of bands, then draw each as an
    # axvspan: those span the full height of the axes whatever its limits,
    # now or later, because their vertical extent is expressed as a
    # fraction of the frame rather than in data coordinates.
    edges = np.linspace(float(time_ms[0]), float(time_ms[-1]), bands + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    opacity = np.interp(centres, time_ms, normalised) * max_alpha

    colour = to_rgb(FLASH_COLOR)
    for start, stop, alpha in zip(edges[:-1], edges[1:], opacity):
        if alpha < 0.004:
            # Below this the band is invisible and only weighs on the file.
            continue
        axes.axvspan(
            start, stop, facecolor=colour, alpha=float(alpha), linewidth=0, zorder=0
        )

    axes.axvline(
        peak_time_ms,
        color=FLASH_LINE_COLOR,
        linestyle=":",
        linewidth=1.0,
        zorder=1,
    )
    return peak_time_ms


def zernike_legend_handles(labels, include_flash=True):
    """Build legend handles for a set of aberrations.

    Parameters
    ----------
    labels : sequence of str
        Aberration labels as exported by the SID4.
    include_flash : bool, optional
        Append the two entries describing the fluorescence shading and the
        flash marker.

    Returns
    -------
    handles, texts : list, list
        Ready to be passed to ``figure.legend``.
    """
    handles, texts = [], []
    for label in labels:
        name = str(label)
        handles.append(Line2D([0], [0], linewidth=1.4, **zernike_line_style(name)))
        texts.append(PRETTY_LABELS.get(name, name))

    if include_flash:
        handles.append(Patch(facecolor=FLASH_COLOR, alpha=0.35, linewidth=0))
        texts.append("Pump fluorescence")
        handles.append(
            Line2D([0], [0], color=FLASH_LINE_COLOR, linestyle=":", linewidth=1.0)
        )
        texts.append("Flash maximum")

    return handles, texts


def add_bottom_legend(figure, handles, texts, ncol=4, bottom=0.22):
    """Place a shared legend below the panels, never on top of the data.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
    handles, texts : list
        As returned by :func:`zernike_legend_handles`.
    ncol : int, optional
        Number of legend columns.
    bottom : float, optional
        Fraction of the figure height reserved for the legend.
    """
    figure.subplots_adjust(bottom=bottom)
    return figure.legend(
        handles,
        texts,
        loc="lower center",
        ncol=ncol,
        frameon=True,
        fontsize=7,
        handlelength=2.2,
        columnspacing=1.2,
        bbox_to_anchor=(0.5, 0.005),
    )


def add_flash_background(axes, root, ids, beam="A", max_alpha=0.35):
    """Shade an axes with the average pump fluorescence of a set of shots.

    Convenience wrapper around :func:`chocolate.results_io.mean_fluorescence`
    and :func:`shade_fluorescence`, so that any script can add the flash
    background in a single call.

    Parameters
    ----------
    axes : matplotlib.axes.Axes
        Target axes, with its vertical limits already final.
    root : str or pathlib.Path
        Data directory.
    ids : sequence of str
        Measurements whose fluorescence is averaged.
    beam : {'A', 'B'}, optional
        Which intensity array to use.
    max_alpha : float, optional
        Opacity of the shading at the peak of the flash.

    Returns
    -------
    float
        Instant of the flash maximum in milliseconds, or nan when no
        intensity array was available.
    """
    time_ms, signal = mean_fluorescence(root, ids, beam)
    if signal is None:
        return float("nan")
    return shade_fluorescence(axes, time_ms, signal, max_alpha=max_alpha)
