"""House graphical style for every figure produced by this package.

Importing this module does **not** change anything by itself. Call
:func:`apply_style` once, at the top of an analysis script:

    from chocolate.style import apply_style
    apply_style()

Every figure in the thesis and in this repository follows the same
conventions: Open Sans at 9 pt, inward ticks, no grid, framed legends,
and English labels. Two colormaps are used consistently:

* ``batlow`` (perceptually uniform, colour-vision friendly) for any
  quantity that is not a direct physical image, and for wavefront maps;
* :data:`CMAP_INTENSITY` (black to deep red to orange) for beam
  intensity images, which mimics the appearance of a Nd:glass
  fluorescence image.
"""

from __future__ import annotations

import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

__all__ = [
    "apply_style",
    "CMAP_INTENSITY",
    "CMAP_WAVEFRONT",
    "PRUNE",
    "COLOR_LIST",
    "ZERNIKE_COLORS",
    "ZERNIKE_LINESTYLES",
    "PISTON_COLOR",
    "FLASH_COLOR",
    "FLASH_LINE_COLOR",
    "zernike_line_style",
    "FIGSIZE_SINGLE",
    "FIGSIZE_WIDE",
    "FIGSIZE_TALL",
    "FIGURE_WIDTH",
    "save_figure",
]

# ---------------------------------------------------------------------------
# Institutional colour (Universite Paris-Saclay "Prune"), used for accents
# and for highlighting a single curve among many.
# ---------------------------------------------------------------------------
PRUNE = "#63003C"

# ---------------------------------------------------------------------------
# Figure sizes in inches, matched to the text width of the manuscript.
# FIGSIZE_SINGLE fits one figure per page width, FIGSIZE_WIDE is for
# two-panel rows, FIGSIZE_TALL for stacked panels.
# ---------------------------------------------------------------------------
FIGSIZE_SINGLE = (6.5, 4.0)
FIGSIZE_WIDE = (6.5, 2.6)
FIGSIZE_TALL = (6.5, 7.0)

#: Text width of the manuscript, in inches. Every figure keeps this width so
#: that font sizes stay consistent from one figure to the next; only the
#: height is adapted to the content.
FIGURE_WIDTH = 6.5

# ---------------------------------------------------------------------------
CMAP_INTENSITY = LinearSegmentedColormap.from_list(
    "chocolate_intensity",
    [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.85, 0.12, 0.0), (1.0, 0.55, 0.0)],
)


def _resolve_wavefront_cmap():
    """Return the batlow colormap, falling back to viridis if unavailable.

    ``cmcrameri`` is an optional-looking dependency in practice: some users
    install the package without it. Rather than crashing on import, we warn
    once and fall back to a perceptually uniform matplotlib colormap.
    """
    try:
        from cmcrameri import cm as cmc

        return cmc.batlow
    except ImportError:  # pragma: no cover - depends on the user environment
        warnings.warn(
            "cmcrameri is not installed, falling back to 'viridis' for "
            "wavefront maps. Install it with: pip install cmcrameri",
            stacklevel=2,
        )
        return plt.get_cmap("viridis")


CMAP_WAVEFRONT = _resolve_wavefront_cmap()

# ---------------------------------------------------------------------------
# Qualitative colour cycle for line plots (matplotlib tab10, then a lighter
# extension so that up to 20 curves stay distinguishable).
# ---------------------------------------------------------------------------
COLOR_LIST = [
    "#1f77b4",  # 0  blue
    "#ff7f0e",  # 1  orange
    "#2ca02c",  # 2  green
    "#d62728",  # 3  red
    "#9467bd",  # 4  purple
    "#8c564b",  # 5  brown
    "#e377c2",  # 6  pink
    "#7f7f7f",  # 7  grey
    "#bcbd22",  # 8  olive
    "#17becf",  # 9  cyan
    "#aec7e8",  # 10 light blue
    "#ffbb78",  # 11 light orange
    "#98df8a",  # 12 light green
    "#ff9896",  # 13 light red
    "#c5b0d5",  # 14 light purple
    "#c49c94",  # 15 light brown
    "#f7b6d2",  # 16 light pink
    "#c7c7c7",  # 17 light grey
    "#dbdb8d",  # 18 light olive
    "#9edae5",  # 19 light cyan
]

# ---------------------------------------------------------------------------
# The rcParams of the house style.
# ---------------------------------------------------------------------------
# Canonical colour code of the 14 Zernike aberrations.
#
# One hue per aberration, taken in order from COLOR_LIST, and solid lines
# throughout. Dashes and lightness variations were tried to encode the
# orientation within a family, but they become unreadable as soon as many
# semi-transparent shots are overlaid, which is the normal mode of these
# figures. Black is deliberately absent: it is reserved for the piston.
#
# This palette is the reference for every figure of the project. Use
# zernike_line_style() rather than picking colours by hand.
# ---------------------------------------------------------------------------
ZERNIKE_COLORS = {
    "Tilt X": COLOR_LIST[0],
    "Tilt Y": COLOR_LIST[1],
    "Defocus": COLOR_LIST[2],
    "Astig 0_deg": COLOR_LIST[3],
    "Astig 45_deg": COLOR_LIST[4],
    "Coma X": COLOR_LIST[5],
    "Coma Y": COLOR_LIST[6],
    "Trefoil 0_deg": COLOR_LIST[7],
    "Trefoil 30_deg": COLOR_LIST[8],
    "Spherical Aberr.": COLOR_LIST[9],
    "Second. Astig 0_deg": COLOR_LIST[10],
    "Second. Astig 45_deg": COLOR_LIST[11],
    "Tetrafoil 0_deg": COLOR_LIST[12],
    "Tetrafoil 25.5_deg": COLOR_LIST[13],
}

#: Line style of each aberration. All solid: dashes become illegible once
#: many semi-transparent shots are overlaid.
ZERNIKE_LINESTYLES = {name: "-" for name in ZERNIKE_COLORS}

#: Reserved for the piston, which is never a Zernike aberration here: it is
#: measured by the interferometric diagnostic, so it is set apart visually.
PISTON_COLOR = "#000000"

#: Single hue of the pump fluorescence shading and of the flash marker.
FLASH_COLOR = "#F49A9A"
FLASH_LINE_COLOR = "#D62728"


def zernike_line_style(name):
    """Return ``dict(color=..., linestyle=...)`` for an aberration.

    Parameters
    ----------
    name : str
        Aberration label as exported by the SID4, for instance
        ``'Astig 45_deg'``.

    Returns
    -------
    dict
        Keyword arguments ready to be splatted into a ``plot`` call.
        Unknown names fall back to grey so that a mislabelled column is
        visible rather than silently coloured like a real mode.
    """
    name = str(name)
    return {
        "color": ZERNIKE_COLORS.get(name, "#7f7f7f"),
        "linestyle": ZERNIKE_LINESTYLES.get(name, ":"),
    }


# ---------------------------------------------------------------------------
# Intensity colormap: black -> deep red -> orange.
# ---------------------------------------------------------------------------
RC_PARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Open Sans", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.2,
    "lines.markersize": 5,
    "axes.grid": False,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.frameon": True,
    # Computer Modern for mathtext, so that a symbol such as $\mathcal{Z}$
    # rendered in a figure matches the same symbol typeset in the manuscript.
    "mathtext.fontset": "cm",
    # Default image colormap. Intensity images must override this explicitly
    # with cmap=CMAP_INTENSITY.
    "image.cmap": "batlow",
}


def apply_style(figsize=FIGSIZE_SINGLE):
    """Apply the house graphical style to the current matplotlib session.

    Parameters
    ----------
    figsize : tuple of float, optional
        Default figure size in inches. Individual figures may still override
        it with ``plt.figure(figsize=...)``.

    Notes
    -----
    If Open Sans is not installed on the machine, matplotlib silently falls
    back to DejaVu Sans, whose metrics are close enough that figures remain
    usable. Open Sans is available at https://fonts.google.com/specimen/Open+Sans
    """
    # Register batlow under its short name so that "image.cmap" resolves.
    try:
        from cmcrameri import cm as cmc

        # Matplotlib >= 3.9 removed plt.register_cmap; use the registry.
        if "batlow" not in mpl.colormaps:
            mpl.colormaps.register(cmc.batlow, name="batlow")
    except ImportError:
        pass

    params = dict(RC_PARAMS)
    if "batlow" not in mpl.colormaps:
        params["image.cmap"] = "viridis"
    params["figure.figsize"] = figsize
    mpl.rcParams.update(params)


def save_figure(fig, path, **kwargs):
    """Save a figure with the house defaults (300 dpi, tight bounding box).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path : str or pathlib.Path
        Destination file. The extension selects the format; use ``.pdf`` for
        anything that goes into the manuscript and ``.png`` for quick looks.
    **kwargs
        Forwarded to ``fig.savefig``.
    """
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, **kwargs)
    return path
