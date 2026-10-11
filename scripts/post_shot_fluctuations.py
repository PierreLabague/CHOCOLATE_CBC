"""How the wavefront settles after a slab shot, and two long acquisitions.

A slab shot leaves the amplifier thermally disturbed, and the wavefront
keeps fluctuating for a while afterwards. This script follows that
relaxation over a series of acquisitions taken minute after minute, and
compares it with the quiet state recorded before the perturbation.

Three output figures:

    post_shot_fluctuations.pdf
        Top: mean and shot-to-shot dispersion of a set of Zernike
        coefficients, one point per acquisition, against elapsed time, with
        the undisturbed level as a dashed line.
        Bottom: power spectral density of one chosen aberration, one curve
        per acquisition coloured by that same elapsed time, with the
        undisturbed spectrum in red.
    long_acquisition_<id>_beam<X>.pdf
        One figure per long acquisition: the aberrations against time.

Together they say two different things: the first figure shows how far the
wavefront still wanders and at which frequencies, the others how it drifts
over a much longer stretch.
"""

# =============================================================================
# %% PARAMETERS
# =============================================================================

ROOT = None


BEAM = "A"  # ← "A" or "B"

# ---- Relaxation series ------------------------------------------------------
# The acquisitions, in chronological order. The first is the time origin.
SHOT_IDS = [
    "2025-12-01-16-15",
    "2025-12-01-16-18",
    "2025-12-01-16-19",
    "2025-12-01-16-20",
    "2025-12-01-16-21",
    "2025-12-01-16-22",
    "2025-12-01-16-23",
    "2025-12-01-16-24",
    "2025-12-01-16-25",
    "2025-12-01-16-26",
    "2025-12-01-16-27",
    "2025-12-01-16-28",
    "2025-12-01-16-29",
    "2025-12-01-16-30",
    "2025-12-01-16-31",
    "2025-12-01-16-32",
    "2025-12-01-16-33",
    "2025-12-01-16-34",
    "2025-12-01-16-35",
    "2025-12-01-16-36",
    "2025-12-01-16-37",
    "2025-12-01-16-38",
    "2025-12-01-16-39",
    "2025-12-01-16-40",
]

# Acquisition holding the undisturbed reference, normally the first of the
# series: its beginning was recorded before the slab was fired.
REFERENCE_SHOT = "2025-12-01-16-15"

# Frames analysed per acquisition. The disturbed state comes from the END
# of each acquisition, the undisturbed reference from the BEGINNING of
# REFERENCE_SHOT.
N_FRAMES = 3201

# Aberrations of the relaxation figure, split over two panels. The tilts
# are an order of magnitude larger than the rest, so drawing them together
# would flatten the higher orders into a single line; giving each group its
# own vertical scale is what makes both readable.
TILT_ABERRATIONS = [
    "Tilt X",
    "Tilt Y",
]
HIGHER_ORDER_ABERRATIONS = [
    "Defocus",
    "Astig 45_deg",
    "Coma X",
    "Second. Astig 0_deg",
]

OUTPUT = "post_shot_fluctuations.pdf"

# ---- Long acquisitions ------------------------------------------------------
# One section per acquisition below, each drawing its own figure. They were
# recorded at different frame rates, which the script reads from the
# database rather than assuming; the time axis and the smoothing window
# adapt accordingly.

# How the traces are drawn:
#   "raw"       every sample as recorded, no smoothing and no thinning.
#               This is the honest view: what the sensor actually returned.
#   "smoothed"  the moving average alone, over the window set below. This
#               is the readable view: the drift without the measurement
#               noise on top of it.
# The two are deliberately exclusive. Overlaying them makes it impossible
# to tell, on a printed figure, which of the two a feature belongs to.
LONG_SHOT_MODE = "raw"

# Length of the moving average, in milliseconds. Only used by "smoothed".
# Expressed in time rather than in samples so that it means the same thing
# at every frame rate.
LONG_SHOT_SMOOTH_MS = 1.0

# Aberrations drawn. None takes the fifteen modes of the SID4 export;
# whether piston is among them is decided per acquisition below.
LONG_SHOT_ABERRATIONS = None

# Bring every trace to zero at the start, so the figures show the drift
# rather than the absolute level of the daily wavefront reference.
LONG_SHOT_ZERO_AT_START = False

# Flash instant set by hand, in seconds counted from the START of the
# acquisition, for the shots whose cine range is missing from the database.
# Read the value off a first run: with no reference, the figure is drawn
# with its origin at the start of the recording, so the flash can be
# located by eye and its abscissa entered here. Setting it both moves the
# time origin onto the flash and draws the red marker.
#
# A value given here overrides the database, so it also serves to correct
# a trigger position known to be wrong.
MANUAL_FLASH_TIME = {
    "2025-12-01-17-57": 88.6,
    # "2025-12-01-12-29": 42.5,
}

# Draw the dense traces as a rasterised layer inside the PDF. The axes and
# the text stay vector. A 130 000-sample trace cannot be resolved on a
# 6.5-inch figure anyway — about two thousand pixels at 300 dpi — so the
# rasterised layer looks identical while keeping the file small enough to
# open. Set to False for a fully vector figure, at the cost of a heavy and
# slow PDF.
LONG_SHOT_RASTERIZE = True

SHOW_PROGRESS = True  # progress bars; needs tqdm, ignored if absent


# =============================================================================
# %% IMPORTS AND SETUP
# =============================================================================

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.ndimage import uniform_filter1d

from chocolate.config import load_config
from chocolate.database import load_database
from chocolate.results_io import load_wavefront_results
from chocolate.plotting import FLASH_LINE_COLOR
from chocolate.style import (
    FIGURE_WIDTH,
    PISTON_COLOR,
    apply_style,
    save_figure,
    zernike_line_style,
)
from chocolate.zernike import (
    PRETTY_LABELS,
    ZERNIKE_MODES,
    convert_zernike_mode,
    format_zernike_mode,
)

#: Colour of the undisturbed reference level, drawn dashed on both panels.
REFERENCE_COLOR = "#7f7f7f"

#: Colours of the aberration curves. Unlike the on-shot figures, which draw
#: all fourteen modes and therefore need a fixed colour per mode, these
#: figures show a handful of chosen aberrations; taking the tab palette in
#: order keeps them as far apart as possible and avoids the pale entries at
#: the far end of the fixed code.
SERIES_COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]

apply_style()

config = load_config()
if ROOT is not None:
    config.root = Path(ROOT)
config.check()

db = load_database(config.root)
unit = format_zernike_mode(config.zernike_mode)

print(f"Data directory : {config.root}")
print(f"Beam           : {BEAM}")
print(f"Normalisation  : {config.zernike_mode}")


# =============================================================================
# %% HELPERS
# =============================================================================


def progress(iterable, description, total=None):
    """Wrap an iterable in a progress bar when tqdm is available.

    Reading a couple of dozen acquisitions and computing their spectra
    takes long enough to be worth reporting. tqdm is optional, so its
    absence must never stop the script; the iterable is then returned
    unchanged.
    """
    if not SHOW_PROGRESS:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=description, total=total, unit="shot", leave=False)


def series_color(rank):
    """Colour of the ``rank``-th curve of the relaxation figure."""
    return SERIES_COLORS[rank % len(SERIES_COLORS)]


def long_shot_style(name):
    """Line style of one mode on the long-acquisition figures.

    These figures draw every mode, so they use the project's fixed colour
    code rather than the sequential palette: with fifteen curves, a colour
    that means the same thing from one figure to the next is worth more
    than maximal contrast between neighbours. Piston keeps the black that
    is reserved for it throughout the project.
    """
    if name == "Piston":
        return {"color": PISTON_COLOR, "linestyle": "-"}
    return zernike_line_style(name)


def minutes_since(shot_id, origin_id):
    """Elapsed time between two acquisitions, in minutes.

    Identifiers are ``YYYY-MM-DD-HH-MM``, so the arithmetic is done on the
    date and the clock time directly. Acquisitions spanning midnight are
    not handled, and never occur in this campaign.
    """

    def as_minutes(identifier):
        day = int(identifier[8:10])
        hour = int(identifier[11:13])
        minute = int(identifier[14:16])
        return (day * 24 + hour) * 60 + minute

    return as_minutes(shot_id) - as_minutes(origin_id)


def sampling_frequency(database, shot_ids):
    """Frame rate shared by the acquisitions, in hertz.

    Reading it from the database rather than hard-coding it means the
    spectra are correct even if the series was recorded at another rate.

    Raises
    ------
    ValueError
        If the acquisitions do not share a single frame rate: their spectra
        would not be comparable, and drawing them together would be
        misleading.
    """
    rates = {
        database[shot_id]["PHASICS"]["frame_rate"]
        for shot_id in shot_ids
        if shot_id in database
    }
    if len(rates) != 1:
        raise ValueError(
            f"Les acquisitions n'ont pas toutes la même fréquence "
            f"d'échantillonnage ({sorted(rates)} Hz), leurs spectres ne sont "
            f"pas comparables."
        )
    return float(rates.pop())


def mode_index(labels, name):
    """Column of an aberration, by name.

    Raises
    ------
    SystemExit
        With the list of available names, which is more useful than a bare
        ``ValueError`` when the name was typed in the parameter block.
    """
    try:
        return labels.index(name)
    except ValueError:
        raise SystemExit(
            f"Aberration inconnue : {name!r}\nDisponibles : "
            + ", ".join(repr(x) for x in labels)
        ) from None


#: Frame resolutions of the campaign, as written in the database. Only the
#: taller format carries the fringe band, hence a meaningful piston; the
#: shorter one is cropped on the beams alone.
RESOLUTION_WITH_PISTON = "384x232"
RESOLUTION_WITHOUT_PISTON = "384x184"


def acquisition_time(shot_id, n_samples, stored, flash_at=None):
    """Time base of an acquisition, in seconds, and its origin.

    The stored ``time__W<beam>.npy`` is built by the partitioning stage
    from ``cine_beginning``, ``cine_end`` and the frame rate. Those fields
    are missing for a handful of acquisitions, and for the long ones they
    disagree with the number of frames actually exported, so the stored
    vector cannot be trusted blindly: it has to be checked against the
    data and rebuilt when it does not match.

    Parameters
    ----------
    shot_id : str
    n_samples : int
        Number of frames actually present in the data array.
    stored : numpy.ndarray or None
        The time vector read from disk.
    flash_at : float, optional
        Instant of the flash, in seconds from the start of the
        acquisition, set by hand. Overrides both the stored vector and the
        database. Defaults to the entry of MANUAL_FLASH_TIME, if any.

    Returns
    -------
    time : numpy.ndarray, shape (n_samples,)
        Time of each frame, in seconds.
    triggered : bool
        True when the origin is the laser shot, False when it is the start
        of the acquisition because the trigger position is unknown.
    """
    meta = db.get(shot_id, {}).get("PHASICS", {})
    rate = meta.get("frame_rate")
    rate = (
        float(rate)
        if isinstance(rate, (int, float)) and not isinstance(rate, bool)
        else 0.0
    )

    if flash_at is None:
        flash_at = MANUAL_FLASH_TIME.get(shot_id)

    if flash_at is not None:
        if rate > 0:
            print(
                f"  [info] {shot_id} : origine des temps fixée à la main "
                f"au flash, {flash_at:g} s après le début de l'acquisition."
            )
            return np.arange(n_samples) / rate - float(flash_at), True
        print(
            f"  [info] {shot_id} : cadence inconnue, la référence de flash "
            f"donnée à la main ne peut pas être appliquée."
        )

    if stored is not None and len(stored) == n_samples:
        return np.asarray(stored, dtype=float), True

    first, last = meta.get("cine_beginning"), meta.get("cine_end")
    numeric = all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in (first, last)
    )

    if stored is not None:
        print(
            f"  [info] {shot_id} : le vecteur temps enregistré compte "
            f"{len(stored)} points pour {n_samples} images, il est "
            f"reconstruit depuis la base."
        )

    if rate > 0 and numeric:
        # Keep the offset of the first frame, spread over the real count.
        return first / rate + np.arange(n_samples) / rate, True
    if rate > 0:
        print(
            f"  [info] {shot_id} : plage cine absente de la base, "
            f"l'origine des temps est le début de l'acquisition."
        )
        return np.arange(n_samples) / rate, False

    print(
        f"  [info] {shot_id} : ni plage cine ni cadence exploitables, "
        f"l'axe est gradué en numéro d'image."
    )
    return np.arange(n_samples, dtype=float), False


def moving_average(values, window):
    """Smooth a trace with a centred moving average of ``window`` samples."""
    window = max(1, int(window))
    if window <= 1:
        return values
    return uniform_filter1d(values, size=window, mode="nearest")


def draw_long_acquisition(
    shot_id, include_piston=None, beam=None, flash_at=None, title=None
):
    """Draw one long acquisition: every aberration against time.

    Each acquisition gets its own section below and its own call, rather
    than a loop, so that a figure can be rebuilt on its own while the rest
    of the session stays loaded.

    Parameters
    ----------
    shot_id : str
        Identifier ``YYYY-MM-DD-HH-MM``.
    include_piston : bool, optional
        Draw the piston, in the black reserved for it. None, the default,
        reads the acquisition type from the database and decides.
    beam : {'A', 'B'}, optional
        Defaults to the BEAM of the parameter block.
    flash_at : float, optional
        Instant of the flash, in seconds from the start of the
        acquisition. Overrides MANUAL_FLASH_TIME for this call.
    title : str, optional
        Extra note appended to the figure title.

    Returns
    -------
    pathlib.Path or None
        The file written, or None when the acquisition could not be read.
    """
    beam = BEAM if beam is None else beam

    if shot_id not in db:
        print(f"  [info] {shot_id} absent de data_base.json, ignoré.")
        return None

    loaded = load_wavefront_results(config.root, beam, [shot_id], verbose=True)
    if shot_id not in loaded:
        print(
            f"  [info] {shot_id} : aucun fichier lisible pour le faisceau "
            f"{beam}, figure ignorée."
        )
        return None

    entry = loaded[shot_id]
    mode_labels = [str(t) for t in entry["tag"]]
    values = convert_zernike_mode(entry["data"], mode_labels, mode=config.zernike_mode)

    time, triggered = acquisition_time(
        shot_id, len(values), entry.get("time"), flash_at=flash_at
    )
    duration = float(time[-1] - time[0])

    rate = db[shot_id]["PHASICS"].get("frame_rate")
    rate = (
        float(rate)
        if isinstance(rate, (int, float)) and not isinstance(rate, bool)
        else 0.0
    )

    # Seconds for a long acquisition, milliseconds for a short one, so the
    # axis never reads as a string of tiny or huge numbers.
    if rate <= 0:
        time_axis, time_unit = time, "frame"
    elif duration >= 1.0:
        time_axis, time_unit = time, "s"
    else:
        time_axis, time_unit = time * 1e3, "ms"

    wanted = LONG_SHOT_ABERRATIONS or list(ZERNIKE_MODES)
    if not include_piston:
        wanted = [name for name in wanted if name != "Piston"]
    # A mode absent from this export is skipped rather than fatal: the two
    # acquisition types do not always carry the same columns.
    wanted = [name for name in wanted if name in mode_labels]

    if LONG_SHOT_MODE == "smoothed":
        window = max(1, int(LONG_SHOT_SMOOTH_MS * 1e-3 * rate)) if rate > 0 else 1
        if window < 2:
            # At 400 Hz a millisecond is less than one sample: the average
            # would be a no-op, and saying so is better than a figure
            # labelled "smoothed" that is not.
            print(
                f"  [info] {shot_id} : à {rate:.0f} Hz, "
                f"{LONG_SHOT_SMOOTH_MS:g} ms ne couvre pas deux images, "
                f"aucun lissage n'est appliqué."
            )
            window = None
    elif LONG_SHOT_MODE == "raw":
        window = None
    else:
        raise SystemExit(
            f"LONG_SHOT_MODE doit valoir 'raw' ou 'smoothed', pas "
            f"{LONG_SHOT_MODE!r}."
        )

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 3.5))

    for name in progress(wanted, shot_id, total=len(wanted)):
        column = mode_labels.index(name)
        trace = np.asarray(values[:, column], dtype=float)

        if LONG_SHOT_ZERO_AT_START:
            # Median of the first per cent of the acquisition: robust to a
            # stray first sample, and short enough to be a true baseline.
            baseline = max(1, len(trace) // 100)
            trace = trace - np.nanmedian(trace[:baseline])

        if window is not None:
            trace = moving_average(trace, window)

        ax.plot(
            time_axis,
            trace,
            linewidth=0.9 if window is not None else 0.5,
            alpha=1.0 if window is not None else 0.7,
            rasterized=LONG_SHOT_RASTERIZE,
            **long_shot_style(name),
        )

    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle=":")

    # The flash can only be marked when the trigger position is known. For
    # the acquisitions whose cine range is missing from the database, the
    # time origin is the start of the recording and drawing a line at zero
    # would claim something false.
    if triggered:
        ax.axvline(0.0, color=FLASH_LINE_COLOR, linestyle=":", linewidth=1.1, zorder=4)

    ax.set_xlabel(f"Time ({time_unit})")
    ax.set_ylabel(f"Zernike coefficient {unit} (waves)")

    description = (
        "raw samples"
        if window is None
        else f"{LONG_SHOT_SMOOTH_MS:g} ms moving average ({window} samples)"
    )
    rate_text = f"{rate / 1e3:.3g} kHz" if rate > 0 else "cadence inconnue"
    span_text = f"{duration:.2f} s" if rate > 0 else f"{len(time)} images"
    origin = "" if triggered else ", origine = début d'acquisition"
    heading = (
        f"{shot_id} — beam {beam} — {rate_text}, {span_text}" f"{origin}, {description}"
    )
    if title:
        heading += f" — {title}"
    ax.set_title(heading, fontsize=9)

    # Legend below the panel rather than inside it: with fifteen entries it
    # would cover a good part of the curves wherever it were placed.
    legend_handles = [
        Line2D([0], [0], linewidth=1.1, **long_shot_style(name)) for name in wanted
    ]
    legend_texts = [PRETTY_LABELS.get(name, name) for name in wanted]
    if triggered:
        legend_handles.append(
            Line2D([0], [0], color=FLASH_LINE_COLOR, linestyle=":", linewidth=1.1)
        )
        legend_texts.append("Flash")

    fig.subplots_adjust(bottom=0.3)
    fig.legend(
        legend_handles,
        legend_texts,
        loc="lower center",
        ncol=4,
        fontsize=6.5,
        frameon=True,
        handlelength=1.8,
        columnspacing=1.0,
        bbox_to_anchor=(0.5, 0.005),
    )

    print(
        f"  {shot_id} : {len(time)} images, {rate_text}, {span_text}, "
        f"{len(wanted)} modes, {description}"
    )

    return save_figure(
        fig, config.figures_dir / f"long_acquisition_{shot_id}_beam{beam}.pdf"
    )


# =============================================================================
# %% LOADING THE RELAXATION SERIES
# =============================================================================

frame_rate = sampling_frequency(db, SHOT_IDS)
print(f"Frame rate     : {frame_rate / 1e3:.1f} kHz")
print(
    f"Frames         : {N_FRAMES} "
    f"({N_FRAMES / frame_rate * 1e3:.1f} ms per acquisition)"
)
print()

results = load_wavefront_results(config.root, BEAM, SHOT_IDS, verbose=True)
if not results:
    raise SystemExit(
        f"Aucune des {len(SHOT_IDS)} acquisitions demandées n'a de fichier "
        f"lisible pour le faisceau {BEAM}."
    )

# Keep the order given above rather than the dictionary order.
ordered_ids = [sid for sid in SHOT_IDS if sid in results]

labels = None
blocks = []
shot_ids = []

for sid in progress(ordered_ids, "Conversion", total=len(ordered_ids)):
    entry = results[sid]
    tags = [str(t) for t in entry["tag"]]
    values = convert_zernike_mode(entry["data"], tags, mode=config.zernike_mode)

    if len(values) < N_FRAMES:
        print(
            f"  [info] {sid} ne compte que {len(values)} images, moins que "
            f"les {N_FRAMES} demandées : ignoré."
        )
        continue

    entry["converted"] = values
    blocks.append(values[-N_FRAMES:, :].T)
    shot_ids.append(sid)
    labels = tags

if not blocks:
    raise SystemExit(f"Aucune acquisition ne compte au moins {N_FRAMES} images.")

data = np.array(blocks)
print(f"Loaded         : {len(shot_ids)} acquisitions")

# ---- Undisturbed reference, from the beginning of the reference shot -------
if REFERENCE_SHOT not in results or "converted" not in results[REFERENCE_SHOT]:
    raise SystemExit(
        f"L'acquisition de référence {REFERENCE_SHOT} n'a pas pu être lue."
    )
reference = results[REFERENCE_SHOT]["converted"][:N_FRAMES, :].T

elapsed = np.array([minutes_since(sid, shot_ids[0]) for sid in shot_ids], dtype=float)
print(f"Time span      : {elapsed[0]:.0f} to {elapsed[-1]:.0f} min")
print()


# =============================================================================
# %% FIGURE 1 — Relaxation: level and dispersion, tilts and higher orders
# =============================================================================
#
# The two groups are drawn on separate panels because their amplitudes are
# an order of magnitude apart: sharing an axis would flatten the higher
# orders into a single line at zero. The colours are unique across both
# panels, so one legend below serves them together.

fig = plt.figure(figsize=(FIGURE_WIDTH, 6.8))
gs = fig.add_gridspec(3, 1, height_ratios=[5, 5, 1.1], hspace=0.28)
ax_tilt = fig.add_subplot(gs[0])
ax_high = fig.add_subplot(gs[1], sharex=ax_tilt)
ax_leg = fig.add_subplot(gs[2])
ax_leg.axis("off")

handles = []
texts = []
rank = 0

for axes, group, heading in (
    (ax_tilt, TILT_ABERRATIONS, "Tilts"),
    (ax_high, HIGHER_ORDER_ABERRATIONS, "Higher orders"),
):
    for name in group:
        column = mode_index(labels, name)
        means = data[:, column, :].mean(axis=1)
        deviations = data[:, column, :].std(axis=1)
        colour = series_color(rank)
        rank += 1
        axes.errorbar(
            elapsed,
            means,
            yerr=deviations,
            fmt="+",
            markersize=4,
            capsize=3,
            capthick=1,
            color=colour,
            alpha=0.9,
        )
        # Undisturbed level of the same aberration, for comparison.
        axes.hlines(
            reference[column].mean(),
            xmin=elapsed[0],
            xmax=elapsed[-1],
            color=colour,
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
        )
        handles.append(Line2D([0], [0], color=colour, marker="+", linewidth=1.0))
        texts.append(PRETTY_LABELS.get(name, name))

    axes.axhline(0.0, color="grey", linewidth=0.8, linestyle=":")
    axes.set_ylabel(f"Zernike {unit} (waves)")
    axes.set_title(heading, fontsize=9)

ax_tilt.tick_params(labelbottom=False)
ax_high.set_xlabel("Time since first acquisition (min)")

# The dashed level means the same thing on every curve, so it earns one
# entry rather than one per aberration.
handles.append(Line2D([0], [0], color=REFERENCE_COLOR, linestyle="--", linewidth=1.0))
texts.append("Level before the perturbation")

ax_leg.legend(
    handles,
    texts,
    loc="center",
    ncol=4,
    fontsize=7,
    frameon=True,
    columnspacing=1.2,
    handlelength=2.0,
)

written = [save_figure(fig, config.figures_dir / OUTPUT)]


# =============================================================================
# %% FIGURE 2 — Long acquisition, 2025-12-04-09-28 (high frame rate)
# =============================================================================

print()
print("Long acquisitions")

path = draw_long_acquisition("2025-12-04-09-28")
if path is not None:
    written.append(path)


# =============================================================================
# %% FIGURE 3 — Long acquisition, 2025-12-03-11-58 (lower frame rate)
# =============================================================================
"""
path = draw_long_acquisition("2025-12-03-11-58")
if path is not None:
    written.append(path)

"""
# =============================================================================
# %% FIGURE 4 — Long acquisition, 2025-12-01-17-57 (no piston)
# =============================================================================

# No cine range in the database for this one: run it once as it is, read
# the flash off the figure, then pass its abscissa as flash_at= to place
# the marker and move the time origin onto it.
path = draw_long_acquisition("2025-12-01-17-57")
# path = draw_long_acquisition("2025-12-01-17-57", flash_at=88.0)
if path is not None:
    written.append(path)


# =============================================================================
# %% FIGURE 5 — Long acquisition, 2025-12-01-12-29 (no piston)
# =============================================================================
"""
path = draw_long_acquisition("2025-12-01-12-29")
# path = draw_long_acquisition("2025-12-01-12-29", flash_at=42.5)
if path is not None:
    written.append(path)
"""

# =============================================================================
# %% SHOW
# =============================================================================

print()
print("Figures written:")
for path in written:
    print(f"  {path}")

plt.show()
