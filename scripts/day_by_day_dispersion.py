"""Day-by-day dispersion of an aberration, against the pooled one.

The wavefront reference of the SID4 is set once each morning, and the laser
chain is realigned with it. Two shots taken on the same day therefore share
an alignment that two shots taken a week apart do not, and a dispersion
computed over the whole campaign mixes two different things: how much a
shot wanders within a day, and how much the daily setup itself moves.

This script separates them. For a chosen aberration it draws, at every
instant, the mean over the shots of each day with its one-sigma band, and
the same over every shot of every retained day in black.

Reading it: coloured means well apart while each band stays narrow says
the setup is repeatable within a day and moves between days, so a daily
recalibration would buy most of the stability. Coloured means on top of
each other, with the black band no wider than theirs, says the day plays
no part and the scatter is intrinsic to the shots.

Days with too few shots are excluded from both, since a standard deviation
over three shots says almost nothing and would drag the pooled figure with
it.

Run this script directly, or cell by cell.
"""

# =============================================================================
# %% PARAMETERS
# =============================================================================

ROOT = None


# Aberrations to look at, one figure each. Any label of the SID4 export
# works; see chocolate.zernike.ZERNIKE_MODES for the full list.
ABERRATIONS = ["Tilt Y"]

# What the dispersion is computed on:
#   "differential"  beam A minus beam B, the quantity combining depends on
#   "A" or "B"      one beam on its own, which says where the motion is
QUANTITY = "differential"

# Instant every trace is referenced to, in milliseconds.
TIME_REF_MS = -0.5

# A day contributing fewer shots than this is left out of the per-day
# curves AND of the pooled one. A standard deviation over three shots is
# mostly noise, and letting such a day into the pool would move the black
# curve for no good reason.
MIN_SHOTS_PER_DAY = 5

# ---- Shot selection ---------------------------------------------------------
CINE_BEGINNING_RANGE = (-4000, -500)  # inclusive bounds, in frames
CINE_END_MAX = 4000
VALID_FRAME_RATES = {66000, 82000}

# Amplifier configuration, and it is mandatory: leaving it open mixes
# shots that have nothing in common. A set that contains both fired and
# unfired shots has a dispersion dominated by that difference, and the
# day-to-day term this script is after would be buried under it.
#
# All four off is the default because it is the clean case for this
# measurement: no flash, no thermal disturbance, so what is left is the
# repeatability of the alignment itself, which is exactly the question.
AMPLIFIERS = {"45A": 0, "45B": 0, "108A": 1, "108B": 1}  # slabs fired
# AMPLIFIERS = {"45A": 1, "45B": 1, "108A": 0, "108B": 0}   # rods fired
# AMPLIFIERS = None                                        # no condition

EXCLUDED_IDS = {}

# Pump fluorescence shaded behind the curves, in red, with a dotted line
# at the flash. It comes from the IA / IB traces of the interferometric
# branch; if those are missing for the selected shots the shading is simply
# skipped and the legend does not promise it.
SHOW_FLUORESCENCE = True
FLASH_SHADE_ALPHA = 0.22

# Opacity of the one-sigma bands. With several days drawn together they
# overlap, so this stays low; the means are what carry the figure.
BAND_ALPHA = 0.18

# ---- Axes, one entry per aberration -----------------------------------------
# A missing entry means automatic scaling. Bounds are in milliseconds and
# in waves.
AXES = {
    "Tilt Y": {"xlim": (-0.7, 0.20), "ylim": (-0.1, 0.2)},
}

LEGEND_FONTSIZE = 6.5
TITLE_FONTSIZE = 9.0

SHOW_PROGRESS = True


# =============================================================================
# %% IMPORTS AND SETUP
# =============================================================================

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from chocolate.config import load_config
from chocolate.database import load_database, select_ids, shot_date
from chocolate.efficiency import zero_at_reference
from chocolate.results_io import (
    ZERNIKE_COLUMNS,
    load_beam_pair,
    mean_fluorescence,
)
from chocolate.style import COLOR_LIST, FIGURE_WIDTH, apply_style, save_figure
from chocolate.zernike import PRETTY_LABELS, format_zernike_mode

#: The pooled curve is black, which is otherwise reserved for the piston;
#: no piston appears here, and the pooled figure is the one the eye should
#: go to first.
COLOR_POOLED = "black"

#: Everything about the flash is red: the shading of the pump fluorescence
#: and the line marking its maximum.
COLOR_FLASH = "#D62728"

apply_style()

config = load_config()
if ROOT is not None:
    config.root = Path(ROOT)
config.check()

db = load_database(config.root)
unit = format_zernike_mode(config.zernike_mode)

print(f"Data directory : {config.root}")
print(f"Normalisation  : {config.zernike_mode}")
print(f"Quantity       : {QUANTITY}")
print(f"Reference time : {TIME_REF_MS} ms")
print(f"Minimum shots  : {MIN_SHOTS_PER_DAY} per day")
print()


# =============================================================================
# %% SHOT SELECTION
# =============================================================================

conditions = [
    lambda _, meta: isinstance(meta["PHASICS"].get("cine_beginning"), (int, float)),
    lambda _, meta: isinstance(meta["PHASICS"].get("cine_end"), (int, float)),
    lambda _, meta: (
        CINE_BEGINNING_RANGE[0]
        <= meta["PHASICS"]["cine_beginning"]
        <= CINE_BEGINNING_RANGE[1]
    ),
    lambda _, meta: meta["PHASICS"]["cine_end"] <= CINE_END_MAX,
    lambda _, meta: meta["PHASICS"]["frame_rate"] in VALID_FRAME_RATES,
]

if not AMPLIFIERS:
    raise SystemExit(
        "AMPLIFIERS ne peut pas être vide : la configuration des "
        "amplificateurs doit être choisie explicitement, sans quoi la "
        "sélection mélange des tirs sans rapport."
    )

for amplifier, state in AMPLIFIERS.items():
    # The default argument freezes the loop variables: without it every
    # lambda would close over the last pair and test the same thing.
    conditions.append(lambda _, meta, a=amplifier, s=state: meta["AMPLI"].get(a) == s)

ids = select_ids(db, conditions, exclude=EXCLUDED_IDS)
print(
    "Amplificateurs demandés : "
    + ", ".join(f"{name}={state}" for name, state in AMPLIFIERS.items())
)
print(f"Selected       : {len(ids)} tirs")
if not ids:
    raise SystemExit("Aucun tir ne satisfait les conditions ci-dessus.")

# What was actually selected, so that a condition silently matching the
# wrong thing shows up here rather than in the figure.
found = {}
for shot in ids:
    ampli = db[shot]["AMPLI"]
    key = tuple(ampli.get(name) for name in ("45A", "45B", "108A", "108B"))
    found[key] = found.get(key, 0) + 1
print("Configurations présentes (45A, 45B, 108A, 108B) :")
for key, count in sorted(found.items(), key=lambda item: -item[1]):
    print(f"  {key} : {count} tirs")
if len(found) > 1:
    raise SystemExit(
        "Plusieurs configurations dans la sélection, ce qui ne devrait pas "
        "arriver avec AMPLIFIERS renseigné. Vérifiez les champs AMPLI de la "
        "base."
    )


# =============================================================================
# %% LOADING
# =============================================================================
#
# Everything is loaded in one call so that every shot, whatever its day,
# lands on the same time base. Grouping by day afterwards is then just a
# matter of slicing the first axis.

data_a, data_b, common_time, labels, shot_ids = load_beam_pair(
    config.root,
    ids,
    zernike_mode=config.zernike_mode,
    columns=ZERNIKE_COLUMNS,
    verbose=True,
)
labels = [str(name) for name in labels]

if QUANTITY == "differential":
    values = data_a - data_b
elif QUANTITY == "A":
    values = data_a
elif QUANTITY == "B":
    values = data_b
else:
    raise SystemExit(
        f"QUANTITY doit valoir 'differential', 'A' ou 'B', pas {QUANTITY!r}."
    )

values = zero_at_reference(values, common_time, TIME_REF_MS)
time_ms = common_time * 1e3

print(
    f"Loaded         : {values.shape[0]} tirs, {values.shape[1]} modes, "
    f"{values.shape[2]} instants"
)
print()


# =============================================================================
# %% GROUPING BY DAY
# =============================================================================

by_day = defaultdict(list)
for index, shot in enumerate(shot_ids):
    by_day[shot_date(shot)].append(index)

kept_days = sorted(
    day for day, rows in by_day.items() if len(rows) >= MIN_SHOTS_PER_DAY
)
dropped_days = sorted(
    day for day, rows in by_day.items() if len(rows) < MIN_SHOTS_PER_DAY
)

print("Tirs par jour")
for day in sorted(by_day):
    mark = "" if day in kept_days else f"   écarté (< {MIN_SHOTS_PER_DAY})"
    print(f"  {day} : {len(by_day[day]):2d} tirs{mark}")

if not kept_days:
    raise SystemExit(
        f"Aucun jour n'atteint {MIN_SHOTS_PER_DAY} tirs. Baissez "
        f"MIN_SHOTS_PER_DAY ou élargissez la sélection."
    )

# The pool is built from the retained days only, so that the black curve
# and the coloured ones rest on exactly the same shots.
pooled_rows = [index for day in kept_days for index in by_day[day]]
print()
print(f"Retenus        : {len(kept_days)} jours, {len(pooled_rows)} tirs")
if dropped_days:
    print(
        f"Écartés        : {len(dropped_days)} jours, "
        f"{sum(len(by_day[day]) for day in dropped_days)} tirs"
    )
print()


# =============================================================================
# %% HELPERS
# =============================================================================


def day_color(rank):
    """Colour of the rank-th day, from the project palette."""
    return COLOR_LIST[rank % len(COLOR_LIST)]


def shade_flash(axes, shot_ids, beam="A", bands=2400):
    """Shade the background with the pump fluorescence, in red.

    The opacity follows the signal, so the background stays transparent
    away from the flash and is tinted at its peak. Drawn as vertical bands
    whose height is given in axes coordinates, so it fills the frame
    whatever the vertical limits are, set before or after this call.

    Returns
    -------
    float
        Instant of the flash maximum, in milliseconds, or nan when no
        intensity trace was available.
    """
    time_axis, signal = mean_fluorescence(config.root, shot_ids, beam)
    if signal is None:
        return float("nan")

    finite = np.isfinite(signal)
    if not finite.any():
        return float("nan")
    baseline = np.nanpercentile(signal[finite], 5)
    amplitude = np.nanmax(signal[finite]) - baseline
    if amplitude <= 0:
        return float("nan")
    normalised = np.clip((signal - baseline) / amplitude, 0.0, 1.0)

    edges = np.linspace(float(time_axis[0]), float(time_axis[-1]), bands + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    opacity = np.interp(centres, time_axis, normalised) * FLASH_SHADE_ALPHA

    colour = to_rgb(COLOR_FLASH)
    for start, stop, alpha in zip(edges[:-1], edges[1:], opacity):
        if alpha < 0.004:
            # Invisible, and would only weigh on the file.
            continue
        axes.axvspan(
            start, stop, facecolor=colour, alpha=float(alpha), linewidth=0, zorder=0
        )

    return float(time_axis[np.nanargmax(signal)])


def mode_index(name):
    """Column of an aberration, by name."""
    try:
        return labels.index(name)
    except ValueError:
        raise SystemExit(
            f"Aberration inconnue : {name!r}\nDisponibles : "
            + ", ".join(repr(x) for x in labels)
        ) from None


def dispersion_figure(aberration):
    """One figure: per-day dispersion against the pooled one."""
    column = mode_index(aberration)
    pretty = PRETTY_LABELS.get(aberration, aberration)

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 3.6))

    handles, texts = [], []
    for rank, day in enumerate(kept_days):
        rows = by_day[day]
        mean = np.nanmean(values[rows, column, :], axis=0)
        # ddof=1: each day is a sample of the shots that day could have
        # produced, not the whole population, and with five to ten shots
        # the difference is a few per cent.
        deviation = np.nanstd(values[rows, column, :], axis=0, ddof=1)
        colour = day_color(rank)

        ax.fill_between(
            time_ms,
            mean + deviation,
            mean - deviation,
            color=colour,
            alpha=BAND_ALPHA,
            linewidth=0,
            zorder=2,
        )
        ax.plot(time_ms, mean, color=colour, linewidth=1.1, zorder=3)
        handles.append(Line2D([0], [0], color=colour, linewidth=1.4))
        texts.append(f"{day} ({len(rows)} shots)")

    pooled_mean = np.nanmean(values[pooled_rows, column, :], axis=0)
    pooled = np.nanstd(values[pooled_rows, column, :], axis=0, ddof=1)
    ax.fill_between(
        time_ms,
        pooled_mean + pooled,
        pooled_mean - pooled,
        color=COLOR_POOLED,
        alpha=BAND_ALPHA,
        linewidth=0,
        zorder=4,
    )
    ax.plot(time_ms, pooled_mean, color=COLOR_POOLED, linewidth=1.8, zorder=5)
    handles.append(Line2D([0], [0], color=COLOR_POOLED, linewidth=1.8))
    texts.append(f"All days pooled ({len(pooled_rows)} shots)")

    ax.axhline(0.0, color="grey", linewidth=0.6, linestyle=":")

    shaded = False
    if SHOW_FLUORESCENCE:
        peak = shade_flash(ax, [shot_ids[i] for i in pooled_rows])
        shaded = bool(np.isfinite(peak))
    # The shot is the origin of the time base, so the flash sits at zero
    # whether or not the fluorescence could be drawn.
    ax.axvline(0.0, color=COLOR_FLASH, linestyle=":", linewidth=1.1, zorder=6)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(f"{pretty} {unit} (waves)")
    ax.set_title(
        f"{pretty} — mean and one-sigma band, day by day and pooled "
        f"({QUANTITY}, referenced at {TIME_REF_MS} ms)",
        fontsize=TITLE_FONTSIZE,
    )

    axes_spec = AXES.get(aberration, {})
    if axes_spec.get("xlim") is not None:
        ax.set_xlim(left=axes_spec["xlim"][0], right=axes_spec["xlim"][1])
    if axes_spec.get("ylim") is not None:
        ax.set_ylim(bottom=axes_spec["ylim"][0], top=axes_spec["ylim"][1])

    handles.append(Line2D([0], [0], color=COLOR_FLASH, linestyle=":", linewidth=1.1))
    texts.append("Flash")
    if shaded:
        handles.append(
            Patch(facecolor=COLOR_FLASH, alpha=FLASH_SHADE_ALPHA, linewidth=0)
        )
        texts.append("Pump fluorescence")

    ax.legend(
        handles, texts, fontsize=LEGEND_FONTSIZE, loc="best", ncol=2, framealpha=0.9
    )

    # The number that summarises the figure, at the shot.
    index_shot = int(np.argmin(np.abs(common_time)))
    per_day = [
        float(np.nanstd(values[by_day[day], column, index_shot], ddof=1))
        for day in kept_days
    ]
    print(f"{pretty} à t = 0 :")
    print(f"  moyenne globale        {pooled_mean[index_shot]:+.5f} vague")
    print(
        f"  dispersion intra-jour  {np.mean(per_day):.5f} vague "
        f"(moyenne des {len(kept_days)} jours)"
    )
    print(f"  dispersion globale     {pooled[index_shot]:.5f} vague")
    if np.mean(per_day) > 0:
        print(
            f"  rapport global / intra-jour : "
            f"{pooled[index_shot] / np.mean(per_day):.2f}"
        )
    print()

    safe = aberration.replace(" ", "_").replace(".", "").replace("/", "_")
    return save_figure(fig, config.figures_dir / f"day_dispersion_{safe}.pdf")


# =============================================================================
# %% FIGURES — One per aberration
# =============================================================================

written = [dispersion_figure(name) for name in ABERRATIONS]


# =============================================================================
# %% SHOW
# =============================================================================

print("Figures written:")
for path in written:
    print(f"  {path}")

plt.show()
