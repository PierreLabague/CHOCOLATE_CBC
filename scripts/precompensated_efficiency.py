"""What combining efficiency would be left after pre-compensation.

A static correction can remove whatever is the same on every shot: a plate,
a deformable mirror set once, a fixed delay line. What it cannot remove is
the shot-to-shot scatter. Subtracting the mean differential across shots
therefore separates the two, and what remains is the efficiency a perfectly
pre-compensated system would still lose.

The question is asked separately for the two things that spoil the
combination, because they are corrected by different hardware and it is
worth knowing which one dominates:

    Higher orders, no piston
        The differential Zernike coefficients, piston excluded, run through
        the ZAA model of :mod:`chocolate.efficiency`. This is what a
        deformable mirror would have to handle.
    Piston, no higher orders
        The optical path difference measured by the fringe analysis, on its
        own. This is what a delay line would have to handle.

and for both amplifier stages, the 45 mm rods and the 108 mm slabs. That
makes four combinations, and each gets its own pair of figures.

Eight output figures:

    precompensated_<stage>_<quantity>_time.pdf
        Efficiency against time, zoomed on the flash and over the first
        few milliseconds after it.
    precompensated_<stage>_<quantity>_correction.pdf
        Efficiency at the shot as a function of when the last correction
        was applied. Reading it backwards gives the delay a correction
        system may tolerate.

Run this script directly, or cell by cell.
"""

# =============================================================================
# %% PARAMETERS
# =============================================================================

ROOT = None


# Instant the two beams are taken to be phased, in milliseconds. Everything
# is a departure from that moment.
TIME_REF_MS = -0.5

# Subtract the mean differential across shots before computing the
# efficiency. True is the point of this script: it is what a static
# pre-compensation would remove, leaving only the shot-to-shot scatter.
# Set to False to see what the same shots give with no correction at all.
PRECOMPENSATE = True

# ---- Axes, one entry per figure ---------------------------------------------
# Every figure has its own entry and nothing falls back to a shared value:
# what is written here is what the figure uses. A bound left as None is the
# only exception and means "let matplotlib decide", which is useful for the
# start of an acquisition whose length you do not know by heart.
#
# For the temporal figures:
#   zoom  time window of the left panel, in ms
#   wide  time window of the right panel, in ms
#   ylim  vertical range, in per cent
# For the correction figures:
#   xlim  range of correction instants shown, in ms
#   ylim  vertical range, in per cent
#
# Below 85 % the second-order model is well outside its domain and the
# numbers would be wrong in any case, hence the usual lower bound. The
# piston figures often need a lower one: a piston left uncorrected drops
# far below that, and cutting the axis there would hide the result rather
# than protect it.
AXES = {
    "rods_wavefront_time": {
        "zoom": (-0.5, 0.5),
        "wide": (None, 5.0),
        "ylim": (85.0, 100.5),
    },
    "rods_piston_time": {
        "zoom": (-0.5, 0.5),
        "wide": (None, 5.0),
        "ylim": (85.0, 100.5),
    },
    "slabs_wavefront_time": {
        "zoom": (-0.5, 0.5),
        "wide": (None, 5.0),
        "ylim": (85.0, 100.5),
    },
    "slabs_piston_time": {
        "zoom": (-0.5, 0.5),
        "wide": (None, 5.0),
        "ylim": (85.0, 100.5),
    },
    "slabs_tilt_corrected_time": {
        "zoom": (-0.5, 0.5),
        "wide": (None, 5.0),
        "ylim": (85.0, 100.5),
    },
    # The two correction figures each stack the two amplifier stages, rods
    # above slabs, sharing both axes: what matters is how they compare, and
    # that reading is only legitimate on common scales.
    "wavefront_correction": {
        "xlim": (-50.0, 0.0),
        "ylim": (85.0, 100.5),
    },
    "piston_correction": {
        "xlim": (-50.0, 0.0),
        "ylim": (85.0, 100.5),
    },
    # The three-row budget figures. Each row has its own vertical range,
    # because the three quantities do not live on the same scale: the
    # piston swings over the whole interval while the wavefront losses stay
    # in the first few per cent.
    "rods_budget": {
        "xlim": (-50.0, 0.0),
        "ylim_piston": (0.0, 100.0),
        "ylim_tilt": (75.0, 100.5),
        "ylim_higher": (97.0, 100.5),
    },
    "slabs_budget": {
        "xlim": (-50.0, 0.0),
        "ylim_piston": (0.0, 100.0),
        "ylim_tilt": (75.0, 100.5),
        "ylim_higher": (97.0, 100.5),
    },
}

# ---- Perfectly corrected modes ----------------------------------------------
# Modes assumed to be taken out by a dedicated actuator, a tip-tilt mirror
# in this case. They are zeroed in the differential before the efficiency
# is computed, which is what a perfect correction of those modes would do.
# The gap to the uncorrected curve is what such an actuator would buy.
CORRECTED_MODES = ["Tilt X", "Tilt Y"]

# ---- Type sizes -------------------------------------------------------------
# The house style sets 8 pt legends; these figures carry six entries in a
# small frame, so they override it. Everything else, axis labels and ticks,
# comes from chocolate.style and is left alone.
LEGEND_FONTSIZE = 6.0
TITLE_FONTSIZE = 9.0
PANEL_FONTSIZE = 8.0

# Opacity of the fluorescence shading at the peak of the flash. The zoomed
# panel sits entirely inside the flash, so the whole frame is tinted there;
# keep it light enough for the curves to stay legible.
FLASH_SHADE_ALPHA = 0.22

# ---- Shot selection ---------------------------------------------------------
CINE_BEGINNING = -4000  # frames, exact value required
CINE_END_MAX = 4000
VALID_FRAME_RATES = {66000, 82000}

# Shots excluded by hand, with the reason. An exclusion nobody can check is
# an exclusion nobody should trust.
EXCLUDED_IDS = {
    # "2025-11-28-17-07": "pas de sauvegarde",
    # "2025-11-28-17-34": "tir qui part dans le mauvais sens",
    # "2025-12-04-10-26": "pas de sauvegarde",
    # "2025-12-05-14-47": "pas de piston",
}

# ---- Marechal reference -----------------------------------------------------
# The classical criterion for a "well corrected" system is a residual of
# lambda/14 RMS. Converted into a combining efficiency with the same model
# as the curves, so that the line and the data are comparable.
MARECHAL_RMS = 1.0 / 14.0

# ---- Last-correction-point sweep --------------------------------------------
# Reference instants swept, in milliseconds before the shot. The efficiency
# is then read at t = 0: the further back the last correction, the more the
# wavefront has had time to drift away from it.
SWEEP_RANGE_MS = (-50.0, -0.5)
SWEEP_POINTS = 300

SHOW_PROGRESS = True


# =============================================================================
# %% IMPORTS AND SETUP
# =============================================================================

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from chocolate.config import load_config
from chocolate.database import load_database, select_ids
from chocolate.efficiency import (
    combining_efficiency_small_fluctuations,
    reference_index,
)
from chocolate.results_io import (
    ZERNIKE_COLUMNS,
    load_beam_pair,
    load_piston_results,
    mean_fluorescence,
)
from chocolate.style import FIGURE_WIDTH, apply_style, save_figure

#: Everything about the flash is red: the shading of the pump fluorescence
#: and the line marking its maximum.
COLOR_FLASH = "#D62728"

#: The Marechal criterion is a target, so it is green: a curve above the
#: line is a shot that meets it.
COLOR_MARECHAL = "tab:green"

#: One colour per amplifier stage, kept identical wherever the two are
#: shown together.
COLOR_RODS = "tab:blue"
COLOR_SLABS = "tab:orange"

COLOR_MEAN = "black"
COLOR_SIGMA_1 = "tab:purple"
COLOR_SIGMA_3 = "tab:pink"

apply_style()

config = load_config()
if ROOT is not None:
    config.root = Path(ROOT)
config.check()

db = load_database(config.root)

# The small-fluctuation model reduces the pupil average to a sum of squared
# coefficients only in an orthonormal basis. Fed 'amp' or 'ptv'
# coefficients it would still return a number, a wrong one, without
# complaint: the array carries no unit and the function cannot tell. The
# check therefore belongs here, where the convention is chosen.
#
# The piston efficiency is exact whatever the convention, since a piston is
# not a Zernike coefficient here but a measured optical path difference.
if config.zernike_mode != "rms":
    raise SystemExit(
        f"Ce script utilise le modèle ZAA, qui suppose une base "
        f"orthonormée. config.yaml demande zernike_mode = "
        f"{config.zernike_mode!r} ; mettez 'rms'."
    )

print(f"Data directory   : {config.root}")
print(f"Normalisation    : {config.zernike_mode}")
print(f"Reference time   : {TIME_REF_MS} ms")
print(f"Pre-compensation : {'on' if PRECOMPENSATE else 'off'}")
print()


def progress(iterable, description, total=None):
    """Wrap an iterable in a progress bar when tqdm is available."""
    if not SHOW_PROGRESS:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=description, total=total, leave=False)


def clip_to_one(values):
    """Cap an efficiency at 100 %.

    The dispersion bands are drawn as the mean plus or minus a multiple of
    the standard deviation, which can exceed one where the scatter is
    large. An efficiency above one is meaningless, so the band is cut there
    rather than drawn into a region that cannot exist.
    """
    return np.minimum(np.asarray(values, dtype=float), 100.0)


# The Marechal criterion, expressed with the model used for the curves so
# that the line and the data mean the same thing: a single mode of
# MARECHAL_RMS carried by the differential wavefront.
_marechal_block = np.zeros((1, 1, 1))
_marechal_block[0, 0, 0] = MARECHAL_RMS
MARECHAL_PERCENT = 100.0 * float(
    combining_efficiency_small_fluctuations(_marechal_block)[0, 0]
)
print(
    f"Seuil de Marechal : {MARECHAL_RMS:.4f} vague RMS "
    f"→ efficacité {MARECHAL_PERCENT:.2f} %"
)
print()


# =============================================================================
# %% SHOT SELECTION — Rods and slabs
# =============================================================================
#
# The two stages are selected by the same conditions except for which
# amplifiers fired. Requiring the other pair to be off is what makes each
# set attributable to one stage alone.


def amplifier_conditions(rods_on, slabs_on):
    """Conditions selecting the shots of one amplifier stage."""
    return [
        lambda _, meta: isinstance(meta["PHASICS"].get("cine_beginning"), (int, float)),
        lambda _, meta: isinstance(meta["PHASICS"].get("cine_end"), (int, float)),
        lambda _, meta: meta["PHASICS"]["cine_beginning"] == CINE_BEGINNING,
        lambda _, meta: meta["PHASICS"]["cine_end"] <= CINE_END_MAX,
        lambda _, meta: meta["AMPLI"]["45A"] == int(rods_on),
        lambda _, meta: meta["AMPLI"]["45B"] == int(rods_on),
        lambda _, meta: meta["AMPLI"]["108A"] == int(slabs_on),
        lambda _, meta: meta["AMPLI"]["108B"] == int(slabs_on),
        lambda _, meta: meta["PHASICS"]["frame_rate"] in VALID_FRAME_RATES,
    ]


ids_rods = select_ids(db, amplifier_conditions(True, False), exclude=EXCLUDED_IDS)
ids_slabs = select_ids(db, amplifier_conditions(False, True), exclude=EXCLUDED_IDS)

print(f"Rods  (45 mm)  : {len(ids_rods)} tirs")
print(f"Slabs (108 mm) : {len(ids_slabs)} tirs")
if not ids_rods or not ids_slabs:
    raise SystemExit(
        "Une des deux sélections est vide. Relâchez CINE_BEGINNING, qui "
        "demande une valeur exacte, ou vérifiez les champs AMPLI."
    )
print()


# =============================================================================
# %% HIGHER ORDERS — Loading and differential
# =============================================================================
#
# ZERNIKE_COLUMNS starts at Tilt X, so the piston is already absent. The
# check below guards against a change of that constant: this script treats
# the piston separately and must never see it here.


def load_higher_orders(shot_ids, label):
    """Differential Zernike coefficients of one stage.

    Returns
    -------
    raw : numpy.ndarray, shape (n_shots, n_modes, n_times)
        Beam A minus beam B, before any referencing, so that the sweep can
        re-reference it at three hundred different instants without
        reloading anything.
    common_time : numpy.ndarray
        In seconds.
    shots : list of str
    labels : list of str
        Name of each mode, in the order of the second axis.
    """
    data_a, data_b, common_time, labels, shots = load_beam_pair(
        config.root,
        shot_ids,
        zernike_mode=config.zernike_mode,
        columns=ZERNIKE_COLUMNS,
        verbose=True,
    )
    if "Piston" in [str(name) for name in labels]:
        raise SystemExit(
            "La colonne Piston est présente dans la sélection de modes : ce "
            "script traite le piston séparément et ne doit pas le voir ici."
        )
    print(
        f"{label:<6}: {data_a.shape[0]} tirs, {data_a.shape[1]} modes, "
        f"{data_a.shape[2]} instants"
    )
    return data_a - data_b, common_time, shots, [str(name) for name in labels]


raw_rods, time_rods, shots_rods, labels_rods = load_higher_orders(ids_rods, "Rods")
raw_slabs, time_slabs, shots_slabs, labels_slabs = load_higher_orders(
    ids_slabs, "Slabs"
)


def wavefront_efficiency(raw, common_time, time_ref_ms, precompensate):
    """Efficiency of the higher orders, in per cent, shot by shot."""
    index = reference_index(common_time, time_ref_ms)
    differential = raw - raw[:, :, [index]]
    if precompensate:
        # The mean over shots is what a static correction removes; the
        # scatter about it is what no static correction can.
        differential = differential - np.nanmean(differential, axis=0, keepdims=True)
    return 100.0 * combining_efficiency_small_fluctuations(differential)


def without_modes(raw, labels, names):
    """Copy of the differential with the named modes set to zero.

    Zeroing a mode is exactly what a perfect correction of it would do: the
    actuator removes that component of the differential wavefront and
    leaves the rest untouched.

    Raises
    ------
    SystemExit
        If a name is not among the modes, with the list of those that are.
    """
    missing = [name for name in names if name not in labels]
    if missing:
        raise SystemExit(
            f"Mode(s) inconnu(s) dans CORRECTED_MODES : {missing}\n"
            f"Disponibles : {', '.join(labels)}"
        )
    corrected = raw.copy()
    for name in names:
        corrected[:, labels.index(name), :] = 0.0
    return corrected


eff_wf_rods = wavefront_efficiency(raw_rods, time_rods, TIME_REF_MS, PRECOMPENSATE)
eff_wf_slabs = wavefront_efficiency(raw_slabs, time_slabs, TIME_REF_MS, PRECOMPENSATE)

# Same shots, with the tip-tilt actuator assumed perfect.
eff_wf_slabs_corrected = wavefront_efficiency(
    without_modes(raw_slabs, labels_slabs, CORRECTED_MODES),
    time_slabs,
    TIME_REF_MS,
    PRECOMPENSATE,
)
print()


# =============================================================================
# %% PISTON — Loading
# =============================================================================
#
# The piston comes from the fringe analysis, not from the Zernike
# decomposition: a wavefront sensor cannot measure an absolute optical
# path. Each shot has its own time base, so they are interpolated onto a
# common one before being compared.


def load_pistons(shot_ids, label):
    """Piston of a set of shots, on a common time base, in waves."""
    results = load_piston_results(config.root, shot_ids, unwrap=True, verbose=False)
    traces = {
        shot: entry
        for shot, entry in results.items()
        if entry.get("piston") is not None and entry.get("time") is not None
    }
    missing = len(shot_ids) - len(traces)
    if missing:
        print(f"  {label} : {missing} tir(s) sans piston de franges, écartés.")
    if not traces:
        print(f"  {label} : aucun piston disponible.")
        return None, None, []

    kept = [shot for shot in shot_ids if shot in traces]
    start = max(float(np.min(traces[shot]["time"])) for shot in kept)
    stop = min(float(np.max(traces[shot]["time"])) for shot in kept)
    if stop <= start:
        print(f"  {label} : les tirs ne se recouvrent pas en temps.")
        return None, None, []

    step = min(float(np.min(np.diff(traces[shot]["time"]))) for shot in kept)
    grid = np.arange(start, stop + step / 2, step)

    pistons = np.array(
        [
            np.interp(
                grid,
                np.asarray(traces[shot]["time"], dtype=float),
                np.asarray(traces[shot]["piston"], dtype=float),
            )
            for shot in kept
        ]
    )
    print(f"  {label} : {len(kept)} tirs, {pistons.shape[1]} instants")
    return pistons, grid, kept


print("Piston de franges")
piston_rods, ptime_rods, pshots_rods = load_pistons(ids_rods, "rods ")
piston_slabs, ptime_slabs, pshots_slabs = load_pistons(ids_slabs, "slabs")
print()


def piston_efficiency(pistons, common_time, time_ref_ms, precompensate):
    """Efficiency of a piston alone, in per cent, shot by shot.

    A uniform phase offset gives an interference of
    ``(1 + cos(2 pi delta)) / 2`` everywhere, so the efficiency is that
    expression evaluated at the measured optical path difference. No pupil
    average is needed and no approximation is made: this one is exact.
    """
    index = reference_index(common_time, time_ref_ms)
    delta = pistons - pistons[:, [index]]
    if precompensate:
        delta = delta - np.nanmean(delta, axis=0, keepdims=True)
    return 100.0 * 0.5 * (1.0 + np.cos(2 * np.pi * delta))


eff_piston_rods = (
    piston_efficiency(piston_rods, ptime_rods, TIME_REF_MS, PRECOMPENSATE)
    if piston_rods is not None
    else None
)
eff_piston_slabs = (
    piston_efficiency(piston_slabs, ptime_slabs, TIME_REF_MS, PRECOMPENSATE)
    if piston_slabs is not None
    else None
)


# =============================================================================
# %% HELPERS — Panels
# =============================================================================


def shade_flash(axes, shot_ids, beam="A", max_alpha=FLASH_SHADE_ALPHA):
    """Shade the background with the pump fluorescence, in red.

    The opacity follows the signal, so the background is transparent away
    from the flash and tinted at its peak. Drawn after the axis limits are
    final, since the shading is sized to fill them.

    Returns True when something was actually drawn.
    """
    time_ms, signal = mean_fluorescence(config.root, shot_ids, beam)
    if signal is None:
        return False

    finite = np.isfinite(signal)
    if not finite.any():
        return False
    baseline = np.nanpercentile(signal[finite], 5)
    amplitude = np.nanmax(signal[finite]) - baseline
    if amplitude <= 0:
        return False
    normalised = np.clip((signal - baseline) / amplitude, 0.0, 1.0)

    low, high = axes.get_ylim()
    rgba = np.empty((normalised.size, 4))
    rgba[:, :3] = to_rgb(COLOR_FLASH)
    rgba[:, 3] = normalised * max_alpha

    axes.imshow(
        rgba[None, :, :],
        extent=(float(time_ms[0]), float(time_ms[-1]), low, high),
        aspect="auto",
        origin="lower",
        interpolation="bilinear",
        zorder=0,
    )
    axes.set_ylim(low, high)
    return True


def draw_dispersion(axes, x_values, efficiency, heading, xlim, ylim):
    """Mean efficiency with its one and three sigma bands, in per cent.

    Three sigma covers 99.7 % of a normal distribution, so the outer band
    is what a shot would have to beat to be called exceptional. It is drawn
    because a mean alone says nothing about whether a given shot will
    combine.
    """
    mean = np.nanmean(efficiency, axis=0)
    deviation = np.nanstd(efficiency, axis=0)

    axes.fill_between(
        x_values,
        clip_to_one(mean + 3 * deviation),
        mean - 3 * deviation,
        color=COLOR_SIGMA_3,
        alpha=0.4,
        linewidth=0,
        zorder=2,
    )
    axes.fill_between(
        x_values,
        clip_to_one(mean + deviation),
        mean - deviation,
        color=COLOR_SIGMA_1,
        alpha=0.4,
        linewidth=0,
        zorder=3,
    )
    axes.plot(x_values, mean, color=COLOR_MEAN, linewidth=1.2, zorder=4)

    axes.axhline(
        MARECHAL_PERCENT, color=COLOR_MARECHAL, linestyle="--", linewidth=1.1, zorder=5
    )

    # A bound of None leaves that side to matplotlib.
    if xlim is not None:
        axes.set_xlim(left=xlim[0], right=xlim[1])
    if ylim is not None:
        axes.set_ylim(bottom=ylim[0], top=ylim[1])
    axes.set_title(heading, fontsize=PANEL_FONTSIZE)
    return mean, deviation


def legend_entries(with_flash):
    """Handles and labels shared by every figure of this script."""
    handles = [
        Line2D([0], [0], color=COLOR_MEAN, linewidth=1.2),
        Patch(facecolor=COLOR_SIGMA_1, alpha=0.4, linewidth=0),
        Patch(facecolor=COLOR_SIGMA_3, alpha=0.4, linewidth=0),
        Line2D([0], [0], color=COLOR_MARECHAL, linestyle="--", linewidth=1.1),
    ]
    texts = [
        "Mean over shots",
        r"$\pm\,\sigma$",
        r"$\pm\,3\sigma$",
        rf"Marechal, $\lambda/14$ ({MARECHAL_PERCENT:.1f} %)",
    ]
    if with_flash:
        handles.append(
            Line2D([0], [0], color=COLOR_FLASH, linestyle=":", linewidth=1.1)
        )
        texts.append("Flash")
        handles.append(
            Patch(facecolor=COLOR_FLASH, alpha=FLASH_SHADE_ALPHA, linewidth=0)
        )
        texts.append("Pump fluorescence")
    return handles, texts


def _window_title(window, time_ms, fallback):
    """Readable heading for a panel, whatever bounds were left to None."""
    low = time_ms[0] if window[0] is None else window[0]
    high = time_ms[-1] if window[1] is None else window[1]
    return f"{fallback} ({low:.3g} to {high:.3g} ms)"


def temporal_figure(key, efficiency, common_time, shot_ids, beam, title, filename):
    """One figure: efficiency against time, zoomed and over a wider window.

    The axes are read from ``AXES[key]``, so every figure is framed by the
    entry that bears its name in the parameter block and by nothing else.

    Parameters
    ----------
    key : str
        Entry of :data:`AXES`.
    efficiency : numpy.ndarray, shape (n_shots, n_times)
        In per cent.
    common_time : numpy.ndarray
        In seconds.
    shot_ids : sequence of str
        Used only for the fluorescence shading.
    beam : {'A', 'B'}
    title, filename : str
    """
    if key not in AXES:
        raise SystemExit(
            f"Aucune entrée {key!r} dans AXES. Ajoutez-la au bloc de "
            f"paramètres : chaque figure a ses propres échelles."
        )
    axes_spec = AXES[key]
    zoom, wide, ylim = axes_spec["zoom"], axes_spec["wide"], axes_spec["ylim"]

    time_ms = common_time * 1e3
    fig, axs = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 3.2), sharey=True)

    windows = (
        (zoom, _window_title(zoom, time_ms, "Around the flash")),
        (wide, _window_title(wide, time_ms, "Wider view")),
    )

    shaded = False
    for ax, (xlim, heading) in zip(axs, windows):
        draw_dispersion(ax, time_ms, efficiency, heading, xlim=xlim, ylim=ylim)
        ax.axvline(0.0, color=COLOR_FLASH, linestyle=":", linewidth=1.1, zorder=5)
        ax.set_xlabel("Time (ms)")
        shaded = shade_flash(ax, shot_ids, beam) or shaded

    axs[0].set_ylabel("Combining efficiency (%)")
    handles, texts = legend_entries(shaded)
    axs[0].legend(
        handles, texts, fontsize=LEGEND_FONTSIZE, loc="lower left", framealpha=0.9
    )
    fig.suptitle(title, fontsize=TITLE_FONTSIZE)

    return save_figure(fig, config.figures_dir / filename)


def correction_figure(key, panels, title, filename):
    """Efficiency at the shot against the last correction instant.

    Two panels stacked, rods above slabs, sharing their time axis and their
    vertical scale. Reading a point: if the correction is set at that
    instant and nothing is done afterwards, this is what the combination
    gives when the shot arrives. Sharing the scales is what makes the two
    stages comparable at a glance.

    Parameters
    ----------
    key : str
        Entry of :data:`AXES`.
    panels : sequence of (str, numpy.ndarray, numpy.ndarray)
        One tuple per panel, top first: heading, mean and standard
        deviation in per cent over the swept instants. A mean of None draws
        an empty panel carrying its explanation, which is what happens when
        a stage has no fringe piston.
    title, filename : str
    """
    if key not in AXES:
        raise SystemExit(
            f"Aucune entrée {key!r} dans AXES. Ajoutez-la au bloc de "
            f"paramètres : chaque figure a ses propres échelles."
        )
    axes_spec = AXES[key]

    fig, axs = plt.subplots(
        len(panels), 1, figsize=(FIGURE_WIDTH, 4.8), sharex=True, sharey=True
    )
    axs = np.atleast_1d(axs)

    for ax, (heading, mean, deviation) in zip(axs, panels):
        if mean is None:
            ax.text(
                0.5,
                0.5,
                "No fringe piston for these shots",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=PANEL_FONTSIZE,
            )
            ax.set_title(heading, fontsize=PANEL_FONTSIZE)
            continue

        ax.fill_between(
            sweep_times,
            clip_to_one(mean + 3 * deviation),
            mean - 3 * deviation,
            color=COLOR_SIGMA_3,
            alpha=0.4,
            linewidth=0,
        )
        ax.fill_between(
            sweep_times,
            clip_to_one(mean + deviation),
            mean - deviation,
            color=COLOR_SIGMA_1,
            alpha=0.4,
            linewidth=0,
        )
        ax.plot(sweep_times, mean, color=COLOR_MEAN, linewidth=1.2)
        ax.axhline(
            MARECHAL_PERCENT, color=COLOR_MARECHAL, linestyle="--", linewidth=1.1
        )
        ax.set_title(heading, fontsize=PANEL_FONTSIZE)
        ax.set_ylabel("Efficiency at the shot (%)")

    if axes_spec["ylim"] is not None:
        axs[0].set_ylim(bottom=axes_spec["ylim"][0], top=axes_spec["ylim"][1])
    if axes_spec["xlim"] is not None:
        axs[0].set_xlim(left=axes_spec["xlim"][0], right=axes_spec["xlim"][1])
    axs[-1].set_xlabel("Last correction point (ms before the shot)")

    handles, texts = legend_entries(False)
    axs[0].legend(handles, texts, fontsize=LEGEND_FONTSIZE, loc="lower left")
    fig.suptitle(title, fontsize=TITLE_FONTSIZE)

    return save_figure(fig, config.figures_dir / filename)


written = []


# =============================================================================
# %% FIGURE 1 — Rods, higher orders, against time
# =============================================================================

# Axes: AXES["rods_wavefront_time"] in the parameter block.
written.append(
    temporal_figure(
        "rods_wavefront_time",
        eff_wf_rods,
        time_rods,
        shots_rods,
        "A",
        f"Rods 45 mm — higher orders, piston excluded ({len(shots_rods)} shots)"
        + (", pre-compensated" if PRECOMPENSATE else ""),
        "precompensated_rods_wavefront_time.pdf",
    )
)


# =============================================================================
# %% FIGURE 2 — Rods, piston, against time
# =============================================================================

if eff_piston_rods is not None:
    written.append(
        temporal_figure(
            "rods_piston_time",
            eff_piston_rods,
            ptime_rods,
            pshots_rods,
            "A",
            f"Rods 45 mm — piston alone ({len(pshots_rods)} shots)"
            + (", pre-compensated" if PRECOMPENSATE else ""),
            "precompensated_rods_piston_time.pdf",
        )
    )
else:
    print("[info] pas de piston pour les rods, figure 2 ignorée.")


# =============================================================================
# %% FIGURE 3 — Slabs, higher orders, against time
# =============================================================================

written.append(
    temporal_figure(
        "slabs_wavefront_time",
        eff_wf_slabs,
        time_slabs,
        shots_slabs,
        "A",
        f"Slabs 108 mm — higher orders, piston excluded ({len(shots_slabs)} shots)"
        + (", pre-compensated" if PRECOMPENSATE else ""),
        "precompensated_slabs_wavefront_time.pdf",
    )
)


# =============================================================================
# %% FIGURE 4 — Slabs, piston, against time
# =============================================================================

if eff_piston_slabs is not None:
    written.append(
        temporal_figure(
            "slabs_piston_time",
            eff_piston_slabs,
            ptime_slabs,
            pshots_slabs,
            "A",
            f"Slabs 108 mm — piston alone ({len(pshots_slabs)} shots)"
            + (", pre-compensated" if PRECOMPENSATE else ""),
            "precompensated_slabs_piston_time.pdf",
        )
    )
else:
    print("[info] pas de piston pour les slabs, figure 4 ignorée.")


# =============================================================================
# %% FIGURE 5 — Slabs, what a perfect tip-tilt correction would buy
# =============================================================================
#
# The same shots twice: the higher orders as measured, and the same with
# the tilt modes zeroed, which is what a perfect tip-tilt actuator would
# leave. The gap between the two curves is the case for that actuator, and
# it is worth more than either curve alone.
#
# Only the one sigma band is drawn here. Two sets of three sigma bands
# would overlap into a single wash and hide the very comparison the figure
# is for.

COLOR_AS_MEASURED = "black"
COLOR_CORRECTED = "tab:blue"

axes_spec = AXES["slabs_tilt_corrected_time"]
time_ms = time_slabs * 1e3

fig, axs = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 3.2), sharey=True)

windows = (
    (axes_spec["zoom"], _window_title(axes_spec["zoom"], time_ms, "Around the flash")),
    (axes_spec["wide"], _window_title(axes_spec["wide"], time_ms, "Wider view")),
)

shaded = False
for ax, (xlim, heading) in zip(axs, windows):
    for efficiency, colour in (
        (eff_wf_slabs, COLOR_AS_MEASURED),
        (eff_wf_slabs_corrected, COLOR_CORRECTED),
    ):
        mean = np.nanmean(efficiency, axis=0)
        deviation = np.nanstd(efficiency, axis=0)
        ax.fill_between(
            time_ms,
            clip_to_one(mean + deviation),
            mean - deviation,
            color=colour,
            alpha=0.20,
            linewidth=0,
            zorder=2,
        )
        ax.plot(time_ms, mean, color=colour, linewidth=1.2, zorder=4)

    ax.axhline(
        MARECHAL_PERCENT, color=COLOR_MARECHAL, linestyle="--", linewidth=1.1, zorder=5
    )
    ax.axvline(0.0, color=COLOR_FLASH, linestyle=":", linewidth=1.1, zorder=5)

    if xlim is not None:
        ax.set_xlim(left=xlim[0], right=xlim[1])
    if axes_spec["ylim"] is not None:
        ax.set_ylim(bottom=axes_spec["ylim"][0], top=axes_spec["ylim"][1])
    ax.set_title(heading, fontsize=PANEL_FONTSIZE)
    ax.set_xlabel("Time (ms)")
    shaded = shade_flash(ax, shots_slabs, "A") or shaded

axs[0].set_ylabel("Combining efficiency (%)")

handles = [
    Line2D([0], [0], color=COLOR_AS_MEASURED, linewidth=1.2),
    Line2D([0], [0], color=COLOR_CORRECTED, linewidth=1.2),
    Patch(facecolor="grey", alpha=0.20, linewidth=0),
    Line2D([0], [0], color=COLOR_MARECHAL, linestyle="--", linewidth=1.1),
    Line2D([0], [0], color=COLOR_FLASH, linestyle=":", linewidth=1.1),
]
texts = [
    "Higher orders, as measured",
    " and ".join(CORRECTED_MODES) + " perfectly corrected",
    r"$\pm\,\sigma$ (68 %)",
    rf"Marechal, $\lambda/14$ ({MARECHAL_PERCENT:.1f} %)",
    "Flash",
]
if shaded:
    handles.append(Patch(facecolor=COLOR_FLASH, alpha=FLASH_SHADE_ALPHA, linewidth=0))
    texts.append("Pump fluorescence")

axs[0].legend(
    handles, texts, fontsize=LEGEND_FONTSIZE, loc="lower left", framealpha=0.9
)
fig.suptitle(
    f"Slabs 108 mm — effect of a perfect tip-tilt correction "
    f"({len(shots_slabs)} shots)" + (", pre-compensated" if PRECOMPENSATE else ""),
    fontsize=TITLE_FONTSIZE,
)

written.append(
    save_figure(fig, config.figures_dir / "precompensated_slabs_tilt_corrected.pdf")
)

index_shot_slabs = int(np.argmin(np.abs(time_slabs)))
print("Slabs au tir, ordres élevés :")
print(
    f"  tels quels                  "
    f"{np.nanmean(eff_wf_slabs[:, index_shot_slabs]):6.2f} % "
    f"± {np.nanstd(eff_wf_slabs[:, index_shot_slabs]):5.2f} %"
)
print(
    f"  {' et '.join(CORRECTED_MODES)} corrigés  "
    f"{np.nanmean(eff_wf_slabs_corrected[:, index_shot_slabs]):6.2f} % "
    f"± {np.nanstd(eff_wf_slabs_corrected[:, index_shot_slabs]):5.2f} %"
)
print()


# =============================================================================
# %% SWEEP — How long before the shot can the last correction be?
# =============================================================================
#
# The reference instant is moved back step by step and the efficiency read
# at the shot each time. Reading a point of the curve: if the correction is
# set at that instant and nothing is done afterwards, this is what the
# combination gives when the shot arrives. The curve is therefore the
# answer to a design question, and its slope is the bandwidth an active
# system would need.

sweep_times = np.linspace(*SWEEP_RANGE_MS, SWEEP_POINTS)


def sweep_wavefront(raw, common_time, precompensate):
    """Efficiency at the shot against the last correction instant."""
    index_shot = int(np.argmin(np.abs(common_time)))
    mean = np.empty(SWEEP_POINTS)
    deviation = np.empty(SWEEP_POINTS)

    for i, instant in enumerate(
        progress(sweep_times, "Balayage ordres élevés", total=SWEEP_POINTS)
    ):
        index = reference_index(common_time, instant)
        differential = raw - raw[:, :, [index]]
        if precompensate:
            differential = differential - np.nanmean(
                differential, axis=0, keepdims=True
            )
        efficiency = combining_efficiency_small_fluctuations(
            differential[:, :, [index_shot]]
        )[:, 0]
        mean[i] = 100.0 * np.nanmean(efficiency)
        deviation[i] = 100.0 * np.nanstd(efficiency)

    return mean, deviation


def sweep_piston(pistons, common_time, precompensate):
    """Same sweep, for the piston alone."""
    index_shot = int(np.argmin(np.abs(common_time)))
    mean = np.empty(SWEEP_POINTS)
    deviation = np.empty(SWEEP_POINTS)

    for i, instant in enumerate(
        progress(sweep_times, "Balayage piston", total=SWEEP_POINTS)
    ):
        index = reference_index(common_time, instant)
        delta = pistons[:, index_shot] - pistons[:, index]
        if precompensate:
            delta = delta - np.nanmean(delta)
        efficiency = 0.5 * (1.0 + np.cos(2 * np.pi * delta))
        mean[i] = 100.0 * np.nanmean(efficiency)
        deviation[i] = 100.0 * np.nanstd(efficiency)

    return mean, deviation


sweep_wf_rods = sweep_wavefront(raw_rods, time_rods, PRECOMPENSATE)
sweep_wf_slabs = sweep_wavefront(raw_slabs, time_slabs, PRECOMPENSATE)
sweep_p_rods = (
    sweep_piston(piston_rods, ptime_rods, PRECOMPENSATE)
    if piston_rods is not None
    else (None, None)
)
sweep_p_slabs = (
    sweep_piston(piston_slabs, ptime_slabs, PRECOMPENSATE)
    if piston_slabs is not None
    else (None, None)
)


# =============================================================================
# %% FIGURE 6 — Higher orders, rods above slabs
# =============================================================================

written.append(
    correction_figure(
        "wavefront_correction",
        (
            (f"Rods 45 mm ({len(shots_rods)} shots)", *sweep_wf_rods),
            (f"Slabs 108 mm ({len(shots_slabs)} shots)", *sweep_wf_slabs),
        ),
        "Higher orders, piston excluded — efficiency left as a function of the "
        "last correction",
        "precompensated_wavefront_correction.pdf",
    )
)


# =============================================================================
# %% FIGURE 7 — Piston, rods above slabs
# =============================================================================

written.append(
    correction_figure(
        "piston_correction",
        (
            (f"Rods 45 mm ({len(pshots_rods)} shots)", *sweep_p_rods),
            (f"Slabs 108 mm ({len(pshots_slabs)} shots)", *sweep_p_slabs),
        ),
        "Piston alone — efficiency left as a function of the last correction",
        "precompensated_piston_correction.pdf",
    )
)


# =============================================================================
# %% BUDGET — Where the efficiency goes, stage by stage
# =============================================================================
#
# Three rows sharing the same abscissa, the instant of the last correction:
#
#   1. what the piston alone leaves, on the full scale, since it swings
#      over the whole interval;
#   2. what the tilt alone leaves;
#   3. what everything else leaves, piston and tilt taken out.
#
# All three are efficiencies, so the three rows read the same way; only
# their vertical ranges differ, the piston needing the whole interval and
# the wavefront rows sitting near the top. The dashed curve on the third
# row is the same quantity without the static pre-compensation, so the gap
# between the two is exactly what pre-compensating buys.

COLOR_TILT = "tab:blue"
COLOR_HIGHER = "tab:orange"
COLOR_PISTON_BAND = "tab:purple"


def only_modes(raw, labels, names):
    """Copy of the differential keeping only the named modes.

    The complement of :func:`without_modes`: everything except those modes
    is set to zero, which isolates what they alone would cost.
    """
    missing = [name for name in names if name not in labels]
    if missing:
        raise SystemExit(
            f"Mode(s) inconnu(s) : {missing}\nDisponibles : " f"{', '.join(labels)}"
        )
    kept = np.zeros_like(raw)
    for name in names:
        index = labels.index(name)
        kept[:, index, :] = raw[:, index, :]
    return kept


def budget_figure(key, raw, common_time, labels, pistons, piston_time, title, filename):
    """Three rows: what the piston, the tilt and the rest each leave.

    Every row carries the same pair of curves: solid with the static
    pre-compensation, dashed without it. The gap between them is what
    pre-compensating that contribution buys, which is the quantity a
    correction budget is actually built on.

    Parameters
    ----------
    key : str
        Entry of :data:`AXES`.
    raw : numpy.ndarray, shape (n_shots, n_modes, n_times)
        Differential Zernike coefficients before any referencing.
    common_time : numpy.ndarray
        Time base of ``raw``, in seconds.
    labels : list of str
        Mode names, in the order of the second axis of ``raw``.
    pistons : numpy.ndarray or None
        Fringe piston of the same stage, on its own time base.
    piston_time : numpy.ndarray or None
    title, filename : str
    """
    if key not in AXES:
        raise SystemExit(
            f"Aucune entrée {key!r} dans AXES. Ajoutez-la au bloc de " f"paramètres."
        )
    axes_spec = AXES[key]

    # Every row is computed twice, with and without the pre-compensation,
    # so that the two can be drawn together and their gap read directly.
    tilt_only = only_modes(raw, labels, CORRECTED_MODES)
    tilt_mean, tilt_std = sweep_wavefront(tilt_only, common_time, PRECOMPENSATE)
    tilt_raw_mean, _ = sweep_wavefront(tilt_only, common_time, False)

    higher = without_modes(raw, labels, CORRECTED_MODES)
    higher_mean, higher_std = sweep_wavefront(higher, common_time, PRECOMPENSATE)
    higher_raw_mean, _ = sweep_wavefront(higher, common_time, False)

    if pistons is None:
        piston_mean = piston_std = piston_raw_mean = None
    else:
        piston_mean, piston_std = sweep_piston(pistons, piston_time, PRECOMPENSATE)
        piston_raw_mean, _ = sweep_piston(pistons, piston_time, False)

    fig, axs = plt.subplots(3, 1, figsize=(FIGURE_WIDTH, 6.4), sharex=True)

    # ---- Row 1: what the piston leaves --------------------------------
    if piston_mean is None:
        axs[0].text(
            0.5,
            0.5,
            "No fringe piston for these shots",
            ha="center",
            va="center",
            transform=axs[0].transAxes,
            fontsize=PANEL_FONTSIZE,
        )
    else:
        axs[0].fill_between(
            sweep_times,
            clip_to_one(piston_mean + piston_std),
            piston_mean - piston_std,
            color=COLOR_PISTON_BAND,
            alpha=0.4,
            linewidth=0,
        )
        axs[0].plot(sweep_times, piston_mean, color=COLOR_MEAN, linewidth=1.2)
        axs[0].plot(
            sweep_times,
            piston_raw_mean,
            color=COLOR_MEAN,
            linestyle="--",
            linewidth=1.2,
        )
    axs[0].set_ylim(*axes_spec["ylim_piston"])
    axs[0].set_ylabel("Efficiency (%)")
    axs[0].set_title("Piston alone", fontsize=PANEL_FONTSIZE)

    # ---- Row 2: what the tilt costs ------------------------------------
    axs[1].fill_between(
        sweep_times,
        clip_to_one(tilt_mean + tilt_std),
        tilt_mean - tilt_std,
        color=COLOR_TILT,
        alpha=0.4,
        linewidth=0,
    )
    axs[1].plot(sweep_times, tilt_mean, color=COLOR_TILT, linewidth=1.3)
    axs[1].plot(
        sweep_times, tilt_raw_mean, color=COLOR_TILT, linestyle="--", linewidth=1.2
    )
    axs[1].plot(
        sweep_times, tilt_raw_mean, color=COLOR_TILT, linestyle="--", linewidth=1.2
    )
    axs[1].axhline(
        MARECHAL_PERCENT, color=COLOR_MARECHAL, linestyle="--", linewidth=1.1
    )
    axs[1].set_ylim(*axes_spec["ylim_tilt"])
    axs[1].set_ylabel("Efficiency (%)")
    axs[1].set_title(" and ".join(CORRECTED_MODES) + " only", fontsize=PANEL_FONTSIZE)

    # ---- Row 3: what the rest costs ------------------------------------
    axs[2].fill_between(
        sweep_times,
        clip_to_one(higher_mean + higher_std),
        higher_mean - higher_std,
        color=COLOR_HIGHER,
        alpha=0.4,
        linewidth=0,
    )
    axs[2].plot(sweep_times, higher_mean, color=COLOR_HIGHER, linewidth=1.3)
    axs[2].plot(
        sweep_times, higher_raw_mean, color=COLOR_HIGHER, linestyle="--", linewidth=1.2
    )
    axs[2].axhline(
        MARECHAL_PERCENT, color=COLOR_MARECHAL, linestyle="--", linewidth=1.1
    )
    axs[2].set_ylim(*axes_spec["ylim_higher"])
    axs[2].set_ylabel("Efficiency (%)")
    axs[2].set_title(
        "All other aberrations, piston and tilt removed", fontsize=PANEL_FONTSIZE
    )

    axs[0].set_xlim(*axes_spec["xlim"])
    axs[2].set_xlabel("Last correction point (ms before the shot)")

    # Every curve gets its own entry: with three colours and two line
    # styles, a legend that explained the styles separately would ask the
    # reader to combine them mentally.
    handles = [
        Line2D([0], [0], color=COLOR_MEAN, linewidth=1.2),
        Line2D([0], [0], color=COLOR_MEAN, linestyle="--", linewidth=1.2),
        Line2D([0], [0], color=COLOR_TILT, linewidth=1.3),
        Line2D([0], [0], color=COLOR_TILT, linestyle="--", linewidth=1.2),
        Line2D([0], [0], color=COLOR_HIGHER, linewidth=1.3),
        Line2D([0], [0], color=COLOR_HIGHER, linestyle="--", linewidth=1.2),
        Line2D([0], [0], color=COLOR_MARECHAL, linestyle="--", linewidth=1.1),
        Patch(facecolor="grey", alpha=0.4, linewidth=0),
    ]
    texts = [
        "Piston, pre-compensated",
        "Piston, no pre-compensation",
        " and ".join(CORRECTED_MODES) + ", pre-compensated",
        " and ".join(CORRECTED_MODES) + ", no pre-compensation",
        "Other aberrations, pre-compensated",
        "Other aberrations, no pre-compensation",
        rf"Marechal, $\lambda/14$ ({MARECHAL_PERCENT:.1f} %)",
        r"$\pm\,\sigma$",
    ]
    # One legend for the three rows, placed below them: the curves reach
    # the top of every frame, so a legend inside any of them would sit on
    # the data.
    fig.subplots_adjust(bottom=0.15)
    fig.legend(
        handles,
        texts,
        loc="lower center",
        ncol=3,
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        columnspacing=1.2,
        handlelength=1.8,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle(title, fontsize=TITLE_FONTSIZE)

    return save_figure(fig, config.figures_dir / filename)


written.append(
    budget_figure(
        "rods_budget",
        raw_rods,
        time_rods,
        labels_rods,
        piston_rods,
        ptime_rods,
        f"Rods 45 mm — learning based : ({len(shots_rods)} shots)",
        "precompensated_rods_budget.pdf",
    )
)

written.append(
    budget_figure(
        "slabs_budget",
        raw_slabs,
        time_slabs,
        labels_slabs,
        piston_slabs,
        ptime_slabs,
        f"Slabs 108 mm — learning based ({len(shots_slabs)} shots)",
        "precompensated_slabs_budget.pdf",
    )
)


# =============================================================================
# %% SHOW
# =============================================================================

print()
print("Efficacité au tir, correction à " f"{TIME_REF_MS} ms :")
for label, efficiency, common_time in (
    ("rods  ordres élevés", eff_wf_rods, time_rods),
    ("slabs ordres élevés", eff_wf_slabs, time_slabs),
    ("rods  piston", eff_piston_rods, ptime_rods),
    ("slabs piston", eff_piston_slabs, ptime_slabs),
):
    if efficiency is None:
        continue
    index = int(np.argmin(np.abs(common_time)))
    print(
        f"  {label:<20} {np.nanmean(efficiency[:, index]):6.2f} % "
        f"± {np.nanstd(efficiency[:, index]):5.2f} %"
    )

print()
print("Figures written:")
for path in written:
    print(f"  {path}")

plt.show()
