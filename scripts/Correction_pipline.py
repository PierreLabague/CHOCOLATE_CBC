"""Pre-shot correction synchronized on the dominant vibration period.

Pipeline, on shot 2025-12-04-09-28 (flash maximum at t = 0):

    1. Load the shot and normalize every aberration to its first frame.
    2. Find the dominant frequency of Tilt Y before the flash, T = 1/f.
    3. On the learning base (target excluded), measure for each aberration
       the mean change between -T and the flash:
           ABB_SHOT = < x(0) - x(-T) >
    4. Place the nodes every T, synchronized on the flash (t = -kT).
    5. Do nothing before LOOP_START_MS. From there, at every node t_k:
           C <- C + actuator(PRECOMP - y(t_k)),   PRECOMP = -ABB_SHOT
       with y = x + C the corrected aberration. The correction is applied
       instantly and held until the next node, so that y(-T) = PRECOMP and
       the shot brings it back to y(0) ~ 0.
    6. The actuator is perfect: every correction is applied exactly and
       instantly. With a gain of one, each node erases the history, so the
       residual at the flash is y(0) = [x(0) - x(-T)] - ABB_SHOT whatever
       the loop start.
    7. Same pipeline and same T for every aberration, then efficiency.
    8. Each learning shot is corrected the same way with the two nodes its
       record offers (-2T and -T), its ABB_SHOT learned on the other shots.

Figures
    node_correction_spectrum.pdf     spectrum of Tilt Y and its main lobes
    node_correction_tilt_y.pdf       Tilt Y uncorrected and corrected, three
                                     time windows
    node_correction_efficiency.pdf   combining efficiency (small-fluctuation
                                     model), uncorrected and corrected
"""

# %% === 1. PARAMETERS ===

ROOT = None


TARGET_ID = "2025-12-04-09-28"
EXAMPLE_MODE = "Tilt Y"

# First frame kept on the target shot. Everything before it is dropped and
# the normalization is made on this frame. 0 keeps the whole record. Only the
# uncorrected curves depend on it: the corrected residual at the flash is
# built from differences, x(0) - x(-P), and does not see the reference.
START_FRAME = 500

SAVE_FIGURES = True

# ---- Dominant frequency -------------------------------------------------------
FREQ_WINDOW_MS = (None, -0.5)  # None: start of the record
FREQ_BAND_HZ = (5.0, 200.0)  # below 5 Hz is slow drift, not a vibration
ZERO_PADDING = 8  # finer frequency grid for the peak position
N_LOBES = 4  # strongest lobes listed and marked on the spectrum
LOBE_MIN_SPACING_HZ = 2.0  # two maxima closer than this belong to the same lobe
SPECTRUM_XLIM_HZ = (1.0, 2000.0)  # log panel
SPECTRUM_ZOOM_HZ = (15.0, 80.0)  # linear panel

# ---- Correction -----------------------------------------------------------------
LOOP_START_MS = -1000.0  # nothing is done before this instant
RESPONSE_TIME_MS = 10.0  # after a node, the value is not guaranteed

SHOW_PROGRESS = True

# ---- Learning base -----------------------------------------------------------------
# Any acquisition at a valid frame rate is kept as long as it covers the
# instants the learning base is read at: -2T (first node of the two-node
# correction of section 8), -T and the flash, with a small margin on both
# sides.
VALID_FRAME_RATES = {66000, 82000}
PRE_FLASH_MARGIN_MS = 1.0
POST_FLASH_MARGIN_MS = 1.0
EXCLUDED_IDS = {
    # "2025-11-28-17-07": "no save",
}

# ---- Figures -----------------------------------------------------------------------
WIDE_WINDOW_MS = (None, 50.0)  # None: start of the kept record
LOBES_WINDOW_MS = (-100.0, 5.0)
FLASH_WINDOW_MS = (-4.0, 0.5)
WIDE_STRIDE = 20  # one frame out of WIDE_STRIDE is drawn on the wide window
EFF_YLIM_LOBES = (80.0, 100.5)
EFF_YLIM_FLASH = (80.0, 100.5)
MARECHAL_RMS = 1.0 / 14.0
LEGEND_FONTSIZE = 7.0
PANEL_FONTSIZE = 8.0
TITLE_FONTSIZE = 9.0

RCPARAMS = {
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
}


# %% === 2. IMPORTS AND SETUP ===

import os
import sys

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass  # executed cell by cell, no __file__

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm as cmc
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.signal import find_peaks

from chocolate.config import load_config
from chocolate.database import load_database
from chocolate.efficiency import combining_efficiency_small_fluctuations
from chocolate.results_io import ZERNIKE_COLUMNS, load_beam_pair
from chocolate.style import FIGURE_WIDTH, apply_style, save_figure

if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

apply_style()
plt.rcParams.update(RCPARAMS)

COLOR_SPECTRUM = "tab:orange"
COLOR_TILT = "tab:orange"  # example aberration, uncorrected
COLOR_ALL = "tab:red"  # all aberrations, uncorrected
COLOR_CORRECTED = "tab:green"
COLOR_NODE = "black"
COLOR_RESPONSE = "grey"
COLOR_FLASH_LINE = "#D62728"
COLOR_MARECHAL = "tab:blue"  # tab:green is taken by the corrected curves
COLOR_LOBE = cmc.batlow(0.08)  # annotations

config = load_config()
if ROOT is not None:
    config.root = Path(ROOT)
config.check()

if config.zernike_mode != "rms":
    raise SystemExit(
        f"The efficiency model needs an orthonormal basis; config.yaml asks "
        f"for zernike_mode = {config.zernike_mode!r}, set it to 'rms'."
    )

db = load_database(config.root)


def progress(iterable, description, total=None):
    """Wrap an iterable in a progress bar when tqdm is available."""
    if not SHOW_PROGRESS:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=description, total=total, leave=False)


def efficiency_percent(values):
    """Combining efficiency of one set of RMS coefficients, in per cent."""
    values = np.asarray(values, dtype=float)
    return 100.0 * float(
        combining_efficiency_small_fluctuations(values[None, :, None])[0, 0]
    )


_marechal_block = np.zeros((1, 1, 1))
_marechal_block[0, 0, 0] = MARECHAL_RMS
MARECHAL_PERCENT = 100.0 * float(
    combining_efficiency_small_fluctuations(_marechal_block)[0, 0]
)


# %% === 3. TARGET SHOT: LOADING AND NORMALIZATION ===

data_a, data_b, target_time_s, target_names, _ = load_beam_pair(
    config.root,
    [TARGET_ID],
    zernike_mode=config.zernike_mode,
    columns=ZERNIKE_COLUMNS,
    verbose=True,
)
labels = [str(name) for name in target_names]
n_modes = len(labels)
i_example = labels.index(EXAMPLE_MODE)

time_ms = np.asarray(target_time_s, dtype=float) * 1e3
dt_ms = float(np.median(np.diff(time_ms)))
x = (data_a - data_b)[0].astype(float)  # (n_modes, n_times), waves RMS

# The first frame of this shot is empty (NaN on every Zernike column). Empty
# frames are replaced by linear interpolation in time, so that the first
# frame used for the normalization is the value of the second one.
n_filled = 0
for m in range(n_modes):
    bad = ~np.isfinite(x[m])
    if bad.any():
        x[m, bad] = np.interp(time_ms[bad], time_ms[~bad], x[m, ~bad])
        n_filled += int(bad.sum())

# Truncation, after the interpolation so that the kept frames never carry an
# empty value.
n_frames_total = x.shape[1]
if not 0 <= START_FRAME < n_frames_total - 1:
    raise SystemExit(
        f"START_FRAME = {START_FRAME} is outside the record (0 to {n_frames_total - 2})."
    )
x = x[:, START_FRAME:]
time_ms = time_ms[START_FRAME:]
if time_ms[0] > LOOP_START_MS:
    raise SystemExit(
        f"With START_FRAME = {START_FRAME} the record starts at {time_ms[0]:.1f} ms, "
        f"after LOOP_START_MS = {LOOP_START_MS} ms."
    )

x = x - x[:, [0]]
i_flash = int(np.argmin(np.abs(time_ms)))

print(
    f"Shot {TARGET_ID}: {n_frames_total} frames, {n_filled} empty value(s) interpolated"
)
print(
    f"  kept from frame {START_FRAME}: {time_ms[0]:.3f} to {time_ms[-1]:.1f} ms, "
    f"{x.shape[1]} frames, normalized on this first frame"
)


# %% === 4. DOMINANT FREQUENCY BEFORE THE FLASH ===

freq_start_ms = time_ms[0] if FREQ_WINDOW_MS[0] is None else FREQ_WINDOW_MS[0]
in_window = (time_ms >= freq_start_ms) & (time_ms <= FREQ_WINDOW_MS[1])
segment = x[i_example, in_window]

# A straight line is removed first, otherwise the slow drift leaks into the
# spectrum and hides the lines.
segment = segment - np.polyval(
    np.polyfit(time_ms[in_window], segment, 1), time_ms[in_window]
)

# Normalized so that a sinusoid of amplitude A reads A on the spectrum.
window = np.hanning(segment.size)
n_fft = ZERO_PADDING * segment.size
spectrum = 2.0 * np.abs(np.fft.rfft(segment * window, n=n_fft)) / window.sum()
freq_hz = np.fft.rfftfreq(n_fft, d=dt_ms * 1e-3)
in_band = (freq_hz >= FREQ_BAND_HZ[0]) & (freq_hz <= FREQ_BAND_HZ[1])

f_main_hz = float(freq_hz[in_band][np.argmax(spectrum[in_band])])
T_MS = 1000.0 / f_main_hz

# The other lobes, for the record and for the figure. Zero padding creates
# many local maxima around each line, hence the minimum spacing.
band_indices = np.flatnonzero(in_band)
lobe_positions, _ = find_peaks(
    spectrum[band_indices],
    distance=max(1, int(round(LOBE_MIN_SPACING_HZ / (freq_hz[1] - freq_hz[0])))),
)
lobe_indices = band_indices[lobe_positions]
lobe_indices = lobe_indices[np.argsort(spectrum[lobe_indices])[::-1][:N_LOBES]]
lobe_freqs_hz = freq_hz[lobe_indices]
lobe_amplitudes = spectrum[lobe_indices]

print(
    f"Dominant frequency of {EXAMPLE_MODE} on [{freq_start_ms:.0f}, "
    f"{FREQ_WINDOW_MS[1]}] ms: {f_main_hz:.2f} Hz, T = {T_MS:.2f} ms"
)
PERIOD_MS = T_MS  # node period

# A correction read at -P and held until the flash leaves x(0) - x(-P). For a
# line of frequency f this is a comb filter of gain |H| = 2 |sin(pi f P)|:
# zero on the harmonics of 1/P, doubled halfway between them. With a random
# phase, a line of amplitude A leaves an RMS residual A |H| / sqrt(2).
comb_gain_t = 2.0 * np.abs(np.sin(np.pi * lobe_freqs_hz * T_MS * 1e-3))

print(f"Strongest lobes in {FREQ_BAND_HZ} Hz, and the loss each leaves at the flash:")
print(f"  {'f (Hz)':>8} {'period':>9} {'amplitude':>10} {'|H|':>6} {'loss':>8}")
for f_lobe, a_lobe, h_t in zip(lobe_freqs_hz, lobe_amplitudes, comb_gain_t):
    loss_t = 100.0 - efficiency_percent([a_lobe * h_t / np.sqrt(2)])
    print(
        f"  {f_lobe:8.2f} {1000.0 / f_lobe:6.2f} ms {a_lobe:10.4f} {h_t:6.2f} {loss_t:7.3f}%"
    )


# %% === 5. LEARNING BASE: MEAN CHANGE BETWEEN -P AND THE FLASH ===

target_amps = db[TARGET_ID].get("AMPLI", {})

required_before_ms = 2.0 * T_MS + PRE_FLASH_MARGIN_MS

ids_learning = [
    shot
    for shot, meta in db.items()
    if shot != "__meta"
    and shot != TARGET_ID  # the corrected shot is never learned
    and shot not in EXCLUDED_IDS  # hand exclusions, see the parameter block
    and isinstance(
        meta.get("PHASICS", {}).get("cine_beginning"), (int, float)
    )  # numeric start
    and isinstance(meta.get("PHASICS", {}).get("cine_end"), (int, float))  # numeric end
    and meta["PHASICS"].get("frame_rate") in VALID_FRAME_RATES  # usable frame rate
    and 1000.0 * meta["PHASICS"]["cine_beginning"] / meta["PHASICS"]["frame_rate"]
    <= -required_before_ms  # starts early enough
    and 1000.0 * meta["PHASICS"]["cine_end"] / meta["PHASICS"]["frame_rate"]
    >= POST_FLASH_MARGIN_MS  # ends after the flash
    and meta.get("AMPLI", {}).get("45A") == target_amps.get("45A")  # same rod A state
    and meta.get("AMPLI", {}).get("45B") == target_amps.get("45B")  # same rod B state
    and meta.get("AMPLI", {}).get("108A")
    == target_amps.get("108A")  # same slab A state
    and meta.get("AMPLI", {}).get("108B")
    == target_amps.get("108B")  # same slab B state
]

learn_a, learn_b, learn_time_s, learn_names, learn_shots = load_beam_pair(
    config.root,
    ids_learning,
    zernike_mode=config.zernike_mode,
    columns=ZERNIKE_COLUMNS,
    verbose=False,
)
if [str(name) for name in learn_names] != labels:
    raise SystemExit("Learning base and target do not share the same modes.")

learn = (learn_a - learn_b).astype(float)  # (n_shots, n_modes, n_times)
learn_time_ms = np.asarray(learn_time_s, dtype=float) * 1e3
n_learn = learn.shape[0]
if n_learn < 2:
    raise SystemExit("At least two learning shots are needed (leave-one-out).")

print(
    f"\nLearning selection: {len(ids_learning)} shots start before "
    f"{-required_before_ms:.1f} ms; common grid {learn.shape[2]} frames"
)

# Same treatment of empty frames as for the target.
n_filled_learn = 0
finite_time = np.isfinite(learn_time_ms)
for s_ in range(n_learn):
    for m in range(n_modes):
        bad = ~np.isfinite(learn[s_, m]) & finite_time
        if bad.any():
            good = np.isfinite(learn[s_, m]) & finite_time
            learn[s_, m, bad] = np.interp(
                learn_time_ms[bad], learn_time_ms[good], learn[s_, m, good]
            )
            n_filled_learn += int(bad.sum())
print(f"  {n_filled_learn} empty value(s) interpolated in the learning base")

# nanargmin: np.argmin returns the position of the first NaN if the time
# vector has one, which silently gives index 0.
if not np.all(np.isfinite(learn_time_ms)):
    print(
        f"  WARNING: {np.sum(~np.isfinite(learn_time_ms))} non-finite value(s) in the learning time vector"
    )
learn_dt_ms = float(np.nanmedian(np.diff(learn_time_ms)))
learn_i_flash = int(np.nanargmin(np.abs(learn_time_ms)))
print(
    f"\nLearning base: {len(learn_shots)} shots, {learn_time_ms.size} frames, "
    f"step {learn_dt_ms * 1e3:.2f} us, from {np.nanmin(learn_time_ms):.2f} "
    f"to {np.nanmax(learn_time_ms):.2f} ms"
)
print(f"  flash : index {learn_i_flash}, t = {learn_time_ms[learn_i_flash]:.4f} ms")
if abs(learn_time_ms[learn_i_flash]) > learn_dt_ms:
    raise SystemExit(
        "The learning time vector does not reach 0 within one frame: check "
        "that it is in seconds and centered on the flash."
    )


def learned_abb(lag_ms):
    """Mean and spread over the learning base of x(0) - x(-lag), per aberration.

    Returns
    -------
    mean, std : numpy.ndarray, shape (n_modes,)
    index : int
        Frame of the learning grid used for -lag.
    """
    index = int(np.nanargmin(np.abs(learn_time_ms + lag_ms)))
    if abs(learn_time_ms[index] + lag_ms) > learn_dt_ms:
        raise SystemExit(
            f"-{lag_ms:.2f} ms is not on the learning grid, which starts at "
            f"{np.nanmin(learn_time_ms):.2f} ms."
        )
    per_shot = learn[:, :, learn_i_flash] - learn[:, :, index]
    return np.nanmean(per_shot, axis=0), np.nanstd(per_shot, axis=0), index


ABB_SHOT, ABB_SHOT_STD, learn_i_minus_p = learned_abb(PERIOD_MS)
PRECOMP = -ABB_SHOT  # value the corrected aberration must hold at -P
print(f"  -P    : index {learn_i_minus_p}, t = {learn_time_ms[learn_i_minus_p]:.4f} ms")

# This shot's own change over the same lag. If it is far from what the base
# learned, in units of the spread of the base, no loop start and no period
# can save the correction: the residual at the flash is exactly this gap.
own_i_minus_p = int(np.argmin(np.abs(time_ms + PERIOD_MS)))
own_delta = x[:, i_flash] - x[:, own_i_minus_p]

print(
    f"\n  {'aberration':<22} {'ABB_SHOT':>9} {'std':>8} {'this shot':>10} {'gap/std':>8}   (waves RMS)"
)
for m, name in enumerate(labels):
    gap = (
        (own_delta[m] - ABB_SHOT[m]) / ABB_SHOT_STD[m]
        if ABB_SHOT_STD[m] > 0
        else np.nan
    )
    print(
        f"  {name:<22} {ABB_SHOT[m]:9.4f} {ABB_SHOT_STD[m]:8.4f} {own_delta[m]:10.4f} {gap:8.1f}"
    )


# %% === 6. CORRECTION LOOP ===
#
# Written for all aberrations at once: every line acts on a vector with one
# entry per aberration, Tilt Y being one of them.


def run_correction(period_ms, start_ms, offset_ms=0.0, full=False):
    """Node correction of every aberration.

    Nodes sit at offset_ms - k * period_ms, from the first one after start_ms
    down to offset_ms - period_ms. At each node the corrected value is read
    and brought to -ABB_SHOT, learned at lag period_ms; the correction is
    applied exactly and instantly, then held. offset_ms shifts the schedule and the evaluation instant
    together, as a clock offset between this shot and the learning base would.

    Returns
    -------
    y_eval : numpy.ndarray, shape (n_modes,)
        Corrected aberrations at offset_ms.
    node_times_ms : numpy.ndarray
    precomp : numpy.ndarray, shape (n_modes,)
    y : numpy.ndarray, shape (n_modes, n_times), or None when full is False
    """
    abb, _, _ = learned_abb(period_ms)
    precomp = -abb

    k_first = int(np.floor((offset_ms - start_ms) / period_ms + 1e-9))
    if k_first < 1:
        raise SystemExit(f"No node between {start_ms} ms and the flash.")
    node_times = offset_ms - period_ms * np.arange(k_first, 0, -1)
    node_idx = np.clip(
        np.rint((node_times - time_ms[0]) / dt_ms).astype(int), 0, x.shape[1] - 1
    )

    current = np.zeros(n_modes)
    held = []
    for k, i_node in enumerate(node_idx):
        measured = x[:, i_node] + current  # y(t_k), what the sensor reads
        current = current + (precomp - measured)
        held.append(current.copy())

    i_eval = int(np.argmin(np.abs(time_ms - offset_ms)))
    y_eval = x[:, i_eval] + current
    if not full:
        return y_eval, node_times, precomp, None

    correction = np.zeros_like(x)  # C(t), held between nodes
    bounds = list(node_idx) + [x.shape[1]]
    for j, level in enumerate(held):
        correction[:, bounds[j] : bounds[j + 1]] = level[:, None]
    return y_eval, node_times, precomp, x + correction


_, node_times_ms, _, y = run_correction(PERIOD_MS, LOOP_START_MS, full=True)

print(
    f"\n{node_times_ms.size} correction nodes, from {node_times_ms[0]:.1f} "
    f"to {node_times_ms[-1]:.2f} ms"
)
print(f"At the flash (t = {time_ms[i_flash]:.3f} ms), waves RMS:")
print(f"  {'aberration':<22} {'raw':>9} {'corrected':>10}")
for m, name in enumerate(labels):
    print(f"  {name:<22} {x[m, i_flash]:9.4f} {y[m, i_flash]:10.4f}")


# %% === 7. COMBINING EFFICIENCY ===

eff_raw = 100.0 * combining_efficiency_small_fluctuations(x[None, :, :])[0]
eff_corrected = 100.0 * combining_efficiency_small_fluctuations(y[None, :, :])[0]

# Share of each aberration in the corrected loss at the flash.
loss_per_mode = np.array(
    [
        100.0
        - efficiency_percent(np.where(np.arange(n_modes) == m, y[:, i_flash], 0.0))
        for m in range(n_modes)
    ]
)

print("\nEfficiency at the flash, piston excluded:")
print(f"  raw (normalized to the first frame) {eff_raw[i_flash]:7.2f} %")
print(f"  corrected                           {eff_corrected[i_flash]:7.2f} %")
print("  corrected loss per aberration:")
for name, loss in zip(labels, loss_per_mode):
    print(f"    {name:<22} {loss:6.3f} %")


# %% === 8. LEARNING BASE: TWO-NODE CORRECTION SHOT BY SHOT ===
#
# Each learning shot is corrected like the target, with the only two nodes
# its record offers before the flash, -2T and -T. Each shot is normalized to
# its own first frame, and its ABB_SHOT is learned on all the other shots
# (leave one out), so that no shot corrects itself. With a perfect actuator
# the node at -2T does not change the result, it is kept to mirror the loop.

learn_i_minus_t = int(np.nanargmin(np.abs(learn_time_ms + T_MS)))
learn_i_minus_2t = int(np.nanargmin(np.abs(learn_time_ms + 2.0 * T_MS)))
if abs(learn_time_ms[learn_i_minus_2t] + 2.0 * T_MS) > learn_dt_ms:
    raise SystemExit(f"-2T = {-2 * T_MS:.2f} ms is not on the learning grid.")

# Change between -T and the flash, shot by shot, and its sum over the base:
# the leave-one-out mean is (sum - own) / (n - 1).
change_per_shot = learn[:, :, learn_i_flash] - learn[:, :, learn_i_minus_t]
change_sum = change_per_shot.sum(axis=0)

eff_learn_raw = np.empty(n_learn)
eff_learn_corrected = np.empty(n_learn)

for s_ in progress(range(n_learn), "Learning shots", total=n_learn):
    shot_x = learn[s_] - learn[s_, :, 0][:, None]  # normalized to its first frame
    precomp_loo = -(change_sum - change_per_shot[s_]) / (n_learn - 1)

    # Node -2T: read and bring the shot to the target.
    current = precomp_loo - shot_x[:, learn_i_minus_2t]
    # Node -T, the last one: read and bring it back to the target.
    current = current + (precomp_loo - (shot_x[:, learn_i_minus_t] + current))

    eff_learn_raw[s_] = efficiency_percent(shot_x[:, learn_i_flash])
    eff_learn_corrected[s_] = efficiency_percent(shot_x[:, learn_i_flash] + current)

print("\nLearning base, two nodes, efficiency at the flash (piston excluded):")
print(
    f"  uncorrected  {np.mean(eff_learn_raw):7.2f} % +/- {np.std(eff_learn_raw):5.2f} %"
)
print(
    f"  corrected    {np.mean(eff_learn_corrected):7.2f} % +/- {np.std(eff_learn_corrected):5.2f} %"
)
print(
    f"  corrected above Marechal: {np.sum(eff_learn_corrected >= MARECHAL_PERCENT)}/{n_learn}, "
    f"above 99 %: {np.sum(eff_learn_corrected >= 99.0)}/{n_learn}"
)


# %% === 9. FIGURE 1: SPECTRUM AND MAIN LOBES ===

fig, axs = plt.subplots(2, 1, figsize=(FIGURE_WIDTH, 5.0))
in_log = (freq_hz >= SPECTRUM_XLIM_HZ[0]) & (freq_hz <= SPECTRUM_XLIM_HZ[1])
in_zoom = (freq_hz >= SPECTRUM_ZOOM_HZ[0]) & (freq_hz <= SPECTRUM_ZOOM_HZ[1])

# ---- Panel (a): whole spectrum, log scales -----------------------------------
ax = axs[0]
ax.axvspan(*FREQ_BAND_HZ, color=COLOR_RESPONSE, alpha=0.12, linewidth=0)
ax.loglog(freq_hz[in_log], spectrum[in_log], color=COLOR_SPECTRUM, linewidth=0.7)
for f_lobe in lobe_freqs_hz:
    ax.axvline(f_lobe, color=COLOR_LOBE, linestyle=":", linewidth=0.8)
ax.axvline(f_main_hz, color=COLOR_LOBE, linestyle="--", linewidth=1.1)
ax.set_xlim(*SPECTRUM_XLIM_HZ)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Amplitude (waves RMS)")
ax.set_title(
    f"{EXAMPLE_MODE} on [{freq_start_ms:.0f}, {FREQ_WINDOW_MS[1]}] ms, "
    f"resolution {1000.0 / (segment.size * dt_ms):.2f} Hz",
    fontsize=PANEL_FONTSIZE,
)

# ---- Panel (b): zoom on the lobes, linear scales -----------------------------
ax = axs[1]
ax.plot(freq_hz[in_zoom], spectrum[in_zoom], color=COLOR_SPECTRUM, linewidth=0.9)
for f_lobe, a_lobe in zip(lobe_freqs_hz, lobe_amplitudes):
    if SPECTRUM_ZOOM_HZ[0] <= f_lobe <= SPECTRUM_ZOOM_HZ[1]:
        ax.axvline(f_lobe, color=COLOR_LOBE, linestyle=":", linewidth=0.8)
        ax.annotate(
            f"{f_lobe:.2f} Hz\n{1000.0 / f_lobe:.2f} ms",
            xy=(f_lobe, a_lobe),
            xytext=(4, -2),
            textcoords="offset points",
            fontsize=LEGEND_FONTSIZE,
            color=COLOR_LOBE,
            va="top",
        )
ax.axvline(f_main_hz, color=COLOR_LOBE, linestyle="--", linewidth=1.1)
ax.set_xlim(*SPECTRUM_ZOOM_HZ)
ax.set_ylim(bottom=0.0)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Amplitude (waves RMS)")
ax.set_title("Main lobes", fontsize=PANEL_FONTSIZE)

handles = [
    Line2D([0], [0], color=COLOR_SPECTRUM),
    Patch(facecolor=COLOR_RESPONSE, alpha=0.12, linewidth=0),
    Line2D([0], [0], color=COLOR_LOBE, linestyle=":", linewidth=0.8),
    Line2D([0], [0], color=COLOR_LOBE, linestyle="--", linewidth=1.1),
]
texts = [
    f"{EXAMPLE_MODE}, Hann window, linear trend removed",
    f"Search band, {FREQ_BAND_HZ[0]:.0f} to {FREQ_BAND_HZ[1]:.0f} Hz",
    f"{N_LOBES} strongest lobes",
    f"Main lobe, {f_main_hz:.2f} Hz, T = {T_MS:.2f} ms",
]
fig.tight_layout(rect=(0, 0.09, 1, 0.97))
fig.legend(
    handles,
    texts,
    loc="lower center",
    ncol=2,
    fontsize=LEGEND_FONTSIZE,
    bbox_to_anchor=(0.5, 0.0),
)
fig.suptitle(
    f"Shot {TARGET_ID}: dominant vibration before the flash", fontsize=TITLE_FONTSIZE
)
if SAVE_FIGURES:
    save_figure(fig, config.figures_dir / "node_correction_spectrum.pdf")


# %% === 10. FIGURE 2: TILT Y, UNCORRECTED AND CORRECTED ===

wide_start = time_ms[0] if WIDE_WINDOW_MS[0] is None else WIDE_WINDOW_MS[0]
wide = np.flatnonzero((time_ms >= wide_start) & (time_ms <= WIDE_WINDOW_MS[1]))[
    ::WIDE_STRIDE
]
lobes = np.flatnonzero(
    (time_ms >= LOBES_WINDOW_MS[0]) & (time_ms <= LOBES_WINDOW_MS[1])
)
flash = np.flatnonzero(
    (time_ms >= FLASH_WINDOW_MS[0]) & (time_ms <= FLASH_WINDOW_MS[1])
)

# The wide window is scaled on the part before the flash: the ringing after
# the shot is ten times larger and would flatten the rest.
before_flash = time_ms < 0.0
wide_low = min(x[i_example, before_flash].min(), y[i_example, before_flash].min())
wide_high = max(x[i_example, before_flash].max(), y[i_example, before_flash].max())
wide_margin = 0.1 * (wide_high - wide_low)

fig, axs = plt.subplots(3, 1, figsize=(FIGURE_WIDTH, 6.6))

# ---- Row 0: whole kept record ----------------------------------------------
ax = axs[0]
for t_node in node_times_ms[node_times_ms >= wide_start]:
    ax.axvspan(
        t_node, t_node + RESPONSE_TIME_MS, color=COLOR_RESPONSE, alpha=0.06, linewidth=0
    )
    ax.axvline(t_node, color=COLOR_NODE, linestyle="--", linewidth=0.3, alpha=0.4)
# Uncorrected drawn on top: before the loop starts the two curves coincide.
ax.plot(
    time_ms[wide], y[i_example, wide], color=COLOR_CORRECTED, linewidth=0.6, zorder=2
)
ax.plot(time_ms[wide], x[i_example, wide], color=COLOR_TILT, linewidth=0.6, zorder=3)
ax.axhline(PRECOMP[i_example], color=COLOR_CORRECTED, linestyle=":", linewidth=1.0)
ax.axvline(0.0, color=COLOR_FLASH_LINE, linestyle="--", linewidth=1.0)
ax.set_xlim(wide_start, WIDE_WINDOW_MS[1])
ax.set_ylim(wide_low - wide_margin, wide_high + wide_margin)
ax.set_ylabel(f"{EXAMPLE_MODE} (waves RMS)")
ax.set_title("2 s range)", fontsize=PANEL_FONTSIZE)

# ---- Row 1: last lobes -----------------------------------------------------
ax = axs[1]
for t_node in node_times_ms[node_times_ms >= LOBES_WINDOW_MS[0] - RESPONSE_TIME_MS]:
    ax.axvspan(
        t_node, t_node + RESPONSE_TIME_MS, color=COLOR_RESPONSE, alpha=0.25, linewidth=0
    )
    ax.axvline(t_node, color=COLOR_NODE, linestyle="--", linewidth=0.9)
ax.plot(time_ms[lobes], x[i_example, lobes], color=COLOR_TILT, linewidth=0.9)
ax.plot(time_ms[lobes], y[i_example, lobes], color=COLOR_CORRECTED, linewidth=0.9)
ax.axhline(PRECOMP[i_example], color=COLOR_CORRECTED, linestyle=":", linewidth=1.0)
ax.axhline(0.0, color="grey", linewidth=0.5, alpha=0.6)
ax.axvline(0.0, color=COLOR_FLASH_LINE, linestyle="--", linewidth=1.0)
ax.set_xlim(*LOBES_WINDOW_MS)
ax.set_ylabel(f"{EXAMPLE_MODE} (waves RMS)")
ax.set_title("100 ms range", fontsize=PANEL_FONTSIZE)

# ---- Row 2: around the flash -----------------------------------------------
ax = axs[2]
ax.plot(time_ms[flash], x[i_example, flash], color=COLOR_TILT, linewidth=1.0)
ax.plot(time_ms[flash], y[i_example, flash], color=COLOR_CORRECTED, linewidth=1.0)
ax.axhline(PRECOMP[i_example], color=COLOR_CORRECTED, linestyle=":", linewidth=1.0)
ax.axhline(0.0, color="grey", linewidth=0.5, alpha=0.6)
ax.axvline(0.0, color=COLOR_FLASH_LINE, linestyle="--", linewidth=1.0)
ax.set_xlim(*FLASH_WINDOW_MS)
ax.set_xlabel("Time (ms)")
ax.set_ylabel(f"{EXAMPLE_MODE} (waves RMS)")
ax.set_title("5 ms range : aound the flash", fontsize=PANEL_FONTSIZE)

handles = [
    Line2D([0], [0], color=COLOR_TILT),
    Line2D([0], [0], color=COLOR_CORRECTED),
    Line2D([0], [0], color=COLOR_CORRECTED, linestyle=":"),
    Line2D([0], [0], color=COLOR_NODE, linestyle="--", linewidth=0.9),
    Patch(facecolor=COLOR_RESPONSE, alpha=0.25, linewidth=0),
    Line2D([0], [0], color=COLOR_FLASH_LINE, linestyle="--"),
]
texts = [
    f"{EXAMPLE_MODE}, uncorrected (normalized at frame {START_FRAME})",
    f"{EXAMPLE_MODE}, corrected, perfect actuator",
    f"Precompensation target, -ABB_SHOT = {PRECOMP[i_example]:.3f} wave",
    f"Correction nodes, every {PERIOD_MS:.2f} ms",
    f"Response time, {RESPONSE_TIME_MS:.0f} ms",
    "Flash maximum",
]
fig.tight_layout(rect=(0, 0.1, 1, 0.97))
fig.legend(
    handles,
    texts,
    loc="lower center",
    ncol=2,
    fontsize=LEGEND_FONTSIZE,
    bbox_to_anchor=(0.5, 0.0),
)

if SAVE_FIGURES:
    save_figure(fig, config.figures_dir / "node_correction_tilt_y.pdf")


# %% === 11. FIGURE 3: COMBINING EFFICIENCY, ALL ABERRATIONS ===
#
# Small-fluctuation model of chocolate.efficiency, piston excluded.

fig, axs = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 3.4))

# ---- Left: last lobes -------------------------------------------------------
ax = axs[0]
for t_node in node_times_ms[node_times_ms >= LOBES_WINDOW_MS[0] - RESPONSE_TIME_MS]:
    ax.axvspan(
        t_node, t_node + RESPONSE_TIME_MS, color=COLOR_RESPONSE, alpha=0.25, linewidth=0
    )
    ax.axvline(t_node, color=COLOR_NODE, linestyle="--", linewidth=0.9)
ax.plot(time_ms[lobes], eff_raw[lobes], color=COLOR_ALL, linewidth=0.9)
ax.plot(time_ms[lobes], eff_corrected[lobes], color=COLOR_CORRECTED, linewidth=0.9)
ax.axvline(0.0, color=COLOR_FLASH_LINE, linestyle="--", linewidth=1.0)
ax.set_xlim(*LOBES_WINDOW_MS)
ax.set_ylim(*EFF_YLIM_LOBES)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Combining efficiency, no piston (%)")
ax.set_title("100 ms range+", fontsize=PANEL_FONTSIZE)

# ---- Right: around the flash ------------------------------------------------
ax = axs[1]
ax.plot(time_ms[flash], eff_raw[flash], color=COLOR_ALL, linewidth=1.0)
ax.plot(time_ms[flash], eff_corrected[flash], color=COLOR_CORRECTED, linewidth=1.0)
ax.axvline(0.0, color=COLOR_FLASH_LINE, linestyle="--", linewidth=1.0)
ax.set_xlim(*FLASH_WINDOW_MS)
ax.set_ylim(*EFF_YLIM_FLASH)
ax.set_xlabel("Time (ms)")
ax.set_title("5 ms range : around the flash", fontsize=PANEL_FONTSIZE)

handles = [
    Line2D([0], [0], color=COLOR_ALL),
    Line2D([0], [0], color=COLOR_CORRECTED),
    Line2D([0], [0], color=COLOR_NODE, linestyle="--", linewidth=0.9),
    Patch(facecolor=COLOR_RESPONSE, alpha=0.25, linewidth=0),
    Line2D([0], [0], color=COLOR_FLASH_LINE, linestyle="--"),
]
texts = [
    f"All aberrations, uncorrected ({eff_raw[i_flash]:.1f} % at the flash)",
    f"All aberrations, corrected ({eff_corrected[i_flash]:.2f} % at the flash)",
    "Correction nodes",
    f"Response time, {RESPONSE_TIME_MS:.0f} ms",
    "Flash maximum",
]
fig.tight_layout(rect=(0, 0.17, 1, 0.95))
fig.legend(
    handles,
    texts,
    loc="lower center",
    ncol=2,
    fontsize=LEGEND_FONTSIZE,
    bbox_to_anchor=(0.5, 0.0),
)

if SAVE_FIGURES:
    save_figure(fig, config.figures_dir / "node_correction_efficiency.pdf")


# %% === 12. SHOW ===

plt.show()
