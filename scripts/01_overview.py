"""Overview of the on-shot behaviour of the 108 mm slabs, beam A.

Quick tour of the condensed results, built from the shots where only the
108 mm slabs are fired and whose acquisition is long enough for the zoom
window. It reuses the building blocks of on_shot_distortion1.py,
spectral_distortion.py and std_over_time.py.

Four output figures:
    01_overview_WF_A.pdf      Zernike coefficients of beam A, every shot
    01_overview_PSD_A.pdf     their power spectral density, before / after flash
    01_overview_piston.pdf    unwrapped differential piston, every shot
    01_overview_std.pdf       shot-to-shot std, beam A (top) and piston (bottom)
"""

# =============================================================================
# %% PARAMETERS
# =============================================================================

# Directory containing data_base.json and RESULTS/.
# None = use the value of config/config.yaml.
ROOT = None

# Restrict to the shots of one day, e.g. "2025-12-02".  None = every day.
DAY = None
# At most this many shots, to keep a first run fast.  None = no limit.
MAX_SHOTS = None

WINDOW_ZOOM = 1000  # frames — half-width required around the shot
SHOT_ALPHA = 0.2  # transparency of individual shot traces
MAX_ALPHA_BACKGROUND = 0.5  # maximum transparency of the background

VALID_FRAME_RATES = (82000, 66000)
# The piston is only reliable at 66000 fps, as in on_shot_distortion1.py.
PISTON_FRAME_RATES = (66000,)

# Displayed time axis limits in ms.
XLIM_ZOOM = (-1.0, 1.0)

# Displayed amplitude limits in waves.  None = automatic.
YLIM_WF = (-0.25, 0.50)
YLIM_PISTON = (-2, 2)

# Time boundary separating pre- and post-flash segments of the PSD, in ms.
SPLIT_MS = -0.5
# Frequency axis display limits in Hz, capped to the Nyquist frequency.
XLIM_FREQ = (1, 50000)

# =============================================================================
# %% IMPORTS AND SETUP
# =============================================================================

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d

from chocolate.config import load_config
from chocolate.database import load_database
from chocolate.efficiency import zero_at_reference
from chocolate.plotting import (
    add_bottom_legend,
    add_flash_background,
    zernike_legend_handles,
)
from chocolate.results_io import ZERNIKE_COLUMNS, load_beam_pair, load_piston_results
from chocolate.style import (
    FIGURE_WIDTH,
    PISTON_COLOR,
    apply_style,
    save_figure,
    zernike_line_style,
)
from chocolate.zernike import format_zernike_mode

apply_style()

config = load_config()
if ROOT is not None:
    config.root = Path(ROOT)
config.check()

db = load_database(config.root)  # dict {shot_id: metadata}

NYQUIST_HZ = min(VALID_FRAME_RATES) / 2.0
XLIM_FREQ = (XLIM_FREQ[0], min(XLIM_FREQ[1], NYQUIST_HZ))

print(f"Data directory : {config.root}")
print(f"Figures        : {config.figures_dir}")
print(f"Normalisation  : {config.zernike_mode}")
print(f"Reference time : {config.time_ref_ms} ms")

unit = format_zernike_mode(config.zernike_mode)


# =============================================================================
# %% HELPER — PSD
# =============================================================================


def compute_psd(signal, time):
    """One-sided PSD in [unit²/Hz] with Hann windowing.
    Returns (freqs_Hz, psd) or (None, None) if the segment is too short.
    """
    n = len(signal)
    if n < 4:
        return None, None
    dt = float(np.mean(np.diff(time)))
    fs = 1.0 / dt
    win = np.hanning(n)
    raw = np.abs(np.fft.rfft((signal - signal.mean()) * win)) ** 2
    psd = 2.0 * raw / (fs * np.sum(win**2))
    psd[0] /= 2.0
    if n % 2 == 0:
        psd[-1] /= 2.0
    return np.fft.rfftfreq(n, d=dt), psd


# =============================================================================
# %% SHOT SELECTION AND LOADING
# =============================================================================


def slab_zoom_shots(frame_rates):
    """Slab-only shots long enough for the zoom window, chronological."""
    ids = sorted(
        sid
        for sid, meta in db.items()
        if sid != "__meta"
        and meta["AMPLI"].get("45A") == False  # rods A off
        and meta["AMPLI"].get("45B") == False  # rods B off
        and meta["AMPLI"].get("108A") == True  # slab A fired
        and meta["AMPLI"].get("108B") == True  # slab B fired
        and meta["PHASICS"].get("frame_rate") in frame_rates
        and meta["PHASICS"].get("cine_beginning") <= -WINDOW_ZOOM
        and meta["PHASICS"].get("cine_end") >= WINDOW_ZOOM
        and (DAY is None or sid.startswith(DAY))
    )
    return ids[:MAX_SHOTS] if MAX_SHOTS is not None else ids


ids_wf = slab_zoom_shots(VALID_FRAME_RATES)
ids_piston = slab_zoom_shots(PISTON_FRAME_RATES)
print(f"Selected       : {len(ids_wf)} shots (wavefront), "
      f"{len(ids_piston)} shots (piston)")
if not ids_wf:
    raise SystemExit("No slab shot matches the selection, relax the parameters.")

# ---- beam A wavefront -------------------------------------------------------
data_a, _, time, labels, shot_ids = load_beam_pair(
    config.root,
    ids_wf,
    zernike_mode=config.zernike_mode,
    columns=ZERNIKE_COLUMNS,
    verbose=False,
)
data_a = zero_at_reference(data_a, time, config.time_ref_ms)
time_ms = time * 1e3
in_zoom = (time_ms >= XLIM_ZOOM[0]) & (time_ms <= XLIM_ZOOM[1])

# ---- differential piston, interpolated on the beam A time base --------------
piston = load_piston_results(config.root, ids_piston, unwrap=True, verbose=False)
piston_traces = []  # list of (time_s, piston_waves), one per shot
for entry in piston.values():
    if entry.get("piston") is None:
        continue
    t = np.asarray(entry["time"], dtype=float)
    p = zero_at_reference(
        np.asarray(entry["piston"], dtype=float), t, config.time_ref_ms
    )
    piston_traces.append((t, p))
piston_ids = list(piston)

print(f"Loaded         : {data_a.shape[0]} shots beam A, "
      f"{len(piston_traces)} piston traces")


# =============================================================================
# %% FIGURE 1 — Wavefront distortions, slab A, zoom
# =============================================================================

fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 4.2))

for shot in range(data_a.shape[0]):
    for mode in range(data_a.shape[1]):
        ax.plot(
            time_ms,
            data_a[shot, mode],
            alpha=SHOT_ALPHA,
            linewidth=0.8,
            **zernike_line_style(labels[mode]),
        )
ax.set_xlim(*XLIM_ZOOM)
if YLIM_WF:
    ax.set_ylim(*YLIM_WF)
add_flash_background(ax, config.root, shot_ids, "A", max_alpha=MAX_ALPHA_BACKGROUND)
ax.set_title(f"Slab A — zoom  ({len(shot_ids)} shots)", fontsize=8)
ax.set_ylabel(f"Zernike {unit} (waves)")
ax.set_xlabel("Time (ms)")

handles, texts = zernike_legend_handles(list(labels), include_flash=True)
add_bottom_legend(fig, handles, texts, ncol=4, bottom=0.32)
save_figure(fig, config.figures_dir / "01_overview_WF_A.pdf")


# =============================================================================
# %% FIGURE 2 — PSD of the wavefront distortions, slab A
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 4.2), sharex=True, sharey=True)

mask_before = time_ms < SPLIT_MS
mask_after = time_ms > SPLIT_MS
n_seg = mask_before.sum()  # same length before and after, for a fair comparison

# ---- left : before flash ----------------------------------------------------
for shot in range(data_a.shape[0]):
    for mode in range(data_a.shape[1]):
        freqs, psd = compute_psd(data_a[shot, mode, mask_before], time[mask_before])
        if freqs is None:
            continue
        axes[0].plot(
            freqs,
            psd,
            alpha=SHOT_ALPHA,
            linewidth=0.8,
            **zernike_line_style(labels[mode]),
        )
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlim(*XLIM_FREQ)
axes[0].set_title(f"Slab A — before flash  ({len(shot_ids)} shots)", fontsize=8)
axes[0].set_ylabel(f"PSD ({unit} waves²·Hz⁻¹)")
axes[0].set_xlabel("Frequency (Hz)")

# ---- right : after flash ----------------------------------------------------
for shot in range(data_a.shape[0]):
    for mode in range(data_a.shape[1]):
        freqs, psd = compute_psd(
            data_a[shot, mode, mask_after][:n_seg], time[mask_after][:n_seg]
        )
        if freqs is None:
            continue
        axes[1].plot(
            freqs,
            psd,
            alpha=SHOT_ALPHA,
            linewidth=0.8,
            **zernike_line_style(labels[mode]),
        )
axes[1].set_title(f"Slab A — after flash  ({len(shot_ids)} shots)", fontsize=8)
axes[1].set_xlabel("Frequency (Hz)")

handles, texts = zernike_legend_handles(list(labels), include_flash=False)
add_bottom_legend(fig, handles, texts, ncol=4, bottom=0.32)
save_figure(fig, config.figures_dir / "01_overview_PSD_A.pdf")


# =============================================================================
# %% FIGURE 3 — Differential piston, slabs, zoom
# =============================================================================

fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 3.0))

for t, p in piston_traces:
    ax.plot(t * 1e3, p, color=PISTON_COLOR, alpha=SHOT_ALPHA, linewidth=0.8)
ax.set_xlim(*XLIM_ZOOM)
if YLIM_PISTON:
    ax.set_ylim(*YLIM_PISTON)
add_flash_background(ax, config.root, piston_ids, "A", max_alpha=MAX_ALPHA_BACKGROUND)
ax.set_title(f"Slab piston — zoom  ({len(piston_traces)} shots)", fontsize=8)
ax.set_ylabel("Differential piston (waves)")
ax.set_xlabel("Time (ms)")

handles = [plt.Line2D([0], [0], color=PISTON_COLOR, linewidth=1.2)]
texts = ["Unwrapped differential piston"]
handles_f, texts_f = zernike_legend_handles([], include_flash=True)
add_bottom_legend(fig, handles + handles_f, texts + texts_f, ncol=3, bottom=0.28)
save_figure(fig, config.figures_dir / "01_overview_piston.pdf")


# =============================================================================
# %% FIGURE 4 — Shot-to-shot std, slab A (top) and piston (bottom)
# =============================================================================

fig = plt.figure(figsize=(FIGURE_WIDTH, 6.0))
gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.10)
ax_A = fig.add_subplot(gs[0])
ax_piston = fig.add_subplot(gs[1], sharex=ax_A)

# Only the zoom window is plotted, so that the automatic vertical scale is
# set by what is visible and not by the much larger late-time excursions.

# ---- beam A — std across shots ----------------------------------------------
std_A = np.std(data_a[:, :, in_zoom], axis=0)  # shape (n_modes, n_times)
for mode in range(std_A.shape[0]):
    ax_A.plot(
        time_ms[in_zoom],
        std_A[mode],
        linewidth=1.0,
        zorder=3,
        **zernike_line_style(labels[mode]),
    )
add_flash_background(ax_A, config.root, shot_ids, "A", max_alpha=MAX_ALPHA_BACKGROUND)
ax_A.set_ylabel(f"Std {unit} (waves)")
ax_A.set_title(f"Slabs — shot-to-shot std  ({len(shot_ids)} shots)", fontsize=8)
ax_A.text(0.01, 0.97, "Beam A", transform=ax_A.transAxes, va="top", fontsize=8)
plt.setp(ax_A.get_xticklabels(), visible=False)

# ---- piston — std across shots ----------------------------------------------
traces = [
    interp1d(t, p, bounds_error=False, fill_value=np.nan)(time[in_zoom])
    for t, p in piston_traces
]
if traces:
    std_P = np.nanstd(np.array(traces), axis=0)
    ax_piston.plot(
        time_ms[in_zoom], std_P, color=PISTON_COLOR, linewidth=1.0, zorder=3
    )
add_flash_background(ax_piston, config.root, piston_ids, "A", max_alpha=MAX_ALPHA_BACKGROUND)
ax_piston.set_xlim(*XLIM_ZOOM)
ax_piston.set_ylabel("Std piston (waves)")
ax_piston.set_xlabel("Time (ms)")
ax_piston.text(
    0.01,
    0.97,
    f"Piston ({len(traces)} shots)",
    transform=ax_piston.transAxes,
    va="top",
    fontsize=8,
)

# ---- common legend ----------------------------------------------------------
handles, texts = zernike_legend_handles(list(labels), include_flash=True)
handles.append(plt.Line2D([0], [0], color=PISTON_COLOR, linewidth=1.2))
texts.append("Piston")
add_bottom_legend(fig, handles, texts, ncol=5, bottom=0.22)
save_figure(fig, config.figures_dir / "01_overview_std.pdf")

print(f"Four figures written to {config.figures_dir}")


# =============================================================================
# %% SHOW
# =============================================================================
plt.show()
