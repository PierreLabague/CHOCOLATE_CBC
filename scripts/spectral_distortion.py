"""Power spectral density of the on-shot distortions.

Pre- and post-flash PSDs are shown side by side for visual comparison.
Only high-cadence shots are selected (frame_rate in VALID_FRAME_RATES) so
that the Nyquist frequency is consistent across all panels.

Four output figures:
    spectral_psd_WF_45.pdf
    spectral_psd_piston_45.pdf
    spectral_psd_WF_108.pdf
    spectral_psd_piston_108.pdf
"""

# =============================================================================
# %% PARAMETERS
# =============================================================================

ROOT = None


WINDOW_FULL = 4000  # frames — half-width for shot selection

# Only shots recorded at one of these frame rates are included.
# At 82000 fps: Nyquist = 41000 Hz.
# At 66000 fps: Nyquist = 33000 Hz.
VALID_FRAME_RATES = [82000, 66000]

# Time boundary separating pre- and post-flash segments, in ms.
SPLIT_MS = -0.5

ISOLATE_45 = True

SHOT_ALPHA = 0.2

# Frequency axis display limits in Hz.  Automatically capped to the
# Nyquist of the selected shots at runtime — see setup section below.
XLIM_FREQ = (1, 50000)

# =============================================================================
# %% IMPORTS AND SETUP
# =============================================================================

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from chocolate.config import load_config
from chocolate.database import load_database, select_ids
from chocolate.efficiency import zero_at_reference
from chocolate.plotting import add_bottom_legend, zernike_legend_handles
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

# Nyquist = min(fps)/2 so the axis is valid for all selected frame rates
NYQUIST_HZ = min(VALID_FRAME_RATES) / 2.0
XLIM_FREQ = (XLIM_FREQ[0], min(XLIM_FREQ[1], NYQUIST_HZ))

print(f"Data directory  : {config.root}")
print(f"Normalisation   : {config.zernike_mode}")
print(f"Reference time  : {config.time_ref_ms} ms")
print(f"Valid frame rates: {VALID_FRAME_RATES} fps")
print(f"Nyquist cap     : {NYQUIST_HZ:.0f} Hz  →  XLIM_FREQ = {XLIM_FREQ}")

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
# %% FIGURE 1 — PSD, Zernike WF, 45 mm rods
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH, 5.6), sharex=True, sharey=True)
labels = []

# ---- top-left : beam A, before flash ----------------------------------------
ids = [
    sid
    for sid, meta in db.items()
    if sid != "__meta"
    and meta["AMPLI"].get("45A") == True  # rods A fired
    and meta["AMPLI"].get("45B") == True  # rods B fired
    and meta["AMPLI"].get("108A") == False  # slab A off
    and meta["AMPLI"].get("108B") == False  # slab B off
    and meta["PHASICS"].get("frame_rate") in VALID_FRAME_RATES
    and meta["PHASICS"].get("cine_beginning") <= -WINDOW_FULL
    and meta["PHASICS"].get("cine_end") >= WINDOW_FULL
]
data_a, _, time, labels, _ = load_beam_pair(
    config.root,
    ids,
    zernike_mode=config.zernike_mode,
    columns=ZERNIKE_COLUMNS,
    verbose=False,
)
data_a = zero_at_reference(data_a, time, config.time_ref_ms)
time_ms = time * 1e3
mask_before = time_ms < SPLIT_MS
n_seg = mask_before.sum()
for shot in range(data_a.shape[0]):
    for mode in range(data_a.shape[1]):
        freqs, psd = compute_psd(data_a[shot, mode, mask_before], time[mask_before])
        if freqs is None:
            continue
        axes[0, 0].plot(
            freqs,
            psd,
            alpha=SHOT_ALPHA,
            linewidth=0.8,
            **zernike_line_style(labels[mode]),
        )
axes[0, 0].set_xscale("log")
axes[0, 0].set_yscale("log")
axes[0, 0].set_xlim(*XLIM_FREQ)
axes[0, 0].set_title(f"Rod A — before flash  ({len(ids)} shots)", fontsize=8)
axes[0, 0].set_ylabel(f"PSD ({unit} waves²·Hz⁻¹)")

# ---- top-right : beam A, after flash ----------------------------------------

mask_after = time_ms > SPLIT_MS
for shot in range(data_a.shape[0]):
    for mode in range(data_a.shape[1]):
        freqs, psd = compute_psd(
            data_a[shot, mode, mask_after][:n_seg], time[mask_after][:n_seg]
        )
        if freqs is None:
            continue
        axes[0, 1].plot(
            freqs,
            psd,
            alpha=SHOT_ALPHA,
            linewidth=0.8,
            **zernike_line_style(labels[mode]),
        )
axes[0, 1].set_xscale("log")
axes[0, 1].set_yscale("log")
axes[0, 1].set_xlim(*XLIM_FREQ)
axes[0, 1].set_title(f"Rod A — after flash  ({len(ids)} shots)", fontsize=8)

# ---- bottom-left : beam B, before flash -------------------------------------
ids = [
    sid
    for sid, meta in db.items()
    if sid != "__meta"
    and meta["AMPLI"].get("45A") == True  # rods A fired
    and meta["AMPLI"].get("45B") == True  # rods B fired
    and meta["AMPLI"].get("108A") == False  # slab A off
    and meta["AMPLI"].get("108B") == False  # slab B off
    and meta["PHASICS"].get("frame_rate") in VALID_FRAME_RATES
    and meta["PHASICS"].get("cine_beginning") <= -WINDOW_FULL
    and meta["PHASICS"].get("cine_end") >= WINDOW_FULL
]
_, data_b, time, labels, _ = load_beam_pair(
    config.root,
    ids,
    zernike_mode=config.zernike_mode,
    columns=ZERNIKE_COLUMNS,
    verbose=False,
)
data_b = zero_at_reference(data_b, time, config.time_ref_ms)
time_ms = time * 1e3
mask_before = time_ms < SPLIT_MS
n_seg = mask_before.sum()
for shot in range(data_b.shape[0]):
    for mode in range(data_b.shape[1]):
        freqs, psd = compute_psd(data_b[shot, mode, mask_before], time[mask_before])
        if freqs is None:
            continue
        axes[1, 0].plot(
            freqs,
            psd,
            alpha=SHOT_ALPHA,
            linewidth=0.8,
            **zernike_line_style(labels[mode]),
        )
axes[1, 0].set_xlim(*XLIM_FREQ)
axes[1, 0].set_title(f"Rod B — before flash  ({len(ids)} shots)", fontsize=8)
axes[1, 0].set_ylabel(f"PSD ({unit} waves²·Hz⁻¹)")
axes[1, 0].set_xlabel("Frequency (Hz)")

# ---- bottom-right : beam B, after flash -------------------------------------
mask_after = time_ms > SPLIT_MS
for shot in range(data_b.shape[0]):
    for mode in range(data_b.shape[1]):
        freqs, psd = compute_psd(
            data_b[shot, mode, mask_after][:n_seg], time[mask_after][:n_seg]
        )
        if freqs is None:
            continue
        axes[1, 1].plot(
            freqs,
            psd,
            alpha=SHOT_ALPHA,
            linewidth=0.8,
            **zernike_line_style(labels[mode]),
        )
axes[1, 1].set_xlim(*XLIM_FREQ)
axes[1, 1].set_title(f"Rod B — after flash  ({len(ids)} shots)", fontsize=8)
axes[1, 1].set_xlabel("Frequency (Hz)")

handles, texts = zernike_legend_handles(list(labels), include_flash=False)
add_bottom_legend(fig, handles, texts, ncol=4, bottom=0.20)
save_figure(fig, config.figures_dir / "spectral_psd_WF_45.pdf")


# =============================================================================
# %% FIGURE 2 — PSD, piston, 45 mm rods
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 3.0), sharex=True, sharey=True)

# ---- left : before flash ----------------------------------------------------
ids = [
    sid
    for sid, meta in db.items()
    if sid != "__meta"
    and meta["AMPLI"].get("45A") == True  # rods A fired
    and meta["AMPLI"].get("45B") == True  # rods B fired
    and meta["AMPLI"].get("108A") == False  # slab A off
    and meta["AMPLI"].get("108B") == False  # slab B off
    and meta["PHASICS"].get("frame_rate") == 66000
    and meta["PHASICS"].get("cine_beginning") <= -WINDOW_FULL
    and meta["PHASICS"].get("cine_end") >= WINDOW_FULL
]
piston = load_piston_results(config.root, ids, unwrap=True, verbose=False)
n_seg = None
for entry in piston.values():
    if entry.get("piston") is None:
        continue
    t = np.asarray(entry["time"], dtype=float)
    p = zero_at_reference(
        np.asarray(entry["piston"], dtype=float), t, config.time_ref_ms
    )
    mask_before = (t * 1e3) < SPLIT_MS
    if n_seg is None:
        n_seg = mask_before.sum()
    freqs, psd = compute_psd(p[mask_before], t[mask_before])
    if freqs is None:
        continue
    axes[0].plot(freqs, psd, color=PISTON_COLOR, alpha=SHOT_ALPHA, linewidth=0.8)
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlim(*XLIM_FREQ)
axes[0].set_title(f"Rod piston — before flash  ({len(ids)} shots)", fontsize=8)
axes[0].set_ylabel("PSD (waves²·Hz⁻¹)")
axes[0].set_xlabel("Frequency (Hz)")

# ---- right : after flash ----------------------------------------------------
for entry in piston.values():
    if entry.get("piston") is None:
        continue
    t = np.asarray(entry["time"], dtype=float)
    p = zero_at_reference(
        np.asarray(entry["piston"], dtype=float), t, config.time_ref_ms
    )
    mask_after = (t * 1e3) > SPLIT_MS
    freqs, psd = compute_psd(p[mask_after][:n_seg], t[mask_after][:n_seg])
    if freqs is None:
        continue
    axes[1].plot(freqs, psd, color=PISTON_COLOR, alpha=SHOT_ALPHA, linewidth=0.8)
axes[1].set_xlim(*XLIM_FREQ)
axes[1].set_title(f"Rod piston — after flash  ({len(ids)} shots)", fontsize=8)
axes[1].set_xlabel("Frequency (Hz)")

handles = [plt.Line2D([0], [0], color=PISTON_COLOR, linewidth=1.2)]
add_bottom_legend(fig, handles, ["Unwrapped differential piston"], ncol=1, bottom=0.22)
save_figure(fig, config.figures_dir / "spectral_psd_piston_45.pdf")


# =============================================================================
# %% FIGURE 3 — PSD, Zernike WF, 108 mm slabs
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH, 5.6), sharex=True, sharey=True)
labels = []

# ---- top-left : beam A, before flash ----------------------------------------
ids = [
    sid
    for sid, meta in db.items()
    if sid != "__meta"
    and meta["AMPLI"].get("45A") == False  # rods A off
    and meta["AMPLI"].get("45B") == False  # rods B off
    and meta["AMPLI"].get("108A") == True  # slab A fired
    and meta["AMPLI"].get("108B") == True  # slab B fired
    and meta["PHASICS"].get("frame_rate") in VALID_FRAME_RATES
    and meta["PHASICS"].get("cine_beginning") <= -WINDOW_FULL
    and meta["PHASICS"].get("cine_end") >= WINDOW_FULL
]
data_a, _, time, labels, _ = load_beam_pair(
    config.root,
    ids,
    zernike_mode=config.zernike_mode,
    columns=ZERNIKE_COLUMNS,
    verbose=False,
)
data_a = zero_at_reference(data_a, time, config.time_ref_ms)
time_ms = time * 1e3
mask_before = time_ms < SPLIT_MS
n_seg = mask_before.sum()
for shot in range(data_a.shape[0]):
    for mode in range(data_a.shape[1]):
        freqs, psd = compute_psd(data_a[shot, mode, mask_before], time[mask_before])
        if freqs is None:
            continue
        axes[0, 0].plot(
            freqs,
            psd,
            alpha=SHOT_ALPHA,
            linewidth=0.8,
            **zernike_line_style(labels[mode]),
        )
axes[0, 0].set_xscale("log")
axes[0, 0].set_yscale("log")
axes[0, 0].set_xlim(*XLIM_FREQ)
axes[0, 0].set_title(f"Slab A — before flash  ({len(ids)} shots)", fontsize=8)
axes[0, 0].set_ylabel(f"PSD ({unit} waves²·Hz⁻¹)")

# ---- top-right : beam A, after flash ----------------------------------------
mask_after = time_ms > SPLIT_MS
for shot in range(data_a.shape[0]):
    for mode in range(data_a.shape[1]):
        freqs, psd = compute_psd(
            data_a[shot, mode, mask_after][:n_seg], time[mask_after][:n_seg]
        )
        if freqs is None:
            continue
        axes[0, 1].plot(
            freqs,
            psd,
            alpha=SHOT_ALPHA,
            linewidth=0.8,
            **zernike_line_style(labels[mode]),
        )
axes[0, 1].set_xscale("log")
axes[0, 1].set_yscale("log")
axes[0, 1].set_xlim(*XLIM_FREQ)
axes[0, 1].set_title(f"Slab A — after flash  ({len(ids)} shots)", fontsize=8)

# ---- bottom-left : beam B, before flash -------------------------------------
ids = [
    sid
    for sid, meta in db.items()
    if sid != "__meta"
    and meta["AMPLI"].get("45A") == False  # rods A off
    and meta["AMPLI"].get("45B") == False  # rods B off
    and meta["AMPLI"].get("108A") == True  # slab A fired
    and meta["AMPLI"].get("108B") == True  # slab B fired
    and meta["PHASICS"].get("frame_rate") in VALID_FRAME_RATES
    and meta["PHASICS"].get("cine_beginning") <= -WINDOW_FULL
    and meta["PHASICS"].get("cine_end") >= WINDOW_FULL
]

_, data_b, time, labels, _ = load_beam_pair(
    config.root,
    ids,
    zernike_mode=config.zernike_mode,
    columns=ZERNIKE_COLUMNS,
    verbose=False,
)
data_b = zero_at_reference(data_b, time, config.time_ref_ms)
time_ms = time * 1e3
mask_before = time_ms < SPLIT_MS
n_seg = mask_before.sum()
for shot in range(data_b.shape[0]):
    for mode in range(data_b.shape[1]):
        freqs, psd = compute_psd(data_b[shot, mode, mask_before], time[mask_before])
        if freqs is None:
            continue
        axes[1, 0].plot(
            freqs,
            psd,
            alpha=SHOT_ALPHA,
            linewidth=0.8,
            **zernike_line_style(labels[mode]),
        )
axes[1, 0].set_xlim(*XLIM_FREQ)
axes[1, 0].set_title(f"Slab B — before flash  ({len(ids)} shots)", fontsize=8)
axes[1, 0].set_ylabel(f"PSD ({unit} waves²·Hz⁻¹)")
axes[1, 0].set_xlabel("Frequency (Hz)")

# ---- bottom-right : beam B, after flash -------------------------------------
mask_after = time_ms > SPLIT_MS
for shot in range(data_b.shape[0]):
    for mode in range(data_b.shape[1]):
        freqs, psd = compute_psd(
            data_b[shot, mode, mask_after][:n_seg], time[mask_after][:n_seg]
        )
        if freqs is None:
            continue
        axes[1, 1].plot(
            freqs,
            psd,
            alpha=SHOT_ALPHA,
            linewidth=0.8,
            **zernike_line_style(labels[mode]),
        )
axes[1, 1].set_xlim(*XLIM_FREQ)
axes[1, 1].set_title(f"Slab B — after flash  ({len(ids)} shots)", fontsize=8)
axes[1, 1].set_xlabel("Frequency (Hz)")

handles, texts = zernike_legend_handles(list(labels), include_flash=False)
add_bottom_legend(fig, handles, texts, ncol=4, bottom=0.20)
save_figure(fig, config.figures_dir / "spectral_psd_WF_108.pdf")


# =============================================================================
# %% FIGURE 4 — PSD, piston, 108 mm slabs
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 3.0), sharex=True, sharey=True)

# ---- left : before flash ----------------------------------------------------
ids = [
    sid
    for sid, meta in db.items()
    if sid != "__meta"
    and meta["AMPLI"].get("45A") == False  # rods A off
    and meta["AMPLI"].get("45B") == False  # rods B off
    and meta["AMPLI"].get("108A") == True  # slab A fired
    and meta["AMPLI"].get("108B") == True  # slab B fired
    and meta["PHASICS"].get("frame_rate") == 66000
    and meta["PHASICS"].get("cine_beginning") <= -WINDOW_FULL
    and meta["PHASICS"].get("cine_end") >= WINDOW_FULL
]
piston = load_piston_results(config.root, ids, unwrap=True, verbose=False)
n_seg = None
for entry in piston.values():
    if entry.get("piston") is None:
        continue
    t = np.asarray(entry["time"], dtype=float)
    p = zero_at_reference(
        np.asarray(entry["piston"], dtype=float), t, config.time_ref_ms
    )
    mask_before = (t * 1e3) < SPLIT_MS
    if n_seg is None:
        n_seg = mask_before.sum()
    freqs, psd = compute_psd(p[mask_before], t[mask_before])
    if freqs is None:
        continue
    axes[0].plot(freqs, psd, color=PISTON_COLOR, alpha=SHOT_ALPHA, linewidth=0.8)
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlim(*XLIM_FREQ)
axes[0].set_title(f"Slab piston — before flash  ({len(ids)} shots)", fontsize=8)
axes[0].set_ylabel("PSD (waves²·Hz⁻¹)")
axes[0].set_xlabel("Frequency (Hz)")

# ---- right : after flash ----------------------------------------------------
for entry in piston.values():
    if entry.get("piston") is None:
        continue
    t = np.asarray(entry["time"], dtype=float)
    p = zero_at_reference(
        np.asarray(entry["piston"], dtype=float), t, config.time_ref_ms
    )
    mask_after = (t * 1e3) > SPLIT_MS
    freqs, psd = compute_psd(p[mask_after][:n_seg], t[mask_after][:n_seg])
    if freqs is None:
        continue
    axes[1].plot(freqs, psd, color=PISTON_COLOR, alpha=SHOT_ALPHA, linewidth=0.8)
axes[1].set_xlim(*XLIM_FREQ)
axes[1].set_title(f"Slab piston — after flash  ({len(ids)} shots)", fontsize=8)
axes[1].set_xlabel("Frequency (Hz)")

handles = [plt.Line2D([0], [0], color=PISTON_COLOR, linewidth=1.2)]
add_bottom_legend(fig, handles, ["Unwrapped differential piston"], ncol=1, bottom=0.22)
save_figure(fig, config.figures_dir / "spectral_psd_piston_108.pdf")


# =============================================================================
# %% SHOW
# =============================================================================
plt.show()
