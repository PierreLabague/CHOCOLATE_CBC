"""Shot-to-shot standard deviation of the Zernike coefficients and piston.

For each amplifier size (45 mm rods and 108 mm slabs), one figure with
three vertically stacked panels sharing a common time axis:

    top    : std across shots of the Zernike RMS coefficients, beam A
    middle : std across shots of the Zernike RMS coefficients, beam B
    bottom : std across shots of the unwrapped differential piston

Output files:
    std_over_time_45.pdf
    std_over_time_108.pdf
"""

# =============================================================================
# %% PARAMETERS
# =============================================================================

ROOT = None


WINDOW_FULL = 4000  # frames — half-width for shot selection and display
VALID_FRAME_RATES = (82000, 66000)
# Vertical limits (waves).  None = automatic.
YLIM_WF45 = None
YLIM_WF108 = (-0.1, 0.6)
YLIM_PISTON = None

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

db = load_database(config.root)

print(f"Data directory : {config.root}")
print(f"Normalisation  : {config.zernike_mode}")
print(f"Reference time : {config.time_ref_ms} ms")

unit = format_zernike_mode(config.zernike_mode)


# =============================================================================
# %% FIGURE 1 — STD over time, 45 mm rods
# =============================================================================

fig = plt.figure(figsize=(FIGURE_WIDTH, 8))
gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.10)
ax_A = fig.add_subplot(gs[0])
ax_B = fig.add_subplot(gs[1], sharex=ax_A)
ax_piston = fig.add_subplot(gs[2], sharex=ax_A)

# ---- beam A — std across shots ----------------------------------------------
ids_A = [
    sid
    for sid, meta in db.items()
    if sid != "__meta"
    and meta["AMPLI"].get("108A") == False  # slab A not fired (rods only)
    and meta["AMPLI"].get("108B") == False  # slab B not fired (rods only)
    and meta["AMPLI"].get("45A") == True  # rod A fired
    and meta["AMPLI"].get("45B") == True  # rod B fired
    and meta["PHASICS"].get("frame_rate") in VALID_FRAME_RATES
    and meta["PHASICS"].get("cine_beginning") <= -WINDOW_FULL
    and meta["PHASICS"].get("cine_end") >= WINDOW_FULL
]
data_a, _, time_A, labels, shot_ids_A = load_beam_pair(
    config.root,
    ids_A,
    zernike_mode=config.zernike_mode,
    columns=ZERNIKE_COLUMNS,
    verbose=False,
)
data_a = zero_at_reference(data_a, time_A, config.time_ref_ms)
std_A = np.std(data_a, axis=0)  # shape (n_modes, n_times)
for mode in range(std_A.shape[0]):
    ax_A.plot(
        time_A * 1e3,
        std_A[mode],
        linewidth=1.0,
        zorder=3,
        **zernike_line_style(labels[mode]),
    )
add_flash_background(ax_A, config.root, shot_ids_A, "A")
ax_A.set_ylabel(f"Std {unit} (waves)")
ax_A.set_title(f"Rods - shot-to-shot std  [A & B: {len(ids_A)} shots]", fontsize=8)
ax_A.text(0.01, 0.97, "Beam A", transform=ax_A.transAxes, va="top", fontsize=8)
if YLIM_WF45:
    ax_A.set_ylim(*YLIM_WF45)
plt.setp(ax_A.get_xticklabels(), visible=False)

# ---- beam B — std across shots ----------------------------------------------
ids_B = [
    sid
    for sid, meta in db.items()
    if sid != "__meta"
    and meta["AMPLI"].get("108A") == False  # slab A not fired (rods only)
    and meta["AMPLI"].get("108B") == False  # slab B not fired (rods only)
    and meta["AMPLI"].get("45A") == True  # rod A fired
    and meta["AMPLI"].get("45B") == True  # rod B fired
    and meta["PHASICS"].get("frame_rate") in VALID_FRAME_RATES
    and meta["PHASICS"].get("cine_beginning") <= -WINDOW_FULL
    and meta["PHASICS"].get("cine_end") >= WINDOW_FULL
]
_, data_b, time_B, _, shot_ids_B = load_beam_pair(
    config.root,
    ids_B,
    zernike_mode=config.zernike_mode,
    columns=ZERNIKE_COLUMNS,
    verbose=False,
)
data_b = zero_at_reference(data_b, time_B, config.time_ref_ms)
std_B = np.std(data_b, axis=0)
for mode in range(std_B.shape[0]):
    ax_B.plot(
        time_B * 1e3,
        std_B[mode],
        linewidth=1.0,
        zorder=3,
        **zernike_line_style(labels[mode]),
    )
add_flash_background(ax_B, config.root, shot_ids_B, "B")
ax_B.set_ylabel(f"Std {unit} (waves)")
ax_B.text(0.01, 0.97, "Beam B", transform=ax_B.transAxes, va="top", fontsize=8)
if YLIM_WF45:
    ax_B.set_ylim(*YLIM_WF45)
plt.setp(ax_B.get_xticklabels(), visible=False)

# ---- piston — std across shots ----------------------------------------------
ids_P = [
    sid
    for sid, meta in db.items()
    if sid != "__meta"
    and meta["AMPLI"].get("108A") == False  # slab A not fired (rods only)
    and meta["AMPLI"].get("108B") == False  # slab B not fired (rods only)
    and meta["AMPLI"].get("45A") == True  # rod A fired
    and meta["AMPLI"].get("45B") == True  # rod B fired
    and meta["PHASICS"].get("frame_rate") in VALID_FRAME_RATES
    and meta["PHASICS"].get("cine_beginning") <= -WINDOW_FULL
    and meta["PHASICS"].get("cine_end") >= WINDOW_FULL
]
piston = load_piston_results(config.root, ids_P, unwrap=True, verbose=False)
# Interpolate every piston trace onto the beam-A time grid then compute std
traces = []
for entry in piston.values():
    if entry.get("piston") is None:
        continue
    t = np.asarray(entry["time"], dtype=float)
    p = zero_at_reference(
        np.asarray(entry["piston"], dtype=float), t, config.time_ref_ms
    )
    traces.append(interp1d(t, p, bounds_error=False, fill_value=np.nan)(time_A))
if traces:
    std_P = np.nanstd(np.array(traces), axis=0)
    ax_piston.plot(time_A * 1e3, std_P, color=PISTON_COLOR, linewidth=1.0, zorder=3)
add_flash_background(ax_piston, config.root, ids_P, "A")
ax_piston.set_ylabel("Std piston (waves)")
ax_piston.set_xlabel("Time (ms)")
ax_piston.text(
    0.01,
    0.97,
    f"Piston ({len(ids_P)} shots)",
    transform=ax_piston.transAxes,
    va="top",
    fontsize=8,
)
if YLIM_PISTON:
    ax_piston.set_ylim(*YLIM_PISTON)

# ---- common legend ----------------------------------------------------------
handles, texts = zernike_legend_handles(list(labels), include_flash=True)
handles.append(plt.Line2D([0], [0], color=PISTON_COLOR, linewidth=1.2))
texts.append("Piston")
add_bottom_legend(fig, handles, texts, ncol=5, bottom=0.15)
plt.tight_layout()
save_figure(fig, config.figures_dir / "std_over_time_45.pdf")


# =============================================================================
# %% FIGURE 2 — STD over time, 108 mm slabs
# =============================================================================

fig = plt.figure(figsize=(FIGURE_WIDTH, 8))
gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.10)
ax_A = fig.add_subplot(gs[0])
ax_B = fig.add_subplot(gs[1], sharex=ax_A)
ax_piston = fig.add_subplot(gs[2], sharex=ax_A)

# ---- beam A — std across shots ----------------------------------------------
ids_A = [
    sid
    for sid, meta in db.items()
    if sid != "__meta"
    and meta["AMPLI"].get("108A") == True  # slab A fired
    and meta["AMPLI"].get("108B") == True  # slab B fired
    and meta["AMPLI"].get("45A") == False  # rod A not fired (slab only)
    and meta["AMPLI"].get("45B") == False  # rod B not fired (slab only)
    and meta["PHASICS"].get("frame_rate") in VALID_FRAME_RATES
    and meta["PHASICS"].get("cine_beginning") <= -WINDOW_FULL
    and meta["PHASICS"].get("cine_end") >= WINDOW_FULL
]
data_a, _, time_A, labels, shot_ids_A = load_beam_pair(
    config.root,
    ids_A,
    zernike_mode=config.zernike_mode,
    columns=ZERNIKE_COLUMNS,
    verbose=False,
)
data_a = zero_at_reference(data_a, time_A, config.time_ref_ms)
std_A = np.std(data_a, axis=0)
for mode in range(std_A.shape[0]):
    ax_A.plot(
        time_A * 1e3,
        std_A[mode],
        linewidth=1.0,
        zorder=3,
        **zernike_line_style(labels[mode]),
    )
add_flash_background(ax_A, config.root, shot_ids_A, "A")
ax_A.set_ylabel(f"Std {unit} (waves)")
ax_A.set_title(f"Slabs — shot-to-shot std  [A & B: {len(ids_A)} shots]", fontsize=8)
ax_A.text(0.01, 0.97, "Beam A", transform=ax_A.transAxes, va="top", fontsize=8)
if YLIM_WF108:
    ax_A.set_ylim(*YLIM_WF108)
plt.setp(ax_A.get_xticklabels(), visible=False)

# ---- beam B — std across shots ----------------------------------------------
ids_B = [
    sid
    for sid, meta in db.items()
    if sid != "__meta"
    and meta["AMPLI"].get("108A") == True  # slab A fired
    and meta["AMPLI"].get("108B") == True  # slab B fired
    and meta["AMPLI"].get("45A") == False  # rod A not fired (slab only)
    and meta["AMPLI"].get("45B") == False  # rod B not fired (slab only)
    and meta["PHASICS"].get("frame_rate") in VALID_FRAME_RATES
    and meta["PHASICS"].get("cine_beginning") <= -WINDOW_FULL
    and meta["PHASICS"].get("cine_end") >= WINDOW_FULL
]

_, data_b, time_B, _, shot_ids_B = load_beam_pair(
    config.root,
    ids_B,
    zernike_mode=config.zernike_mode,
    columns=ZERNIKE_COLUMNS,
    verbose=False,
)
data_b = zero_at_reference(data_b, time_B, config.time_ref_ms)
std_B = np.std(data_b, axis=0)
for mode in range(std_B.shape[0]):
    ax_B.plot(
        time_B * 1e3,
        std_B[mode],
        linewidth=1.0,
        zorder=3,
        **zernike_line_style(labels[mode]),
    )
add_flash_background(ax_B, config.root, shot_ids_B, "B")
ax_B.set_ylabel(f"Std {unit} (waves)")
ax_B.text(0.01, 0.97, "Beam B", transform=ax_B.transAxes, va="top", fontsize=8)
if YLIM_WF108:
    ax_B.set_ylim(*YLIM_WF108)
plt.setp(ax_B.get_xticklabels(), visible=False)

# ---- piston — std across shots ----------------------------------------------
ids_P = [
    sid
    for sid, meta in db.items()
    if sid != "__meta"
    and meta["AMPLI"].get("108A") == True  # slab A fired
    and meta["AMPLI"].get("108B") == True  # slab B fired
    and meta["AMPLI"].get("45A") == False  # rod A not fired (slab only)
    and meta["AMPLI"].get("45B") == False  # rod B not fired (slab only)
    and meta["PHASICS"].get("frame_rate") in VALID_FRAME_RATES
    and meta["PHASICS"].get("cine_beginning") <= -WINDOW_FULL
    and meta["PHASICS"].get("cine_end") >= WINDOW_FULL
]
piston = load_piston_results(config.root, ids_P, unwrap=True, verbose=False)
traces = []
for entry in piston.values():
    if entry.get("piston") is None:
        continue
    t = np.asarray(entry["time"], dtype=float)
    p = zero_at_reference(
        np.asarray(entry["piston"], dtype=float), t, config.time_ref_ms
    )
    traces.append(interp1d(t, p, bounds_error=False, fill_value=np.nan)(time_A))
if traces:
    std_P = np.nanstd(np.array(traces), axis=0)
    ax_piston.plot(time_A * 1e3, std_P, color=PISTON_COLOR, linewidth=1.0, zorder=3)
add_flash_background(ax_piston, config.root, ids_P, "A")
ax_piston.set_ylabel("Std piston (waves)")
ax_piston.set_xlabel("Time (ms)")
ax_piston.text(
    0.01,
    0.97,
    f"Piston ({len(ids_P)} shots)",
    transform=ax_piston.transAxes,
    va="top",
    fontsize=8,
)
if YLIM_PISTON:
    ax_piston.set_ylim(*YLIM_PISTON)

# ---- common legend ----------------------------------------------------------
handles, texts = zernike_legend_handles(list(labels), include_flash=True)
handles.append(plt.Line2D([0], [0], color=PISTON_COLOR, linewidth=1.2))
texts.append("Piston")
add_bottom_legend(fig, handles, texts, ncol=5, bottom=0.15)
plt.tight_layout()

save_figure(fig, config.figures_dir / "std_over_time_108.pdf")


# =============================================================================
# %% SHOW
# =============================================================================
plt.show()
