"""On-shot distortions introduced by the amplifiers.

Four output figures:
    on-shot_distortion_WF_45.pdf
    on-shot_distortion_piston_45.pdf
    on-shot_distortion_WF_108.pdf
    on-shot_distortion_piston_108.pdf
    on-shot_distortion_WF_45_108.pdf
    on-shot_distortion_piston_45_108.pdf
"""

# =============================================================================
# %% PARAMETERS
# =============================================================================

ROOT = None


WINDOW_ZOOM = 1000  # frames — half-width for zoom panels
WINDOW_FULL = 4000  # frames — half-width for full-window panels

SHOT_ALPHA = 0.2  # transparency of individual shot traces

VALID_FRAME_RATES = (82000, 66000)

# Displayed time axis limits in ms.  None = automatic.
XLIM_ZOOM = (-1.0, 1.0)
XLIM_FULL = None

# Displayed amplitude limits in waves.  None = automatic per panel.
YLIM_WF = {
    "45A_zoom": (-0.25, 0.25),
    "45A_full": (-0.25, 0.25),
    "45B_zoom": (-0.25, 0.25),
    "45B_full": (-0.25, 0.25),
    "108A_zoom": (-0.25, 0.50),
    "108A_full": (-2.50, 3.00),
    "108B_zoom": (-0.25, 0.50),
    "108B_full": (-2.50, 3.00),
}
YLIM_PISTON = {
    "45_zoom": (-2, 2),
    "45_full": (-4, 4),
    "108_zoom": (-2, 2),
    "108_full": (-80, 20),
}

# =============================================================================
# %% IMPORTS AND SETUP
# =============================================================================

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

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

print(f"Data directory : {config.root}")
print(f"Normalisation  : {config.zernike_mode}")
print(f"Reference time : {config.time_ref_ms} ms")

unit = format_zernike_mode(config.zernike_mode)


# =============================================================================
# %% FIGURE 1 — Wavefront distortions, 45 mm rods
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH, 5.6), sharex="col")
labels = []

# ---- top-left : beam A, zoom ------------------------------------------------
ids = [
    sid
    for sid, meta in db.items()
    if sid != "__meta"
    and meta["AMPLI"].get("45A") == True  # rods A fired
    and meta["AMPLI"].get("45B") == True  # rods B fired
    and meta["AMPLI"].get("108A") == False  # slab A off
    and meta["AMPLI"].get("108B") == False  # slab B off
    and meta["PHASICS"].get("frame_rate") in VALID_FRAME_RATES
    and meta["PHASICS"].get("cine_beginning") <= -WINDOW_ZOOM
    and meta["PHASICS"].get("cine_end") >= WINDOW_ZOOM
]

data_a, _, time, labels, shot_ids = load_beam_pair(
    config.root,
    ids,
    zernike_mode=config.zernike_mode,
    columns=ZERNIKE_COLUMNS,
    verbose=False,
)
data_a = zero_at_reference(data_a, time, config.time_ref_ms)
for shot in range(data_a.shape[0]):
    for mode in range(data_a.shape[1]):
        axes[0, 0].plot(
            time * 1e3,
            data_a[shot, mode],
            alpha=SHOT_ALPHA,
            linewidth=0.8,
            **zernike_line_style(labels[mode]),
        )
axes[0, 0].set_xlim(*XLIM_ZOOM)
axes[0, 0].set_ylim(*YLIM_WF["45A_zoom"])
add_flash_background(axes[0, 0], config.root, shot_ids, "A")
axes[0, 0].set_title(f"Rod A — zoom  ({len(ids)} shots)", fontsize=8)
axes[0, 0].set_ylabel(f"Zernike {unit} (waves)")

# ---- top-right : beam A, full -----------------------------------------------
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
data_a, _, time, labels, shot_ids = load_beam_pair(
    config.root,
    ids,
    zernike_mode=config.zernike_mode,
    columns=ZERNIKE_COLUMNS,
    verbose=False,
)
data_a = zero_at_reference(data_a, time, config.time_ref_ms)
for shot in range(data_a.shape[0]):
    for mode in range(data_a.shape[1]):
        axes[0, 1].plot(
            time * 1e3,
            data_a[shot, mode],
            alpha=SHOT_ALPHA,
            linewidth=0.8,
            **zernike_line_style(labels[mode]),
        )
if XLIM_FULL:
    axes[0, 1].set_xlim(*XLIM_FULL)
axes[0, 1].set_ylim(*YLIM_WF["45A_full"])
add_flash_background(axes[0, 1], config.root, shot_ids, "A")
axes[0, 1].set_title(f"Rod A — full window  ({len(ids)} shots)", fontsize=8)

# ---- bottom-left : beam B, zoom ---------------------------------------------
ids = [
    sid
    for sid, meta in db.items()
    if sid != "__meta"
    and meta["AMPLI"].get("45A") == True  # rods A fired
    and meta["AMPLI"].get("45B") == True  # rods B fired
    and meta["AMPLI"].get("108A") == False  # slab A off
    and meta["AMPLI"].get("108B") == False  # slab B off
    and meta["PHASICS"].get("frame_rate") in VALID_FRAME_RATES
    and meta["PHASICS"].get("cine_beginning") <= -WINDOW_ZOOM
    and meta["PHASICS"].get("cine_end") >= WINDOW_ZOOM
]
_, data_b, time, labels, shot_ids = load_beam_pair(
    config.root,
    ids,
    zernike_mode=config.zernike_mode,
    columns=ZERNIKE_COLUMNS,
    verbose=False,
)
data_b = zero_at_reference(data_b, time, config.time_ref_ms)
for shot in range(data_b.shape[0]):
    for mode in range(data_b.shape[1]):
        axes[1, 0].plot(
            time * 1e3,
            data_b[shot, mode],
            alpha=SHOT_ALPHA,
            linewidth=0.8,
            **zernike_line_style(labels[mode]),
        )
axes[1, 0].set_xlim(*XLIM_ZOOM)
axes[1, 0].set_ylim(*YLIM_WF["45B_zoom"])
add_flash_background(axes[1, 0], config.root, shot_ids, "B")
axes[1, 0].set_title(f"Rod B — zoom  ({len(ids)} shots)", fontsize=8)
axes[1, 0].set_ylabel(f"Zernike {unit} (waves)")
axes[1, 0].set_xlabel("Time (ms)")

# ---- bottom-right : beam B, full --------------------------------------------
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
_, data_b, time, labels, shot_ids = load_beam_pair(
    config.root,
    ids,
    zernike_mode=config.zernike_mode,
    columns=ZERNIKE_COLUMNS,
    verbose=False,
)
data_b = zero_at_reference(data_b, time, config.time_ref_ms)
for shot in range(data_b.shape[0]):
    for mode in range(data_b.shape[1]):
        axes[1, 1].plot(
            time * 1e3,
            data_b[shot, mode],
            alpha=SHOT_ALPHA,
            linewidth=0.8,
            **zernike_line_style(labels[mode]),
        )
if XLIM_FULL:
    axes[1, 1].set_xlim(*XLIM_FULL)
axes[1, 1].set_ylim(*YLIM_WF["45B_full"])
add_flash_background(axes[1, 1], config.root, shot_ids, "B")
axes[1, 1].set_title(f"Rod B — full window  ({len(ids)} shots)", fontsize=8)
axes[1, 1].set_xlabel("Time (ms)")

handles, texts = zernike_legend_handles(list(labels), include_flash=True)
add_bottom_legend(fig, handles, texts, ncol=4, bottom=0.20)
save_figure(fig, config.figures_dir / "on-shot_distortion_WF_45.pdf")


# =============================================================================
# %% FIGURE 2 — Piston, 45 mm rods
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 3.0))

# ---- left : zoom ------------------------------------------------------------
ids = [
    sid
    for sid, meta in db.items()
    if sid != "__meta"
    and meta["AMPLI"].get("45A") == True  # rods A fired
    and meta["AMPLI"].get("45B") == True  # rods B fired
    and meta["AMPLI"].get("108A") == False  # slab A off
    and meta["AMPLI"].get("108B") == False  # slab B off
    and meta["PHASICS"].get("frame_rate") == 66000
    and meta["PHASICS"].get("cine_beginning") <= -WINDOW_ZOOM
    and meta["PHASICS"].get("cine_end") >= WINDOW_ZOOM
]
piston = load_piston_results(config.root, ids, unwrap=True, verbose=False)
for entry in piston.values():
    if entry.get("piston") is None:
        continue
    t = np.asarray(entry["time"], dtype=float)
    p = zero_at_reference(
        np.asarray(entry["piston"], dtype=float), t, config.time_ref_ms
    )
    axes[0].plot(t * 1e3, p, color=PISTON_COLOR, alpha=SHOT_ALPHA, linewidth=0.8)
axes[0].set_xlim(*XLIM_ZOOM)
axes[0].set_ylim(*YLIM_PISTON["45_zoom"])
add_flash_background(axes[0], config.root, list(piston), "A")
axes[0].set_title(f"Rod piston — zoom  ({len(ids)} shots)", fontsize=8)
axes[0].set_ylabel("Differential piston (waves)")
axes[0].set_xlabel("Time (ms)")

# ---- right : full -----------------------------------------------------------
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
for entry in piston.values():
    if entry.get("piston") is None:
        continue
    t = np.asarray(entry["time"], dtype=float)
    p = zero_at_reference(
        np.asarray(entry["piston"], dtype=float), t, config.time_ref_ms
    )
    axes[1].plot(t * 1e3, p, color=PISTON_COLOR, alpha=SHOT_ALPHA, linewidth=0.8)
if XLIM_FULL:
    axes[1].set_xlim(*XLIM_FULL)
axes[1].set_ylim(*YLIM_PISTON["45_full"])
add_flash_background(axes[1], config.root, list(piston), "A")
axes[1].set_title(f"Rod piston — full window  ({len(ids)} shots)", fontsize=8)
axes[1].set_xlabel("Time (ms)")

handles = [plt.Line2D([0], [0], color=PISTON_COLOR, linewidth=1.2)]
texts = ["Unwrapped differential piston"]
handles_f, texts_f = zernike_legend_handles([], include_flash=True)
add_bottom_legend(fig, handles + handles_f, texts + texts_f, ncol=3, bottom=0.3)
save_figure(fig, config.figures_dir / "on-shot_distortion_piston_45.pdf")


# =============================================================================
# %% FIGURE 3 — Wavefront distortions, 108 mm slabs
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH, 5.6), sharex="col")
labels = []

# ---- top-left : beam A, zoom ------------------------------------------------
ids = [
    sid
    for sid, meta in db.items()
    if sid != "__meta"
    and meta["AMPLI"].get("45A") == False  # rods A off
    and meta["AMPLI"].get("45B") == False  # rods B off
    and meta["AMPLI"].get("108A") == True  # slab A fired
    and meta["AMPLI"].get("108B") == True  # slab B fired
    and meta["PHASICS"].get("frame_rate") in VALID_FRAME_RATES
    and meta["PHASICS"].get("cine_beginning") <= -WINDOW_ZOOM
    and meta["PHASICS"].get("cine_end") >= WINDOW_ZOOM
]
data_a, _, time, labels, shot_ids = load_beam_pair(
    config.root,
    ids,
    zernike_mode=config.zernike_mode,
    columns=ZERNIKE_COLUMNS,
    verbose=False,
)
data_a = zero_at_reference(data_a, time, config.time_ref_ms)
for shot in range(data_a.shape[0]):
    for mode in range(data_a.shape[1]):
        axes[0, 0].plot(
            time * 1e3,
            data_a[shot, mode],
            alpha=SHOT_ALPHA,
            linewidth=0.8,
            **zernike_line_style(labels[mode]),
        )
axes[0, 0].set_xlim(*XLIM_ZOOM)
axes[0, 0].set_ylim(*YLIM_WF["108A_zoom"])
add_flash_background(axes[0, 0], config.root, shot_ids, "A")
axes[0, 0].set_title(f"Slab A — zoom  ({len(ids)} shots)", fontsize=8)
axes[0, 0].set_ylabel(f"Zernike {unit} (waves)")

# ---- top-right : beam A, full -----------------------------------------------
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
data_a, _, time, labels, shot_ids = load_beam_pair(
    config.root,
    ids,
    zernike_mode=config.zernike_mode,
    columns=ZERNIKE_COLUMNS,
    verbose=False,
)
data_a = zero_at_reference(data_a, time, config.time_ref_ms)
for shot in range(data_a.shape[0]):
    for mode in range(data_a.shape[1]):
        axes[0, 1].plot(
            time * 1e3,
            data_a[shot, mode],
            alpha=SHOT_ALPHA,
            linewidth=0.8,
            **zernike_line_style(labels[mode]),
        )
if XLIM_FULL:
    axes[0, 1].set_xlim(*XLIM_FULL)
axes[0, 1].set_ylim(*YLIM_WF["108A_full"])
add_flash_background(axes[0, 1], config.root, shot_ids, "A")
axes[0, 1].set_title(f"Slab A — full window  ({len(ids)} shots)", fontsize=8)

# ---- bottom-left : beam B, zoom ---------------------------------------------
ids = [
    sid
    for sid, meta in db.items()
    if sid != "__meta"
    and meta["AMPLI"].get("45A") == False  # rods A off
    and meta["AMPLI"].get("45B") == False  # rods B off
    and meta["AMPLI"].get("108A") == True  # slab A fired
    and meta["AMPLI"].get("108B") == True  # slab B fired
    and meta["PHASICS"].get("frame_rate") in VALID_FRAME_RATES
    and meta["PHASICS"].get("cine_beginning") <= -WINDOW_ZOOM
    and meta["PHASICS"].get("cine_end") >= WINDOW_ZOOM
]
_, data_b, time, labels, shot_ids = load_beam_pair(
    config.root,
    ids,
    zernike_mode=config.zernike_mode,
    columns=ZERNIKE_COLUMNS,
    verbose=False,
)
data_b = zero_at_reference(data_b, time, config.time_ref_ms)
for shot in range(data_b.shape[0]):
    for mode in range(data_b.shape[1]):
        axes[1, 0].plot(
            time * 1e3,
            data_b[shot, mode],
            alpha=SHOT_ALPHA,
            linewidth=0.8,
            **zernike_line_style(labels[mode]),
        )
axes[1, 0].set_xlim(*XLIM_ZOOM)
axes[1, 0].set_ylim(*YLIM_WF["108B_zoom"])
add_flash_background(axes[1, 0], config.root, shot_ids, "B")
axes[1, 0].set_title(f"Slab B — zoom  ({len(ids)} shots)", fontsize=8)
axes[1, 0].set_ylabel(f"Zernike {unit} (waves)")
axes[1, 0].set_xlabel("Time (ms)")

# ---- bottom-right : beam B, full --------------------------------------------
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
_, data_b, time, labels, shot_ids = load_beam_pair(
    config.root,
    ids,
    zernike_mode=config.zernike_mode,
    columns=ZERNIKE_COLUMNS,
    verbose=False,
)
data_b = zero_at_reference(data_b, time, config.time_ref_ms)
for shot in range(data_b.shape[0]):
    for mode in range(data_b.shape[1]):
        axes[1, 1].plot(
            time * 1e3,
            data_b[shot, mode],
            alpha=SHOT_ALPHA,
            linewidth=0.8,
            **zernike_line_style(labels[mode]),
        )
if XLIM_FULL:
    axes[1, 1].set_xlim(*XLIM_FULL)
axes[1, 1].set_ylim(*YLIM_WF["108B_full"])
add_flash_background(axes[1, 1], config.root, shot_ids, "B")
axes[1, 1].set_title(f"Slab B — full window  ({len(ids)} shots)", fontsize=8)
axes[1, 1].set_xlabel("Time (ms)")

handles, texts = zernike_legend_handles(list(labels), include_flash=True)
add_bottom_legend(fig, handles, texts, ncol=4, bottom=0.20)
save_figure(fig, config.figures_dir / "on-shot_distortion_WF_108.pdf")


# =============================================================================
# %% FIGURE 4 — Piston, 108 mm slabs
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 3.0))

# ---- left : zoom ------------------------------------------------------------
ids = [
    sid
    for sid, meta in db.items()
    if sid != "__meta"
    and meta["AMPLI"].get("45A") == False  # rods A off
    and meta["AMPLI"].get("45B") == False  # rods B off
    and meta["AMPLI"].get("108A") == True  # slab A fired
    and meta["AMPLI"].get("108B") == True  # slab B fired
    and meta["PHASICS"].get("frame_rate") == 66000
    and meta["PHASICS"].get("cine_beginning") <= -WINDOW_ZOOM
    and meta["PHASICS"].get("cine_end") >= WINDOW_ZOOM
]
piston = load_piston_results(config.root, ids, unwrap=True, verbose=False)
for entry in piston.values():
    if entry.get("piston") is None:
        continue
    t = np.asarray(entry["time"], dtype=float)
    p = zero_at_reference(
        np.asarray(entry["piston"], dtype=float), t, config.time_ref_ms
    )
    axes[0].plot(t * 1e3, p, color=PISTON_COLOR, alpha=SHOT_ALPHA, linewidth=0.8)
axes[0].set_xlim(*XLIM_ZOOM)
axes[0].set_ylim(*YLIM_PISTON["108_zoom"])
add_flash_background(axes[0], config.root, list(piston), "A")
axes[0].set_title(f"Slab piston — zoom  ({len(ids)} shots)", fontsize=8)
axes[0].set_ylabel("Differential piston (waves)")
axes[0].set_xlabel("Time (ms)")

# ---- right : full -----------------------------------------------------------
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
for entry in piston.values():
    if entry.get("piston") is None:
        continue
    t = np.asarray(entry["time"], dtype=float)
    p = zero_at_reference(
        np.asarray(entry["piston"], dtype=float), t, config.time_ref_ms
    )
    axes[1].plot(t * 1e3, p, color=PISTON_COLOR, alpha=SHOT_ALPHA, linewidth=0.8)
if XLIM_FULL:
    axes[1].set_xlim(*XLIM_FULL)
axes[1].set_ylim(*YLIM_PISTON["108_full"])
add_flash_background(axes[1], config.root, list(piston), "A")
axes[1].set_title(f"Slab piston — full window  ({len(ids)} shots)", fontsize=8)
axes[1].set_xlabel("Time (ms)")

handles = [plt.Line2D([0], [0], color=PISTON_COLOR, linewidth=1.2)]
texts = ["Unwrapped differential piston"]
handles_f, texts_f = zernike_legend_handles([], include_flash=True)
add_bottom_legend(fig, handles + handles_f, texts + texts_f, ncol=3, bottom=0.3)
save_figure(fig, config.figures_dir / "on-shot_distortion_piston_108.pdf")


# =============================================================================
# %% FIGURE 5 — WF A, 108 mm slabs et 45 mm rods
# =============================================================================

fig, axes = plt.subplots(2, 1, figsize=(FIGURE_WIDTH, 5.6), sharex="col")
labels = []

# ---- top : beam A, rods 45 mm, zoom ------------------------------------------------
ids = [
    sid
    for sid, meta in db.items()
    if sid != "__meta"
    and meta["AMPLI"].get("45A") == True  # rods A fired
    and meta["AMPLI"].get("45B") == True  # rods B fired
    and meta["AMPLI"].get("108A") == False  # slab A off
    and meta["AMPLI"].get("108B") == False  # slab B off
    and meta["PHASICS"].get("frame_rate") in VALID_FRAME_RATES
    and meta["PHASICS"].get("cine_beginning") <= -WINDOW_ZOOM
    and meta["PHASICS"].get("cine_end") >= WINDOW_ZOOM
]
data_a, _, time, labels, shot_ids = load_beam_pair(
    config.root,
    ids,
    zernike_mode=config.zernike_mode,
    columns=ZERNIKE_COLUMNS,
    verbose=False,
)
data_a = zero_at_reference(data_a, time, config.time_ref_ms)
for shot in range(data_a.shape[0]):
    for mode in range(data_a.shape[1]):
        axes[0].plot(
            time * 1e3,
            data_a[shot, mode],
            alpha=SHOT_ALPHA,
            linewidth=0.8,
            **zernike_line_style(labels[mode]),
        )
axes[0].set_xlim(*XLIM_ZOOM)
axes[0].set_ylim(*YLIM_WF["45A_zoom"])
add_flash_background(axes[0], config.root, shot_ids, "A")
axes[0].set_title(f"Rod A — ({len(ids)} shots)", fontsize=8)
axes[0].set_ylabel(f"Zernike {unit} (waves)")


# ---- bottom : beam A, slabs 108 mm, zoom ------------------------------------------------
ids = [
    sid
    for sid, meta in db.items()
    if sid != "__meta"
    and meta["AMPLI"].get("45A") == False  # rods A off
    and meta["AMPLI"].get("45B") == False  # rods B off
    and meta["AMPLI"].get("108A") == True  # slab A fired
    and meta["AMPLI"].get("108B") == True  # slab B fired
    and meta["PHASICS"].get("frame_rate") in VALID_FRAME_RATES
    and meta["PHASICS"].get("cine_beginning") <= -WINDOW_ZOOM
    and meta["PHASICS"].get("cine_end") >= WINDOW_ZOOM
]
data_a, _, time, labels, shot_ids = load_beam_pair(
    config.root,
    ids,
    zernike_mode=config.zernike_mode,
    columns=ZERNIKE_COLUMNS,
    verbose=False,
)
data_a = zero_at_reference(data_a, time, config.time_ref_ms)
for shot in range(data_a.shape[0]):
    for mode in range(data_a.shape[1]):
        axes[1].plot(
            time * 1e3,
            data_a[shot, mode],
            alpha=SHOT_ALPHA,
            linewidth=0.8,
            **zernike_line_style(labels[mode]),
        )
axes[1].set_xlim(*XLIM_ZOOM)
axes[1].set_ylim(*YLIM_WF["108A_zoom"])
add_flash_background(axes[1], config.root, shot_ids, "A")
axes[1].set_title(f"Slab A — ({len(ids)} shots)", fontsize=8)
axes[1].set_ylabel(f"Zernike {unit} (waves)")
axes[1].set_xlabel("Time (ms)")

handles, texts = zernike_legend_handles(list(labels), include_flash=True)
add_bottom_legend(fig, handles, texts, ncol=4, bottom=0.20)
save_figure(fig, config.figures_dir / "on-shot_distortion_WF_45_108.pdf")

# =============================================================================
# %% FIGURE 4 — Piston, 108 mm slabs and 45 mm rods
# =============================================================================

fig, axes = plt.subplots(2, 1, figsize=(FIGURE_WIDTH, 5.6), sharex="col")
labels = []

# ---- left : zoom ------------------------------------------------------------
ids = [
    sid
    for sid, meta in db.items()
    if sid != "__meta"
    and meta["AMPLI"].get("45A") == True  # rods A fired
    and meta["AMPLI"].get("45B") == True  # rods B fired
    and meta["AMPLI"].get("108A") == False  # slab A off
    and meta["AMPLI"].get("108B") == False  # slab B off
    and meta["PHASICS"].get("frame_rate") == 66000
    and meta["PHASICS"].get("cine_beginning") <= -WINDOW_ZOOM
    and meta["PHASICS"].get("cine_end") >= WINDOW_ZOOM
]
piston = load_piston_results(config.root, ids, unwrap=True, verbose=False)
for entry in piston.values():
    if entry.get("piston") is None:
        continue
    t = np.asarray(entry["time"], dtype=float)
    p = zero_at_reference(
        np.asarray(entry["piston"], dtype=float), t, config.time_ref_ms
    )
    axes[0].plot(t * 1e3, p, color=PISTON_COLOR, alpha=SHOT_ALPHA, linewidth=0.8)
axes[0].set_xlim(*XLIM_ZOOM)
axes[0].set_ylim(*YLIM_PISTON["45_zoom"])
add_flash_background(axes[0], config.root, list(piston), "A")
axes[0].set_title(f"Rods piston — ({len(ids)} shots)", fontsize=8)
axes[0].set_ylabel("Differential piston (waves)")

# ---- right : full -----------------------------------------------------------
ids = [
    sid
    for sid, meta in db.items()
    if sid != "__meta"
    and meta["AMPLI"].get("45A") == False  # rods A off
    and meta["AMPLI"].get("45B") == False  # rods B off
    and meta["AMPLI"].get("108A") == True  # slab A fired
    and meta["AMPLI"].get("108B") == True  # slab B fired
    and meta["PHASICS"].get("frame_rate") == 66000
    and meta["PHASICS"].get("cine_beginning") <= -WINDOW_ZOOM
    and meta["PHASICS"].get("cine_end") >= WINDOW_ZOOM
]

piston = load_piston_results(config.root, ids, unwrap=True, verbose=False)
for entry in piston.values():
    if entry.get("piston") is None:
        continue
    t = np.asarray(entry["time"], dtype=float)
    p = zero_at_reference(
        np.asarray(entry["piston"], dtype=float), t, config.time_ref_ms
    )
    axes[1].plot(t * 1e3, p, color=PISTON_COLOR, alpha=SHOT_ALPHA, linewidth=0.8)
if XLIM_ZOOM:
    axes[1].set_xlim(*XLIM_ZOOM)
axes[1].set_ylim(*YLIM_PISTON["108_zoom"])
add_flash_background(axes[1], config.root, list(piston), "A")
axes[1].set_title(f"Slab piston — ({len(ids)} shots)", fontsize=8)
axes[1].set_xlabel("Time (ms)")

handles = [plt.Line2D([0], [0], color=PISTON_COLOR, linewidth=1.2)]
texts = ["Unwrapped differential piston"]
handles_f, texts_f = zernike_legend_handles([], include_flash=True)
add_bottom_legend(fig, handles + handles_f, texts + texts_f, ncol=3, bottom=0.15)

save_figure(fig, config.figures_dir / "on-shot_distortion_piston_45_108.pdf")


# =============================================================================
# %% SHOW
# =============================================================================
plt.show()
