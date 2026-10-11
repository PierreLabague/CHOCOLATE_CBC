"""Daily variability of the on-shot distortions — 108 mm slabs, beam B.

Two figures:

``daily_defocus_108B.pdf``
    Defocus time traces for all shots of the zoomed window, one colour
    per acquisition day, persistence rendering.

``daily_std_108B.pdf``
    Shot-to-shot standard deviation over time for the four dominant
    aberrations (Tilt X, Tilt Y, Defocus, Astig 0°) and the differential
    piston.  Each panel shows the std computed over all days pooled
    (dark) then over each day individually (one colour per day).  Flash
    background and peak marker are drawn on every panel.  A single
    common legend sits below the five panels.

Only shots that have BOTH wavefront and piston data are used, so that the
five panels always describe the same shot population.

Run this script directly.
"""

# ===========================================================================
#  EDIT HERE
# ===========================================================================
ROOT = None


WINDOW_ZOOM = 1000  # frames — half-width of the zoomed window

# Names to search for in the Zernike label list (case-insensitive substring).
# Adjust if your label strings differ.
MODE_NAMES = {
    "Tilt X": "tilt x",
    "Tilt Y": "tilt y",
    "Defocus": "defocus",
    "Astig 0°": "astig 0",
}

# Axis limits (waves).  None = automatic.
YLIM_TRACES = None  # for fig 1 individual traces
YLIM_STD_WF = (0, 0.15)  # for fig 2 WF std panels
YLIM_STD_P = (0, 4)  # for fig 2 piston std panel
XLIM_ZOOM = (-1.0, 1.0)  # ms

SHOT_ALPHA = 0.25  # individual traces
ALL_COLOR = "k"  # colour used for the "all days pooled" std curve
# ===========================================================================

from collections import defaultdict
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from chocolate.config import load_config
from chocolate.database import load_database
from chocolate.efficiency import zero_at_reference
from chocolate.plotting import add_bottom_legend, add_flash_background
from chocolate.results_io import ZERNIKE_COLUMNS, load_beam_pair, load_piston_results
from chocolate.style import FIGURE_WIDTH, apply_style, save_figure, zernike_line_style
from chocolate.zernike import format_zernike_mode

# One colour per day — tab10 gives 10 distinct colours which is enough for
# a typical beam-time campaign.
DAY_COLORS = plt.cm.tab10.colors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_day(shot_id):
    """Return the date string 'YYYY-MM-DD' from a shot ID 'YYYY-MM-DD-HH-MM'."""
    parts = shot_id.split("-")
    return "-".join(parts[:3])


def ids_for_108B_zoom(database):
    """Shot IDs: 108B amplifier fired, acquisition covers the zoom window."""
    selected = []
    for shot_id, meta in database.items():
        if shot_id == "__meta":
            continue
        ampli = meta.get("AMPLI", {})
        if not ampli.get("108B"):
            continue
        phasics = meta.get("PHASICS", {})
        start = phasics.get("cine_beginning")
        stop = phasics.get("cine_end")
        if not isinstance(start, (int, float)) or isinstance(start, bool):
            continue
        if not isinstance(stop, (int, float)) or isinstance(stop, bool):
            continue
        if start <= -WINDOW_ZOOM and stop >= WINDOW_ZOOM:
            selected.append(shot_id)
    return selected


def find_mode(labels, substring):
    """Return the first index whose label contains `substring` (case-insensitive)."""
    for i, lbl in enumerate(labels):
        if substring.lower() in str(lbl).lower():
            return i
    return None


def group_by_day(shot_ids):
    """Return an OrderedDict {day_str: [shot_id, ...]} sorted chronologically."""
    groups = defaultdict(list)
    for sid in shot_ids:
        groups[extract_day(sid)].append(sid)
    return dict(sorted(groups.items()))


def interpolate_piston_to_grid(piston_dict, common_time, time_ref_ms):
    """Return array (n_shots, n_times) of unwrapped piston on common_time."""
    traces = []
    for entry in piston_dict.values():
        if entry.get("piston") is None:
            continue
        t = np.asarray(entry["time"], dtype=float)
        p = zero_at_reference(np.asarray(entry["piston"], dtype=float), t, time_ref_ms)
        # Restrict interpolation to the measured range; fill outside with NaN.
        f = interp1d(t, p, bounds_error=False, fill_value=np.nan)
        traces.append(f(common_time))
    if not traces:
        return np.empty((0, len(common_time)))
    return np.array(traces)  # (n_shots, n_times)


# ---------------------------------------------------------------------------
# Fig 1 — Defocus traces coloured by day
# ---------------------------------------------------------------------------
def make_fig_defocus(config, database, ids_all, days):
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 3.5))

    for day_idx, (day, day_ids) in enumerate(days.items()):
        color = DAY_COLORS[day_idx % len(DAY_COLORS)]

        if not day_ids:
            continue

        _, data_b, common_time, labels, shot_ids = load_beam_pair(
            config.root,
            day_ids,
            zernike_mode=config.zernike_mode,
            columns=ZERNIKE_COLUMNS,
            verbose=False,
        )
        data_b = zero_at_reference(data_b, common_time, config.time_ref_ms)
        time_ms = common_time * 1e3

        defocus_idx = find_mode(labels, MODE_NAMES["Defocus"])
        if defocus_idx is None:
            print(f"  [WARN] Defocus mode not found for day {day}")
            continue

        for shot in range(data_b.shape[0]):
            ax.plot(
                time_ms,
                data_b[shot, defocus_idx],
                color=color,
                alpha=SHOT_ALPHA,
                linewidth=0.8,
                label=day if shot == 0 else None,  # label only on first trace
            )

    ax.set_xlim(*XLIM_ZOOM)
    if YLIM_TRACES:
        ax.set_ylim(*YLIM_TRACES)
    add_flash_background(ax, config.root, ids_all, "B")
    ax.set_xlabel("Time (ms)")
    unit = format_zernike_mode(config.zernike_mode)
    ax.set_ylabel(f"Defocus {unit} (waves)")
    ax.set_title("Defocus — Slab B, zoomed window", fontsize=8)

    # Compact legend: one entry per day showing shot count
    handles, texts = [], []
    for day_idx, (day, day_ids) in enumerate(days.items()):
        color = DAY_COLORS[day_idx % len(DAY_COLORS)]
        handles.append(plt.Line2D([0], [0], color=color, linewidth=1.5))
        texts.append(f"{day}  ({len(day_ids)} shots)")
    add_bottom_legend(fig, handles, texts, ncol=len(days), bottom=0.22)

    path = config.figures_dir / "daily_defocus_108B.pdf"
    save_figure(fig, path)
    print(f"  written: {path}")
    return path


# ---------------------------------------------------------------------------
# Fig 2 — STD over time, 4 WF modes + piston
# ---------------------------------------------------------------------------
def make_fig_std(config, database, ids_all, days):
    unit = format_zernike_mode(config.zernike_mode)

    fig = plt.figure(figsize=(FIGURE_WIDTH, 9.0))
    gs = gridspec.GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[1, 1, 1],
        hspace=0.45,
        wspace=0.35,
    )
    ax_tiltx = fig.add_subplot(gs[0, 0])
    ax_tilty = fig.add_subplot(gs[0, 1])
    ax_defoc = fig.add_subplot(gs[1, 0])
    ax_astig = fig.add_subplot(gs[1, 1])
    ax_piston = fig.add_subplot(gs[2, :])  # spans both columns

    wf_axes = {
        "Tilt X": ax_tiltx,
        "Tilt Y": ax_tilty,
        "Defocus": ax_defoc,
        "Astig 0°": ax_astig,
    }

    # ------------------------------------------------------------------
    # 1) ALL DAYS POOLED — wavefront
    # ------------------------------------------------------------------
    _, data_b_all, common_time_all, labels, shot_ids_all = load_beam_pair(
        config.root,
        ids_all,
        zernike_mode=config.zernike_mode,
        columns=ZERNIKE_COLUMNS,
        verbose=False,
    )
    data_b_all = zero_at_reference(data_b_all, common_time_all, config.time_ref_ms)
    time_ms_all = common_time_all * 1e3

    mode_indices = {}
    for mode_name, substring in MODE_NAMES.items():
        idx = find_mode(labels, substring)
        if idx is None:
            print(f"  [WARN] Mode '{mode_name}' not found in labels")
        mode_indices[mode_name] = idx

    n_shots_all = data_b_all.shape[0]
    for mode_name, ax in wf_axes.items():
        idx = mode_indices.get(mode_name)
        if idx is None:
            continue
        std_all = np.std(data_b_all[:, idx, :], axis=0)
        ax.plot(
            time_ms_all,
            std_all,
            color=ALL_COLOR,
            linewidth=1.4,
            zorder=5,
            label=f"All days  ({n_shots_all} shots)",
        )

    # ------------------------------------------------------------------
    # 2) ALL DAYS POOLED — piston
    # ------------------------------------------------------------------
    piston_all = load_piston_results(config.root, ids_all, unwrap=True, verbose=False)
    piston_mat_all = interpolate_piston_to_grid(
        piston_all, common_time_all, config.time_ref_ms
    )
    if piston_mat_all.shape[0] > 0:
        std_p_all = np.nanstd(piston_mat_all, axis=0)
        ax_piston.plot(
            time_ms_all,
            std_p_all,
            color=ALL_COLOR,
            linewidth=1.4,
            zorder=5,
            label=f"All days  ({piston_mat_all.shape[0]} shots)",
        )

    # ------------------------------------------------------------------
    # 3) DAY BY DAY — wavefront + piston
    # ------------------------------------------------------------------
    for day_idx, (day, day_ids) in enumerate(days.items()):
        color = DAY_COLORS[day_idx % len(DAY_COLORS)]

        if not day_ids:
            continue

        # Wavefront
        _, data_b_day, common_time_day, _, shot_ids_day = load_beam_pair(
            config.root,
            day_ids,
            zernike_mode=config.zernike_mode,
            columns=ZERNIKE_COLUMNS,
            verbose=False,
        )
        data_b_day = zero_at_reference(data_b_day, common_time_day, config.time_ref_ms)
        time_ms_day = common_time_day * 1e3
        n_shots_day = data_b_day.shape[0]

        for mode_name, ax in wf_axes.items():
            idx = mode_indices.get(mode_name)
            if idx is None:
                continue
            std_day = np.std(data_b_day[:, idx, :], axis=0)
            ax.plot(
                time_ms_day,
                std_day,
                color=color,
                linewidth=1.0,
                zorder=4,
                label=f"{day}  ({n_shots_day} shots)",
            )

        # Piston
        piston_day = load_piston_results(
            config.root, day_ids, unwrap=True, verbose=False
        )
        piston_mat_day = interpolate_piston_to_grid(
            piston_day, common_time_day, config.time_ref_ms
        )
        if piston_mat_day.shape[0] > 0:
            std_p_day = np.nanstd(piston_mat_day, axis=0)
            ax_piston.plot(
                time_ms_day,
                std_p_day,
                color=color,
                linewidth=1.0,
                zorder=4,
                label=f"{day}  ({piston_mat_day.shape[0]} shots)",
            )

    # ------------------------------------------------------------------
    # 4) Axes formatting — WF panels
    # ------------------------------------------------------------------
    for mode_name, ax in wf_axes.items():
        ax.set_xlim(*XLIM_ZOOM)
        if YLIM_STD_WF:
            ax.set_ylim(*YLIM_STD_WF)
        add_flash_background(ax, config.root, ids_all, "B")
        ax.set_title(f"{mode_name}", fontsize=8)
        ax.set_ylabel(f"Std {unit} (waves)")

    ax_tiltx.set_xlabel("")
    ax_tilty.set_xlabel("")
    ax_defoc.set_xlabel("Time (ms)")
    ax_astig.set_xlabel("Time (ms)")

    # ------------------------------------------------------------------
    # 5) Axes formatting — piston panel
    # ------------------------------------------------------------------
    ax_piston.set_xlim(*XLIM_ZOOM)
    if YLIM_STD_P:
        ax_piston.set_ylim(*YLIM_STD_P)
    add_flash_background(ax_piston, config.root, ids_all, "B")
    ax_piston.set_title("Differential piston", fontsize=8)
    ax_piston.set_ylabel("Std piston (waves)")
    ax_piston.set_xlabel("Time (ms)")

    # ------------------------------------------------------------------
    # 6) Common legend — built from the piston panel which has all labels
    # ------------------------------------------------------------------
    all_handles, all_texts = ax_piston.get_legend_handles_labels()
    # Deduplicate while preserving order
    seen = {}
    for h, t in zip(all_handles, all_texts):
        if t not in seen:
            seen[t] = h
    add_bottom_legend(
        fig,
        list(seen.values()),
        list(seen.keys()),
        ncol=len(seen),
        bottom=0.08,
    )

    path = config.figures_dir / "daily_std_108B.pdf"
    save_figure(fig, path)
    print(f"  written: {path}")
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    apply_style()

    config = load_config()
    if ROOT is not None:
        config.root = Path(ROOT)
    config.check()

    print(f"Data directory : {config.root}")
    print(f"Normalisation  : {config.zernike_mode}")
    print(f"Reference time : {config.time_ref_ms} ms")
    print()

    database = load_database(config.root)

    ids_all = ids_for_108B_zoom(database)
    days = group_by_day(ids_all)

    print(f"  {len(ids_all)} shots total across {len(days)} days:")
    for day, ids in days.items():
        print(f"    {day}  →  {len(ids)} shots")
    print()

    make_fig_defocus(config, database, ids_all, days)
    make_fig_std(config, database, ids_all, days)

    plt.show()
