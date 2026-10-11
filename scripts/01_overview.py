"""Overview of a set of shots: Zernike traces and combining efficiency.

Run this script directly. It reads the condensed RESULTS directory, puts
every selected shot on a common time base, and produces two figures in the
output directory:

* ``01_zernike_traces.pdf`` - temporal evolution of the differential
  Zernike coefficients, one curve per mode, averaged over the shots, with
  the shot-to-shot dispersion as a shaded band;
* ``01_combining_efficiency.pdf`` - resulting coherent beam combining
  efficiency as a function of time, per shot and on average.

It also prints a summary in the terminal, which is the quickest way to
check that the data are read as expected.
"""

# ===========================================================================
#  EDIT HERE
# ===========================================================================
# Path to the directory containing data_base.json, RESULTS/ and PHASICS/.
# Leave it as None to use the value of config/config.yaml instead.
#
# Windows: use forward slashes, or prefix the string with r as below.
ROOT = None
# ROOT = r"D:\DATA CHOCOLATE2"

# Restrict the analysis to the shots of one day, for instance "2025-12-02".
# None means every shot of the database.
DAY = "2025-12-02"  # None

# At most this many shots, to keep a first run fast. None means no limit.
MAX_SHOTS = 10
# ===========================================================================

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from chocolate.config import load_config
from chocolate.database import list_measurement_ids, load_database, select_ids
from chocolate.efficiency import differential_coefficients, zernike_analysis
from chocolate.results_io import load_beam_pair
from chocolate.style import COLOR_LIST, FIGSIZE_SINGLE, apply_style, save_figure
from chocolate.zernike import PRETTY_LABELS, format_zernike_mode


def main():
    apply_style()

    # -- Configuration ------------------------------------------------------
    config = load_config()
    if ROOT is not None:
        config.root = Path(ROOT)
    config.check()

    print(f"Data directory : {config.root}")
    print(f"Figures        : {config.figures_dir}")
    print(f"Normalisation  : {config.zernike_mode}")
    print(f"Reference time : {config.time_ref_ms} ms")
    print()

    # -- Shot selection -----------------------------------------------------
    database = load_database(config.root)
    print(f"Database       : {len(list_measurement_ids(database))} measurements")

    conditions = []
    if DAY is not None:
        conditions.append(lambda shot_id, meta: shot_id.startswith(DAY))
    ids = select_ids(database, conditions)
    if MAX_SHOTS is not None:
        ids = ids[:MAX_SHOTS]
    print(f"Selected       : {len(ids)} measurements")
    if not ids:
        raise SystemExit("Nothing to analyse, relax the selection above.")

    # -- Loading ------------------------------------------------------------
    data_a, data_b, common_time, labels, shot_ids = load_beam_pair(
        config.root, ids, zernike_mode=config.zernike_mode
    )
    print(
        f"Loaded         : {data_a.shape[0]} shots, {data_a.shape[1]} modes, "
        f"{data_a.shape[2]} time samples"
    )
    print(
        f"Time span      : {common_time[0] * 1e3:.2f} to "
        f"{common_time[-1] * 1e3:.2f} ms"
    )
    print()

    # -- Differential and efficiency ---------------------------------------
    differential, mean, variance = differential_coefficients(
        data_a, data_b, common_time, config.time_ref_ms
    )
    efficiency, efficiency_mean, efficiency_std = zernike_analysis(
        differential, model="small_fluctuations", mode=config.zernike_mode
    )

    index_shot = int(np.argmin(np.abs(common_time)))
    print(
        f"Efficiency at t = 0 : {efficiency_mean[index_shot] * 100:.2f} % "
        f"+/- {efficiency_std[index_shot] * 100:.2f} %"
    )
    print(
        f"Minimum efficiency  : {np.nanmin(efficiency_mean) * 100:.2f} % "
        f"at t = {common_time[np.nanargmin(efficiency_mean)] * 1e3:.2f} ms"
    )
    print()

    time_ms = common_time * 1e3
    unit = format_zernike_mode(config.zernike_mode)

    # -- Figure 1: differential Zernike traces ------------------------------
    figure, axes = plt.subplots(figsize=FIGSIZE_SINGLE)
    for mode in range(mean.shape[0]):
        label = PRETTY_LABELS.get(str(labels[mode]), str(labels[mode]))
        colour = COLOR_LIST[mode % len(COLOR_LIST)]
        axes.plot(time_ms, mean[mode], color=colour, label=label)
        axes.fill_between(
            time_ms,
            mean[mode] - np.sqrt(variance[mode]),
            mean[mode] + np.sqrt(variance[mode]),
            color=colour,
            alpha=0.15,
            linewidth=0,
        )
    axes.axvline(0.0, color="black", linewidth=0.6, linestyle="--")
    axes.set_xlabel("Time (ms)")
    axes.set_ylabel(f"Differential Zernike coefficient {unit} (waves)")
    axes.set_title(f"Beam A minus beam B, {len(shot_ids)} shots")
    axes.legend(ncol=2, fontsize=6, loc="best")
    save_figure(figure, config.figures_dir / "01_zernike_traces.pdf")

    # -- Figure 2: combining efficiency -------------------------------------
    figure, axes = plt.subplots(figsize=FIGSIZE_SINGLE)
    for shot in range(efficiency.shape[0]):
        axes.plot(
            time_ms, efficiency[shot] * 100, color="grey", alpha=0.35, linewidth=0.6
        )
    axes.plot(
        time_ms,
        efficiency_mean * 100,
        color=COLOR_LIST[0],
        linewidth=1.6,
        label="Mean over shots",
    )
    axes.fill_between(
        time_ms,
        (efficiency_mean - efficiency_std) * 100,
        (efficiency_mean + efficiency_std) * 100,
        color=COLOR_LIST[0],
        alpha=0.2,
        linewidth=0,
        label="Standard deviation",
    )
    axes.axvline(0.0, color="black", linewidth=0.6, linestyle="--")
    axes.set_xlabel("Time (ms)")
    axes.set_ylabel("Combining efficiency (%)")
    axes.set_title(f"Small fluctuation model, {len(shot_ids)} shots")
    axes.legend(loc="lower left")
    save_figure(figure, config.figures_dir / "01_combining_efficiency.pdf")

    print(f"Two figures written to {config.figures_dir}")
    plt.show()


if __name__ == "__main__":
    main()
