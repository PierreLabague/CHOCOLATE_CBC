"""Comparing three ways of computing the coherent combining efficiency.

Two beams whose wavefronts differ by a map D interfere with a local
intensity (1 + cos(2 pi D)) / 2, and the combining efficiency is that
quantity averaged over the pupil. Three routes lead to it, and the whole
campaign rests on the cheapest of them being trustworthy.

    RWMC — raw wavefront map computation
        The exported wavefront maps of the two beams are subtracted pixel
        by pixel and the cosine averaged. Nothing is projected, nothing is
        approximated; this is the reference.
    ZWR — Zernike wavefront reconstruction
        The wavefront is rebuilt from its fifteen Zernike coefficients,
        then treated exactly as above. The gap to RWMC measures what the
        truncation of the expansion costs.
    ZAA — Zernike analytic approximation
        The cosine is expanded to second order, which reduces the pupil
        average to a sum of squared coefficients. The gap to ZWR measures
        what the expansion costs, on top of the truncation.

Both beams are analysed through masks of the same size, so the comparison
is between methods and nothing else: no resampling, no difference of
pupil area to argue about.

Piston is excluded
------------------
A wavefront sensor reads the shape of a wavefront, not its absolute
optical path: the constant term it reports is set by its own reference and
is not a measurement. The Zernike piston, the coefficient of the mode Z0,
is therefore removed on all three routes, and what is compared here is the
efficiency the higher orders alone would give.

Removing it means subtracting the mean over the **analysed pupil**, not
over the square crop that contains it: Z0 equals one across the disc, so
its coefficient is exactly that mean, and averaging over anything else
would take out the wrong quantity.

The true optical path difference is measured separately, by the fringe
analysis of the interferometric branch, and combines with what is computed
here. This acquisition has no such measurement, which is another reason to
keep the two apart rather than let a meaningless constant into the
comparison.

Run this script directly.
"""

# =============================================================================
# %% PARAMETERS
# =============================================================================

ROOT = None


# Sub-directory holding the analysis of the test acquisition, and the name
# of the raw directory inside it.
TEST_SUBDIR = "test_polynome"
RAW_DIRNAME = "PHASICS_NEW"

# The analysis where both beams went through masks of the same size.
SHOT = "2025-11-28-17-34_PHA_SAME"

# Frames analysed, as a slice of the acquisition.
FRAME_RANGE = (0, 3201)

# Frame the efficiency is referenced to: the instant the two beams are
# taken to be phased, from which everything is a departure.
REFERENCE_FRAME = 0

# ---- Pupil sampling of the Zernike reconstruction ---------------------------
# The mask size the SID4 reports is in camera pixels; a phase pixel of the
# exported map covers PHASE_BINNING of them, so the pupil is
# Width / PHASE_BINNING phase pixels across. Reconstructing on that grid is
# what makes the comparison fair: a finer grid adds no information the
# sensor ever had, a coarser one throws some away.
#
# This is deliberately NOT taken from the crop measured on the map: the
# circular pupil does not quite touch the edges of the square export, so
# the crop is a little wider than the pupil itself.
PHASE_BINNING = 3

# None derives the resolution from the mask width, as above. Set a number
# to override it.
RECONSTRUCTION_RESOLUTION = None

# ---- Analysis masks ---------------------------------------------------------
# Read from the SID4 export of this acquisition, in camera pixels:
#
#     beam A   166 x 166 px, centred on X = 90,  Y = 94
#     beam B   166 x 166 px, centred on X = 284, Y = 92
#
# Same size for both, which is what makes the comparison clean: the two
# pupils cover the same number of pixels, so no resampling is needed and no
# difference of area has to be argued about.
#
# The exported maps are smaller than the camera frame by the binning factor
# of the SID4, so those coordinates have to be divided by it. The crops
# below are in MAP pixels and are written out explicitly rather than
# derived, so that what the script uses is exactly what is on the page.
# Figure 0 draws them on the maps; adjust them there until the two boxes
# sit on the two pupils, then leave them alone.
#
# CAREFUL: the SID4 exports only the analysed region, not the whole camera
# frame, so the two beams usually come out as two separate small maps and
# the coordinates above do NOT apply to them directly. The script prints
# the actual map size at load time and refuses a crop that does not fit.
#
# Start from the whole map on both beams:
#     CROP_A = (slice(0, None), slice(0, None))
# then read the right box off figure 0 and write it in.
CROP_A = (slice(0, 10_000), slice(0, 10_000))
CROP_B = (slice(0, 10_000), slice(0, 10_000))

# ---- Scale factor of the exported maps --------------------------------------
# The exported map is a 16-bit integer image spanning the full peak-to-
# valley excursion of the wavefront, so it comes back to waves as
#
#     waves = tiff / 65536 * PtV
#
# with PtV the per-frame value the export carries. Columns are looked up by
# tag, not by index: this analysis exports two extra columns at the front,
# "PtV Phase" and "RMS Phase", which the standard one does not, and an
# index taken from one layout points at the wrong quantity in the other.
#
# Getting this wrong rescales the whole measured wavefront and nothing
# else, so RWMC moves while the two Zernike models stay put. The check
# below catches it: the scaled map must have the RMS the export reports.
PHA_SCALE_TAG = "PtV Phase"
PHA_RMS_TAG = "RMS Phase"

# ---- Manuscript summary figure ----------------------------------------------
# Acquisition whose intensity traces carry the pump fluorescence, under the
# main RESULTS directory. Used only to shade the background of the summary
# figure; set to None, or leave it missing, and the shading is skipped.
FLUORESCENCE_SHOT = "2025-11-28-17-34"

# Time window of the zoomed panel, in milliseconds.
SUMMARY_ZOOM_MS = (-1.0, 1.0)

# Vertical range shared by the two temporal panels.
SUMMARY_YLIM = (0.6, 1.1)

# ---- Correction -------------------------------------------------------------
CORRECTION_DEGREE = 4  # degree of the polynomial fitted from a model onto RWMC

# Frames of each end used for the two separate fits. The point of splitting
# is to see whether one correction holds over the whole shot or drifts.
HALF_LENGTH = 15000

SHOW_PROGRESS = True


# =============================================================================
# %% IMPORTS AND SETUP
# =============================================================================

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from chocolate.config import load_config
from chocolate.efficiency import (
    combining_efficiency_reconstruction,
    combining_efficiency_small_fluctuations,
)
from chocolate.plotting import add_flash_background
from chocolate.results_io import load_wavefront_results
from chocolate.style import (
    CMAP_WAVEFRONT,
    CMAP_INTENSITY,
    FIGURE_WIDTH,
    FLASH_COLOR,
    FLASH_LINE_COLOR,
    apply_style,
    save_figure,
)
from chocolate.wavefront_maps import TIFF_FULL_SCALE, read_phasics_maps
from chocolate.zernike import (
    RMS_TO_AMP,
    ZERNIKE_MODES,
    convert_zernike_mode,
    zernike_wavefront_on,
)

#: Colours of the three methods, kept identical on every figure.
COLOR_RWMC = "tab:green"
COLOR_ZWR = "tab:blue"
COLOR_ZAA = "tab:orange"
COLOR_IDEAL = "tab:red"

#: Interference patterns are shown in grey: they are images of light, the
#: quantity a detector records and the eye reads as brightness directly.
#: It also keeps that panel apart from the three wavefront maps around it,
#: which carry a signed quantity and need a diverging scale.
CMAP_INTERFERENCE = CMAP_INTENSITY

apply_style()

config = load_config()
if ROOT is not None:
    config.root = Path(ROOT)
config.check()

test_root = config.root / TEST_SUBDIR
first, last = FRAME_RANGE

print(f"Data directory : {config.root}", flush=True)
print(f"Test directory : {test_root}")
print(f"Shot           : {SHOT}")
print(f"Frames         : {first} to {last}")


def progress(iterable, description, total=None):
    """Wrap an iterable in a progress bar when tqdm is available."""
    if not SHOW_PROGRESS:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, desc=description, total=total, unit="frame", leave=False)


# =============================================================================
# %% LOADING — Zernike coefficients
# =============================================================================

results_a = load_wavefront_results(test_root, "A", [SHOT], verbose=True)
results_b = load_wavefront_results(test_root, "B", [SHOT], verbose=True)

if SHOT not in results_a or SHOT not in results_b:
    raise SystemExit(f"L'analyse {SHOT} n'a pas pu être lue pour les deux faisceaux.")

labels = [str(t) for t in results_a[SHOT]["tag"]]
time = np.asarray(results_a[SHOT]["time"], dtype=float)[first:last]
time_ms = time * 1e3

# Columns found by name: an index that is off by one silently mixes up the
# modes, and the two analyses need not export the same column order.
zernike_columns = [labels.index(name) for name in ZERNIKE_MODES]
piston_column = ZERNIKE_MODES.index("Piston")

raw_a = results_a[SHOT]["data"][first:last, :]
raw_b = results_b[SHOT]["data"][first:last, :]

# Mask geometry as exported, for the check further down.
for beam, block in (("A", raw_a), ("B", raw_b)):
    geometry = {
        name: block[0, labels.index(name)]
        for name in ("Width (pix)", "Height (pix)", "Center X (pix)", "Center Y (pix)")
        if name in labels
    }
    print(
        f"  masque {beam} : " + ", ".join(f"{k} = {v:.0f}" for k, v in geometry.items())
    )

# Pupil diameter in phase pixels, from the mask the SID4 actually used.
mask_width = float(raw_a[0, labels.index("Width (pix)")])
if RECONSTRUCTION_RESOLUTION is None:
    resolution = int(round(mask_width / PHASE_BINNING))
else:
    resolution = int(RECONSTRUCTION_RESOLUTION)
print(
    f"  pupille : {mask_width:.0f} px caméra / {PHASE_BINNING} = "
    f"{mask_width / PHASE_BINNING:.1f} px de phase → reconstruction sur "
    f"{resolution}x{resolution}"
)

print(f"Loaded         : {len(time)} frames", flush=True)


# =============================================================================
# %% LOADING — Exported wavefront maps, cropped to their pupil
# =============================================================================


def maps_folder(beam):
    """Path of the exported maps of one beam."""
    return (
        test_root
        / RAW_DIRNAME
        / f"{SHOT}_PHASICS"
        / "SID4"
        / f"{SHOT}_W{beam}"
        / "Results"
    )


# The slow step of the whole script: a full acquisition is a few thousand
# TIFF files per beam, and nothing can be drawn before they are all read.
print("Lecture des cartes de front d'onde (étape la plus longue)…", flush=True)
maps_a = read_phasics_maps(maps_folder("A"), description="Cartes A")[first:last]
maps_b = read_phasics_maps(maps_folder("B"), description="Cartes B")[first:last]

print(
    f"  taille des cartes : A {maps_a.shape[1]}x{maps_a.shape[2]}, "
    f"B {maps_b.shape[1]}x{maps_b.shape[2]} px"
)

# Numpy truncates a slice that runs past the end of an array instead of
# raising, so a crop that does not fit produces a smaller block and the
# error only surfaces much later, as a broadcasting failure. It is checked
# here, where the message can still say what to do about it.
for beam, block, crop in (("A", maps_a, CROP_A), ("B", maps_b, CROP_B)):
    rows, columns = crop
    height, width = block.shape[1], block.shape[2]
    fits = rows.stop <= height and columns.stop <= width
    takes_all = rows.stop >= height and columns.stop >= width
    if not fits and not takes_all:
        raise SystemExit(
            f"CROP_{beam} = [{rows.start}:{rows.stop}, "
            f"{columns.start}:{columns.stop}] déborde des cartes du faisceau "
            f"{beam}, qui font {height}x{width} px.\n"
            f"Les cartes exportées ne couvrent que la région analysée, pas "
            f"la trame caméra entière : les coordonnées du masque "
            f"(166 px centré sur X = 90 ou 284) ne s'y appliquent pas "
            f"directement.\n"
            f"Mettez CROP_{beam} = (slice(0, {height}), slice(0, {width})) "
            f"pour prendre la carte entière, puis resserrez à la vue de la "
            f"figure 0."
        )

crop_a_preview = maps_a[REFERENCE_FRAME][CROP_A[0], CROP_A[1]]
crop_b_preview = maps_b[REFERENCE_FRAME][CROP_B[0], CROP_B[1]]
shape_a = crop_a_preview.shape
shape_b = crop_b_preview.shape
# Effective bounds, after the slices have been clipped to the map.
bounds_a = (CROP_A[0].indices(maps_a.shape[1]), CROP_A[1].indices(maps_a.shape[2]))
bounds_b = (CROP_B[0].indices(maps_b.shape[1]), CROP_B[1].indices(maps_b.shape[2]))
print(
    f"  découpe A : lignes {bounds_a[0][0]}-{bounds_a[0][1]}, "
    f"colonnes {bounds_a[1][0]}-{bounds_a[1][1]}  ({shape_a[0]}x{shape_a[1]})"
)
print(
    f"  découpe B : lignes {bounds_b[0][0]}-{bounds_b[0][1]}, "
    f"colonnes {bounds_b[1][0]}-{bounds_b[1][1]}  ({shape_b[0]}x{shape_b[1]})"
)

if shape_a != shape_b:
    raise SystemExit(
        f"Les deux découpes ne font pas la même taille ({shape_a} contre "
        f"{shape_b}). Elles doivent l'être : les deux masques font 166 px de "
        f"côté, donc les deux boîtes aussi."
    )


# =============================================================================
# %% FIGURE 0 — What the crops actually select
# =============================================================================
#
# Drawn before anything is computed, because every number below depends on
# these two boxes landing on the two pupils. A box that is off by a few
# pixels lets background into the average and quietly lowers the
# efficiency, without any error being raised.

fig, axs = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 2.6))

for ax, block, crop, beam in (
    (axs[0], maps_a, CROP_A, "A"),
    (axs[1], maps_b, CROP_B, "B"),
):
    ax.imshow(block[REFERENCE_FRAME], cmap=CMAP_WAVEFRONT)
    rows = slice(*crop[0].indices(block.shape[1]))
    columns = slice(*crop[1].indices(block.shape[2]))
    ax.add_patch(
        Rectangle(
            (columns.start - 0.5, rows.start - 0.5),
            columns.stop - columns.start,
            rows.stop - rows.start,
            fill=False,
            edgecolor=COLOR_IDEAL,
            linewidth=1.2,
        )
    )
    ax.set_title(
        f"Beam {beam} — crop [{rows.start}:{rows.stop}, "
        f"{columns.start}:{columns.stop}]",
        fontsize=8,
    )
    ax.set_xlabel("Column (map px)")
axs[0].set_ylabel("Row (map px)")

fig.suptitle(f"Selected pupils, frame {REFERENCE_FRAME}", fontsize=9)
written_zero = save_figure(fig, config.figures_dir / "efficiency_models_crops.pdf")
print(f"  vérification des découpes : {written_zero}")

# Back to waves: 16-bit integers times the per-frame scale of the export.
if PHA_SCALE_TAG not in labels:
    raise SystemExit(
        f"Colonne {PHA_SCALE_TAG!r} absente de l'export. Colonnes "
        f"disponibles : {', '.join(labels)}"
    )
scale_column = labels.index(PHA_SCALE_TAG)
print(
    f"  facteur d'échelle : colonne {scale_column}, {PHA_SCALE_TAG!r}, "
    f"médiane {np.nanmedian(raw_a[:, scale_column]):.4g} vague"
)

scale_a = raw_a[:, scale_column]
scale_b = raw_b[:, scale_column]

crop_a = maps_a[:, CROP_A[0], CROP_A[1]] * scale_a[:, None, None] / TIFF_FULL_SCALE
crop_b = maps_b[:, CROP_B[0], CROP_B[1]] * scale_b[:, None, None] / TIFF_FULL_SCALE

# Pixels the mask rejected come back as exactly zero and carry no
# measurement; they must not be averaged in.
crop_a[crop_a == 0] = np.nan
crop_b[crop_b == 0] = np.nan


# ---- Is the scale factor right? ---------------------------------------------
# The export states the RMS of each frame, so the scaled map must have that
# RMS. This is a direct check with nothing fitted: a ratio away from one
# means the wrong column is being used as the scale, and every efficiency
# computed from the maps is then wrong.
#
# It only holds once the piston is removed, since the RMS the export
# reports is that of the wavefront about its own mean.
if PHA_RMS_TAG in labels:
    rms_column = labels.index(PHA_RMS_TAG)
    print("Vérification du facteur d'échelle des cartes…")
    for beam, crop, block in (("A", crop_a, raw_a), ("B", crop_b, raw_b)):
        frame = crop[REFERENCE_FRAME]
        measured_rms = float(np.nanstd(frame - np.nanmean(frame)))
        stated_rms = float(block[REFERENCE_FRAME, rms_column])
        ratio = measured_rms / stated_rms if stated_rms else np.nan
        flag = "" if 0.9 < ratio < 1.1 else "   <<< à vérifier"
        print(
            f"  faisceau {beam} : carte {measured_rms:.4f} vague RMS, "
            f"export {stated_rms:.4f}, rapport {ratio:.3f}{flag}"
        )
else:
    print(
        f"  [info] colonne {PHA_RMS_TAG!r} absente, l'échelle des cartes "
        f"n'est pas vérifiée."
    )
print()


# =============================================================================
# %% RWMC — Efficiency from the raw maps
# =============================================================================
#
# The two maps are subtracted pixel by pixel, then the Zernike piston is
# removed from each frame. Since Z0 is one over the whole disc, its
# coefficient is the mean over the pupil, and nanmean over the crop gives
# exactly that: the pixels outside the mask are nan and are left out of the
# average, so the square that contains the disc does not bias it.

print("Calcul RWMC…", flush=True)
differential = crop_a - crop_b
differential = differential - np.nanmean(differential, axis=(1, 2))[:, None, None]
differential = differential - differential[REFERENCE_FRAME]

support = int(np.sum(np.isfinite(differential[REFERENCE_FRAME])))
if support == 0:
    raise SystemExit(
        "Aucun pixel exploitable : la carte de référence est vide. Vérifiez "
        "REFERENCE_FRAME et le contenu des cartes exportées."
    )

eff_rwmc = (1 + np.nansum(np.cos(2 * np.pi * differential), axis=(1, 2)) / support) / 2

print(
    f"RWMC : support {support} px, efficacité de {np.nanmin(eff_rwmc):.3f} "
    f"à {np.nanmax(eff_rwmc):.3f}"
)


# =============================================================================
# %% ZWR — Efficiency from the reconstructed wavefronts
# =============================================================================
#
# This is a single call to chocolate.efficiency.combining_efficiency_
# reconstruction, deliberately: the whole point of the comparison is to
# test the function the rest of the project uses, not a copy of it written
# for the occasion. Anything the function does differently would otherwise
# go unnoticed here and turn up later.
#
# The coefficients go to the bare polynomial convention, which is the one
# the reconstruction uses. Piston is left out by default, and the
# reference instant is handled inside the function, on the coefficients:
# the reconstruction is linear, so subtracting them before is the same as
# subtracting the maps after, and much cheaper.

amp_a = convert_zernike_mode(raw_a, labels, mode="amp")[:, zernike_columns]
amp_b = convert_zernike_mode(raw_b, labels, mode="amp")[:, zernike_columns]
amp_differential = amp_a - amp_b

print("Reconstruction des fronts d'onde Zernike…", flush=True)
eff_zwr = combining_efficiency_reconstruction(
    amp_differential,
    resolution=resolution,
    mode="amp",
    include_piston=False,
    reference=REFERENCE_FRAME,
)

print(
    f"ZWR  : grille {resolution}x{resolution}, efficacité de "
    f"{np.nanmin(eff_zwr):.3f} à {np.nanmax(eff_zwr):.3f}"
)


# =============================================================================
# %% ZAA — Efficiency from the analytic approximation
# =============================================================================
#
# The second-order expansion only reduces to a sum of squares in an
# orthonormal basis, so the coefficients go back to the RMS convention.
# Piston is left out of the sum, consistently with the two other routes;
# its coefficient is already zero, but dropping the column says so
# explicitly rather than relying on it.

rms_factors = np.array([RMS_TO_AMP[name] for name in ZERNIKE_MODES])
rms_differential = amp_differential / rms_factors

higher_orders = [index for index in range(len(ZERNIKE_MODES)) if index != piston_column]

# Shape (1 shot, n_modes, n_times), referenced to the chosen frame.
delta = rms_differential[:, higher_orders].T[None, :, :]
delta = delta - delta[:, :, [REFERENCE_FRAME]]
eff_zaa = combining_efficiency_small_fluctuations(delta)[0]

print(f"ZAA  : efficacité de {np.nanmin(eff_zaa):.3f} à {np.nanmax(eff_zaa):.3f}")
print()


# =============================================================================
# %% FIGURE 1 — The three methods against time
# =============================================================================

fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 3.4))

ax.plot(
    time_ms,
    eff_rwmc,
    color=COLOR_RWMC,
    alpha=0.85,
    linewidth=1.0,
    label="RWMC — raw wavefront maps",
)
ax.plot(
    time_ms,
    eff_zwr,
    color=COLOR_ZWR,
    alpha=0.85,
    linewidth=1.0,
    label="ZWR — Zernike reconstruction",
)
ax.plot(
    time_ms,
    eff_zaa,
    color=COLOR_ZAA,
    alpha=0.85,
    linewidth=1.0,
    label="ZAA — analytic approximation",
)

ax.set_ylim(0, 1.05)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("CBC efficiency")
ax.set_title(f"Temporal efficiency, higher orders only — {SHOT}", fontsize=9)
ax.legend(fontsize=7, loc="lower left")

written = [
    written_zero,
    save_figure(fig, config.figures_dir / "efficiency_models_time.pdf"),
]


# =============================================================================
# %% FIGURE 2 — Model against reference, coloured by time
# =============================================================================
#
# Each point is one frame. A method that agreed with the reference would
# lie on the diagonal; the shape of the departure says more than a single
# error figure, and colouring by time shows whether it drifts during the
# shot.

fig, axs = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 3.2), sharex=True, sharey=True)
normalise = Normalize(vmin=time_ms.min(), vmax=time_ms.max())

for ax, model, heading in (
    (axs[0], eff_zwr, "ZWR — Zernike reconstruction"),
    (axs[1], eff_zaa, "ZAA — analytic approximation"),
):
    scatter = ax.scatter(
        eff_rwmc,
        model,
        c=time_ms,
        cmap=CMAP_WAVEFRONT,
        norm=normalise,
        marker="o",
        alpha=0.8,
        s=3,
        linewidths=0,
    )
    ax.plot([0, 1], [0, 1], color=COLOR_IDEAL, linewidth=1.0)
    ax.set_xlabel("RWMC efficiency")
    ax.set_title(heading, fontsize=8)

axs[0].set_ylabel("Model efficiency")

bar = fig.colorbar(scatter, ax=axs, orientation="vertical", fraction=0.04, pad=0.02)
bar.set_label("Time (ms)", fontsize=8)

written.append(save_figure(fig, config.figures_dir / "efficiency_models_scatter.pdf"))


# =============================================================================
# %% POLYNOMIAL CORRECTION — Fitting each model onto the reference
# =============================================================================
#
# Both models depart from the reference in a smooth, monotonic way, so a
# low-order polynomial fitted from the model onto RWMC recovers most of the
# gap. Whether that correction is a property of the method or of this
# particular shot is what the stability figures below settle.

coef_zwr = np.polyfit(eff_zwr, eff_rwmc, deg=CORRECTION_DEGREE)
coef_zaa = np.polyfit(eff_zaa, eff_rwmc, deg=CORRECTION_DEGREE)

corrected_zwr = np.poly1d(coef_zwr)(eff_zwr)
corrected_zaa = np.poly1d(coef_zaa)(eff_zaa)

for name, model, corrected in (
    ("ZWR", eff_zwr, corrected_zwr),
    ("ZAA", eff_zaa, corrected_zaa),
):
    before = np.nanmean(np.abs(model - eff_rwmc)) * 100
    after = np.nanmean(np.abs(corrected - eff_rwmc)) * 100
    print(
        f"{name} : écart moyen {before:5.2f} % → {after:5.2f} % après "
        f"correction de degré {CORRECTION_DEGREE}"
    )
print()


# =============================================================================
# %% FIGURE 3 — Effect of the polynomial correction
# =============================================================================

fig, axs = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 3.2), sharex=True, sharey=True)

for ax, model, corrected, heading in (
    (axs[0], eff_zwr, corrected_zwr, "ZWR"),
    (axs[1], eff_zaa, corrected_zaa, "ZAA"),
):
    ax.scatter(
        eff_rwmc,
        model,
        color=COLOR_ZWR,
        marker="o",
        alpha=0.3,
        s=4,
        linewidths=0,
        label="Uncorrected",
    )
    ax.scatter(
        eff_rwmc,
        corrected,
        color=COLOR_ZAA,
        marker="o",
        alpha=0.3,
        s=4,
        linewidths=0,
        label="Corrected",
    )
    ax.plot([0, 1], [0, 1], color=COLOR_IDEAL, linewidth=1.0, label="Ideal")
    ax.set_xlabel("RWMC efficiency")
    ax.set_title(heading, fontsize=8)

axs[0].set_ylabel("Model efficiency")
axs[0].legend(fontsize=6.5, loc="upper left")
fig.suptitle(f"Polynomial correction, degree {CORRECTION_DEGREE}", fontsize=10)

written.append(
    save_figure(fig, config.figures_dir / "efficiency_models_correction.pdf")
)


# =============================================================================
# %% FIGURE 4 — Is the ZWR correction stable over the shot?
# =============================================================================
#
# The same fit is repeated on the first and last stretch of the
# acquisition. If the three correction curves coincide, the correction is a
# property of the method and can be applied elsewhere; if they separate, it
# belongs to this shot and must not be reused.

coef_zwr_first = np.polyfit(
    eff_zwr[:HALF_LENGTH], eff_rwmc[:HALF_LENGTH], deg=CORRECTION_DEGREE
)
coef_zwr_last = np.polyfit(
    eff_zwr[-HALF_LENGTH:], eff_rwmc[-HALF_LENGTH:], deg=CORRECTION_DEGREE
)

sweep = np.linspace(0, 1, 200)

fig, axs = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 3.0))

axs[0].scatter(
    eff_rwmc,
    eff_zwr,
    color=COLOR_ZWR,
    marker="o",
    alpha=0.3,
    s=4,
    linewidths=0,
    label="Uncorrected",
)
axs[0].scatter(
    eff_rwmc[:HALF_LENGTH],
    np.poly1d(coef_zwr_first)(eff_zwr[:HALF_LENGTH]),
    color=COLOR_ZAA,
    marker="o",
    alpha=0.3,
    s=4,
    linewidths=0,
    label="Fitted on the first stretch",
)
axs[0].scatter(
    eff_rwmc[-HALF_LENGTH:],
    np.poly1d(coef_zwr_last)(eff_zwr[-HALF_LENGTH:]),
    color=COLOR_RWMC,
    marker="o",
    alpha=0.3,
    s=4,
    linewidths=0,
    label="Fitted on the last stretch",
)
axs[0].plot([0, 1], [0, 1], color=COLOR_IDEAL, linewidth=1.0, label="Ideal")
axs[0].set_xlabel("RWMC efficiency")
axs[0].set_ylabel("Model efficiency")
axs[0].set_title("ZWR", fontsize=8)
axs[0].legend(fontsize=6, loc="upper left")

axs[1].plot(
    sweep,
    100 * (sweep - np.poly1d(coef_zwr)(sweep)),
    color=COLOR_ZWR,
    label="Whole shot",
)
axs[1].plot(
    sweep,
    100 * (sweep - np.poly1d(coef_zwr_first)(sweep)),
    color=COLOR_ZAA,
    label="First stretch",
)
axs[1].plot(
    sweep,
    100 * (sweep - np.poly1d(coef_zwr_last)(sweep)),
    color=COLOR_RWMC,
    label="Last stretch",
)
axs[1].set_xlabel("Efficiency")
axs[1].set_ylabel("Correction applied (%)")
axs[1].set_title("Correction function", fontsize=8)
axs[1].legend(fontsize=6)

written.append(
    save_figure(fig, config.figures_dir / "efficiency_correction_stability_ZWR.pdf")
)


# =============================================================================
# %% FIGURE 5 — Is the ZAA correction stable over the shot?
# =============================================================================

coef_zaa_first = np.polyfit(
    eff_zaa[:HALF_LENGTH], eff_rwmc[:HALF_LENGTH], deg=CORRECTION_DEGREE
)
coef_zaa_last = np.polyfit(
    eff_zaa[-HALF_LENGTH:], eff_rwmc[-HALF_LENGTH:], deg=CORRECTION_DEGREE
)

fig, axs = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 3.0))

axs[0].scatter(
    eff_rwmc,
    eff_zaa,
    color=COLOR_ZWR,
    marker="o",
    alpha=0.3,
    s=4,
    linewidths=0,
    label="Uncorrected",
)
axs[0].scatter(
    eff_rwmc[:HALF_LENGTH],
    np.poly1d(coef_zaa_first)(eff_zaa[:HALF_LENGTH]),
    color=COLOR_ZAA,
    marker="o",
    alpha=0.3,
    s=4,
    linewidths=0,
    label="Fitted on the first stretch",
)
axs[0].scatter(
    eff_rwmc[-HALF_LENGTH:],
    np.poly1d(coef_zaa_last)(eff_zaa[-HALF_LENGTH:]),
    color=COLOR_RWMC,
    marker="o",
    alpha=0.3,
    s=4,
    linewidths=0,
    label="Fitted on the last stretch",
)
axs[0].plot([0, 1], [0, 1], color=COLOR_IDEAL, linewidth=1.0, label="Ideal")
axs[0].set_xlabel("RWMC efficiency")
axs[0].set_ylabel("Model efficiency")
axs[0].set_title("ZAA", fontsize=8)
axs[0].legend(fontsize=6, loc="upper left")

axs[1].plot(
    sweep,
    100 * (sweep - np.poly1d(coef_zaa)(sweep)),
    color=COLOR_ZWR,
    label="Whole shot",
)
axs[1].plot(
    sweep,
    100 * (sweep - np.poly1d(coef_zaa_first)(sweep)),
    color=COLOR_ZAA,
    label="First stretch",
)
axs[1].plot(
    sweep,
    100 * (sweep - np.poly1d(coef_zaa_last)(sweep)),
    color=COLOR_RWMC,
    label="Last stretch",
)
axs[1].set_xlabel("Efficiency")
axs[1].set_ylabel("Correction applied (%)")
axs[1].set_title("Correction function", fontsize=8)
axs[1].legend(fontsize=6)

written.append(
    save_figure(fig, config.figures_dir / "efficiency_correction_stability_ZAA.pdf")
)


# =============================================================================
# %% FIGURE 6 — Relative error of each method
# =============================================================================
#
# The same information as figure 2, expressed as the relative departure
# from the reference, which is the number a reader wants when deciding
# whether a method is good enough for a given purpose.

fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 3.0))

ax.scatter(
    eff_rwmc,
    100 * (eff_zwr - eff_rwmc) / eff_rwmc,
    color=COLOR_ZWR,
    marker="o",
    alpha=0.3,
    s=4,
    linewidths=0,
    label="ZWR",
)
ax.scatter(
    eff_rwmc,
    100 * (eff_zaa - eff_rwmc) / eff_rwmc,
    color=COLOR_ZAA,
    marker="o",
    alpha=0.3,
    s=4,
    linewidths=0,
    label="ZAA",
)
ax.axhline(0.0, color=COLOR_IDEAL, linewidth=1.0)

ax.set_xlabel("RWMC efficiency")
ax.set_ylabel("Relative error (%)")
ax.set_title("Departure of each model from the reference", fontsize=9)
ax.legend(fontsize=7)

written.append(save_figure(fig, config.figures_dir / "efficiency_models_error.pdf"))


# =============================================================================
# %% FIGURE 7 — One differential wavefront, measured and reconstructed
# =============================================================================
#
# The two maps of the same instant, side by side: what the sensor returned
# and what fifteen Zernike coefficients can say about it. The difference
# between them is the truncation, and it is what separates RWMC from ZWR on
# every figure above.

# Instant compared, in milliseconds. None takes the frame where the two
# methods disagree the most, which is the one worth looking at; set a
# number to inspect a particular moment.
COMPARISON_TIME_MS = 0

if COMPARISON_TIME_MS is None:
    index = int(np.nanargmax(np.abs(eff_zwr - eff_rwmc)))
    chosen = "écart maximal entre les deux méthodes"
else:
    index = int(np.argmin(np.abs(time_ms - COMPARISON_TIME_MS)))
    chosen = f"instant demandé, {COMPARISON_TIME_MS:g} ms"

print(f"Figure 7 : image {index}, t = {time_ms[index]:.3f} ms ({chosen})")
print(
    f"  RWMC {eff_rwmc[index] * 100:.2f} %, ZWR {eff_zwr[index] * 100:.2f} %, "
    f"ZAA {eff_zaa[index] * 100:.2f} %"
)

# ---- Putting the reconstruction on the grid of the measurement --------------
#
# The reconstruction is evaluated on the pixels of the measurement itself,
# not on an ideal disc of its own: a real analysis mask has stepped edges
# and is not exactly inscribed in the exported square, so a reconstruction
# built separately never lines up with it, and their difference would be
# dominated by the mismatch of the two supports rather than by the
# truncation, which is what the panel is meant to show.
#
# The change of coordinates is fixed, not a setting:
#
#   * a TIFF is stored row by row from the top, so the row index runs
#     downwards across the pupil;
#   * the Zernike polynomials are written in the optical convention, where
#     y runs upwards.
#
# The two frames differ by the sign of y and by nothing else, so the row
# offset is negated and the column offset is left alone. Both conventions
# are fixed and documented, so the sign follows from them: a positive
# Tilt Y is highest at the top of the pupil, which is row 0 of the image,
# and only the negation puts it there.
measured = differential[index]
valid = np.isfinite(measured)

rows = np.flatnonzero(valid.any(axis=1))
columns = np.flatnonzero(valid.any(axis=0))
centre_row = 0.5 * (rows[0] + rows[-1])
centre_column = 0.5 * (columns[0] + columns[-1])

row_index, column_index = np.indices(measured.shape)
distance = np.hypot(row_index - centre_row, column_index - centre_column)

# The radius is set by the outermost pixel the mask keeps, not by half the
# bounding box: a stepped mask has corner pixels that stick out past the
# inscribed circle, and normalising on the box would leave them at rho > 1,
# where zernike_wavefront_on returns nan. They would then be missing from
# the reconstruction while present in the measurement, and the two supports
# would differ by a ring of pixels.
radius = float(distance[valid].max())

delta_row = -(row_index - centre_row) / radius  # image rows down, pupil y up
delta_column = (column_index - centre_column) / radius

rho = np.hypot(delta_column, delta_row)
theta = np.arctan2(delta_row, delta_column)

reconstructed = zernike_wavefront_on(
    amp_differential[index] - amp_differential[REFERENCE_FRAME], rho, theta
)
# The measurement's own mask, so that both maps cover identical pixels.
reconstructed[~valid] = np.nan

# Piston is excluded from the model, so it must be from the picture too,
# otherwise the two panels would differ by a constant that plays no part in
# the efficiency. Both are averaged over the same support, so the two means
# are directly comparable.
measured = measured - np.nanmean(measured)
reconstructed = reconstructed - np.nanmean(reconstructed)

residual = measured - reconstructed

# Everything below is relative to the measured wavefront, which is exactly
# zero when the compared instant is the reference instant: the two beams
# are phased there by construction. That is a legitimate choice of
# COMPARISON_TIME_MS, so it is handled rather than left to divide by zero.
rms_measured_raw = float(np.nanstd(measured))
flat = rms_measured_raw < 1e-12
if flat:
    print(
        "  [info] l'instant comparé est l'instant de référence : la carte "
        "différentielle y est nulle par construction. Choisissez un autre "
        "COMPARISON_TIME_MS pour voir quelque chose."
    )
    relative_residual = float("nan")
    explained = float("nan")
else:
    # How much of the measured wavefront the fifteen modes account for. A
    # quality figure, not a test of anything: it is the truncation the
    # difference panel shows, expressed as one number.
    relative_residual = 100 * float(np.nanstd(residual)) / rms_measured_raw
    explained = 100 * (1.0 - np.nanvar(residual) / np.nanvar(measured))

print(
    f"  résidu de troncature : {np.nanstd(residual):.4f} vague RMS, "
    f"pour {rms_measured_raw:.4f} mesurée ({relative_residual:.0f} %)"
)
print(f"  variance expliquée par les 15 modes : {explained:.1f} %")

# One colour scale for the two wavefront panels, so that they can be
# compared by eye. Two independent scales would make a small residual look
# exactly like a large one. The difference gets its own, being smaller.
# A zero range would make imshow draw a uniform grey and hide the panel
# entirely, so a floor is set on both scales.
limit = max(
    float(np.nanmax([np.nanmax(np.abs(measured)), np.nanmax(np.abs(reconstructed))])),
    1e-12,
)
limit_residual = max(float(np.nanmax(np.abs(residual))), 1e-12)


def _statistics(carte):
    """RMS and peak-to-valley of a map, in waves, over the pupil."""
    values = carte[np.isfinite(carte)]
    return float(values.std()), float(values.max() - values.min())


rms_measured, ptv_measured = _statistics(measured)
rms_reconstructed, ptv_reconstructed = _statistics(reconstructed)
rms_residual, ptv_residual = _statistics(residual)

fig, axs = plt.subplots(2, 2, figsize=(FIGURE_WIDTH, 5.4))
fig.subplots_adjust(hspace=0.35)

image = axs[0, 0].imshow(measured, cmap=CMAP_WAVEFRONT, vmin=-limit, vmax=limit)
axs[0, 0].set_title(
    f"Measured, RWMC — {eff_rwmc[index] * 100:.1f} %\n"
    f"{rms_measured:.3f} $\\lambda$ RMS, {ptv_measured:.3f} $\\lambda$ PtV",
    fontsize=8,
)
fig.colorbar(image, ax=axs[0, 0], fraction=0.046, pad=0.04).set_label(
    "Waves", fontsize=7
)

image = axs[0, 1].imshow(reconstructed, cmap=CMAP_WAVEFRONT, vmin=-limit, vmax=limit)
axs[0, 1].set_title(
    f"Reconstructed, ZWR — {eff_zwr[index] * 100:.1f} %\n"
    f"{rms_reconstructed:.3f} $\\lambda$ RMS, "
    f"{ptv_reconstructed:.3f} $\\lambda$ PtV",
    fontsize=8,
)
fig.colorbar(image, ax=axs[0, 1], fraction=0.046, pad=0.04).set_label(
    "Waves", fontsize=7
)

image = axs[1, 0].imshow(
    residual, cmap=CMAP_WAVEFRONT, vmin=-limit_residual, vmax=limit_residual
)
axs[1, 0].set_title(
    f"Difference, truncation — {relative_residual:.0f} % of the measured RMS\n"
    f"{rms_residual:.3f} $\\lambda$ RMS, {ptv_residual:.3f} $\\lambda$ PtV",
    fontsize=8,
)
fig.colorbar(image, ax=axs[1, 0], fraction=0.046, pad=0.04).set_label(
    "Waves", fontsize=7
)

# The interference pattern is a map of light, not of phase, so it gets a
# grey scale: that is what a detector would record, and it keeps the panel
# visually apart from the three wavefront maps around it.
interference = 0.5 * (1 + np.cos(2 * np.pi * measured))
image = axs[1, 1].imshow(interference, cmap=CMAP_INTERFERENCE, vmin=0, vmax=1)
axs[1, 1].set_title(
    f"Interference pattern, measured\n"
    f"mean {np.nanmean(interference) * 100:.1f} % = RWMC efficiency",
    fontsize=8,
)
fig.colorbar(image, ax=axs[1, 1], fraction=0.046, pad=0.04).set_label(
    "Intensity", fontsize=7
)

for ax in axs.flat:
    ax.axis("off")
fig.suptitle(
    f"Differential wavefront at t = {time_ms[index]:.2f} ms, piston removed",
    fontsize=9,
)

written.append(save_figure(fig, config.figures_dir / "efficiency_models_wavefront.pdf"))


# =============================================================================
# %% MANUSCRIPT FIGURE — Efficiency over time, and the error of each model
# =============================================================================
#
# The figure meant for the manuscript. Top row: the three methods against
# time, over the whole acquisition and over the window around the shot, on
# a shared vertical scale so the two panels can be read against each other.
# Bottom: the departure of each model from the reference, which is the
# number a reader needs to decide whether a model is good enough.
#
# Swap the two top panels by exchanging the two entries of the tuple below.

fig = plt.figure(figsize=(FIGURE_WIDTH, 6.0))
gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 0.16, 1.0], hspace=0.45, wspace=0.08)

ax_zoom = fig.add_subplot(gs[0, 0])
ax_full = fig.add_subplot(gs[0, 1], sharey=ax_zoom)
ax_legend = fig.add_subplot(gs[1, :])
ax_error = fig.add_subplot(gs[2, :])
ax_legend.axis("off")

for ax, window, heading in (
    (ax_zoom, SUMMARY_ZOOM_MS, "Around the shot"),
    (ax_full, None, "Whole acquisition"),
):
    ax.plot(time_ms, eff_rwmc, color=COLOR_RWMC, linewidth=1.0, alpha=0.9)
    ax.plot(time_ms, eff_zwr, color=COLOR_ZWR, linewidth=1.0, alpha=0.9)
    ax.plot(time_ms, eff_zaa, color=COLOR_ZAA, linewidth=1.0, alpha=0.9)

    if window is not None:
        ax.set_xlim(*window)
    ax.set_ylim(*SUMMARY_YLIM)
    ax.set_xlabel("Time (ms)")
    ax.set_title(heading, fontsize=8)

    # The shot is the origin of the time base, so the flash is at zero.
    ax.axvline(0.0, color=FLASH_LINE_COLOR, linestyle=":", linewidth=1.1, zorder=4)

ax_zoom.set_ylabel("Combining efficiency")
ax_full.tick_params(labelleft=False)

# Pump fluorescence behind the curves, if the intensity traces exist for
# this acquisition. They live under the main RESULTS directory, not under
# the test one, and this shot may simply not have them. line=False because
# the panels already draw their own line at the shot.
fluorescence_drawn = False
if FLUORESCENCE_SHOT:
    try:
        for ax in (ax_zoom, ax_full):
            peak = add_flash_background(
                ax, config.root, [FLUORESCENCE_SHOT], "A", line=False
            )
            fluorescence_drawn = fluorescence_drawn or np.isfinite(peak)
    except Exception as error:
        print(
            f"  [info] pas de fluorescence pour {FLUORESCENCE_SHOT} "
            f"({error}). Le fond n'est pas ombré."
        )
    if not fluorescence_drawn:
        print(
            f"  [info] aucune trace d'intensité pour {FLUORESCENCE_SHOT}, "
            f"le fond n'est pas ombré."
        )

# One legend for both panels, since they show the same three curves.
legend_handles = [
    Line2D([0], [0], color=COLOR_RWMC, linewidth=1.4),
    Line2D([0], [0], color=COLOR_ZWR, linewidth=1.4),
    Line2D([0], [0], color=COLOR_ZAA, linewidth=1.4),
    Line2D([0], [0], color=FLASH_LINE_COLOR, linestyle=":", linewidth=1.1),
]
legend_texts = [
    "RWMC - raw wavefront maps",
    "ZWR - Zernike reconstruction",
    "ZAA - analytic approximation",
    "Flash",
]
if fluorescence_drawn:
    legend_handles.append(Patch(facecolor=FLASH_COLOR, alpha=0.35, linewidth=0))
    legend_texts.append("Pump fluorescence")

# Below the two panels rather than inside one of them: the curves fill the
# upper part of the frame, and a legend placed there would cover the very
# region the figure is about.
ax_legend.legend(
    legend_handles,
    legend_texts,
    loc="center",
    ncol=2,
    fontsize=6.5,
    frameon=True,
    columnspacing=1.2,
    handlelength=1.8,
)

# Bottom: relative departure of each model from the reference.
ax_error.scatter(
    eff_rwmc,
    100 * (eff_zwr - eff_rwmc) / eff_rwmc,
    color=COLOR_ZWR,
    marker="o",
    alpha=0.3,
    s=4,
    linewidths=0,
    label="ZWR",
)
ax_error.scatter(
    eff_rwmc,
    100 * (eff_zaa - eff_rwmc) / eff_rwmc,
    color=COLOR_ZAA,
    marker="o",
    alpha=0.3,
    s=4,
    linewidths=0,
    label="ZAA",
)
# The zero line is the reference itself, so it wears the colour of RWMC
# rather than a neutral one: a point on it is a model that agrees with the
# green curve above.
ax_error.axhline(0.0, color=COLOR_RWMC, linewidth=1.2)
ax_error.set_xlabel("RWMC efficiency")
ax_error.set_ylabel("Relative error (%)")
ax_error.set_title(
    "Difference from the ZWR and ZAA models to the reference", fontsize=8
)
ax_error.legend(fontsize=6.5)

written.append(
    save_figure(fig, config.figures_dir / "manuscript_efficiency_overview.pdf")
)


# =============================================================================
# %% SHOW
# =============================================================================
#
# The figures are only displayed here, at the end. Run the cells one by one
# to see each of them as it is built; running the whole file shows them
# together once the last one is done.

print()
print("Figures written:")
for path in written:
    print(f"  {path}")

plt.show()
