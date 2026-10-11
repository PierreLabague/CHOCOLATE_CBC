"""Partitioning stage: raw acquisitions to condensed ``.npy`` results.

This is stage C of the processing pipeline. It walks the raw acquisitions
of a measurement and writes a handful of small arrays into
``RESULTS/<measure_id>/``, so that every later analysis runs from a few
hundred kilobytes instead of a gigabyte per shot.

Three entry points, one per kind of acquisition:

:func:`process_phasics_measure`
    Gathers the SID4 XML exports of one beam and stacks the Zernike
    decomposition into ``data__W<letter>.npy``, with its column labels and
    its time base.
:func:`process_piston_measure`
    Reads the raw interferograms, extracts the beam intensities and the
    fringe phase, and writes the piston and intensity arrays. Requires the
    interferometric frame format.
:func:`process_measure_IA_IB`
    Intensities only, for the acquisitions recorded without fringes.

:func:`process_database` runs any of them over a selection of
measurements.

Every function logs to ``RESULTS/RESULTS.txt`` and returns an empty array
rather than raising, so that a single unreadable shot does not interrupt a
run of a hundred.

Directory layout
----------------
The raw data are expected under ``<root>/<raw_dirname>/``, with
``raw_dirname`` defaulting to ``"PHASICS"``::

    <root>/<raw_dirname>/<measure_id>_PHASICS/
        Test1/<camera_folder>/Img000000.tif      raw frames
        SID4/<measure_id>_W<letter>/Results/     ACC Img000000.xml

Pass ``raw_dirname="P"`` (or any other name) if the directory was renamed
on your storage.
"""

from __future__ import annotations

import json
import os
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from PIL import Image

__all__ = [
    "FRAME_WAVEFRONT_ONLY",
    "FRAME_INTERFEROMETRIC",
    "FrameGeometry",
    "frame_geometry",
    "load_database",
    "select_entries",
    "parse_phasics_xml",
    "process_phasics_measure",
    "process_piston_measure",
    "process_measure_IA_IB",
    "process_database",
    "decimate_vector",
]


# ---------------------------------------------------------------------------
# Frame formats.
#
# The campaign produced acquisitions in two frame sizes, and the size is
# what tells them apart:
#
#   384 x 184  the two beams only, no interference fringes. Wavefront
#              measurement alone; only the beam intensities can be
#              extracted from these frames.
#   384 x 232  the two beams in the upper part and, below a short gap, the
#              fringe band that carries the optical path difference. This
#              is the only format from which the piston can be read.
#
# The shapes below follow the numpy convention, (height, width), which is
# the transpose of how the camera software labels them.
#
# In both formats the two beams sit side by side, each occupying half the
# width. In the interferometric format the beams stop at row 190 and the
# fringe band runs from row 200 to the bottom; the top-left corner falls
# outside the circular beam footprint and measures the background. The
# shorter format is cropped tight on the beams, so it has no dark corner
# and no background is subtracted.
# ---------------------------------------------------------------------------
FRAME_WAVEFRONT_ONLY = (184, 384)
FRAME_INTERFEROMETRIC = (232, 384)

KNOWN_FRAME_FORMATS = {
    FRAME_WAVEFRONT_ONLY: {
        "name": "384x184, faisceaux seuls",
        "beam_last_row": 184,
        "fringe_first_row": None,
        "noise_size": 0,
    },
    FRAME_INTERFEROMETRIC: {
        "name": "384x232, faisceaux + franges",
        "beam_last_row": 190,
        "fringe_first_row": 200,
        "noise_size": 20,
    },
}

#: Half-width, in frames, of the zoomed panel of the quick-look figure.
QUICKLOOK_ZOOM = 200


class FrameGeometry(NamedTuple):
    """Where each region sits in a raw frame.

    Built once per measurement by :func:`frame_geometry`, from the shape of
    the first frame, then reused for every frame of that measurement.

    Attributes
    ----------
    height, width : int
        Frame shape, numpy convention.
    beam_last_row : int
        Rows ``0`` to ``beam_last_row`` hold the two beam images.
    split_column : int
        Boundary between beam A and beam B, at half the width.
    fringe_first_row : int or None
        First row of the fringe band, or None for a format without one.
    noise_size : int
        Side of the top-left background estimation corner. Zero means no
        background subtraction.
    name : str
        Human-readable format name.
    recognised : bool
        False when the frame shape is neither of the two known formats.
    """

    height: int
    width: int
    beam_last_row: int
    split_column: int
    fringe_first_row: Optional[int]
    noise_size: int
    name: str
    recognised: bool

    @property
    def beam_pixels(self) -> int:
        """Number of pixels integrated for one beam."""
        return self.beam_last_row * self.split_column

    @property
    def has_fringes(self) -> bool:
        """True when the frame contains a fringe band."""
        return self.fringe_first_row is not None

    @property
    def subtract_background(self) -> bool:
        """True when a dark corner is available to measure the background."""
        return self.noise_size > 0

    def describe(self) -> str:
        """One-line summary, written to the log so the split can be checked."""
        fringes = (
            f"franges lignes {self.fringe_first_row}-{self.height}"
            if self.has_fringes
            else "pas de franges"
        )
        return (
            f"{self.name} — faisceaux lignes 0-{self.beam_last_row}, "
            f"séparation colonne {self.split_column}, {fringes}"
        )


def frame_geometry(
    shape,
    beam_last_row: Optional[int] = None,
    fringe_first_row: Optional[int] = None,
    noise_size: Optional[int] = None,
) -> FrameGeometry:
    """Work out the region layout of a frame from its shape.

    Recognises the two formats of the campaign and returns their documented
    layout. Any other shape is handled conservatively: the beams are still
    split at half the width, but no fringe band is assumed and the geometry
    is flagged as unrecognised so that the caller can warn.

    Parameters
    ----------
    shape : tuple of int
        ``(height, width)`` of the frame.
    beam_last_row, fringe_first_row, noise_size : int, optional
        Override the values of the recognised format. Use these for an
        acquisition whose camera region of interest differs from the two
        standard ones; passing ``fringe_first_row`` is what lets a
        non-standard frame be processed for piston.

    Returns
    -------
    FrameGeometry

    Raises
    ------
    ValueError
        If the frame is too small to hold two beams.

    Examples
    --------
    >>> frame_geometry(FRAME_INTERFEROMETRIC).has_fringes
    True
    >>> frame_geometry(FRAME_WAVEFRONT_ONLY).has_fringes
    False
    """
    height, width = int(shape[0]), int(shape[1])
    if height < 2 or width < 2:
        raise ValueError(f"Image trop petite pour contenir deux faisceaux : {shape}")

    known = KNOWN_FRAME_FORMATS.get((height, width))
    if known is not None:
        defaults = dict(known)
        recognised = True
    else:
        # Unknown format: keep the half-width split, which holds whatever
        # the size, and refuse to guess where a fringe band might start.
        defaults = {
            "name": f"format inconnu {width}x{height}",
            "beam_last_row": height,
            "fringe_first_row": None,
            "noise_size": 0,
        }
        recognised = False

    if beam_last_row is not None:
        defaults["beam_last_row"] = beam_last_row
    if fringe_first_row is not None:
        defaults["fringe_first_row"] = fringe_first_row
    if noise_size is not None:
        defaults["noise_size"] = noise_size

    resolved_fringe = defaults["fringe_first_row"]
    if resolved_fringe is not None and resolved_fringe >= height:
        resolved_fringe = None

    limit = height if resolved_fringe is None else resolved_fringe
    return FrameGeometry(
        height=height,
        width=width,
        beam_last_row=min(defaults["beam_last_row"], limit),
        split_column=width // 2,
        fringe_first_row=resolved_fringe,
        noise_size=min(defaults["noise_size"], height, width),
        name=defaults["name"],
        recognised=recognised,
    )


def _read_frame_shape(path) -> tuple:
    """Return the shape of one frame without keeping the image in memory."""
    with Image.open(path) as img:
        return np.array(img, dtype=np.float32).shape


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _append_log(results_dir: Path, measure_id: str, level: str, msg: str) -> None:
    """Append one line to ``RESULTS/RESULTS.txt``.

    The log is the trace of what happened during a long batch run: which
    shots were skipped, which files were unreadable, which frame counts
    disagreed with the database. A failure to write the log is reported but
    never raised, so that logging cannot break processing.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "RESULTS.txt"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] [{measure_id}] [{level}] {msg}\n")
    except Exception:
        print(f"[ERREUR-LOG] Impossible d'écrire dans {log_path}")


def _warn(results_dir: Path, measure_id: str, msg: str) -> None:
    """Report a recoverable problem to both the console and the log."""
    print(f"[AVERTISSEMENT] {msg}")
    _append_log(results_dir, measure_id, "WARN", msg)


def _error(results_dir: Path, measure_id: str, msg: str) -> None:
    """Report a failure to both the console and the log."""
    print(f"[ERREUR] {msg}")
    _append_log(results_dir, measure_id, "ERROR", msg)


def _info(results_dir: Path, measure_id: str, msg: str) -> None:
    """Report progress to both the console and the log."""
    print(f"[INFO] {msg}")
    _append_log(results_dir, measure_id, "INFO", msg)


def _maybe_import_tqdm():
    """Return the tqdm callable, or None when the package is absent.

    tqdm only adds a progress bar, so its absence must never stop a run.
    """
    try:
        from tqdm.auto import tqdm

        return tqdm
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def decimate_vector(vector, max_length: int = 400):
    """Subsample a vector so that it holds at most ``max_length`` points.

    Used for the quick-look figures only: plotting eight thousand points
    produces a heavy PNG that shows nothing more than four hundred.
    """
    if len(vector) <= max_length:
        return vector
    step = len(vector) // max_length
    return vector[::step]


def _time_vector(database: dict, measure_id: str, n_expected: int = None):
    """Build the time base of a measurement from its metadata.

    The camera records frames ``cine_beginning`` to ``cine_end`` around the
    trigger, so the time of a frame follows from the frame rate.

    Parameters
    ----------
    database : dict
    measure_id : str
    n_expected : int, optional
        Number of samples to generate. Defaults to the frame count implied
        by the metadata.

    Returns
    -------
    time : numpy.ndarray
        Time of each frame, in seconds, negative before the shot.
    n_frames : int
        Frame count implied by the metadata, for comparison with what was
        actually read.
    """
    phasics = database[measure_id]["PHASICS"]
    first, last = phasics["cine_beginning"], phasics["cine_end"]
    frame_rate = phasics["frame_rate"]

    n_frames = last - first + 1
    size = n_frames if n_expected is None else n_expected
    return np.linspace(first / frame_rate, last / frame_rate, size), n_frames


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def load_database(root: str) -> dict:
    """Read ``data_base.json`` from the data directory.

    Raises
    ------
    FileNotFoundError
        If the file is absent.
    """
    db_path = os.path.join(root, "data_base.json")
    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"data_base.json introuvable dans {root}")

    with open(db_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def select_entries(
    database: dict,
    mode: str = "all",
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
) -> List[str]:
    """Select the measurement identifiers to process.

    Parameters
    ----------
    database : dict
    mode : {'all', 'include', 'exclude'}
        ``'include'`` keeps only the listed identifiers, ``'exclude'``
        keeps everything but those.
    include, exclude : list of str, optional
        Required by the corresponding mode.

    Raises
    ------
    ValueError
        On an unknown mode, or when the list the mode needs is missing.
    """
    all_ids = [k for k in database.keys() if k != "__meta"]

    if mode == "all":
        return all_ids

    if mode == "include":
        if not include:
            raise ValueError("mode='include' nécessite une liste include=[]")
        return [x for x in include if x in all_ids]

    if mode == "exclude":
        if not exclude:
            raise ValueError("mode='exclude' nécessite une liste exclude=[]")
        return [x for x in all_ids if x not in exclude]

    raise ValueError("mode doit être 'all', 'include' ou 'exclude'.")


# ---------------------------------------------------------------------------
# SID4 XML parsing
# ---------------------------------------------------------------------------
def parse_phasics_xml(xml_path: str) -> Dict[str, float]:
    """Read one SID4 export and return its ``{label: value}`` mapping.

    Each ``ACC ImgXXXXXX.xml`` holds the analysis of a single frame: pupil
    geometry, Strehl ratio, the normalisation flag and the Zernike
    coefficients.

    Returns
    -------
    dict
        Values are floats where they parse as such, and left as strings
        otherwise, which happens for the empty ``Ellipse`` entry.

    Notes
    -----
    The SID4 writes decimals with a comma, as ``'3,500E+0'``. They are
    converted here, so nothing downstream has to care about the locale.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    data_dict = {}
    for node in root.findall("Data"):
        label = node.findtext("Label")
        value = node.findtext("Value")
        if label is None or value is None:
            continue

        value = value.replace(",", ".")
        try:
            value = float(value)
        except ValueError:
            pass

        data_dict[label] = value

    return data_dict


# ---------------------------------------------------------------------------
# Wavefront: XML exports to condensed arrays
# ---------------------------------------------------------------------------
def process_phasics_measure(
    root: str,
    measure_id: str,
    wavefront_letter: str,
    *,
    raw_dirname: str = "PHASICS",
    use_tqdm: bool = True,
) -> list:
    """Stack the SID4 exports of one beam into condensed arrays.

    Writes into ``RESULTS/<measure_id>/``:

    ``<id>_data__W<letter>.npy``
        Shape ``(n_frames, n_columns)``, one row per frame.
    ``<id>_tag__W<letter>.npy``
        The ``n_columns`` labels, as objects.
    ``<id>_time__W<letter>.npy``
        Time of each frame, in seconds.
    ``<id>_Zernike__W<letter>.png``
        Quick-look figure of the coefficients.

    Parameters
    ----------
    root : str
        Data directory.
    measure_id : str
        Identifier ``YYYY-MM-DD-HH-MM``.
    wavefront_letter : {'A', 'B'}
        Which beam to process.
    raw_dirname : str, optional
        Name of the directory holding the raw acquisitions.
    use_tqdm : bool, optional
        Show a progress bar when tqdm is available.

    Returns
    -------
    list
        Always empty. The function communicates through the files it writes
        and through the log; the return value is kept only for
        compatibility with existing callers.

    Notes
    -----
    This stage reads the XML exports, not the images, so the frame format
    is irrelevant here: it applies equally to both acquisition types.
    """
    results_dir = Path(root) / "RESULTS"

    try:
        base = (
            Path(root)
            / raw_dirname
            / f"{measure_id}_PHASICS"
            / "SID4"
            / f"{measure_id}_W{wavefront_letter}"
            / "Results"
        )

        save_base = results_dir / measure_id
        save_base.mkdir(parents=True, exist_ok=True)

        if not base.exists():
            _warn(
                results_dir,
                measure_id,
                f"Dossier PHASICS introuvable pour {measure_id} ({base})",
            )
            return []

        xml_files = sorted(
            f
            for f in base.iterdir()
            if f.name.startswith("ACC Img") and f.name.endswith(".xml")
        )
        if not xml_files:
            _warn(
                results_dir,
                measure_id,
                f"Aucun fichier XML 'ACC Img*.xml' trouvé dans {base}",
            )
            return []

        _info(
            results_dir,
            measure_id,
            f"{measure_id} — {len(xml_files)} XML détectés. "
            f"Traitement PHASICS W{wavefront_letter}…",
        )

        # -- Progress bar, optional --------------------------------------
        iterator = xml_files
        if use_tqdm:
            tqdm = _maybe_import_tqdm()
            if tqdm is not None:
                iterator = tqdm(
                    xml_files,
                    desc=f"PHASICS {measure_id} W{wavefront_letter}",
                    unit="xml",
                )
            else:
                _info(
                    results_dir,
                    measure_id,
                    "tqdm non installé — traitement sans barre de progression",
                )

        # -- Parse every frame -------------------------------------------
        results = []
        for f in iterator:
            img_number = f.name.replace("ACC Img", "").replace(".xml", "").strip()
            try:
                results.append([img_number, parse_phasics_xml(str(f))])
            except Exception as error:
                _error(results_dir, measure_id, f"Impossible de lire {f} : {error}")

        if not results:
            _error(
                results_dir,
                measure_id,
                f"Aucun XML valide n'a pu être analysé pour {measure_id} "
                f"(W{wavefront_letter})",
            )
            return []

        # -- Build the data matrix ---------------------------------------
        # The first frame sets the column order; a frame missing a key
        # leaves a nan in that column rather than shifting the row.
        tag = list(results[0][1].keys())
        data = np.full((len(results), len(tag)), np.nan, dtype=float)

        for i, (img_number, values) in enumerate(results):
            for j, key in enumerate(tag):
                value = values.get(key, "")
                if value != "":
                    try:
                        data[i, j] = float(value)
                    except Exception:
                        _warn(
                            results_dir,
                            measure_id,
                            f"Valeur non convertible '{key}' dans "
                            f"{img_number} : {value}",
                        )
                elif key not in values:
                    _warn(
                        results_dir,
                        measure_id,
                        f"Clé '{key}' manquante dans {img_number}",
                    )

        np.save(save_base / f"{measure_id}_data__W{wavefront_letter}.npy", data)
        np.save(
            save_base / f"{measure_id}_tag__W{wavefront_letter}.npy",
            np.array(tag, dtype=object),
        )

        # -- Time base ----------------------------------------------------
        database = load_database(root)
        time_vec, n_expected = _time_vector(database, measure_id)

        # A mismatch here usually means the export was interrupted, or the
        # database holds the wrong cine range. The data are kept either
        # way, but the shot deserves a look.
        if len(data) != n_expected:
            _warn(
                results_dir,
                measure_id,
                f"Nombre d'images attendu ({n_expected}) différent du nombre "
                f"lu ({len(data)}) pour {measure_id} W{wavefront_letter}",
            )

        np.save(save_base / f"{measure_id}_time__W{wavefront_letter}.npy", time_vec)

        # -- Quick-look figure -------------------------------------------
        # Columns 0 to 3 hold the pupil geometry, not coefficients, hence
        # the offset of four.
        max_points = 500
        step = max(1, len(time_vec) // max_points)
        t_dec = time_vec[::step][:max_points]
        d_dec = data[::step][:max_points]

        fig_path = save_base / f"{measure_id}_Zernike__W{wavefront_letter}.png"

        plt.figure(figsize=(10, 4))
        if len(tag) > 4:
            for i in range(len(tag) - 4):
                # Referenced to the first frame, so the figure shows the
                # evolution rather than the absolute level.
                plt.plot(
                    t_dec * 1000, d_dec[:, i + 4] - d_dec[0, i + 4], label=tag[i + 4]
                )
            plt.legend()
        else:
            _warn(
                results_dir,
                measure_id,
                f"Trop peu de tags pour tracer les Zernike (len(tag)={len(tag)})",
            )

        plt.xlabel("Time (ms)")
        plt.ylabel("Zernike coefficients (waves)")
        plt.title(f"PHASICS {measure_id} - W{wavefront_letter}")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()

        _info(
            results_dir,
            measure_id,
            f"{measure_id} — Traitement PHASICS terminé (W{wavefront_letter}). "
            f"Figure : {fig_path.name}",
        )
        return []

    except Exception as error:
        _error(
            results_dir,
            measure_id,
            f"Erreur inattendue pendant process_phasics_measure : {error}\n"
            f"{traceback.format_exc()}",
        )
        return []


# ---------------------------------------------------------------------------
# Interferograms: intensities and fringe phase
# ---------------------------------------------------------------------------
def process_single_file(f, geometry: FrameGeometry):
    """Extract from one frame the two beam intensities and the fringe band.

    Parameters
    ----------
    f : pathlib.Path
        A raw ``ImgXXXXXX.tif``.
    geometry : FrameGeometry
        Region layout, built once per measurement.

    Returns
    -------
    IA, IB : float
        Integrated intensity of each beam, background removed when the
        format provides a dark corner. They follow the Nd:glass flash-lamp
        fluorescence and give the residual amplification at 1064 nm.
    fringes : numpy.ndarray or None
        The fringe band, or None for a format that has none.

    Raises
    ------
    ValueError
        If the frame does not have the shape the geometry was built for.
        Sizes differ between acquisitions but never within one, so a
        mismatch means the folder mixes two measurements.
    """
    with Image.open(f) as img:
        picture = np.array(img, dtype=np.float32)

    if picture.shape[:2] != (geometry.height, geometry.width):
        raise ValueError(
            f"Taille {picture.shape} pour {f.name}, attendu "
            f"{(geometry.height, geometry.width)} d'après la première image"
        )

    # The top-left corner falls outside the circular beam footprint in the
    # interferometric format, so it measures the background. The shorter
    # format is cropped tight on the beams and has no such corner.
    offset = 0.0
    if geometry.subtract_background:
        size = geometry.noise_size
        offset = np.mean(picture[:size, :size]) * geometry.beam_pixels

    rows = slice(0, geometry.beam_last_row)
    split = geometry.split_column

    intensity_a = np.sum(picture[rows, :split]) - offset
    intensity_b = np.sum(picture[rows, split : 2 * split]) - offset

    fringes = picture[geometry.fringe_first_row :, :] if geometry.has_fringes else None
    return intensity_a, intensity_b, fringes


def process_piston_measure(
    root: str,
    measure_id: str,
    *,
    raw_dirname: str = "PHASICS",
    n_jobs: int = -1,
    verbose: int = 10,
    use_tqdm: bool = True,
    save_line: bool = False,
    beam_last_row: Optional[int] = None,
    fringe_first_row: Optional[int] = None,
) -> np.ndarray:
    """Extract the piston and the beam intensities of one measurement.

    Each frame is reduced to three numbers and one line: the two beam
    intensities, and the vertically collapsed fringe band. Stacking those
    lines and taking their Fourier transform gives the fringe phase, hence
    the optical path difference between the two arms.

    Only the interferometric format, 384 x 232, carries fringes. A
    measurement recorded in the 384 x 184 format is reported and skipped;
    use :func:`process_measure_IA_IB` for those.

    Writes into ``RESULTS/<measure_id>/``:

    ``<id>_piston.npy``
        Wrapped phase, in radians.
    ``<id>_IA.npy``, ``<id>_IB.npy``
        Beam intensities.
    ``<id>_time.npy``
        Time base, in seconds.
    ``<id>_argmax.npy``
        Fringe frequency bin of each frame, see the notes.
    ``<id>_Ifrange.npy``
        Integrated intensity of the fringe band.
    ``<id>_line.npy``
        Only when ``save_line`` is set, see below.
    ``<id>_Piston_Fluo.png``
        Quick-look figure.

    Parameters
    ----------
    root : str
        Data directory.
    measure_id : str
        Identifier ``YYYY-MM-DD-HH-MM``.
    raw_dirname : str, optional
        Name of the directory holding the raw acquisitions.
    n_jobs : int, optional
        Number of parallel workers, ``-1`` for all cores. Only used when
        the run falls back to joblib.
    verbose : int, optional
        joblib verbosity; 10 or above prints progress.
    use_tqdm : bool, optional
        Process sequentially with a progress bar. Falls back to the
        parallel path when tqdm is missing.
    save_line : bool, optional
        Also save the zero-padded, windowed line matrix that feeds the
        Fourier transform, as ``<id>_line.npy``. Off by default because
        the array is large: shape ``(n_frames, 3 * width)`` in float32, so
        roughly 37 MB for eight thousand frames of 384 pixels, against
        about 130 kB for all the other outputs together. Turn it on to
        re-derive the phase with another estimator, or to inspect the
        fringe contrast frame by frame, without going back to the raw
        TIFFs.
    beam_last_row, fringe_first_row : int, optional
        Override the layout of the recognised format. Needed only for an
        acquisition whose camera region of interest differs from the two
        standard ones.

    Returns
    -------
    numpy.ndarray
        The wrapped phase, in radians. Empty when the measurement could not
        be processed; the reason is in the log.

    Notes
    -----
    The phase is read at a **fixed** frequency bin, the one of the first
    frame, rather than at each frame's own maximum. This is deliberate: it
    is a lock-in at the nominal fringe frequency, and it keeps the phase
    continuous. Letting the bin follow the maximum would make the phase
    jump whenever the peak moves by one bin, which is a sampling artefact
    rather than a physical shift. The per-frame maxima are still saved in
    ``<id>_argmax.npy``, where they serve as an indicator: a bin that
    drifts during the shot flags a significant tilt.
    """
    results_dir = Path(root) / "RESULTS"

    try:
        # -- Locate the raw frames ---------------------------------------
        base = Path(root) / raw_dirname / f"{measure_id}_PHASICS" / "Test1"
        if not base.exists():
            _warn(
                results_dir,
                measure_id,
                f"Dossier PHASICS introuvable pour {measure_id} ({base})",
            )
            return np.array([])

        # Test1 holds a single camera folder, named after the camera and
        # the acquisition number.
        try:
            subfolder = next(base.iterdir())
        except StopIteration:
            _warn(results_dir, measure_id, f"Aucun sous-dossier trouvé dans {base}")
            return np.array([])

        save_base = results_dir / measure_id
        save_base.mkdir(parents=True, exist_ok=True)

        files = sorted(
            f
            for f in subfolder.iterdir()
            if f.name.startswith("Img") and f.name.endswith(".tif")
        )
        if not files:
            _warn(
                results_dir, measure_id, f"Aucun fichier .tif trouvé dans {subfolder}"
            )
            return np.array([])

        n_files = len(files)

        # -- Frame format -------------------------------------------------
        geometry = frame_geometry(
            _read_frame_shape(files[0]),
            beam_last_row=beam_last_row,
            fringe_first_row=fringe_first_row,
        )
        _info(
            results_dir,
            measure_id,
            f"{measure_id} — {n_files} images détectées. {geometry.describe()}",
        )

        if not geometry.recognised:
            _warn(
                results_dir,
                measure_id,
                f"Format d'image non reconnu pour {measure_id} "
                f"({geometry.width}x{geometry.height}). Les formats connus sont "
                f"384x184 (sans franges) et 384x232 (avec franges). Passez "
                f"fringe_first_row=... pour traiter ce tir malgré tout.",
            )

        if not geometry.has_fringes:
            _warn(
                results_dir,
                measure_id,
                f"Les images de {measure_id} sont au format "
                f"{geometry.width}x{geometry.height} et ne contiennent pas de "
                f"bande de franges : utilisez process_measure_IA_IB "
                f"(function='INTENSITY') pour ce tir.",
            )
            return np.array([])

        # -- Read every frame --------------------------------------------
        # Two paths for the same result: sequential with a progress bar, or
        # parallel through joblib. The parallel one is faster on a
        # workstation, the sequential one is easier to follow.
        def _sequential_with_tqdm():
            tqdm = _maybe_import_tqdm()
            if tqdm is None:
                print(
                    "[INFO] tqdm n'est pas installé. Retour au mode parallèle joblib."
                )
                return None

            intensity_a = np.zeros(n_files, dtype=np.float32)
            intensity_b = np.zeros(n_files, dtype=np.float32)
            fringes = np.zeros(
                (n_files, geometry.height - geometry.fringe_first_row, geometry.width),
                dtype=np.float32,
            )

            for i, f in enumerate(
                tqdm(files, desc=f"Traitement {measure_id}", unit="img")
            ):
                intensity_a[i], intensity_b[i], fringes[i, :, :] = process_single_file(
                    f, geometry
                )

            return intensity_a, intensity_b, fringes

        sequential = _sequential_with_tqdm() if use_tqdm else None

        if sequential is not None:
            IA, IB, img_cropped = sequential
        else:
            results = Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(process_single_file)(f, geometry) for f in files
            )
            IA_list, IB_list, img_list = zip(*results)
            IA = np.array(IA_list, dtype=np.float32)
            IB = np.array(IB_list, dtype=np.float32)
            img_cropped = np.stack(img_list).astype(np.float32)

        # -- Collapse the fringe band into one line per frame -------------
        I_frange = np.sum(img_cropped, axis=(1, 2))

        # Hamming apodisation, normalised to unit RMS so that the window
        # does not change the amplitude scale of the transform.
        window = np.hamming(geometry.width)
        window /= np.sqrt(np.mean(window**2))

        # Zero-padding by a factor of three: the signal sits in the middle
        # third, which interpolates the spectrum and refines the frequency
        # estimate without adding information.
        line = np.zeros((n_files, geometry.width * 3), dtype=np.float32)
        line[:, geometry.width : 2 * geometry.width] = (
            np.sum(img_cropped, axis=1) * window
        )

        if save_line:
            line_path = save_base / f"{measure_id}_line.npy"
            np.save(line_path, line)
            _info(
                results_dir,
                measure_id,
                f"Matrice 'line' sauvegardée ({line.nbytes / 1e6:.1f} Mo) : "
                f"{line_path.name}",
            )

        # -- Fringe phase --------------------------------------------------
        # ifftshift then fftshift keeps the centre of the padded line as the
        # phase origin, so the extracted phase is that of the fringes and
        # not of an arbitrary edge.
        tf_line = np.fft.fftshift(
            np.fft.fft(np.fft.ifftshift(line, axes=1), axis=1), axes=1
        )
        tf_line = tf_line[:, : tf_line.shape[1] // 2]

        # Per-frame position of the fringe peak, saved as a drift
        # indicator. The last ten bins are excluded, they are edge
        # artefacts of the padded spectrum.
        arg_max = np.argmax(np.abs(tf_line[:, :-10]), axis=1)

        # Fixed-bin lock-in: every frame is read at the frequency of the
        # first one. See the notes in the docstring.
        phase_complex = tf_line[np.arange(n_files), arg_max[0]]
        phase = np.angle(phase_complex)

        # -- Time base -----------------------------------------------------
        database = load_database(root)
        time_vec, n_expected = _time_vector(database, measure_id)

        if len(tf_line) != n_expected:
            _warn(
                results_dir,
                measure_id,
                f"Nombre d'images attendu ({n_expected}) différent du nombre "
                f"d'images lues ({len(tf_line)}) pour {measure_id}",
            )

        # -- Quick-look figure ---------------------------------------------
        phase_decimated = decimate_vector(phase)
        phase_unwrapped_decimated = decimate_vector(np.unwrap(phase))
        time_decimated = decimate_vector(time_vec)

        # Zoom centred on the trigger. Taking the index of the sample
        # closest to t = 0 keeps the window centred even when the cine
        # range is not symmetric.
        index_zero = int(np.argmin(np.abs(time_vec)))
        zoom = slice(
            max(0, index_zero - QUICKLOOK_ZOOM),
            min(len(time_vec), index_zero + QUICKLOOK_ZOOM),
        )

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.plot(
            time_decimated * 1000,
            phase_decimated / (2 * np.pi),
            label="Phase (wrapped)",
        )
        ax1.plot(
            time_decimated * 1000,
            phase_unwrapped_decimated / (2 * np.pi),
            label="Phase (unwrapped)",
        )
        ax1.set_ylabel("Optical path difference (waves)")
        ax1.set_xlabel("Time (ms)")
        ax1.set_title(f"Piston, full window — {measure_id}")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(
            time_vec[zoom] * 1000, phase[zoom] / (2 * np.pi), label="Phase (wrapped)"
        )
        ax2.plot(
            time_vec[zoom] * 1000,
            np.unwrap(phase[zoom]) / (2 * np.pi),
            label="Phase (unwrapped)",
        )
        ax2.set_ylabel("Optical path difference (waves)")
        ax2.set_xlabel("Time (ms)")
        ax2.set_title(f"Piston, zoom on the flash — {measure_id}")
        ax2.grid(True)

        ax3 = ax2.twinx()
        ax3.semilogy(
            time_vec[zoom] * 1000, IA[zoom], "g", label="Intensity A", alpha=0.4
        )
        ax3.semilogy(
            time_vec[zoom] * 1000, IB[zoom], "r", label="Intensity B", alpha=0.4
        )
        ax3.set_ylabel("Intensity (counts)")
        ax2.legend(loc="center left")
        ax3.legend(loc="center right")

        plt.tight_layout()
        fig_path = save_base / f"{measure_id}_Piston_Fluo.png"
        plt.savefig(fig_path)
        plt.close()

        # -- Save ------------------------------------------------------------
        np.save(save_base / f"{measure_id}_piston.npy", phase)
        np.save(save_base / f"{measure_id}_IA.npy", IA)
        np.save(save_base / f"{measure_id}_IB.npy", IB)
        np.save(save_base / f"{measure_id}_time.npy", time_vec)
        np.save(save_base / f"{measure_id}_argmax.npy", arg_max)
        np.save(save_base / f"{measure_id}_Ifrange.npy", I_frange)

        _info(
            results_dir,
            measure_id,
            f"{measure_id} — Traitement terminé. Figure : {fig_path.name}",
        )
        return phase

    except Exception as error:
        _error(
            results_dir,
            measure_id,
            f"Exception pendant le traitement : {error}\n{traceback.format_exc()}",
        )
        return np.array([])


#: Kept so that existing scripts importing the misspelled name still work.
process_pistin_measure = process_piston_measure


# ---------------------------------------------------------------------------
# Intensities only, for the acquisitions recorded without fringes
# ---------------------------------------------------------------------------
def process_single_file_IA_IB(f, geometry: FrameGeometry) -> Tuple[float, float]:
    """Integrate the two halves of a frame.

    Parameters
    ----------
    f : pathlib.Path
    geometry : FrameGeometry
        Built once per measurement, so the split follows the actual frame
        width instead of a fixed column.

    Returns
    -------
    IA, IB : float

    Raises
    ------
    ValueError
        If the frame does not have the shape the geometry was built for.

    Notes
    -----
    No background subtraction, unlike :func:`process_single_file`: the
    384 x 184 format is cropped tight on the beams and offers no dark
    corner to measure it from.
    """
    with Image.open(f) as img:
        picture = np.array(img, dtype=np.float32)

    if picture.shape[:2] != (geometry.height, geometry.width):
        raise ValueError(
            f"Taille {picture.shape} pour {f.name}, attendu "
            f"{(geometry.height, geometry.width)} d'après la première image"
        )

    split = geometry.split_column
    return np.sum(picture[:, :split]), np.sum(picture[:, split : 2 * split])


def process_measure_IA_IB(
    root: str,
    measure_id: str,
    *,
    raw_dirname: str = "PHASICS",
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the beam intensities of a measurement.

    Intended for the 384 x 184 acquisitions, which carry no fringes. It
    also runs on the interferometric format, but the fringe band is then
    integrated along with the beams, which is why
    :func:`process_piston_measure` is the right entry point for those.

    Writes ``<id>_IA.npy``, ``<id>_IB.npy``, ``<id>_time.npy`` and a
    quick-look ``<id>_IA_IB.png``.

    Parameters
    ----------
    root : str
        Data directory.
    measure_id : str
    raw_dirname : str, optional

    Returns
    -------
    IA, IB : numpy.ndarray
        Intensities of each beam, with ``nan`` for the frames that could
        not be read. Empty arrays when the measurement failed entirely.

    Notes
    -----
    Deliberately sequential: these acquisitions are short, and the
    per-frame error handling is easier to follow without workers.
    """
    results_dir = Path(root) / "RESULTS"

    try:
        base = Path(root) / raw_dirname / f"{measure_id}_PHASICS" / "Test1"
        if not base.exists():
            _warn(results_dir, measure_id, f"Dossier PHASICS introuvable : {base}")
            return np.array([]), np.array([])

        try:
            subfolder = next(base.iterdir())
        except StopIteration:
            _warn(results_dir, measure_id, f"Aucun sous-dossier dans {base}")
            return np.array([]), np.array([])

        files = sorted(
            f
            for f in subfolder.iterdir()
            if f.name.startswith("Img") and f.name.endswith(".tif")
        )
        if not files:
            _warn(results_dir, measure_id, f"Aucun fichier .tif dans {subfolder}")
            return np.array([]), np.array([])

        geometry = frame_geometry(_read_frame_shape(files[0]))
        _info(
            results_dir,
            measure_id,
            f"{measure_id} — {len(files)} images détectées. {geometry.describe()}",
        )

        if not geometry.recognised:
            _warn(
                results_dir,
                measure_id,
                f"Format d'image non reconnu pour {measure_id} "
                f"({geometry.width}x{geometry.height}). Les intensités sont "
                f"quand même extraites en séparant l'image en deux moitiés.",
            )
        elif geometry.has_fringes:
            _warn(
                results_dir,
                measure_id,
                f"{measure_id} est au format {geometry.width}x{geometry.height} "
                f"et contient des franges : process_piston_measure "
                f"(function='PISTON_INTENSITY') en extrairait aussi le piston.",
            )

        IA = np.zeros(len(files), dtype=np.float32)
        IB = np.zeros(len(files), dtype=np.float32)
        n_failed = 0

        for i, f in enumerate(files):
            try:
                IA[i], IB[i] = process_single_file_IA_IB(f, geometry)
            except Exception as error:
                n_failed += 1
                # Only the first few are logged individually: a systematic
                # problem would otherwise fill the log with one line per
                # frame and hide everything else.
                if n_failed <= 5:
                    _error(
                        results_dir, measure_id, f"Erreur fichier {f.name} : {error}"
                    )
                IA[i] = np.nan
                IB[i] = np.nan

        if n_failed:
            _warn(
                results_dir,
                measure_id,
                f"{n_failed} image(s) sur {len(files)} illisibles pour "
                f"{measure_id}, remplacées par nan",
            )
        if n_failed == len(files):
            _error(
                results_dir,
                measure_id,
                f"Aucune image exploitable pour {measure_id} : les fichiers "
                f"sont tous illisibles ou de taille incohérente. Rien n'est "
                f"sauvegardé.",
            )
            return np.array([]), np.array([])

        database = load_database(root)
        time_vec, _ = _time_vector(database, measure_id, n_expected=len(IA))

        save_base = results_dir / measure_id
        save_base.mkdir(parents=True, exist_ok=True)

        np.save(save_base / f"{measure_id}_IA.npy", IA)
        np.save(save_base / f"{measure_id}_IB.npy", IB)
        np.save(save_base / f"{measure_id}_time.npy", time_vec)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.semilogy(time_vec * 1000, IA, label="IA", alpha=0.8)
        ax.semilogy(time_vec * 1000, IB, label="IB", alpha=0.8)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Intensity (counts)")
        ax.set_title(f"IA / IB — {measure_id}")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        fig_path = save_base / f"{measure_id}_IA_IB.png"
        plt.savefig(fig_path)
        plt.close()

        _info(
            results_dir,
            measure_id,
            f"{measure_id} — IA / IB extraits et sauvegardés. "
            f"Figure : {fig_path.name}",
        )
        return IA, IB

    except Exception as error:
        _error(results_dir, measure_id, f"{error}\n{traceback.format_exc()}")
        return np.array([]), np.array([])


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------
def process_database(
    root: str,
    function: str,
    wavefront_letter: str = "A",
    mode: str = "all",
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, List]:
    """Run one of the processing functions over a selection of measurements.

    Parameters
    ----------
    root : str
        Data directory.
    function : {'PHASICS', 'PISTON_INTENSITY', 'INTENSITY'}
        ``'PHASICS'`` stacks the SID4 exports of one beam,
        ``'PISTON_INTENSITY'`` extracts the piston and the intensities from
        the 384 x 232 acquisitions, ``'INTENSITY'`` the intensities alone
        from the 384 x 184 ones.
    wavefront_letter : {'A', 'B'}, optional
        Only used by ``'PHASICS'``.
    mode : {'all', 'include', 'exclude'}, optional
    include, exclude : list of str, optional
    **kwargs
        Forwarded to the processing function, which is how ``save_line``
        and ``raw_dirname`` reach it.

    Returns
    -------
    dict
        Always empty. Kept for compatibility with existing callers; the
        results are the files written on disk.

    Raises
    ------
    FileNotFoundError
        If the data directory or ``data_base.json`` is missing.

    Examples
    --------
    Every measurement, beam A::

        process_database(ROOT, "PHASICS", "A")

    One measurement, keeping the line matrix::

        process_database(ROOT, "PISTON_INTENSITY", mode="include",
                         include=["2025-12-03-14-10"], save_line=True)
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dossier racine introuvable : {root}")
    if not os.path.isfile(os.path.join(root, "data_base.json")):
        raise FileNotFoundError(f"data_base.json introuvable dans {root}")

    database = load_database(root)
    selected_ids = select_entries(database, mode, include, exclude)

    handlers = {
        "PHASICS": lambda mid: process_phasics_measure(
            root, mid, wavefront_letter, **kwargs
        ),
        "PISTON_INTENSITY": lambda mid: process_piston_measure(root, mid, **kwargs),
        "INTENSITY": lambda mid: process_measure_IA_IB(root, mid, **kwargs),
    }

    if function not in handlers:
        print(
            f"[ERREUR] Fonction inconnue : {function} "
            f"(attendu : {', '.join(handlers)})"
        )
        return {}

    for measure_id in selected_ids:
        print(f"Traitement {function} pour {measure_id}…")
        handlers[function](measure_id)

    return {}
