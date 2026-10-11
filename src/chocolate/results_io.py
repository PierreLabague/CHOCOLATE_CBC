"""Reading the condensed ``RESULTS`` files produced by the partitioning stage.

For each measurement ``YYYY-MM-DD-HH-MM`` the partitioning stage writes a
directory of small ``.npy`` arrays. The ones used by the wavefront analysis
are, for each beam letter ``A`` or ``B``:

``<id>_data__W<letter>.npy``
    Array of shape ``(n_frames, n_columns)``. The columns are the values
    exported by the SID4 for each frame, described by the tag array.
``<id>_tag__W<letter>.npy``
    Array of ``n_columns`` labels, such as ``'Defocus'`` or
    ``'Mode (0=RMS,1=PtV)'``.
``<id>_time__W<letter>.npy``
    Time of each frame, in seconds, relative to the laser shot.

Loading these is enough to reproduce every figure of the interpretation
stage; the raw interferograms are never needed again.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import interp1d

from .database import list_measurement_ids, load_database
from .zernike import convert_zernike_mode

__all__ = [
    "ZERNIKE_COLUMNS",
    "load_wavefront_results",
    "load_piston_results",
    "interpolate_on_common_time",
    "load_beam_pair",
    "mean_fluorescence",
]

#: Column indices of the 14 Zernike coefficients in the exported arrays,
#: from Tilt X to the second tetrafoil. Piston, index 11, is excluded: it is
#: not measurable by the wavefront sensor in absolute terms and is provided
#: by the interferometric diagnostic instead. Indices 0 to 10 hold geometry
#: and quality figures, not coefficients.
ZERNIKE_COLUMNS = tuple(range(12, 26))


def load_wavefront_results(
    root,
    beam: str,
    ids: Optional[Iterable[str]] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load the condensed wavefront arrays of one beam.

    Parameters
    ----------
    root : str or pathlib.Path
        Data directory, the one containing ``RESULTS`` and
        ``data_base.json``.
    beam : {'A', 'B'}
        Which beam to read.
    ids : iterable of str, optional
        Identifiers to load. Defaults to every measurement of the database.
        Identifiers with no file on disk are skipped.
    verbose : bool, optional
        Report skipped measurements.

    Returns
    -------
    dict
        ``{shot_id: {'data': array, 'tag': array, 'time': array}}``, in
        chronological order.

    Raises
    ------
    FileNotFoundError
        If the ``RESULTS`` directory itself does not exist.
    """
    root = Path(root)
    results_root = root / "RESULTS"
    if not results_root.is_dir():
        raise FileNotFoundError(f"RESULTS directory not found: {results_root}")

    if ids is None:
        ids = list_measurement_ids(load_database(root))
    ids = list(ids)

    output: Dict[str, Dict[str, np.ndarray]] = {}
    missing: List[str] = []

    for shot_id in ids:
        folder = results_root / shot_id
        files = {
            key: folder / f"{shot_id}_{key}__W{beam}.npy"
            for key in ("data", "tag", "time")
        }
        if not all(path.is_file() for path in files.values()):
            missing.append(shot_id)
            continue
        try:
            # allow_pickle is required: the tag arrays hold Python strings.
            output[shot_id] = {
                key: np.load(path, allow_pickle=True) for key, path in files.items()
            }
        except Exception as error:  # corrupted or truncated file
            missing.append(shot_id)
            if verbose:
                print(f"  [error] {shot_id} beam {beam}: {error}")

    if verbose and missing:
        print(
            f"  [info] beam {beam}: {len(missing)} of {len(ids)} measurements "
            f"have no readable file and were skipped."
        )
    return output


def interpolate_on_common_time(
    results: Dict[str, Dict[str, np.ndarray]],
    columns: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample several shots onto a single shared time base.

    Each acquisition has its own frame rate and its own trigger delay, so
    the shots must be brought onto a common grid before they can be
    averaged or compared. The grid spans the overlap of all the shots and
    is sampled at the finest step among them.

    Parameters
    ----------
    results : dict
        As returned by :func:`load_wavefront_results`.
    columns : sequence of int
        Column indices to extract, typically :data:`ZERNIKE_COLUMNS`.

    Returns
    -------
    data : numpy.ndarray, shape (n_shots, n_columns, n_times)
        Interpolated values, shots in the order of ``results``.
    common_time : numpy.ndarray, shape (n_times,)
        Shared time base, in seconds.

    Raises
    ------
    ValueError
        If ``results`` is empty, or if the shots have no time overlap.
    """
    if not results:
        raise ValueError(
            "No measurement to interpolate: the selection is empty or none "
            "of the selected shots has readable files."
        )

    times = [results[shot_id]["time"] for shot_id in results]

    t_start = max(float(t.min()) for t in times)
    t_stop = min(float(t.max()) for t in times)
    if t_stop <= t_start:
        raise ValueError(
            "The selected shots do not overlap in time, so no common time "
            "base exists. Narrow down the selection."
        )
    step = min(float(np.min(np.diff(t))) for t in times)
    common_time = np.arange(t_start, t_stop + step / 2, step)

    data = np.empty((len(results), len(columns), common_time.size))
    for i, shot_id in enumerate(results):
        time = results[shot_id]["time"]
        values = results[shot_id]["data"]
        for j, column in enumerate(columns):
            interpolate = interp1d(
                time, values[:, column], kind="linear", fill_value="extrapolate"
            )
            data[i, j] = interpolate(common_time)

    return data, common_time


def load_piston_results(
    root,
    ids: Optional[Iterable[str]] = None,
    unwrap: bool = True,
    verbose: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load the piston and beam intensity arrays of a set of measurements.

    These come from the interferometric branch of the partitioning stage,
    and share a single time base for both beams, unlike the wavefront
    arrays.

    Parameters
    ----------
    root : str or pathlib.Path
        Data directory.
    ids : iterable of str, optional
        Measurements to load, defaults to the whole database.
    unwrap : bool, optional
        Unwrap the piston and express it in waves. The partitioning stage
        stores it wrapped, in radians.
    verbose : bool, optional
        Report skipped measurements.

    Returns
    -------
    dict
        ``{shot_id: {'time': array, 'piston': array or None,
        'IA': array or None, 'IB': array or None}}``. A measurement is
        returned as soon as its time base exists, even if the piston is
        missing: the shots taken before the fringes were set up have
        intensities but no interferometric signal.
    """
    root = Path(root)
    results_root = root / "RESULTS"
    if not results_root.is_dir():
        raise FileNotFoundError(f"RESULTS directory not found: {results_root}")

    if ids is None:
        ids = list_measurement_ids(load_database(root))
    ids = list(ids)

    output: Dict[str, Dict[str, np.ndarray]] = {}
    missing: List[str] = []

    for shot_id in ids:
        folder = results_root / shot_id
        time_file = folder / f"{shot_id}_time.npy"
        if not time_file.is_file():
            missing.append(shot_id)
            continue

        entry: Dict[str, np.ndarray] = {}
        try:
            entry["time"] = np.load(time_file, allow_pickle=True)
        except Exception as error:
            missing.append(shot_id)
            if verbose:
                print(f"  [error] {shot_id}: {error}")
            continue

        for key, filename in (
            ("piston", f"{shot_id}_piston.npy"),
            ("IA", f"{shot_id}_IA.npy"),
            ("IB", f"{shot_id}_IB.npy"),
        ):
            path = folder / filename
            try:
                entry[key] = (
                    np.load(path, allow_pickle=True) if path.is_file() else None
                )
            except Exception:
                entry[key] = None

        if unwrap and entry.get("piston") is not None:
            # Stored wrapped, in radians; unwrap and express in waves so
            # that it is homogeneous with the Zernike coefficients.
            entry["piston"] = np.unwrap(entry["piston"]) / (2 * np.pi)

        output[shot_id] = entry

    if verbose and missing:
        print(
            f"  [info] piston: {len(missing)} of {len(ids)} measurements have "
            f"no time base and were skipped."
        )
    return output


def load_beam_pair(
    root,
    ids: Optional[Iterable[str]] = None,
    zernike_mode: str = "rms",
    columns: Sequence[int] = ZERNIKE_COLUMNS,
    verbose: bool = True,
):
    """Load both beams, convert them and put them on a common time base.

    This is the usual entry point of an analysis script.

    Parameters
    ----------
    root : str or pathlib.Path
        Data directory.
    ids : iterable of str, optional
        Measurements to load. Defaults to the whole database.
    zernike_mode : {'rms', 'amp', 'ptv'}, optional
        Normalisation the coefficients are converted to.
    columns : sequence of int, optional
        Columns to keep, defaults to the 14 Zernike coefficients.
    verbose : bool, optional
        Report progress and skipped measurements.

    Returns
    -------
    data_a, data_b : numpy.ndarray, shape (n_shots, n_columns, n_times)
        Coefficients of each beam.
    common_time : numpy.ndarray, shape (n_times,)
        Shared time base, in seconds.
    labels : numpy.ndarray of str, shape (n_columns,)
        Name of each retained column.
    shot_ids : list of str
        Measurements actually loaded, in the order of the first axis.

    Notes
    -----
    Only the measurements for which **both** beams are available are
    returned, since every downstream quantity is a difference between the
    two.
    """
    results_a = load_wavefront_results(root, "A", ids, verbose=verbose)
    results_b = load_wavefront_results(root, "B", ids, verbose=verbose)

    common_ids = sorted(set(results_a) & set(results_b))
    if not common_ids:
        raise ValueError(
            "No measurement has both beam A and beam B available. Check the "
            "selection and the contents of the RESULTS directory."
        )
    if verbose:
        print(f"  [info] {len(common_ids)} measurements with both beams.")

    results_a = {shot_id: results_a[shot_id] for shot_id in common_ids}
    results_b = {shot_id: results_b[shot_id] for shot_id in common_ids}

    labels = np.asarray(results_a[common_ids[0]]["tag"])

    for results in (results_a, results_b):
        for shot_id, entry in results.items():
            entry["data"] = convert_zernike_mode(
                entry["data"], entry["tag"], mode=zernike_mode
            )

    data_a, common_time = interpolate_on_common_time(results_a, columns)
    data_b, _ = interpolate_on_common_time(results_b, columns)

    return data_a, data_b, common_time, labels[list(columns)], common_ids


def mean_fluorescence(root, ids, beam: str = "A"):
    """Average pump fluorescence of a set of shots, on a common time base.

    The intensity arrays ``IA`` and ``IB`` are extracted by the
    partitioning stage from the raw interferograms; they follow the
    Nd:glass flash-lamp fluorescence and are used as a visual time
    reference on the figures.

    Parameters
    ----------
    root : str or pathlib.Path
        Data directory.
    ids : iterable of str
        Measurements to average.
    beam : {'A', 'B'}, optional
        Which of the two intensity arrays to read.

    Returns
    -------
    time_ms, signal : numpy.ndarray or None
        Time base in milliseconds and averaged signal, or ``(None, None)``
        when none of the measurements carries an intensity array or when
        they do not overlap in time.
    """
    results = load_piston_results(root, ids, unwrap=False, verbose=False)
    traces = [
        (entry["time"], entry[f"I{beam}"])
        for entry in results.values()
        if entry.get(f"I{beam}") is not None and entry.get("time") is not None
    ]
    if not traces:
        return None, None

    start = max(float(np.min(time)) for time, _ in traces)
    stop = min(float(np.max(time)) for time, _ in traces)
    if stop <= start:
        return None, None

    grid = np.linspace(start, stop, 2000)
    stack = [
        np.interp(grid, np.asarray(time, dtype=float), np.asarray(signal, dtype=float))
        for time, signal in traces
    ]
    return grid * 1e3, np.nanmean(np.array(stack), axis=0)
