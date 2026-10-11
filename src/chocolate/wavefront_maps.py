"""Reading and handling the wavefront maps exported by the SID4.

Alongside the Zernike decomposition, the SID4 can export the reconstructed
wavefront of each frame as a 16-bit TIFF, named ``PHA ImgXXXXXX.tif``.
These maps are the reference against which the Zernike-based efficiency
models are checked: they carry the whole measured wavefront, not its
projection onto fifteen polynomials.

The three helpers here are what that comparison needs: reading a folder of
maps, putting two maps of different pixel size onto a common grid, and
restricting a map to a circular pupil.
"""

from __future__ import annotations

import glob
import os

import numpy as np

__all__ = [
    "read_phasics_maps",
    "pupil_bounding_box",
    "resize_nearest",
    "circular_mask",
]

#: Full scale of the exported TIFF. The maps are stored as 16-bit integers
#: and must be divided by this, then multiplied by the per-frame scale
#: factor of the export, to come back to waves.
TIFF_FULL_SCALE = 2**16


def read_phasics_maps(
    folder, pattern="PHA Img*.tif", verbose=True, progress=True, description=None
):
    """Read a folder of exported wavefront maps into one array.

    Parameters
    ----------
    folder : str or pathlib.Path
        The ``Results`` directory of a SID4 analysis.
    pattern : str, optional
        Filename pattern of the maps.
    verbose : bool, optional
        Report how many files were found.
    progress : bool, optional
        Show a progress bar while reading. A full acquisition is several
        thousand files and takes minutes, so a script that reads two of
        them looks frozen without one. Requires tqdm; ignored if absent.
    description : str, optional
        Label of the progress bar. Defaults to the folder name.

    Returns
    -------
    numpy.ndarray, shape (n_frames, height, width)
        The maps, as float32, in the raw units of the export. Multiply by
        the per-frame scale factor and divide by :data:`TIFF_FULL_SCALE` to
        obtain waves.

    Raises
    ------
    FileNotFoundError
        If the folder does not exist or holds no matching file.

    Notes
    -----
    The files are sorted by their frame number rather than
    alphabetically, so that ``Img10`` follows ``Img9`` instead of
    preceding it.
    """
    folder = str(folder)
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Le dossier '{folder}' n'existe pas.")

    files = sorted(
        glob.glob(os.path.join(folder, pattern)),
        key=lambda name: int(name.split("Img")[1].split(".tif")[0]),
    )
    if not files:
        raise FileNotFoundError(
            f"Aucun fichier ne correspond à '{pattern}' dans '{folder}'."
        )

    try:
        import tifffile
    except ImportError as error:  # pragma: no cover
        raise ImportError(
            "La lecture des cartes de front d'onde nécessite tifffile : "
            "pip install tifffile"
        ) from error

    iterator = files
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(
                files,
                desc=description
                or f"Cartes {os.path.basename(os.path.dirname(folder))}",
                unit="img",
                leave=False,
            )
        except ImportError:
            pass

    maps = []
    unreadable = 0
    for path in iterator:
        try:
            maps.append(tifffile.imread(path).astype(np.float32))
        except Exception as error:
            unreadable += 1
            if unreadable <= 5:
                print(f"[ERREUR] Lecture de {path} : {error}")

    if not maps:
        raise FileNotFoundError(f"Aucune image lisible dans '{folder}'.")

    maps = np.array(maps)
    if verbose:
        print(
            f"  {len(files)} cartes trouvées dans {os.path.basename(folder)}, "
            f"forme {maps.shape}" + (f", {unreadable} illisibles" if unreadable else "")
        )
    return maps


def resize_nearest(image, height, width):
    """Resample a 2-D array onto another pixel grid, nearest neighbour.

    Used to bring the two beams onto a common grid: their pupils do not
    cover the same number of pixels on the sensor, so a pixel-by-pixel
    difference is only meaningful once one has been resampled onto the
    other.

    Parameters
    ----------
    image : numpy.ndarray, shape (h, w) or (h, w, c)
    height, width : int
        Target shape.

    Returns
    -------
    numpy.ndarray

    Notes
    -----
    Nearest neighbour is deliberate: it introduces no value that was not
    measured. A smoother interpolation would invent intermediate wavefront
    values and quietly improve the agreement between the two beams, which
    is exactly the quantity being measured here.
    """
    source_height, source_width = image.shape[:2]
    # Index of the source pixel each target pixel samples.
    rows = np.minimum(
        (np.arange(height) * source_height / height).astype(int), source_height - 1
    )
    columns = np.minimum(
        (np.arange(width) * source_width / width).astype(int), source_width - 1
    )
    return image[np.ix_(rows, columns)] if image.ndim == 2 else image[rows][:, columns]


def circular_mask(image, radius):
    """Set everything outside a centred disc to ``nan``.

    Parameters
    ----------
    image : numpy.ndarray, shape (h, w) or (h, w, c)
    radius : float
        Radius of the disc, in pixels, about the centre of the array.

    Returns
    -------
    numpy.ndarray
        A copy; the input is left untouched.
    """
    height, width = image.shape[:2]
    y, x = np.ogrid[:height, :width]
    inside = (x - width // 2) ** 2 + (y - height // 2) ** 2 <= radius**2

    masked = image.astype(float, copy=True)
    if image.ndim == 2:
        masked[~inside] = np.nan
    else:
        masked[~inside, :] = np.nan
    return masked


def pupil_bounding_box(image, tolerance=0.0):
    """Smallest box containing the analysed pupil of an exported map.

    Pixels outside the analysis mask are exported as exactly zero, so the
    pupil is the bounding box of the non-zero ones. Finding it from the
    data means the crop follows the mask that was actually used, instead of
    repeating coordinates read off a screen once and never checked again.

    Parameters
    ----------
    image : numpy.ndarray, shape (h, w)
        One exported map, before any nan substitution.
    tolerance : float, optional
        Values whose absolute value is at or below this are treated as
        outside the mask. Zero is right for a raw export; raise it if the
        maps have been through an operation that turned exact zeros into
        very small numbers.

    Returns
    -------
    tuple of slice
        ``(rows, columns)``, ready to index the map.

    Raises
    ------
    ValueError
        If the map holds no pixel above the tolerance.
    """
    valid = np.abs(np.nan_to_num(image)) > tolerance
    if not valid.any():
        raise ValueError(
            "Aucun pixel exploitable dans la carte : elle est vide, ou le "
            "seuil est trop haut."
        )

    rows = np.flatnonzero(valid.any(axis=1))
    columns = np.flatnonzero(valid.any(axis=0))
    return (
        slice(int(rows[0]), int(rows[-1]) + 1),
        slice(int(columns[0]), int(columns[-1]) + 1),
    )
