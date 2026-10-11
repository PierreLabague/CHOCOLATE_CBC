"""Zernike conventions, normalisation changes and wavefront reconstruction.

Three normalisations of the same basis are used in this work. They differ
only by a per-mode scaling factor, but mixing them silently is the easiest
way to be wrong by a factor of two or four, so the convention in use is
always carried explicitly alongside the data.

=========  ==========================================  ====================
Label      Definition                                  Also known as
=========  ==========================================  ====================
``amp``    bare polynomial, ``max|Z| = 1``              SID4 "PtV mode" (1)
``rms``    Noll / OSA-ANSI orthonormal basis            SID4 "RMS mode" (0)
``ptv``    scaled to the true peak-to-valley of a mode  optical specification
=========  ==========================================  ====================

The SID4 software only exposes the first two. Its ``Mode (0=RMS,1=PtV)``
column takes the value 0 or 1, which maps onto ``rms`` and ``amp``
respectively. The ``ptv`` convention is never produced by the instrument
and only ever appears as a conversion target.

The conversion chain is ``rms -> amp -> ptv``, each step being a
multiplication of the coefficient by a per-mode factor listed in
:data:`RMS_TO_AMP` and :data:`AMP_TO_PTV`. Going the other way divides by
the same factors.

Tilt in milliradians
--------------------
The SID4 export also contains the two columns ``Tilt X (mrad)`` and
``Tilt Y (mrad)``. These are **angles**, not Zernike coefficients, and they
are deliberately left untouched by :func:`convert_zernike_mode`. When the
tilt needs to be compared across representations, the Zernike coefficient
is brought back to milliradians with :func:`zernike_tilt_to_mrad`, never
the opposite: the angle is the directly measured quantity and does not
depend on any normalisation convention.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

__all__ = [
    "ZERNIKE_MODES",
    "RMS_TO_AMP",
    "AMP_TO_PTV",
    "MODE_CODES",
    "PRETTY_LABELS",
    "format_zernike_mode",
    "convert_zernike_mode",
    "zernike_tilt_to_mrad",
    "zernike_wavefront",
    "zernike_wavefront_on",
]

# ---------------------------------------------------------------------------
# Mode ordering.
#
# This is the order in which the SID4 exports the projection, and therefore
# the order of the columns of every array in this package once the
# non-Zernike columns have been dropped. It runs from piston up to radial
# order 4, that is 15 modes.
# ---------------------------------------------------------------------------
ZERNIKE_MODES = (
    "Piston",
    "Tilt X",
    "Tilt Y",
    "Defocus",
    "Astig 0_deg",
    "Astig 45_deg",
    "Coma X",
    "Coma Y",
    "Trefoil 0_deg",
    "Trefoil 30_deg",
    "Spherical Aberr.",
    "Second. Astig 0_deg",
    "Second. Astig 45_deg",
    "Tetrafoil 0_deg",
    "Tetrafoil 25.5_deg",
)

# ---------------------------------------------------------------------------
# Conversion factors.
#
# Multiply a coefficient by RMS_TO_AMP[mode] to go from the orthonormal
# convention to the bare polynomial one; the polynomial itself is divided by
# the same factor, so the wavefront is of course unchanged.
#
# The two "(mrad)" columns are intentionally absent from these tables: they
# hold angles and must never be rescaled by a Zernike normalisation factor.
# ---------------------------------------------------------------------------
RMS_TO_AMP: Dict[str, float] = {
    "Piston": 1.0,
    "Tilt X": 2.0,
    "Tilt Y": 2.0,
    "Defocus": np.sqrt(3),
    "Astig 0_deg": np.sqrt(6),
    "Astig 45_deg": np.sqrt(6),
    "Coma X": np.sqrt(8),
    "Coma Y": np.sqrt(8),
    "Trefoil 0_deg": np.sqrt(8),
    "Trefoil 30_deg": np.sqrt(8),
    "Spherical Aberr.": np.sqrt(5),
    "Second. Astig 0_deg": np.sqrt(10),
    "Second. Astig 45_deg": np.sqrt(10),
    "Tetrafoil 0_deg": np.sqrt(10),
    "Tetrafoil 25.5_deg": np.sqrt(10),
}

# Peak-to-valley excursion of each bare polynomial over the unit disc.
# Spherical aberration is the only mode whose excursion is not 2: the
# polynomial 6r^4 - 6r^2 + 1 ranges from -0.5 at r = 1/sqrt(2) to +1 at the
# centre and the edge, hence 1.5.
AMP_TO_PTV: Dict[str, float] = {
    "Piston": 1.0,
    "Tilt X": 2.0,
    "Tilt Y": 2.0,
    "Defocus": 2.0,
    "Astig 0_deg": 2.0,
    "Astig 45_deg": 2.0,
    "Coma X": 2.0,
    "Coma Y": 2.0,
    "Trefoil 0_deg": 2.0,
    "Trefoil 30_deg": 2.0,
    "Spherical Aberr.": 1.5,
    "Second. Astig 0_deg": 2.0,
    "Second. Astig 45_deg": 2.0,
    "Tetrafoil 0_deg": 2.0,
    "Tetrafoil 25.5_deg": 2.0,
}

# Numerical code stored in the "Mode" column of the SID4 export.
MODE_CODES: Dict[str, int] = {"rms": 0, "amp": 1, "ptv": 2}

# Labels used on figures. The vendor string "Tetrafoil 25.5_deg" is kept as
# the dictionary key because that is what the XML contains, but the physical
# angle of the second tetrafoil term is 22.5 degrees.
PRETTY_LABELS: Dict[str, str] = {
    "Piston": "Piston",
    "Tilt X": "Tilt X",
    "Tilt Y": "Tilt Y",
    "Defocus": "Defocus",
    "Astig 0_deg": "Astigmatism 0°",
    "Astig 45_deg": "Astigmatism 45°",
    "Coma X": "Coma X",
    "Coma Y": "Coma Y",
    "Trefoil 0_deg": "Trefoil 0°",
    "Trefoil 30_deg": "Trefoil 30°",
    "Spherical Aberr.": "Spherical aberration",
    "Second. Astig 0_deg": "Secondary astigmatism 0°",
    "Second. Astig 45_deg": "Secondary astigmatism 45°",
    "Tetrafoil 0_deg": "Tetrafoil 0°",
    "Tetrafoil 25.5_deg": "Tetrafoil 22.5°",
}


def format_zernike_mode(mode: str) -> str:
    """Return the label of a normalisation, for use in axis labels.

    Parameters
    ----------
    mode : {'rms', 'amp', 'ptv'}

    Returns
    -------
    str
        ``'RMS'``, ``'Amplitude'`` or ``'PtV'``.
    """
    labels = {"rms": "RMS", "amp": "Amplitude", "ptv": "PtV"}
    try:
        return labels[mode]
    except KeyError:
        raise ValueError(
            f"Unknown Zernike normalisation {mode!r}, expected one of "
            f"{sorted(labels)}."
        ) from None


def _conversion_factors(current: int, target: int) -> Dict[str, float]:
    """Build the per-mode multiplicative factors from one convention to another.

    Parameters
    ----------
    current, target : int
        Numerical codes, see :data:`MODE_CODES`.
    """
    if current == target:
        return {name: 1.0 for name in RMS_TO_AMP}

    # Factor taking a coefficient from the reference convention 'rms' to
    # each of the three conventions.
    to_amp = RMS_TO_AMP
    to_ptv = {name: RMS_TO_AMP[name] * AMP_TO_PTV[name] for name in RMS_TO_AMP}
    from_rms = {
        0: {name: 1.0 for name in RMS_TO_AMP},
        1: to_amp,
        2: to_ptv,
    }
    # Composing "current -> rms" with "rms -> target" gives the factor.
    return {
        name: from_rms[target][name] / from_rms[current][name] for name in RMS_TO_AMP
    }


def convert_zernike_mode(
    data: np.ndarray,
    tag: Sequence[str],
    mode: str,
    copy: bool = True,
) -> np.ndarray:
    """Convert Zernike coefficients to the requested normalisation.

    The current normalisation is read from the ``Mode`` column of the data
    itself, so calling this function twice with the same target is harmless:
    the second call is a no-op.

    Parameters
    ----------
    data : numpy.ndarray, shape (n_frames, n_columns)
        Block of exported values. Columns are described by ``tag``.
    tag : sequence of str
        Column labels, as read from the SID4 XML export.
    mode : {'rms', 'amp', 'ptv'}
        Target normalisation.
    copy : bool, optional
        If True (default) the input array is left untouched and a converted
        copy is returned. Set to False to convert in place, which saves
        memory on large blocks but modifies the caller's array.

    Returns
    -------
    numpy.ndarray
        The converted block. Its ``Mode`` column is updated to the code of
        the target normalisation.

    Raises
    ------
    ValueError
        If ``mode`` is not a known normalisation, if the ``Mode`` column is
        missing, or if the value it holds is not a known code.

    Notes
    -----
    The ``Tilt X (mrad)`` and ``Tilt Y (mrad)`` columns are angles and are
    deliberately left unchanged. Use :func:`zernike_tilt_to_mrad` to compare
    a Zernike tilt coefficient with them.
    """
    if mode not in MODE_CODES:
        raise ValueError(
            f"Unknown Zernike normalisation {mode!r}, expected one of "
            f"{sorted(MODE_CODES)}."
        )
    target_mode = MODE_CODES[mode]

    tag = list(tag)
    mode_idx = next((i for i, name in enumerate(tag) if "Mode" in name), None)
    if mode_idx is None:
        raise ValueError(
            "No 'Mode' column found in the tag list. The SID4 export should "
            "contain a column labelled 'Mode (0=RMS,1=PtV)'."
        )

    # The normalisation is assumed homogeneous over the whole block, which
    # it always is: it is a property of the export, not of the frame.
    current_mode = int(data[0, mode_idx])
    if current_mode not in MODE_CODES.values():
        raise ValueError(
            f"The 'Mode' column holds the unexpected value {current_mode}, "
            f"expected one of {sorted(MODE_CODES.values())}."
        )

    out = data.copy() if copy else data

    if current_mode != target_mode:
        factors = _conversion_factors(current_mode, target_mode)
        for i, name in enumerate(tag):
            if name in factors:
                out[:, i] *= factors[name]

    out[:, mode_idx] = target_mode
    return out


def zernike_tilt_to_mrad(
    coefficient: np.ndarray,
    mode: str,
    coef_tilt_mrad_to_ptv: float,
) -> np.ndarray:
    """Express a Zernike tilt coefficient as an angle in milliradians.

    The conversion is deliberately one-way. The angle measured by the sensor
    is the reference quantity, independent of any Zernike normalisation, so
    comparisons are always made in milliradians.

    Parameters
    ----------
    coefficient : numpy.ndarray
        Tilt X or Tilt Y coefficient, in waves, expressed in ``mode``.
    mode : {'rms', 'amp', 'ptv'}
        Normalisation of ``coefficient``.
    coef_tilt_mrad_to_ptv : float
        Geometric factor of the setup: peak-to-valley optical path
        difference in waves produced across the pupil by a tilt of one
        milliradian. It depends on the pupil diameter and the wavelength.

    Returns
    -------
    numpy.ndarray
        The same tilt, in milliradians.
    """
    if mode not in MODE_CODES:
        raise ValueError(
            f"Unknown Zernike normalisation {mode!r}, expected one of "
            f"{sorted(MODE_CODES)}."
        )
    if not np.isfinite(coef_tilt_mrad_to_ptv) or coef_tilt_mrad_to_ptv == 0:
        raise ValueError(
            "coef_tilt_mrad_to_ptv must be a non-zero finite number, got "
            f"{coef_tilt_mrad_to_ptv!r}."
        )

    # Bring the coefficient to the ptv convention first, then divide by the
    # geometric factor.
    to_ptv = _conversion_factors(MODE_CODES[mode], MODE_CODES["ptv"])["Tilt X"]
    return coefficient * to_ptv / coef_tilt_mrad_to_ptv


# ---------------------------------------------------------------------------
# Wavefront reconstruction.
# ---------------------------------------------------------------------------
# Bare polynomials, in the order of ZERNIKE_MODES. Each has max|Z| = 1 over
# the unit disc, which is the 'amp' convention: coefficients passed to
# zernike_wavefront must therefore be expressed in that convention.
_POLYNOMIALS = (
    lambda r, t: np.ones_like(r),  # Piston
    lambda r, t: r * np.cos(t),  # Tilt X
    lambda r, t: r * np.sin(t),  # Tilt Y
    lambda r, t: 2 * r**2 - 1,  # Defocus
    lambda r, t: r**2 * np.cos(2 * t),  # Astigmatism 0
    lambda r, t: r**2 * np.sin(2 * t),  # Astigmatism 45
    lambda r, t: (3 * r**2 - 2) * r * np.cos(t),  # Coma X
    lambda r, t: (3 * r**2 - 2) * r * np.sin(t),  # Coma Y
    lambda r, t: r**3 * np.cos(3 * t),  # Trefoil 0
    lambda r, t: r**3 * np.sin(3 * t),  # Trefoil 30
    lambda r, t: 6 * r**4 - 6 * r**2 + 1,  # Spherical aberration
    lambda r, t: (4 * r**2 - 3) * r**2 * np.cos(2 * t),  # Second. astig 0
    lambda r, t: (4 * r**2 - 3) * r**2 * np.sin(2 * t),  # Second. astig 45
    lambda r, t: r**4 * np.cos(4 * t),  # Tetrafoil 0
    lambda r, t: r**4 * np.sin(4 * t),  # Tetrafoil 22.5
)


def zernike_wavefront_on(coefficients, rho, theta):
    """Evaluate a Zernike expansion on a grid you provide.

    :func:`zernike_wavefront` samples the polynomials on its own square
    grid, with the pupil exactly inscribed. That is the wrong thing when
    the map has to be compared with a measurement pixel by pixel: a real
    analysis mask has stepped edges and is not exactly inscribed in its
    export, so a reconstruction built on an ideal disc never lines up with
    it, and the difference of the two is then dominated by the mismatch of
    their supports rather than by the truncation.

    Passing the polar coordinates of the measured grid removes that
    problem: the expansion is evaluated exactly where the measurement
    exists, and masking the result with the measurement's own mask leaves
    the two on identical supports.

    Parameters
    ----------
    coefficients : sequence of float, length 15
        Coefficients in the order of :data:`ZERNIKE_MODES`, **in the 'amp'
        convention**, as for :func:`zernike_wavefront`.
    rho : numpy.ndarray
        Radial coordinate, normalised so that the pupil edge is at 1.
    theta : numpy.ndarray
        Azimuth, in radians, same shape as ``rho``.

    Returns
    -------
    numpy.ndarray
        Wavefront in waves, same shape as ``rho``, with ``nan`` where
        ``rho`` exceeds 1.

    Raises
    ------
    ValueError
        If the number of coefficients is not 15, or if the two coordinate
        arrays do not have the same shape.
    """
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape[-1] != len(_POLYNOMIALS):
        raise ValueError(
            f"Expected {len(_POLYNOMIALS)} coefficients (piston to radial "
            f"order 4), got {coefficients.shape[-1]}."
        )

    rho = np.asarray(rho, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if rho.shape != theta.shape:
        raise ValueError(
            f"rho and theta must have the same shape, got {rho.shape} and "
            f"{theta.shape}."
        )

    wavefront = np.zeros_like(rho)
    for coefficient, polynomial in zip(coefficients, _POLYNOMIALS):
        wavefront += coefficient * polynomial(rho, theta)

    wavefront[rho > 1.0] = np.nan
    return wavefront


def zernike_wavefront(coefficients: Sequence[float], resolution: int = 100):
    """Reconstruct a wavefront map from its Zernike coefficients.

    Parameters
    ----------
    coefficients : sequence of float, length 15
        Coefficients in the order of :data:`ZERNIKE_MODES`, from piston up
        to radial order 4, **expressed in the 'amp' convention**. Passing
        coefficients in another normalisation silently distorts the map;
        convert them first with :func:`convert_zernike_mode`.
    resolution : int, optional
        Number of samples along each axis of the square grid enclosing the
        pupil.

    Returns
    -------
    numpy.ndarray, shape (resolution, resolution)
        Wavefront in waves, with ``nan`` outside the unit disc.

    Raises
    ------
    ValueError
        If the number of coefficients is not 15.

    Notes
    -----
    The grid follows the mathematical convention: row 0 holds y = -1 and
    the last row y = +1, so y increases with the row index. Displayed with
    matplotlib's default ``origin='upper'``, which puts row 0 at the top,
    the map therefore appears **flipped vertically** with respect to a
    camera image, whose row 0 really is the top of the sensor.

    Pass ``origin='lower'`` to ``imshow``, or flip the array with
    ``[::-1]``, when the map has to be shown next to an image. Nothing
    computed from a pupil average is affected: a mirror leaves the mean of
    the interference pattern, and hence the combining efficiency,
    unchanged.
    """
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape[-1] != len(_POLYNOMIALS):
        raise ValueError(
            f"Expected {len(_POLYNOMIALS)} coefficients (piston to radial "
            f"order 4), got {coefficients.shape[-1]}."
        )

    axis = np.linspace(-1.0, 1.0, resolution)
    x, y = np.meshgrid(axis, axis)
    rho = np.hypot(x, y)
    theta = np.arctan2(y, x)

    wavefront = np.zeros_like(rho)
    for coefficient, polynomial in zip(coefficients, _POLYNOMIALS):
        wavefront += coefficient * polynomial(rho, theta)

    wavefront[rho > 1.0] = np.nan
    return wavefront
