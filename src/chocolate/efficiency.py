"""Coherent beam combining efficiency from differential wavefront data.

Two beams whose wavefronts differ by a residual optical path difference
map :math:`\\Delta(x, y)`, expressed in waves, combine on a beam splitter
with the efficiency

.. math::

    \\eta = \\frac{1}{2}\\left(1 + \\langle \\cos(2\\pi\\Delta) \\rangle\\right)

where the average runs over the pupil. Two ways of evaluating it are
provided, and they answer different questions.

:func:`combining_efficiency_brute_force`
    Reconstructs the differential wavefront map from its Zernike
    coefficients and averages the cosine over the pupil. Exact within the
    truncation of the Zernike expansion, but costly.

:func:`combining_efficiency_small_fluctuations`
    Expands the cosine to second order, which for an orthonormal basis
    reduces the pupil average to a sum of squared coefficients:

    .. math::

        \\eta \\simeq 1 - \\frac{1}{4}\\sum_j (2\\pi c_j)^2

    Cheap, and accurate as long as the residual is a small fraction of a
    wave. This is the model used for the results of the campaign.

:func:`combining_efficiency_marechal`
    Same input, but resums the expansion into an exponential. It agrees
    with the second-order model at small residual and, unlike it, stays
    between zero and one however large the residual grows.

All three take the **differential** coefficients, beam A minus beam B.
Piston is excluded throughout: a uniform phase offset between the two
beams is measured by the interferometric diagnostic, not by the wavefront
sensor, and including it here would either double count it or, worse,
inject the sensor's arbitrary reference into the result.
"""

from __future__ import annotations

import numpy as np

from .zernike import (
    MODE_CODES,
    ZERNIKE_MODES,
    _POLYNOMIALS,
    _conversion_factors,
)

__all__ = [
    "reference_index",
    "zero_at_reference",
    "differential_coefficients",
    "combining_efficiency_small_fluctuations",
    "combining_efficiency_marechal",
    "combining_efficiency_reconstruction",
    "combining_efficiency_brute_force",
    "zernike_analysis",
]


def reference_index(common_time, time_ref_ms):
    """Index of the sample closest to a reference instant.

    Parameters
    ----------
    common_time : numpy.ndarray
        Time base, in seconds.
    time_ref_ms : float
        Reference instant, in milliseconds.
    """
    return int(np.argmin(np.abs(np.asarray(common_time) - time_ref_ms / 1000.0)))


def zero_at_reference(data, common_time, time_ref_ms):
    """Subtract from every trace its own value at a reference instant.

    The wavefront sensor measures aberrations relative to a daily
    reference, so the absolute level of a coefficient carries no meaning on
    its own. What the on-shot figures show is the distortion *added* by the
    amplification, which requires every trace to start from zero at a
    common instant taken before the flash.

    Parameters
    ----------
    data : numpy.ndarray, shape (..., n_times)
        Any array whose last axis is time. Applied independently to each
        shot and each mode.
    common_time : numpy.ndarray, shape (n_times,)
        Time base, in seconds.
    time_ref_ms : float
        Instant at which every trace is set to zero, in milliseconds.
        Negative means before the shot.

    Returns
    -------
    numpy.ndarray
        A new array, the input is left untouched.
    """
    index = reference_index(common_time, time_ref_ms)
    data = np.asarray(data, dtype=float)
    return data - data[..., index][..., None]


def differential_coefficients(data_a, data_b, common_time, time_ref_ms):
    """Build the A minus B coefficient difference, zeroed at a reference time.

    Parameters
    ----------
    data_a, data_b : numpy.ndarray, shape (n_shots, n_modes, n_times)
        Zernike coefficients of each beam, interpolated on a common time
        base.
    common_time : numpy.ndarray, shape (n_times,)
        Time base, in seconds.
    time_ref_ms : float
        Instant at which the differential is set to zero, in milliseconds.
        This is the moment the two beams are assumed to be phased, for
        instance just before the shot.

    Returns
    -------
    differential : numpy.ndarray, shape (n_shots, n_modes, n_times)
        Differential coefficients, zero at the reference time for every
        shot and every mode.
    mean : numpy.ndarray, shape (n_modes, n_times)
        Average over the shots.
    variance : numpy.ndarray, shape (n_modes, n_times)
        Variance over the shots.
    """
    # Subtract the value taken at the reference instant, shot by shot and
    # mode by mode, so that every trace starts from zero there.
    differential = zero_at_reference(data_a - data_b, common_time, time_ref_ms)

    mean = np.nanmean(differential, axis=0)
    variance = np.nanvar(differential, axis=0)
    return differential, mean, variance


def combining_efficiency_small_fluctuations(differential: np.ndarray) -> np.ndarray:
    """Combining efficiency in the small residual approximation.

    Parameters
    ----------
    differential : numpy.ndarray, shape (n_shots, n_modes, n_times)
        Differential Zernike coefficients in waves, **in the orthonormal
        'rms' convention**. The second-order expansion only reduces to a
        sum of squares if the basis is orthonormal.

    Returns
    -------
    numpy.ndarray, shape (n_shots, n_times)
        Efficiency of each shot as a function of time.

    Notes
    -----
    The expansion is second order in the residual, so the result drifts
    from the exact value once the differential wavefront approaches a
    quarter of a wave RMS. Compare against
    :func:`combining_efficiency_brute_force` when in doubt.
    """
    n_shots, n_modes, n_times = differential.shape
    efficiency = np.ones((n_shots, n_times))
    for mode in range(n_modes):
        efficiency -= 0.25 * (2 * np.pi * differential[:, mode, :]) ** 2
    return efficiency


def combining_efficiency_marechal(differential: np.ndarray) -> np.ndarray:
    """Combining efficiency, with the expansion resummed as an exponential.

    The second-order model is the first two terms of an expansion; summing
    the series for a phase error with Gaussian statistics gives

    .. math::

        \\eta \\simeq \\frac{1}{2}\\left(1 + e^{-(2\\pi\\sigma)^2}\\right)

    with :math:`\\sigma^2` the sum of the squared coefficients. This is the
    extended Marechal approximation, familiar from Strehl ratios.

    Parameters
    ----------
    differential : numpy.ndarray, shape (n_shots, n_modes, n_times)
        Differential Zernike coefficients in waves, in the orthonormal
        'rms' convention.

    Returns
    -------
    numpy.ndarray, shape (n_shots, n_times)

    Notes
    -----
    It agrees with :func:`combining_efficiency_small_fluctuations` to
    second order, and differs from it exactly where that one fails: it
    stays in [0, 1] whatever the residual, instead of running off to
    negative values. Worth trying before reaching for an empirical
    correction, since it costs nothing and has a reason to be right.
    """
    variance = np.sum(differential**2, axis=1)
    return 0.5 * (1.0 + np.exp(-((2 * np.pi) ** 2) * variance))


def _polynomial_basis(resolution: int):
    """Sample the 15 bare Zernike polynomials on a square grid.

    Returns
    -------
    basis : numpy.ndarray, shape (15, resolution, resolution)
        Polynomials sampled inside the unit disc, zero outside.
    pupil : numpy.ndarray of bool, shape (resolution, resolution)
        Mask of the unit disc.
    """
    axis = np.linspace(-1.0, 1.0, resolution)
    x, y = np.meshgrid(axis, axis)
    rho = np.hypot(x, y)
    theta = np.arctan2(y, x)
    pupil = rho <= 1.0

    basis = np.zeros((len(_POLYNOMIALS), resolution, resolution))
    for i, polynomial in enumerate(_POLYNOMIALS):
        basis[i] = np.where(pupil, polynomial(rho, theta), 0.0)
    return basis, pupil


def combining_efficiency_reconstruction(
    coefficients: np.ndarray,
    resolution: int = 64,
    mode: str = "rms",
    include_piston: bool = False,
    reference=None,
) -> np.ndarray:
    """Combining efficiency by reconstruction of the differential wavefront.

    This is the ZWR route: the differential wavefront is rebuilt from its
    Zernike coefficients, the interference pattern is formed from it, and
    the pupil average of that pattern is the efficiency. No expansion is
    made, so the only approximation left is the truncation of the Zernike
    series itself.

    Parameters
    ----------
    coefficients : array_like
        Differential Zernike coefficients in waves, beam A minus beam B.
        Either one set, shape ``(n_modes,)``, or a series, shape
        ``(n_times, n_modes)``. ``n_modes`` is 14, in which case a zero
        piston is prepended, or 15 in the order of
        :data:`chocolate.zernike.ZERNIKE_MODES`.
    resolution : int, optional
        Sampling of the pupil along each axis. Match it to the number of
        phase pixels across the real pupil: reconstructing on a finer grid
        than the sensor measured adds no information, and on a coarser one
        loses some.
    mode : {'rms', 'amp', 'ptv'}, optional
        Normalisation of the input coefficients. They are converted
        internally to the bare polynomial convention, which is the one the
        reconstruction uses.
    include_piston : bool, optional
        Keep the piston coefficient. False, the default, forces it to zero:
        a uniform offset between the two beams is measured by the
        interferometric diagnostic, and the constant term reported by a
        wavefront sensor is set by its own reference rather than by the
        beams.
    reference : int, optional
        Index of the instant the two beams are taken to be phased. The
        coefficients of that instant are subtracted from every other before
        reconstruction, which is equivalent to subtracting the
        reconstructed maps since the reconstruction is linear, and much
        cheaper.

    Returns
    -------
    numpy.ndarray
        Efficiency, shape ``(n_times,)``, or a scalar array for a single
        set of coefficients.

    Raises
    ------
    ValueError
        If the number of modes is neither 14 nor 15.

    Examples
    --------
    A single residual mode of a tenth of a wave RMS:

    >>> import numpy as np
    >>> coefficients = np.zeros(15)
    >>> coefficients[3] = 0.1                       # defocus
    >>> float(combining_efficiency_reconstruction(coefficients))  # doctest: +SKIP
    0.83...
    """
    coefficients = np.asarray(coefficients, dtype=float)
    single = coefficients.ndim == 1
    coefficients = np.atleast_2d(coefficients)
    n_modes = coefficients.shape[1]

    if n_modes == len(ZERNIKE_MODES) - 1:
        # Piston is absent from the exported columns; prepend it as zero so
        # that column j lines up with polynomial j.
        coefficients = np.hstack([np.zeros((len(coefficients), 1)), coefficients])
    elif n_modes != len(ZERNIKE_MODES):
        raise ValueError(f"Expected 14 or {len(ZERNIKE_MODES)} modes, got {n_modes}.")
    else:
        coefficients = coefficients.copy()

    if reference is not None:
        coefficients = coefficients - coefficients[reference]

    if not include_piston:
        coefficients[:, 0] = 0.0

    # The polynomials below are bare, so the coefficients must be too.
    if mode != "amp":
        factors = _conversion_factors(MODE_CODES[mode], MODE_CODES["amp"])
        coefficients = coefficients * np.array(
            [factors[name] for name in ZERNIKE_MODES]
        )

    basis, pupil = _polynomial_basis(resolution)
    n_pixels = int(pupil.sum())

    # wavefronts[t] is the differential map at instant t, in waves.
    wavefronts = np.tensordot(coefficients, basis, axes=(1, 0))
    cosines = np.where(pupil, np.cos(2 * np.pi * wavefronts), 0.0)
    efficiency = (1.0 + cosines.sum(axis=(1, 2)) / n_pixels) / 2.0

    return efficiency[0] if single else efficiency


#: Former name of :func:`combining_efficiency_reconstruction`, kept so that
#: existing scripts keep working.
combining_efficiency_brute_force = combining_efficiency_reconstruction


def zernike_analysis(
    differential: np.ndarray,
    model: str = "small_fluctuations",
    resolution: int = 64,
    mode: str = "rms",
    verbose: bool = False,
):
    """Combining efficiency of a set of shots, with the chosen model.

    Parameters
    ----------
    differential : numpy.ndarray, shape (n_shots, n_modes, n_times)
        Differential Zernike coefficients, as returned by
        :func:`differential_coefficients`. The array is never modified.
    model : {'small_fluctuations', 'marechal', 'brute_force'}, optional
        Which estimator to use. ``'brute_force'`` is the reconstruction
        route of :func:`combining_efficiency_reconstruction`.
    resolution : int, optional
        Pupil sampling, only used by the brute force model.
    mode : {'rms', 'amp', 'ptv'}, optional
        Normalisation of the coefficients.
    verbose : bool, optional
        Print progress during the brute force computation, which is slow.

    Returns
    -------
    efficiency : numpy.ndarray, shape (n_shots, n_times)
        Efficiency of each shot.
    mean : numpy.ndarray, shape (n_times,)
        Average over the shots.
    std : numpy.ndarray, shape (n_times,)
        Standard deviation over the shots.

    Raises
    ------
    ValueError
        If ``model`` is not one of the two supported estimators.
    """
    differential = np.asarray(differential, dtype=float)

    if model == "small_fluctuations":
        if mode != "rms":
            raise ValueError(
                "The small fluctuation model assumes an orthonormal basis; "
                f"convert the coefficients to 'rms' first, got {mode!r}."
            )
        efficiency = combining_efficiency_small_fluctuations(differential)

    elif model == "marechal":
        if mode != "rms":
            raise ValueError(
                "The Marechal approximation assumes an orthonormal basis; "
                f"convert the coefficients to 'rms' first, got {mode!r}."
            )
        efficiency = combining_efficiency_marechal(differential)

    elif model == "brute_force":
        efficiency = []
        n_shots = differential.shape[0]
        for shot in range(n_shots):
            if verbose:
                print(f"Brute force efficiency, shot {shot + 1}/{n_shots}")
            efficiency.append(
                combining_efficiency_reconstruction(
                    differential[shot].T, resolution=resolution, mode=mode
                )
            )
        efficiency = np.array(efficiency)

    else:
        raise ValueError(
            f"Unknown model {model!r}, expected 'small_fluctuations', "
            f"'marechal' or 'brute_force'."
        )

    # nanmean for the average but plain std for the dispersion: this is the
    # behaviour of the original analysis and it is kept as is so that the
    # published curves are reproduced exactly. A shot containing a gap
    # therefore contributes to the mean but turns the dispersion into nan at
    # that instant.
    return efficiency, np.nanmean(efficiency, axis=0), np.std(efficiency, axis=0)
