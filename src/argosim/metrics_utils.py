"""Metrics utils.

This module contains utility functions to compute metrics between images.

:Authors: Ezequiel Centofanti <ezequiel.centofanti@cea.fr>

"""

import jax
import jax.numpy as jnp
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def mse(img1, img2):
    """Mean squared error.

    Function to compute the mean squared error between two images.

    Parameters
    ----------
    img1 : np.ndarray
        The first image.
    img2 : np.ndarray
        The second image.

    Returns
    -------
    mse : float
        The mean squared error between the two images.
    """
    return np.mean((img1 - img2) ** 2)


def residuals(img1, img2, absolute=True):
    """Residuals.

    Function to compute the residuals between two images.

    Parameters
    ----------
    img1 : np.ndarray
        The first image.
    img2 : np.ndarray
        The second image.

    Returns
    -------
    residual : np.ndarray
        The residuals between the two images.
    """
    res = img1 - img2
    return np.abs(res) if absolute else res


def compute_metrics(img1, img2):
    """Compute metrics.

    Function to compute the metrics between two images.

    Parameters
    ----------
    img1 : np.ndarray
        The first image.
    img2 : np.ndarray
        The second image.

    Returns
    -------
    metrics : dict
        The metrics between the two images.
    """
    mse_val = mse(img1, img2)
    norm_sq = mse(img1, 0.0)
    metrics = {
        "mse": mse_val,
        "rel_mse": mse_val / norm_sq,
        "residual": residuals(img1, img2),
    }
    return metrics


def fit_elliptical_beam(beam, threshold_ratio=0.5):
    """Fit elliptical beam.

    Fit an ellipse to the main lobe of the dirty beam using intensity thresholding.

    Parameters
    ----------
    beam : np.ndarray
        2D beam image.
    threshold_ratio : float
        Threshold fraction of the maximum intensity to define the bright region.

    Returns
    -------
    dict : dictionary
        Ellipse parameters: center, width, height, angle_deg, eccentricity.
        ``width`` and ``height`` are the full widths at half maximum (FWHM), in
        pixels, along the semi-major and semi-minor axes respectively.
    """
    max_val = np.max(np.abs(beam))
    threshold = threshold_ratio * max_val
    mask = np.abs(beam) >= threshold
    coords_yx = np.column_stack(np.where(mask))

    # Compute center and covariance matrix
    center_y, center_x = np.mean(coords_yx, axis=0)
    cov = np.cov(coords_yx, rowvar=False)

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    major_axis = eigvecs[:, 0]
    angle_rad = np.arctan2(major_axis[0], major_axis[1])
    angle_deg = np.degrees(angle_rad)

    # Axes (FWHM). For the half-max region, 2*sqrt(eigval) is the semi-axis
    # (~HWHM, the radius of the region); multiply by 2 to report the full width
    # at half maximum.
    width = 4 * np.sqrt(eigvals[0])
    height = 4 * np.sqrt(eigvals[1])

    # Eccentricity
    a, b = max(width, height), min(width, height)
    eccentricity = np.sqrt(1 - (b / a) ** 2)

    return {
        "center": (center_x, center_y),
        "width": width,
        "height": height,
        "angle_deg": angle_deg,
        "eccentricity": eccentricity,
    }


def _main_lobe_mask_np(beam_np):
    """Compute main lobe mask.

    Compute a binary mask of the main lobe of a beam using watershed segmentation.

    Parameters
    ----------
    beam_np : np.ndarray
        The beam image (2D).

    Returns
    -------
    np.ndarray
        Binary mask of the main lobe.
    """
    beam_np = np.asarray(beam_np)
    peak_idx = np.unravel_index(np.argmax(beam_np), beam_np.shape)

    # One marker per local maximum -> each lobe gets its own catchment basin
    coords = peak_local_max(beam_np, min_distance=1)
    markers = np.zeros(beam_np.shape, dtype=np.int32)
    for i, c in enumerate(coords, start=1):
        markers[tuple(c)] = i

    labels = watershed(-beam_np, markers=markers)  # flood "downhill" from each peak
    main_label = labels[peak_idx]
    return (labels == main_label).astype(np.float32)


def beam_isl(beam):
    """Compute ISL.

    Compute the integrated Sidelobe level (ISL) of a beam using watershed segmentation.

    Parameters
    ----------
    beam : np.ndarray
        The beam image (2D).

    Returns
    -------
    float
        Integrated sidelobe level (ISL) in dB.
    """
    # Non-differentiable segmentation step, wrapped for use inside jit/grad
    mask = jax.pure_callback(
        _main_lobe_mask_np,
        jax.ShapeDtypeStruct(beam.shape, jnp.float32),
        beam,
    )
    mask = jax.lax.stop_gradient(mask)

    main_lobe_power = jnp.sum((beam * mask) ** 2)
    side_lobe_power = jnp.sum((beam * (1 - mask)) ** 2)
    isl_db = 10 * jnp.log10(side_lobe_power / main_lobe_power + 1e-12)
    return isl_db


def beam_psl(beam):
    """Compute PSL.

    Compute the peak sidelobe level (PSL) of a beam using watershed segmentation.

    Parameters
    ----------
    beam : np.ndarray
        The beam image (2D).

    Returns
    -------
    float
        Peak sidelobe level (PSL) in dB.
    """
    # Non-differentiable segmentation step, wrapped for use inside jit/grad
    mask = jax.pure_callback(
        _main_lobe_mask_np,
        jax.ShapeDtypeStruct(beam.shape, jnp.float32),
        beam,
    )
    mask = jax.lax.stop_gradient(mask)

    main_lobe_peak_power = jnp.max((beam * mask) ** 2)
    side_lobe_peak_power = jnp.max((beam * (1 - mask)) ** 2)
    psl_db = 10 * jnp.log10(side_lobe_peak_power / main_lobe_peak_power + 1e-12)
    return psl_db


def compute_fwhm(beam, fit_result=None):
    """Compute fwhm.

    Compute FWHM from beam.

    Parameters
    ----------
    beam : np.ndarray
        The beam image (2D).
    fit_result : dict
        Dictionary containing the ellipse parameters (center, width, height, angle_deg, eccentricity). If None, it is computed from the beam.

    Returns
    -------
    tuple
        FWHM along the semi-major and semi-minor axis of the beam.
    """
    if fit_result is None:
        fit_result = fit_elliptical_beam(beam)
    return fit_result["width"], fit_result["height"]


def compute_eccentricity(beam, fit_result=None):
    """Compute eccentricity.

    Compute the eccentricity of the beam main lobe.

    Parameters
    ----------
    beam : np.ndarray
        The beam image (2D).
    fit_result : dict
        Dictionary containing the ellipse parameters (center, width, height, angle_deg, eccentricity). If None, it is computed from the beam.

    Returns
    -------
    float
        Eccentricity of the beam. Ranges from 0 to 1, where 0 indicates a perfectly circular beam and 1 a completely degenerated beam.
    """
    if fit_result is None:
        fit_result = fit_elliptical_beam(beam)
    fwhm_a, fwhm_b = compute_fwhm(beam, fit_result)
    return np.sqrt(1 - (fwhm_b / fwhm_a) ** 2)


def compute_beam_metrics(beam):
    """Compute beam metrics.

    Compute main beam metrics: peak and integrated sidelobe level (PSL / ISL),
    FWHM, and eccentricity.

    Parameters
    ----------
    beam : np.ndarray
        The beam image (2D).

    Returns
    -------
    dict
        Dictionary containing PSL (dB), ISL (dB), FWHM (x, y), and eccentricity.
    """
    fit_result = fit_elliptical_beam(beam)
    fwhm_a, fwhm_b = compute_fwhm(beam, fit_result)
    eccentricity = compute_eccentricity(beam, fit_result)
    return {
        "psl_db": float(beam_psl(beam)),
        "isl_db": float(beam_isl(beam)),
        "fwhm": (fwhm_a, fwhm_b),
        "eccentricity": eccentricity,
    }
