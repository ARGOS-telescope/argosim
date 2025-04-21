"""Metrics utils.

This module contains utility functions to compute metrics between images.

:Authors: Ezequiel Centofanti <ezequiel.centofanti@cea.fr>

"""

import numpy as np
from scipy.optimize import curve_fit


# from skimage.metrics import structural_similarity as ssim_skimage


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


# def ssim(img1, img2):
#     """Structural similarity index.

#     Function to compute the structural similarity index between two images.

#     Parameters
#     ----------
#     img1 : np.ndarray
#         The first image.
#     img2 : np.ndarray
#         The second image.

#     Returns
#     -------
#     ssim : float
#         The structural similarity index between the two images.
#     """
#     return ssim_skimage(img1, img2)


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
        # "ssim": ssim(img1, img2),
    }
    return metrics


def mask_main_lobe_elliptical(beam, fit_result, scale=1.0):
    """
    Apply an elliptical mask to suppress the main lobe from a beam image.

    Parameters
    ----------
    beam : np.ndarray
        2D beam image.
    fit_result : dict
        Dictionary containing the ellipse parameters (center, width, height, angle_deg).
    scale : float
        Scale factor to enlarge or shrink the ellipse mask (default is 1.0).

    Returns
    -------
    beam_masked : np.ndarray
        Beam image with the main lobe masked (set to 0).
    """
    center_x, center_y = fit_result["center"]
    width = fit_result["width"] * scale
    height = fit_result["height"] * scale
    angle = np.radians(fit_result["angle_deg"])

    y, x = np.indices(beam.shape)
    x_shifted = x - center_x
    y_shifted = y - center_y

    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    x_rot = cos_angle * x_shifted + sin_angle * y_shifted
    y_rot = -sin_angle * x_shifted + cos_angle * y_shifted

    mask = (x_rot**2 / (width / 2) ** 2 + y_rot**2 / (height / 2) ** 2) < 1

    beam_copy = beam.copy()
    beam_copy[mask] = 0
    return beam_copy


def compute_sll(beam, fit_result, scale=1.0):
    """
    Compute the Side-Lobe Level (SLL) of a beam using an elliptical mask.

    Parameters
    ----------
    beam : np.ndarray
        The beam image (2D).
    fit_result : dict
        Elliptical fit result containing center, width, height, angle_deg.
    scale : float
        Scale factor for the ellipse mask.

    Returns
    -------
    sll_db : float
        Side-Lobe Level in dB.
    """
    main_lobe_peak = np.max(np.abs(beam))
    beam_masked = mask_main_lobe_elliptical(np.abs(beam), fit_result, scale=scale)
    side_lobe_peak = np.max(beam_masked)

    sll_db = 10 * np.log10(side_lobe_peak / main_lobe_peak + 1e-12)
    return sll_db


def compute_fwhm(sigma_x, sigma_y):
    """
    Convert standard deviations to FWHM.

    Parameters
    ----------
    sigma_x : float
        Standard deviation along x-axis.
    sigma_y : float
        Standard deviation along y-axis.
    Returns
    -------
    tuple
        FWHM along x and y.
    """
    factor = 2.553  # approximation for FWHM = 2*sqrt(2*ln(2))
    return factor * sigma_x, factor * sigma_y


def compute_eccentricity(fwhm_x, fwhm_y):
    """
    Compute the eccentricity of an ellipse from FWHM values.

    Parameters
    ----------
    fwhm_x : float
        FWHM along x-axis.
    fwhm_y : float
        FWHM along y-axis.

    Returns
    -------
    float
        Ellipticity e ∈ [0, 1], where 0 = circle and 1 = highly elongated.
    """
    a = max(fwhm_x, fwhm_y)
    b = min(fwhm_x, fwhm_y)
    return np.sqrt(1 - (b / a) ** 2)


def fit_elliptical_beam(beam, threshold_ratio=0.5):
    """
    Fit an ellipse to the brightest region of the beam based on intensity thresholding.

    Parameters
    ----------
    beam : np.ndarray
        2D beam image.
    threshold_ratio : float
        Threshold fraction of the maximum intensity to define the bright region.

    Returns
    -------
    dict
        Ellipse parameters: center, width, height, angle_deg, eccentricity.
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

    # Axes
    scale = 2
    width = 2 * np.sqrt(eigvals[0]) * scale
    height = 2 * np.sqrt(eigvals[1]) * scale

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
