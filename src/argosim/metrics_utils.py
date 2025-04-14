"""Metrics utils.

This module contains utility functions to compute metrics between images.

:Authors: Ezequiel Centofanti <ezequiel.centofanti@cea.fr>

"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from matplotlib.patches import Ellipse


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


def compute_mask_radius(fwhm_x, fwhm_y, masking_factor=3):
    """
    Compute the radius (in pixels) for masking the main lobe.

    Parameters
    ----------
    fwhm_x : float
        Full Width at Half Maximum along x-axis.
    fwhm_y : float
        Full Width at Half Maximum along y-axis.
    masking_factor : float
        Factor to reduce the mask size (default is 3).

    Returns
    -------
    radius : int
        Radius in pixels used to mask the main lobe.
    """
    return int(np.round((fwhm_x + fwhm_y) / masking_factor))


def mask_main_lobe(beam, center, radius):
    """
    Apply a circular mask to suppress the main lobe from a beam image.

    Parameters
    ----------
    beam : np.ndarray
        2D beam image.
    center : tuple of int
        (x, y) coordinates of the beam center.
    radius : int
        Radius of the exclusion zone in pixels.

    Returns
    -------
    beam_masked : np.ndarray
        Beam image with the main lobe masked (set to 0).
    """
    beam_copy = beam.copy()
    y, x = np.indices(beam.shape)
    mask = ((x - center[0]) ** 2 + (y - center[1]) ** 2) < radius**2
    beam_copy[mask] = 0
    return beam_copy


def compute_sll(beam, center, fwhm_x, fwhm_y, masking_factor=3, plot=False):
    """
    Compute the Side-Lobe Level (SLL) of a beam.

    Parameters
    ----------
    beam : np.ndarray
        The beam image (2D).
    center : tuple of int
        Coordinates of the beam center.
    fwhm_x : float
        Beam width along x.
    fwhm_y : float
        Beam width along y.
    masking_factor : float
        Factor to determine the radius size for the mask.
    plot : bool
        If True, plot the masked beam.

    Returns
    -------
    sll_db : float
        Side-Lobe Level in dB.
    """
    beam_shifted = np.fft.fftshift(beam.copy())
    radius = compute_mask_radius(fwhm_x, fwhm_y, masking_factor)
    beam_masked = mask_main_lobe(beam_shifted, center, radius)

    main_lobe_peak = np.max(beam_shifted)
    side_lobe_peak = np.max(np.abs(beam_masked))
    sll_db = 10 * np.log10(side_lobe_peak / main_lobe_peak + 1e-12)

    if plot:
        plt.figure(figsize=(6, 5))
        plt.imshow(beam_masked, origin="lower", cmap="viridis", vmin=-0.01, vmax=0.02)
        plt.colorbar()
        plt.xlim(
            center[0] - 30, center[0] + 30
        )  # zoom in on an area around the center with more or less 30px : better visibility of the plot
        plt.ylim(center[1] - 30, center[1] + 30)
        plt.title(f"Dirty Beam (main lobe masked)\nSLL = {sll_db:.8f} dB")
        plt.show()

    return sll_db


def gaussian_2d_elliptical(coords, amp, sigma_x, sigma_y, theta, offset, x0, y0):
    """
    2D elliptical Gaussian function with rotation.

    Parameters
    ----------
    coords : tuple of np.ndarray
        Tuple containing 2D coordinate arrays (x, y).
    amp : float
        Amplitude of the Gaussian.
    sigma_x : float
        Standard deviation along x-axis.
    sigma_y : float
        Standard deviation along y-axis.
    theta : float
        Rotation angle (in radians).
    offset : float
        Constant background offset.
    x0, y0 : float
        Center of the Gaussian.

    Returns
    -------
    np.ndarray
        Flattened 2D elliptical Gaussian evaluated at coords.
    """
    x, y = coords
    x0, y0 = float(x0), float(y0)

    xp = (x - x0) * np.cos(theta) + (y - y0) * np.sin(theta)
    yp = -(x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)

    return (
        amp * np.exp(-((xp**2) / (2 * sigma_x**2) + (yp**2) / (2 * sigma_y**2)))
        + offset
    )


def get_beam_crop(beam, center, crop_size):
    """
    Extract a square region from the beam centered at a given pixel.

    Parameters
    ----------
    beam : np.ndarray
        Full 2D beam image.
    center : tuple
        Pixel coordinates (x, y) of the beam center.
    crop_size : int
        Size of the square region to extract.

    Returns
    -------
    np.ndarray
        Cropped square region of shape (crop_size, crop_size).
    """
    x0, y0 = center
    half = crop_size // 2
    x1, x2 = x0 - half, x0 + half
    y1, y2 = y0 - half, y0 + half
    return beam[y1:y2, x1:x2]


def fit_elliptical_gaussian_to_crop(crop, x0_local, y0_local):
    """
    Fit an elliptical 2D Gaussian model to a cropped beam region.

    Parameters
    ----------
    crop : np.ndarray
        2D cropped beam region.
    x0_local, y0_local : int
        Local coordinates of the Gaussian center in the crop.

    Returns
    -------
    popt : list
        Optimal parameters [amp, sigma_x, sigma_y, theta, offset].
    """
    y, x = np.indices(crop.shape)

    def model(coords, amp, sigma_x, sigma_y, theta, offset):
        return gaussian_2d_elliptical(
            coords, amp, sigma_x, sigma_y, theta, offset, x0_local, y0_local
        )

    p0 = (np.max(crop), 4.5, 4.5, 0, 0)
    bounds = ([0, 0.5, 0.5, -np.pi, -np.inf], [np.inf, 10, 10, np.pi, np.inf])

    popt, _ = curve_fit(
        model, (x.ravel(), y.ravel()), crop.ravel(), p0=p0, bounds=bounds
    )
    return popt


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


def fit_elliptical_beam(beam, center=(271, 271), crop_size=4):
    """
    Fit an elliptical 2D Gaussian to the main lobe of the beam.

    Parameters
    ----------
    beam : np.ndarray
        The dirty beam (2D array).
    center : tuple
        Pixel coordinates of the beam center.
    crop_size : int
        Size of the window around the center to fit the Gaussian.

    Returns
    -------
    dict
        Dictionary with fit results: FWHM, eccentricity, angle, sigma.
    """
    crop = get_beam_crop(beam, center, crop_size)
    x0_local = crop.shape[1] // 2
    y0_local = crop.shape[0] // 2

    amp, sigma_x, sigma_y, theta, offset = fit_elliptical_gaussian_to_crop(
        crop, x0_local, y0_local
    )

    fwhm_x, fwhm_y = compute_fwhm(sigma_x, sigma_y)
    angle_deg = np.degrees(theta) % 180
    eccentricity = compute_eccentricity(fwhm_x, fwhm_y)

    print("Elliptical Beam Fit:")
    print(f" - FWHM X         : {fwhm_x:.2f} px")
    print(f" - FWHM Y         : {fwhm_y:.2f} px")
    print(f" - Angle          : {angle_deg:.2f}°")
    print(f" - Eccentricity   : {eccentricity:.4f}")

    return {
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
        "fwhm_x": fwhm_x,
        "fwhm_y": fwhm_y,
        "angle_deg": angle_deg,
        "center": center,
        "eccentricity": eccentricity,
    }
