"""Metrics utils.

This module contains utility functions to compute metrics between images.

:Authors: Ezequiel Centofanti <ezequiel.centofanti@cea.fr>

"""

import matplotlib.pyplot as plt
import numpy as np

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
