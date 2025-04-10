"""Metrics utils.

This module contains utility functions to compute metrics between images.

:Authors: Ezequiel Centofanti <ezequiel.centofanti@cea.fr>

"""

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
    norm_sq = mse(img1, 0.)
    metrics = {
        "mse": mse_val,
        "rel_mse": mse_val / norm_sq,
        "residuals": residuals(img1, img2),
        # "ssim": ssim(img1, img2),
    }
    return metrics
