"""Clean.

This module contains the functions to perform
the Hogbom's clean algorithm on dirty observations.

:Authors: Ezequiel Centofanti <ezequiel.centofanti@cea.fr>

"""

import numpy as np

from argosim.data_utils import gauss_source
from argosim.metrics_utils import fit_elliptical_beam


def shift_beam(beam, shift_x, shift_y):
    """Shift beam.

    Function to shift the beam image by a given amount of pixels in the
    x and y directions.

    Parameters
    ----------
    beam : np.ndarray
        The beam image.
    shift_x : int
        The shift in the x direction.
    shift_y : int
        The shift in the y direction.

    Returns
    -------
    beam_shift : np.ndarray
        The shifted beam image.
    """
    beam_shift = np.roll(beam, shift_x, axis=1)
    if shift_x < 0:
        beam_shift[:, shift_x:] = 0
    else:
        beam_shift[:, :shift_x] = 0

    beam_shift = np.roll(beam_shift, shift_y, axis=0)
    if shift_y < 0:
        beam_shift[shift_y:, :] = 0
    else:
        beam_shift[:shift_y, :] = 0
    return beam_shift


def find_peak(I):
    """Find peak.

    Function to find the brightest (positive) peak of an image. Cleaning is
    positive-only, so the maximum value is used rather than the maximum
    absolute value — negative sidelobes are never selected as components.

    Parameters
    ----------
    I : np.ndarray
        The image.

    Returns
    -------
    max_val : float
        The maximum value of the image.
    x_max : int
        The x coordinate of the maximum value.
    y_max : int
        The y coordinate of the maximum value.
    shift_x : int
        The shift in the x direction from the center of the image.
    shift_y : int
        The shift in the y direction from the center of the image.
    """
    y_max, x_max = np.unravel_index(int(np.argmax(I)), I.shape)
    y_off, x_off = I.shape[0] // 2, I.shape[1] // 2
    shift_x, shift_y = x_max - x_off, y_max - y_off
    max_val = I[y_max, x_max]
    return max_val, x_max, y_max, shift_x, shift_y


def pad_odd(im):
    """Pad odd.

    Function to pad an image with zeros to make it odd in size.

    Parameters
    ----------
    im : np.ndarray
        The image to pad.

    Returns
    -------
    im_padded : np.ndarray
        The padded image.
    """
    n = im.shape[0]
    if n % 2 != 0:
        return im
    else:
        return np.pad(im, ((0, 1), (0, 1)), mode="constant")


def clean_hogbom(
    I_obs,
    B,
    gamma=0.2,
    max_iter=100,
    threshold=None,
    clean_beam_size_px=None,
    res=False,
):
    """Clean Hogbom.

    Function to perform the Hogbom's clean algorithm on a dirty image. Cleaning
    is positive-only: at each iteration the brightest positive residual peak is
    selected as a component, and cleaning stops when that peak falls below
    ``threshold``.

    Parameters
    ----------
    I_obs : np.ndarray
        The dirty image.
    B : np.ndarray
        The beam image (fft shifted).
    gamma : float
        The clean gain.
    max_iter : int
        The maximum number of iterations.
    threshold : float
        The threshold on the residual peak to stop the cleaning process.
    clean_beam_size_px : float, optional
        The size (FWHM, in pixels) of the circular restoring beam. If ``None``
        (default), it is estimated from the dirty beam: an ellipse is fitted to
        the dirty-beam main lobe and a round beam of FWHM equal to the mean of
        the two fitted FWHMs is used (a warning prints the estimated size). A
        round beam is used on purpose, so sources are restored as they are,
        undistorted by the (elliptical) interferometer beam.
    res : bool
        Add residual signal to clean image.

    Returns
    -------
    I_clean : np.ndarray
        The cleaned image.
    sky_model : np.ndarray
        The sky model image.
    """
    # If the observation and beam are even in size, pad them with zeros at the bottom and right
    # An odd beam is easier to place at the image peaks
    if I_obs.shape[0] % 2 == 0:
        I_obs = pad_odd(I_obs)
        B = pad_odd(B)

    I_res = I_obs.copy()
    I_clean = np.zeros_like(I_obs)
    sky_model = np.zeros_like(I_obs)
    B_norm = B / np.max(B)
    if clean_beam_size_px is None:
        # Estimate a round restoring beam from the dirty-beam main lobe:
        # a round beam of FWHM equal to the mean of the two fitted FWHMs.
        fit = fit_elliptical_beam(B_norm)
        clean_beam_size_px = (fit["width"] + fit["height"]) / 2
        print("Estimated clean beam size (FWHM): {:.2f} px".format(clean_beam_size_px))
    B_clean = gauss_source(
        B.shape[1], B.shape[0], np.array([0, 0]), fwhm_pix=clean_beam_size_px
    )

    for i in range(max_iter):
        # Get the brightest positive peak coordinates and flux value
        max_val, x_max, y_max, shift_x, shift_y = find_peak(I_res)
        if threshold is not None and max_val < threshold:
            print("Reached threshold at iteration {}".format(i))
            break
        # Subtract the peak from the dirty image
        I_res -= gamma * max_val * shift_beam(B_norm, shift_x, shift_y)
        sky_model[y_max, x_max] += gamma * max_val
        I_clean += gamma * max_val * shift_beam(B_clean, shift_x, shift_y)

    if I_obs.shape[0] % 2 != 0:
        I_res = I_res[:-1, :-1]
        I_clean = I_clean[:-1, :-1]
        sky_model = sky_model[:-1, :-1]

    if res:
        return I_clean + I_res, sky_model
    else:
        return I_clean, sky_model
