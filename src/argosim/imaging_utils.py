"""Imaging utils.

This module contains functions to perform radio interferometric imaging.

:Authors: Ezequiel Centofanti <ezequiel.centofanti@cea.fr>
          Samuel Gullin <gullin@ia.forth.gr>

"""

import jax.numpy as jnp
import numpy as np
import numpy.random as rnd

from argosim.rand_utils import local_seed


def sky2uv(sky):
    """Sky to uv plane (JAX version).

    Function to compute the Fourier transform of the sky.

    Parameters
    ----------
    sky : np.ndarray
        The sky image.

    Returns
    -------
    sky_uv : np.ndarray
        The Fourier transform of the sky.
    """
    # return np.fft.fft2(sky)
    return jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(sky)))


def scale_uv_samples(uv_samples, sky_uv_shape, fov_size):
    """Scale uv samples (JAX version).

    Function to scale the uv samples to pixel coordinates.

    Parameters
    ----------
    uv_samples : np.ndarray
        The uv samples coordinates in meters.
    sky_uv_shape : tuple
        The shape of the sky model in pixels.
    fov_size : tuple
        The field of view size in degrees.

    Returns
    -------
    uv_samples_indices : np.ndarray
        The indices of the uv samples in pixel coordinates.
    """
    max_u = (180 / jnp.pi) * sky_uv_shape[0] / (2 * fov_size[0])
    max_v = (180 / jnp.pi) * sky_uv_shape[1] / (2 * fov_size[1])
    uv_samples_indices = (
        jnp.rint(
            uv_samples[:, :2] / jnp.array([max_u, max_v]) / 2 * jnp.array(sky_uv_shape)
        )
        + jnp.array(sky_uv_shape) // 2
    )
    return uv_samples_indices


def check_uv_samples_range(uv_samples_indices, uv_samples, sky_uv_shape, fov_size):
    """Check uv samples range (JAX version).

    Function to check if the uv samples are within the uv-plane range.

    Parameters
    ----------
    uv_samples_indices : np.ndarray
        The indices of the uv samples in pixel coordinates.
    sky_uv_shape : tuple
        The shape of the sky model in pixels.
    uv_samples : np.ndarray
        The uv samples coordinates in meters.
    fov_size : tuple
        The field of view size in degrees.
    """
    sky_uv_shape_array = jnp.array(sky_uv_shape)
    if jnp.any(sky_uv_shape_array <= jnp.max(uv_samples_indices, axis=0)):
        max_uv = jnp.max(jnp.abs(uv_samples[:, :2]), axis=0)
        required_npix = jnp.ceil(max_uv * 2 * jnp.pi * jnp.array(fov_size) / 180)
        raise ValueError(
            f"uv samples lie out of the uv-plane. Required Npix > {required_npix}"
        )


def grid_uv_samples(
    uv_samples, sky_uv_shape, fov_size, mask_type="binary", weights=None
):
    """Grid uv samples (JAX version).

    Compute the uv sampling mask from the uv samples.

    Parameters
    ----------
    uv_samples : np.ndarray
        The uv samples coordinates in meters.
    sky_uv_shape : tuple
        The shape of the sky model in pixels.
    fov_size : tuple
        The field of view size in degrees.
    mask_type : str
        The type of mask to use. Choose between 'binary', 'histogram' and 'weighted'.
    weights : np.ndarray
        The weights to use for the mask type 'weighted'.

    Returns
    -------
    uv_mask : np.ndarray
        The uv sampling mask.
    uv_samples_indices : np.ndarray
        The indices of the uv samples in pixel coordinates.
    """
    uv_samples_indices = scale_uv_samples(uv_samples, sky_uv_shape, fov_size)
    # Check if the uv samples are within the uv-plane range
    check_uv_samples_range(uv_samples_indices, uv_samples, sky_uv_shape, fov_size)

    uv_mask = jnp.zeros(sky_uv_shape, dtype=jnp.complex128)

    # Convert uv_samples_indices to integer indices
    indices = jnp.array(uv_samples_indices, dtype=jnp.int32)

    if mask_type == "binary":
        uv_mask = uv_mask.at[indices[:, 1], indices[:, 0]].set(1 + 0j)
    elif mask_type == "histogram":
        uv_mask = uv_mask.at[indices[:, 1], indices[:, 0]].add(1 + 0j)
    elif mask_type == "weighted":
        assert weights is not None, "Weights must be provided for mask type 'weighted'."
        uv_mask = uv_mask.at[indices[:, 1], indices[:, 0]].add(
            weights[indices[:, 0], indices[:, 1]]
        )
    else:
        raise ValueError(
            "Invalid mask type. Choose between 'binary', 'histogram' and 'weighted'."
        )

    return uv_mask, uv_samples_indices


def uv2sky(uv):
    """Uv to sky (JAX version).

    Function to compute the inverse Fourier transform of the uv plane.

    Parameters
    ----------
    uv : np.ndarray
        The image in the uv/Fourier domain.

    Returns
    -------
    sky : np.ndarray
        The image in the sky domain.
    """
    return jnp.fft.fftshift(jnp.fft.ifft2(jnp.fft.ifftshift(uv))).real


def compute_visibilities_grid(sky_uv, uv_mask):
    """Compute visibilities gridded.

    Function to compute the visibilities from the fourier sky and the uv sampling mask.

    Parameters
    ----------
    sky_uv : np.ndarray
        The sky model in Fourier/uv domain.
    uv_mask : np.ndarray
        The uv sampling mask.

    Returns
    -------
    visibilities : np.ndarray
        Gridded visibilities on the uv-plane.
    """
    return sky_uv * uv_mask + 0 + 0.0j


def add_noise_uv(vis, uv_mask, sigma=0.1, seed=None):
    """Add noise in uv-plane.

    Function to add white gaussian noise to the visibilities in the uv-plane.

    Parameters
    ----------
    vis : np.ndarray
        The visibilities.
    mask : np.ndarray
        The uv sampling mask.
    sigma : float
        The standard deviation of the noise.
    seed : int
        Optional seed to set.

    Returns
    -------
    vis : np.ndarray
        The visibilities with added noise.
    """
    if sigma == 0.0:
        return vis

    with local_seed(seed):
        noise_sky = rnd.normal(0, sigma, vis.shape)
    noise_uv = sky2uv(noise_sky)

    return vis + compute_visibilities_grid(noise_uv, uv_mask)


def simulate_dirty_observation(
    sky, track, fov_size, multi_band=False, freqs=None, beam=None, sigma=0.2, seed=None
):
    """Simulate dirty observation.

    Function to simulate a radio observation of the sky model from the track uv-samples.

    Parameters
    ----------
    sky : np.ndarray
        The sky model image.
    track : np.ndarray
        The uv sampling points.
    fov_size : float
        The field of view size in degrees.
    multi_band : bool
        If True, simulate a multi-band observation.
    freqs : list
        The frequency list for the multi-band simulation.
    beam : Beam
        The beam object to apply to the sky, only used in multi-band simulations.
    sigma : float
        The standard deviation of the noise.
    seed : int
        Optional seed to set for reproducibility in noise realisation.

    Returns
    -------
    obs : np.ndarray
        The dirty observation(s).
    dirty_beam : np.ndarray
        The dirty beam(s).
    """
    if multi_band:
        assert freqs is not None, "Frequency list is required for multiband simulation"
        obs_multiband = []
        beam_multiband = []
        # Iterate over the frequency bands
        for f_, track_f in zip(freqs, track):
            # Apply beam to the sky
            if beam is not None:
                beam.set_fov(fov_size)
                beam.set_f(f_ / 1e9)
                beam_amplitude = beam.get_beam()
                sky_obs = sky * beam_amplitude
            else:
                sky_obs = sky
            # Transform to uv domain
            sky_uv = sky2uv(sky_obs)
            # Compute visibilities
            uv_mask, _ = grid_uv_samples(track_f, sky_uv.shape, (fov_size, fov_size))
            vis_f = compute_visibilities_grid(sky_uv, uv_mask)
            # Add noise
            vis_f = add_noise_uv(vis_f, uv_mask, sigma, seed=seed)
            # Back to image domain
            obs = uv2sky(vis_f)
            dirty_beam = uv2sky(uv_mask)

            obs_multiband.append(obs)
            beam_multiband.append(dirty_beam)

        obs = np.array(obs_multiband)
        dirty_beam = np.array(beam_multiband)
    else:
        sky_uv = sky2uv(sky)
        uv_mask, _ = grid_uv_samples(track, sky_uv.shape, (fov_size, fov_size))
        vis = compute_visibilities_grid(sky_uv, uv_mask)
        vis = add_noise_uv(vis, uv_mask, sigma, seed=seed)
        obs = uv2sky(vis)
        dirty_beam = uv2sky(uv_mask)

    return obs, dirty_beam
