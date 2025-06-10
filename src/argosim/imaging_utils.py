"""Imaging utils.

This module contains functions to perform radio interferometric imaging.

:Authors: Ezequiel Centofanti <ezequiel.centofanti@cea.fr>
          Samuel Gullin <gullin@ia.forth.gr>

"""

import numpy as np
import numpy.random as rnd

from argosim.rand_utils import local_seed

# from PIL import Image

########################################
#           Grid uv samples            #
########################################

# def get_uv_plane(baseline, uv_dim=128):
#     """Get uv plane.

#     Function to compute the uv sampling mask from the baselines list.
#     Perform a 2D histogram of the baselines list with uv_dim bins.

#     Parameters
#     ----------
#     baseline : np.ndarray
#         The baselines list.
#     uv_dim : int
#         The uv-plane sampling mask size.

#     Returns
#     -------
#     uv_plane : np.ndarray
#         The uv sampling mask of the antenna array. The dimensions are (uv_dim, uv_dim).
#         The value of each pixel is the number of uv samples in that pixel.

#     """
#     # Count number of samples per uv grid
#     x_lim=np.max(np.absolute(baseline))#*1.1
#     y_lim=x_lim
#     uv_plane, _, _ = np.histogram2d(baseline[:,0],baseline[:,1],bins=uv_dim, range=[[-x_lim,x_lim],[-y_lim,y_lim]])
#     return np.fliplr(uv_plane.T)#/np.sum(uv_plane, axis=(0,1))

# def get_uv_mask(uv_plane):
#     """Get uv mask.

#     Function to compute the binary mask from the uv sampling grid.

#     Parameters
#     ----------
#     uv_plane : np.ndarray
#         The uv sampling mask.

#     Returns
#     -------
#     uv_plane_mask : np.ndarray
#         The binary mask of the uv sampling mask.
#         The value of each pixel is 1 if the pixel is sampled, 0 otherwise.
#     """
#     # Get binary mask from the uv sampled grid
#     uv_plane_mask = uv_plane.copy()
#     uv_plane_mask[np.where(uv_plane>0)] = 1
#     return uv_plane_mask

# def get_beam(uv_mask):
#     """Get beam.

#     Function to compute the telescope beam from the uv sampling mask.

#     Parameters
#     ----------
#     uv_mask : np.ndarray
#         The uv sampling mask.

#     Returns
#     -------
#     beam : np.ndarray
#         The beam image of the antenna array. The beam is fftshifted (non centered).
#     """
#     return np.abs(np.fft.ifft2(uv_mask))


# def load_sky_model(path):
#     """Load sky model.

#     Function to load a sky model image.

#     Parameters
#     ----------
#     path : str
#         The path to the sky model image.

#     Returns
#     -------
#     sky : np.ndarray
#         The sky model image.
#     """
#     return np.array(Image.open(path).convert("L"))


def sky2uv(sky):
    """Sky to uv plane.

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
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(sky)))


def grid_uv_samples(
    uv_samples, sky_uv_shape, fov_size, mask_type="binary", weights=None
):
    """Grid uv samples.

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
    max_u = (180 / np.pi) * sky_uv_shape[0] / (2 * fov_size[0])
    max_v = (180 / np.pi) * sky_uv_shape[1] / (2 * fov_size[1])
    uv_samples_indices = (
        np.rint(
            uv_samples[:, :2] / np.array([max_u, max_v]) / 2 * np.array(sky_uv_shape)
        )
        + np.array(sky_uv_shape) // 2
    )

    if any(np.array(sky_uv_shape) <= np.max(uv_samples_indices, axis=0)):
        raise ValueError(
            "uv samples are out of the uv-plane range. Required Npix > {}".format(
                # np.max(uv_samples_indices, axis=0)
                np.ceil(
                    np.max(np.abs(uv_samples[:, :2]), axis=0)
                    * 2
                    * np.pi
                    * fov_size
                    / 180
                )
            )
        )

    uv_mask = np.zeros(sky_uv_shape, dtype=complex)

    for index in uv_samples_indices:
        if mask_type == "binary":
            uv_mask[int(index[1]), int(index[0])] = 1 + 0j
        elif mask_type == "histogram":
            uv_mask[int(index[1]), int(index[0])] += 1 + 0j
        elif mask_type == "weighted":
            assert (
                weights is not None
            ), "Weights must be provided for mask type 'weighted'."
            uv_mask[int(index[1]), int(index[0])] += weights[
                int(index[0]), int(index[1])
            ]
        else:
            raise ValueError(
                "Invalid mask type. Choose between 'binary', 'histogram' and 'weighted'."
            )

    return uv_mask, uv_samples_indices


def uv2sky(uv):
    """Uv to sky.

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
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(uv))).real


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
