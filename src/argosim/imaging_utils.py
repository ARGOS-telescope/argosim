"""Imaging utils.

This module contains functions to perform radio interferometric imaging.

:Authors: Ezequiel Centofanti <ezequiel.centofanti@cea.fr>
          Samuel Gullin <gullin@ia.forth.gr>

"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.random as rnd
from functools import partial
from jax import jit

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


def kaiser_bessel(x, W, beta):
    """Kaiser-Bessel anti-aliasing kernel.

    Evaluates the Kaiser-Bessel window function used as a gridding convolution
    kernel to suppress aliasing artefacts from non-uniform sampling.

    Parameters
    ----------
    x : jnp.ndarray
        Offset from the kernel centre in pixels.
    W : int
        Kernel support width in pixels (number of pixels the kernel spans).
    beta : float
        Shape parameter controlling the trade-off between main-lobe width and
        side-lobe suppression. A standard choice is ``beta = 2.34 * W``.

    Returns
    -------
    jnp.ndarray
        Kaiser-Bessel kernel values at ``x``. Evaluates to zero outside
        ``[-W/2, W/2]`` by the clip operation on the argument.
    """
    arg = jnp.clip(1.0 - (2.0 * x / W) ** 2, 0.0, None)
    return jax.scipy.special.i0(beta * jnp.sqrt(arg)) / jax.scipy.special.i0(beta)


def scale_uv_samples_continuous(uv_samples, sky_uv_shape, fov_size):
    """Scale UV samples to continuous pixel coordinates.

    Like :func:`scale_uv_samples` but returns floating-point (non-rounded)
    pixel positions, required for sub-pixel-accurate convolutional gridding.

    Parameters
    ----------
    uv_samples : jnp.ndarray
        UV samples in wavelengths, shape ``(n_vis, 3)``.
    sky_uv_shape : tuple
        Shape of the UV grid ``(ny, nx)``.
    fov_size : tuple
        Field of view in degrees ``(fov_y, fov_x)``.

    Returns
    -------
    uv_px : jnp.ndarray
        Continuous pixel coordinates ``(u, v)`` for each visibility,
        shape ``(n_vis, 2)``.
    """
    max_u = (180 / jnp.pi) * sky_uv_shape[0] / (2 * fov_size[0])
    max_v = (180 / jnp.pi) * sky_uv_shape[1] / (2 * fov_size[1])
    uv_px = (
        uv_samples[:, :2] / jnp.array([max_u, max_v]) / 2 * jnp.array(sky_uv_shape)
        + jnp.array(sky_uv_shape) // 2
    )
    return uv_px


def kb_correction(grid_shape, W, beta):
    """Image-domain correction for Kaiser-Bessel convolutional gridding.

    Computes the 2D correction image ``C(l, m)`` such that multiplying the
    dirty image by ``C`` compensates for the amplitude taper introduced by the
    KB convolution kernel. The correction is separable:
    ``C(l, m) = C_1d(l) * C_1d(m)``, where ``C_1d`` is the reciprocal of the
    discrete Fourier transform of the 1D KB kernel.

    Parameters
    ----------
    grid_shape : tuple
        Shape of the grid ``(ny, nx)``.
    W : int
        Kernel support width in pixels.
    beta : float
        KB shape parameter.

    Returns
    -------
    correction : jnp.ndarray
        2D image-domain correction array, shape ``grid_shape``.
    """
    def correction_1d(n):
        kernel_1d = jnp.zeros(n)
        x = jnp.arange(-(W // 2), W // 2 + 1, dtype=float)
        kb_vals = kaiser_bessel(x, W, beta)
        center = n // 2
        kernel_1d = kernel_1d.at[center - W // 2 : center + W // 2 + 1].set(kb_vals)
        kernel_ft = jnp.abs(
            jnp.fft.fftshift(jnp.fft.fft(jnp.fft.ifftshift(kernel_1d)))
        )
        # Guard against near-zero values at the grid edges
        kernel_ft = jnp.where(kernel_ft < 1e-10, jnp.ones_like(kernel_ft), kernel_ft)
        return 1.0 / kernel_ft

    ny, nx = grid_shape
    return jnp.outer(correction_1d(ny), correction_1d(nx))


@partial(jit, static_argnums=(2, 3, 4))
def grid_visibilities_conv(vis, uv_px, grid_shape, W, beta):
    """Convolutional gridding with Kaiser-Bessel anti-aliasing kernel (JAX).

    Distributes complex visibility values onto a regular grid using a
    Kaiser-Bessel convolution kernel. The Python loop over the ``W x W``
    kernel support is unrolled by JAX at JIT-compile time, so the compiled
    kernel is fully vectorised over all visibilities. Differentiable w.r.t.
    ``vis``.

    Parameters
    ----------
    vis : jnp.ndarray
        Complex visibility values to grid, shape ``(n_vis,)``.
    uv_px : jnp.ndarray
        Continuous pixel coordinates ``(u, v)`` for each visibility,
        shape ``(n_vis, 2)``.
    grid_shape : tuple of int
        Shape of the output grid ``(ny, nx)``. Must be a static argument.
    W : int
        Kernel support width in pixels (odd). Must be a static argument.
        Typically ``W = 7``.
    beta : float
        KB shape parameter. Must be a static argument.
        Typically ``beta = 2.34 * W``.

    Returns
    -------
    grid : jnp.ndarray
        Gridded visibilities, shape ``grid_shape``, dtype ``complex128``.
        Out-of-bounds contributions are silently dropped.
    """
    u_px = uv_px[:, 0]
    v_px = uv_px[:, 1]
    u0 = jnp.round(u_px).astype(jnp.int32)
    v0 = jnp.round(v_px).astype(jnp.int32)
    half_W = W // 2
    grid = jnp.zeros(grid_shape, dtype=jnp.complex128)

    for du in range(-half_W, half_W + 1):
        ku = kaiser_bessel(u_px - (u0 + du), W, beta)  # (n_vis,)
        iu = u0 + du                                    # (n_vis,)
        for dv in range(-half_W, half_W + 1):
            kv = kaiser_bessel(v_px - (v0 + dv), W, beta)  # (n_vis,)
            iv = v0 + dv                                    # (n_vis,)
            grid = grid.at[iv, iu].add(
                (ku * kv).astype(jnp.complex128) * vis, mode="drop"
            )
    return grid


@partial(jit, static_argnums=(2, 3, 4))
def degrid_visibilities_conv(sky_uv, uv_px, grid_shape, W, beta):
    """Convolutional degridding with Kaiser-Bessel anti-aliasing kernel (JAX).

    Interpolates complex visibility values from a gridded sky model at the
    given UV positions. This is the adjoint of :func:`grid_visibilities_conv`.
    Differentiable w.r.t. ``sky_uv``.

    Parameters
    ----------
    sky_uv : jnp.ndarray
        Gridded sky model in UV domain, shape ``grid_shape``, complex.
    uv_px : jnp.ndarray
        Continuous pixel coordinates ``(u, v)`` for each visibility,
        shape ``(n_vis, 2)``.
    grid_shape : tuple of int
        Shape of ``sky_uv`` ``(ny, nx)``. Must be a static argument.
    W : int
        Kernel support width in pixels (odd). Must be a static argument.
    beta : float
        KB shape parameter. Must be a static argument.

    Returns
    -------
    vis : jnp.ndarray
        Interpolated complex visibilities, shape ``(n_vis,)``.
        Contributions from out-of-bounds grid points are clipped to the
        nearest edge pixel.
    """
    u_px = uv_px[:, 0]
    v_px = uv_px[:, 1]
    u0 = jnp.round(u_px).astype(jnp.int32)
    v0 = jnp.round(v_px).astype(jnp.int32)
    half_W = W // 2
    ny, nx = grid_shape
    vis = jnp.zeros(uv_px.shape[0], dtype=jnp.complex128)

    for du in range(-half_W, half_W + 1):
        ku = kaiser_bessel(u_px - (u0 + du), W, beta)  # (n_vis,)
        iu = jnp.clip(u0 + du, 0, nx - 1)
        for dv in range(-half_W, half_W + 1):
            kv = kaiser_bessel(v_px - (v0 + dv), W, beta)  # (n_vis,)
            iv = jnp.clip(v0 + dv, 0, ny - 1)
            vis = vis + (ku * kv) * sky_uv[iv, iu]
    return vis


def add_noise_vis(vis, sigma=0.1, seed=None):
    """Add independent complex Gaussian noise to ungridded visibilities.

    Adds independent complex noise to each visibility datum. Real and imaginary
    parts are each drawn from ``N(0, sigma/sqrt(2))`` so that the total noise
    power per visibility is ``sigma^2``. This prepares for the proper
    per-baseline noise model (radiometer equation) to be integrated later.

    Parameters
    ----------
    vis : np.ndarray
        Complex visibility values, shape ``(n_vis,)``.
    sigma : float
        Standard deviation of the complex noise.
    seed : int
        Optional seed for reproducibility.

    Returns
    -------
    vis_noisy : np.ndarray
        Visibilities with added noise.
    """
    if sigma == 0.0:
        return vis
    with local_seed(seed):
        noise = rnd.normal(0, sigma / np.sqrt(2), vis.shape) + 1j * rnd.normal(
            0, sigma / np.sqrt(2), vis.shape
        )
    return vis + noise


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
    sky,
    track,
    fov_size,
    multi_band=False,
    freqs=None,
    beam=None,
    sigma=0.2,
    seed=None,
    kernel_support=7,
):
    """Simulate dirty observation.

    Function to simulate a radio observation of the sky model from the track
    uv-samples using convolutional gridding with a Kaiser-Bessel anti-aliasing
    kernel. The forward model is: degrid true visibilities from the sky FFT,
    add per-visibility complex Gaussian noise, re-grid with the KB kernel, and
    apply the image-domain correction.

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
        The standard deviation of the complex noise per visibility.
    seed : int
        Optional seed to set for reproducibility in noise realisation.
    kernel_support : int
        KB kernel support width in pixels (odd). Default is 7. Larger values
        give better anti-aliasing at the cost of more compute.

    Returns
    -------
    obs : np.ndarray
        The dirty observation(s).
    dirty_beam : np.ndarray
        The dirty beam(s).
    """
    W = kernel_support
    beta = 2.34 * W

    if multi_band:
        assert freqs is not None, "Frequency list is required for multiband simulation"
        obs_multiband = []
        beam_multiband = []
        for f_, track_f in zip(freqs, track):
            # Apply beam to the sky
            if beam is not None:
                beam.set_fov(fov_size)
                beam.set_f(f_ / 1e9)
                sky_obs = sky * beam.get_beam()
            else:
                sky_obs = sky
            # Transform to uv domain
            sky_uv = sky2uv(sky_obs)
            grid_shape = sky_uv.shape
            uv_px = scale_uv_samples_continuous(track_f, grid_shape, (fov_size, fov_size))
            # Degrid true visibilities, add noise, re-grid
            vis_f = degrid_visibilities_conv(sky_uv, uv_px, grid_shape, W, beta)
            vis_f = add_noise_vis(np.array(vis_f), sigma, seed=seed)
            gridded = grid_visibilities_conv(
                jnp.asarray(vis_f), uv_px, grid_shape, W, beta
            )
            psf_grid = grid_visibilities_conv(
                jnp.ones(uv_px.shape[0], dtype=jnp.complex128), uv_px, grid_shape, W, beta
            )
            # Apply image-domain correction
            corr = kb_correction(grid_shape, W, beta)
            obs_multiband.append(uv2sky(gridded) * corr)
            beam_multiband.append(uv2sky(psf_grid) * corr)

        obs = np.array(obs_multiband)
        dirty_beam = np.array(beam_multiband)
    else:
        sky_uv = sky2uv(sky)
        grid_shape = sky_uv.shape
        uv_px = scale_uv_samples_continuous(track, grid_shape, (fov_size, fov_size))
        # Degrid true visibilities, add noise, re-grid
        vis = degrid_visibilities_conv(sky_uv, uv_px, grid_shape, W, beta)
        vis = add_noise_vis(np.array(vis), sigma, seed=seed)
        gridded = grid_visibilities_conv(jnp.asarray(vis), uv_px, grid_shape, W, beta)
        psf_grid = grid_visibilities_conv(
            jnp.ones(uv_px.shape[0], dtype=jnp.complex128), uv_px, grid_shape, W, beta
        )
        # Apply image-domain correction
        corr = kb_correction(grid_shape, W, beta)
        obs = uv2sky(gridded) * corr
        dirty_beam = uv2sky(psf_grid) * corr

    return obs, dirty_beam
