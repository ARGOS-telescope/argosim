"""Imaging utils.

This module contains functions to perform radio interferometric imaging.

The production forward model uses **Kaiser-Bessel (KB) convolutional gridding**
(:func:`simulate_dirty_observation`). A legacy nearest-neighbour (NN)
binary-mask path is kept at the bottom of the module for the GUI and for
teaching; see the "Legacy / educational" section there.

:Authors: Ezequiel Centofanti <ezequiel.centofanti@cea.fr>
          Samuel Gullin <gullin@ia.forth.gr>

"""

import warnings
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import numpy.random as rnd
from jax import jit

from argosim.rand_utils import local_seed


def _as_fov_tuple(fov_size):
    """Normalise a field of view to a ``(fov_y, fov_x)`` tuple.

    Accepts either a scalar (applied to both axes, square field) or a
    ``(fov_y, fov_x)`` pair, so every field-of-view argument can be given in
    either form.

    Parameters
    ----------
    fov_size : float or tuple
        Field of view in degrees, scalar or ``(fov_y, fov_x)``.

    Returns
    -------
    tuple
        ``(fov_y, fov_x)``.
    """
    return (fov_size, fov_size) if np.isscalar(fov_size) else tuple(fov_size)


# ---------------------------------------------------------------------------
# Fourier transforms
# ---------------------------------------------------------------------------
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
    return jnp.fft.fftshift(jnp.fft.fft2(jnp.fft.ifftshift(sky)))


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


# ---------------------------------------------------------------------------
# Kaiser-Bessel convolutional gridding
# ---------------------------------------------------------------------------
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

    Notes
    -----
    The epsilon inside the square root keeps its argument strictly positive.
    Without it ``sqrt`` has an infinite derivative at ``arg == 0`` (the support
    edge ``|x| = W/2``, always hit by the gridding stencil), giving a NaN
    gradient in single precision. The value change is negligible (~1e-12).
    """
    arg = jnp.clip(1.0 - (2.0 * x / W) ** 2, 0.0, None)
    return jax.scipy.special.i0(beta * jnp.sqrt(arg + 1e-12)) / jax.scipy.special.i0(beta)


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
        kernel_ft = jnp.abs(jnp.fft.fftshift(jnp.fft.fft(jnp.fft.ifftshift(kernel_1d))))
        # Guard against near-zero values at the grid edges
        kernel_ft = jnp.where(kernel_ft < 1e-10, jnp.ones_like(kernel_ft), kernel_ft)
        return 1.0 / kernel_ft

    ny, nx = grid_shape
    return jnp.outer(correction_1d(ny), correction_1d(nx))


def scale_uv_samples_continuous(uv_samples, sky_uv_shape, fov_size):
    """Scale UV samples to continuous pixel coordinates.

    Returns floating-point (non-rounded) pixel positions, required for
    sub-pixel-accurate convolutional gridding.

    Parameters
    ----------
    uv_samples : jnp.ndarray
        UV samples in wavelengths, shape ``(n_vis, 3)``.
    sky_uv_shape : tuple
        Shape of the UV grid ``(ny, nx)``.
    fov_size : float or tuple
        Field of view in degrees, scalar or ``(fov_y, fov_x)``.

    Returns
    -------
    uv_px : jnp.ndarray
        Continuous pixel coordinates ``(u, v)`` for each visibility,
        shape ``(n_vis, 2)``.
    """
    fov_size = _as_fov_tuple(fov_size)
    max_u = (180 / jnp.pi) * sky_uv_shape[0] / (2 * fov_size[0])
    max_v = (180 / jnp.pi) * sky_uv_shape[1] / (2 * fov_size[1])
    uv_px = (
        uv_samples[:, :2] / jnp.array([max_u, max_v]) / 2 * jnp.array(sky_uv_shape)
        + jnp.array(sky_uv_shape) // 2
    )
    return uv_px


def check_uv_in_grid(uv_px, grid_shape, fov_size, on_out_of_bounds="warn"):
    """Check that UV samples fall inside the uv grid.

    Convolutional gridding silently drops any visibility whose target pixel
    falls outside the grid (``grid_visibilities_conv`` uses ``mode="drop"``),
    so a grid too small for the longest baselines quietly loses data. This
    function flags that case, using the continuous pixel coordinates returned by
    ``scale_uv_samples_continuous``: a sample is out of the grid when its pixel
    coordinate is ``< 0`` or ``> grid_shape`` along either axis.

    The message therefore reports the largest field of view that keeps every
    sample on the grid for the chosen ``Npix`` and uv coverage.

    Parameters
    ----------
    uv_px : jnp.ndarray
        Continuous pixel coordinates ``(u, v)`` for each visibility,
        shape ``(n_vis, 2)``, as returned by ``scale_uv_samples_continuous``.
    grid_shape : tuple
        Shape of the uv grid ``(ny, nx)``.
    fov_size : float or tuple
        Field of view in degrees corresponding to ``uv_px`` / ``grid_shape``.
        A scalar is applied to both axes; a tuple is read as ``(fov_y, fov_x)``.
        Used to report the maximum field of view that keeps all samples on-grid.
    on_out_of_bounds : {"warn", "error", "ignore"}
        Action when one or more samples fall outside the grid:
        ``"warn"`` (default) emits a ``UserWarning`` reporting the proportion of
        dropped samples, ``"error"`` raises a ``ValueError``, and ``"ignore"``
        returns silently (samples are still dropped downstream).

    Raises
    ------
    ValueError
        If ``on_out_of_bounds`` is not one of the accepted options, or if it is
        ``"error"`` and any sample falls outside the grid.
    """
    if on_out_of_bounds == "ignore":
        return
    if on_out_of_bounds not in ("error", "warn"):
        raise ValueError(
            "on_out_of_bounds must be one of 'error', 'warn' or 'ignore', "
            f"got {on_out_of_bounds!r}."
        )

    ny, nx = grid_shape
    u = np.asarray(uv_px[:, 0])
    v = np.asarray(uv_px[:, 1])
    out_of_bounds = (u < 0) | (u > nx) | (v < 0) | (v > ny)
    n_oob = int(np.count_nonzero(out_of_bounds))
    if n_oob == 0:
        return

    # Largest FoV keeping all samples on-grid: deviation scales linearly with
    # FoV, so fov_max = fov * (grid half-width / max sample deviation), per axis.
    fov_y, fov_x = (fov_size, fov_size) if np.isscalar(fov_size) else fov_size
    du_max = np.max(np.abs(u - nx // 2))
    dv_max = np.max(np.abs(v - ny // 2))
    fov_max = min(
        fov_x * (nx / 2) / max(du_max, 1e-12),
        fov_y * (ny / 2) / max(dv_max, 1e-12),
    )

    fraction = n_oob / u.size
    msg = (
        f"{n_oob}/{u.size} ({100 * fraction:.1f}%) visibilities fall outside "
        f"the uv grid of shape {tuple(grid_shape)} and will be dropped. The "
        f"maximum field of view for this Npix and uv coverage is "
        f"{fov_max:.4g} deg; reduce the field of view or increase Npix."
    )
    if on_out_of_bounds == "error":
        raise ValueError(msg)
    warnings.warn(msg, stacklevel=2)


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
        Gridded visibilities, shape ``grid_shape``, complex dtype (``complex64``
        by default, ``complex128`` if JAX x64 is enabled).
        Out-of-bounds contributions are silently dropped.
    """
    u_px = uv_px[:, 0]
    v_px = uv_px[:, 1]
    u0 = jnp.round(u_px).astype(jnp.int32)
    v0 = jnp.round(v_px).astype(jnp.int32)
    half_W = W // 2
    grid = jnp.zeros(grid_shape, dtype=complex)

    for du in range(-half_W, half_W + 1):
        ku = kaiser_bessel(u_px - (u0 + du), W, beta)  # (n_vis,)
        iu = u0 + du  # (n_vis,)
        for dv in range(-half_W, half_W + 1):
            kv = kaiser_bessel(v_px - (v0 + dv), W, beta)  # (n_vis,)
            iv = v0 + dv  # (n_vis,)
            grid = grid.at[iv, iu].add((ku * kv).astype(complex) * vis, mode="drop")
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
    vis = jnp.zeros(uv_px.shape[0], dtype=complex)

    for du in range(-half_W, half_W + 1):
        ku = kaiser_bessel(u_px - (u0 + du), W, beta)  # (n_vis,)
        iu = jnp.clip(u0 + du, 0, nx - 1)
        for dv in range(-half_W, half_W + 1):
            kv = kaiser_bessel(v_px - (v0 + dv), W, beta)  # (n_vis,)
            iv = jnp.clip(v0 + dv, 0, ny - 1)
            vis = vis + (ku * kv) * sky_uv[iv, iu]
    return vis


def _visibility_weights(uv_px, grid_shape, weighting):
    """Per-visibility imaging weights from a uv-cell sample count.

    Weighting is computed on a **sample-count histogram** (each visibility binned
    to its nearest uv cell), decoupled from the KB anti-aliasing kernel and from
    any oversampling — bin on the native imaging grid. Because every visibility
    is a member of its own cell, the cell count is always ``>= 1``, so the
    weights are bounded and need no threshold.

    - ``"natural"`` -> weight 1: dense uv regions dominate (best sensitivity,
      broadest PSF main lobe).
    - ``"uniform"`` -> weight ``1/count(cell)``: every sampled cell contributes
      equally regardless of how many samples fell in it (finer PSF, higher
      noise).

    This is the seam for a future Briggs/robust scheme
    (``w = 1/(1 + count * f(robust))``).

    Parameters
    ----------
    uv_px : jnp.ndarray
        Continuous pixel coordinates ``(u, v)`` per visibility, on the grid at
        which the count is binned (use the native, un-oversampled grid).
    grid_shape : tuple
        Shape ``(ny, nx)`` of that counting grid.
    weighting : {"natural", "uniform"}
        Weighting scheme.

    Returns
    -------
    jnp.ndarray
        Per-visibility weights, shape ``(n_vis,)``.
    """
    if weighting == "natural":
        return jnp.ones(uv_px.shape[0])
    if weighting != "uniform":
        raise ValueError(
            f"Invalid weighting {weighting!r}. Choose 'natural' or 'uniform'."
        )
    ny, nx = grid_shape
    uv = np.asarray(uv_px)
    iu = np.clip(np.round(uv[:, 0]).astype(int), 0, nx - 1)
    iv = np.clip(np.round(uv[:, 1]).astype(int), 0, ny - 1)
    flat = iv * nx + iu
    counts = np.bincount(flat, minlength=nx * ny)
    return jnp.asarray(1.0 / counts[flat])


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
    beta=None,
    oversampling=2,
    on_out_of_bounds="warn",
    weighting="natural",
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
        Standard deviation of the complex Gaussian noise added independently to
        each visibility (radiometer-style), in sky/flux units. The dirty image
        is normalised so a point source of flux ``S`` peaks at ``S`` (Jy/beam),
        so the image noise scales roughly as ``sigma / sqrt(n_vis)``.
    seed : int
        Optional seed to set for reproducibility in noise realisation.
    kernel_support : int
        KB kernel support width in pixels (odd). Default is 7. Larger values
        give better anti-aliasing at the cost of more compute.
    beta : float, optional
        KB kernel shape parameter. If ``None`` (default), uses the standard
        heuristic ``beta = 2.34 * kernel_support``. Expose it to experiment
        with different main-lobe/side-lobe trade-offs.
    oversampling : float
        UV-grid oversampling factor. The sky is zero-padded to
        ``oversampling * N`` pixels before gridding (finer UV cells, same
        angular pixel scale, FoV scaled by ``oversampling``), and the dirty
        image/beam are cropped back to the original ``N x N`` afterwards. This
        discards the image borders where the KB correction amplifies noise and
        aliasing. Default is 2, matching the ``beta = 2.34 * W`` heuristic. Set
        to 1 to recover the un-padded native-resolution behaviour.
    on_out_of_bounds : {"warn", "error", "ignore"}
        Behaviour when UV samples fall outside the (oversampled) uv grid and
        would be silently dropped by the convolutional gridding. ``"warn"``
        (default) emits a ``UserWarning`` reporting the proportion of dropped
        samples, ``"error"`` raises a ``ValueError``, and ``"ignore"`` disables
        the check. See :func:`check_uv_in_grid`.
    weighting : {"natural", "uniform"}
        Visibility weighting scheme (see :func:`_visibility_weights`).
        ``"natural"`` (default) weights every visibility equally (best
        sensitivity, broadest PSF); ``"uniform"`` weights each by
        ``1/count`` of its uv cell (finer PSF, higher noise).

    Returns
    -------
    obs : np.ndarray
        The dirty observation(s).
    dirty_beam : np.ndarray
        The dirty beam(s).
    """
    W = kernel_support
    beta = 2.34 * W if beta is None else beta
    factor = oversampling
    fov_y, fov_x = _as_fov_tuple(fov_size)

    def _simulate_single(sky_obs, track_f):
        """Pad -> grid -> correct -> crop for a single sky/track pair."""
        ny, nx = sky_obs.shape
        ny_p = int(round(factor * ny))
        nx_p = int(round(factor * nx))
        py0 = (ny_p - ny) // 2
        px0 = (nx_p - nx) // 2
        # Zero-pad the sky in image space: yields a finer (oversampled) UV grid
        # at the same angular pixel scale, with the FoV scaled by the padding.
        sky_pad = jnp.zeros((ny_p, nx_p), dtype=jnp.asarray(sky_obs).dtype)
        sky_pad = sky_pad.at[py0 : py0 + ny, px0 : px0 + nx].set(jnp.asarray(sky_obs))
        grid_shape = (ny_p, nx_p)
        fov_os = (fov_y * ny_p / ny, fov_x * nx_p / nx)
        corr = kb_correction(grid_shape, W, beta)

        # Transform to uv domain. Pre-multiply the model by the KB correction:
        # degridding convolves by the kernel (it is the adjoint of gridding) and
        # so imprints the kernel-FT taper once. Without this pre-correction the
        # recovered sky would be tapered by that factor everywhere off-centre
        # (off-axis sources suppressed); pre-distorting cancels it, leaving only
        # the single gridding taper that the final correction removes.
        sky_uv = sky2uv(sky_pad * corr)
        uv_px = scale_uv_samples_continuous(track_f, grid_shape, fov_os)
        # Bounds check at native resolution: oversampling scales the grid and
        # the sample deviations together, so it never changes the coverage. This
        # keeps the reported Npix / max-FoV in the user's (un-padded) terms.
        uv_px_native = scale_uv_samples_continuous(track_f, (ny, nx), (fov_y, fov_x))
        check_uv_in_grid(uv_px_native, (ny, nx), (fov_y, fov_x), on_out_of_bounds)
        # Per-visibility weights from the native-grid sample counts (decoupled
        # from the KB kernel / oversampling).
        w_vis = _visibility_weights(uv_px_native, (ny, nx), weighting)
        # Degrid true visibilities, add per-visibility noise, weight, grid back.
        vis = np.array(degrid_visibilities_conv(sky_uv, uv_px, grid_shape, W, beta))
        vis = add_noise_vis(vis, sigma, seed=seed)
        gridded = grid_visibilities_conv(
            w_vis * jnp.asarray(vis), uv_px, grid_shape, W, beta
        )
        psf_grid = grid_visibilities_conv(
            w_vis * jnp.ones(uv_px.shape[0], dtype=complex),
            uv_px,
            grid_shape,
            W,
            beta,
        )
        # Image-domain KB correction, then crop away the amplified border region.
        crop = (slice(py0, py0 + ny), slice(px0, px0 + nx))
        obs_c = (uv2sky(gridded) * corr)[crop]
        beam_c = (uv2sky(psf_grid) * corr)[crop]
        # Flux normalisation: divide by the sum of imaging weights (the
        # dirty-beam DC / peak) so the beam peaks at 1 and a point source of flux
        # S images to peak S (Jy/beam), for any weighting scheme.
        w_sum = beam_c[ny // 2, nx // 2]
        return obs_c / w_sum, beam_c / w_sum

    if multi_band:
        assert freqs is not None, "Frequency list is required for multiband simulation"
        obs_multiband = []
        beam_multiband = []
        for f_, track_f in zip(freqs, track):
            # Apply beam to the sky
            if beam is not None:
                beam.set_fov(fov_x)
                beam.set_f(f_ / 1e9)
                sky_obs = sky * beam.get_beam()
            else:
                sky_obs = sky
            obs_c, beam_c = _simulate_single(sky_obs, track_f)
            obs_multiband.append(obs_c)
            beam_multiband.append(beam_c)

        obs = np.array(obs_multiband)
        dirty_beam = np.array(beam_multiband)
    else:
        obs, dirty_beam = _simulate_single(sky, track)

    return obs, dirty_beam


# ===========================================================================
# Legacy / educational imaging path (GUI only)
#
# The functions below implement the older nearest-neighbour (NN) binary-mask
# gridding and its GUI helpers. They are kept ONLY for the interactive GUI and
# for teaching the contrast with the Kaiser-Bessel convolutional gridding
# above -- they are NOT part of the production forward model. For prototyping
# or applications use ``simulate_dirty_observation``, not these.
# ===========================================================================
def scale_uv_samples(uv_samples, sky_uv_shape, fov_size):
    """Scale uv samples to nearest integer pixel coordinates (NN snap).

    Function to scale the uv samples to pixel coordinates, rounding to the
    nearest cell.

    Parameters
    ----------
    uv_samples : np.ndarray
        The uv samples coordinates in meters.
    sky_uv_shape : tuple
        The shape of the sky model in pixels.
    fov_size : float or tuple
        The field of view size in degrees, scalar or ``(fov_y, fov_x)``.

    Returns
    -------
    uv_samples_indices : np.ndarray
        The indices of the uv samples in pixel coordinates.
    """
    fov_size = _as_fov_tuple(fov_size)
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
    """Check uv samples range.

    Function to check if the uv samples are within the uv-plane range.

    Parameters
    ----------
    uv_samples_indices : np.ndarray
        The indices of the uv samples in pixel coordinates.
    sky_uv_shape : tuple
        The shape of the sky model in pixels.
    uv_samples : np.ndarray
        The uv samples coordinates in meters.
    fov_size : float or tuple
        The field of view size in degrees, scalar or ``(fov_y, fov_x)``.
    """
    fov_size = _as_fov_tuple(fov_size)
    sky_uv_shape_array = jnp.array(sky_uv_shape)
    if jnp.any(sky_uv_shape_array <= jnp.max(uv_samples_indices, axis=0)):
        max_uv = jnp.max(jnp.abs(uv_samples[:, :2]), axis=0)
        required_npix = jnp.ceil(max_uv * 2 * jnp.pi * jnp.array(fov_size) / 180)
        raise ValueError(
            f"uv samples lie out of the uv-plane. Required Npix > {required_npix}"
        )


def grid_uv_samples(uv_samples, sky_uv_shape, fov_size, mask_type="binary"):
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
        The type of mask to use. Choose between 'binary' and 'histogram'.

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

    uv_mask = jnp.zeros(sky_uv_shape, dtype=complex)

    # Convert uv_samples_indices to integer indices
    indices = jnp.array(uv_samples_indices, dtype=jnp.int32)

    if mask_type == "binary":
        uv_mask = uv_mask.at[indices[:, 1], indices[:, 0]].set(1 + 0j)
    elif mask_type == "histogram":
        uv_mask = uv_mask.at[indices[:, 1], indices[:, 0]].add(1 + 0j)
    else:
        raise ValueError("Invalid mask type. Choose between 'binary' and 'histogram'.")

    return uv_mask, uv_samples_indices


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


def _add_cell_noise(vis_grid, uv_mask, sigma, seed=None):
    """Add per-cell complex Gaussian noise to gridded NN visibilities.

    The nearest-neighbour path snaps each visibility to a uv-grid cell, so noise
    is injected on the sampled cells (visibility domain), matching the
    per-visibility injection of :func:`add_noise_vis` used by the KB path. The
    std scales as ``sigma * sqrt(|uv_mask|)`` so a cell that accumulates ``n``
    samples (histogram weighting) gets noise variance ``n * sigma**2``, as if
    ``n`` independent visibilities were summed; for a binary mask this is just
    ``sigma`` per sampled cell.

    Parameters
    ----------
    vis_grid : jnp.ndarray
        Gridded visibilities ``sky_uv * uv_mask``.
    uv_mask : jnp.ndarray
        The uv sampling mask (binary / histogram).
    sigma : float
        Per-visibility complex noise std.
    seed : int
        Optional seed for reproducibility.

    Returns
    -------
    jnp.ndarray
        The gridded visibilities with added per-cell noise.
    """
    if sigma == 0.0:
        return vis_grid
    with local_seed(seed):
        noise = rnd.normal(0, sigma / np.sqrt(2), uv_mask.shape) + 1j * rnd.normal(
            0, sigma / np.sqrt(2), uv_mask.shape
        )
    return vis_grid + jnp.asarray(noise) * jnp.sqrt(jnp.abs(uv_mask))


def simulate_dirty_observation_nn(
    sky,
    track,
    fov_size,
    multi_band=False,
    freqs=None,
    beam=None,
    sigma=0.2,
    seed=None,
    weighting="natural",
):
    """Simulate a dirty observation with nearest-neighbour (NN) gridding.

    Educational counterpart to :func:`simulate_dirty_observation` (which uses
    Kaiser-Bessel convolutional gridding). Here each visibility is snapped to
    its nearest uv-grid cell: the sky FFT is multiplied by the sampling mask,
    per-visibility (per-cell) noise is added, and the result is inverse
    Fourier-transformed. Intended to contrast NN vs KB gridding in the GUI; the
    KB forward model is the one to use for prototyping and applications.

    Parameters
    ----------
    sky : np.ndarray
        The sky model image.
    track : np.ndarray
        The uv sampling points.
    fov_size : float or tuple
        The field of view size in degrees, scalar or ``(fov_y, fov_x)``.
    multi_band : bool
        If True, simulate a multi-band observation.
    freqs : list
        The frequency list for the multi-band simulation.
    beam : Beam
        The beam object to apply to the sky, only used in multi-band simulations.
    sigma : float
        Standard deviation of the complex Gaussian noise added independently to
        each sampled uv cell (visibility domain, matching the KB forward model),
        in sky/flux units. The dirty image is normalised so a point source of
        flux ``S`` peaks at ``S`` (Jy/beam). See :func:`_add_cell_noise`.
    seed : int
        Optional seed to set for reproducibility in the noise realisation.
    weighting : {"natural", "uniform"}
        Visibility weighting. For NN this is exactly the sampling-mask choice:
        ``"natural"`` uses the ``"histogram"`` mask (each cell weighted by its
        sample count), ``"uniform"`` uses the ``"binary"`` mask (each sampled
        cell weighted equally). See :func:`simulate_dirty_observation`.

    Returns
    -------
    obs : np.ndarray
        The dirty observation(s).
    dirty_beam : np.ndarray
        The dirty beam(s).
    """
    mask_type = {"natural": "histogram", "uniform": "binary"}.get(weighting)
    if mask_type is None:
        raise ValueError(
            f"Invalid weighting {weighting!r}. Choose 'natural' or 'uniform'."
        )
    fov_y, fov_x = _as_fov_tuple(fov_size)

    def _simulate_single(sky_obs, track_f):
        """Sample the sky FFT on the NN mask, add per-cell noise, invert."""
        ny, nx = sky_obs.shape
        sky_uv = sky2uv(jnp.asarray(sky_obs))
        uv_mask, _ = grid_uv_samples(
            track_f, (ny, nx), (fov_y, fov_x), mask_type=mask_type
        )
        vis = compute_visibilities_grid(sky_uv, uv_mask)
        vis = _add_cell_noise(vis, uv_mask, sigma, seed=seed)
        obs_c = uv2sky(vis)
        beam_c = uv2sky(uv_mask)
        # Flux normalisation: point source of flux S -> peak S.
        w_sum = beam_c[ny // 2, nx // 2]
        return obs_c / w_sum, beam_c / w_sum

    if multi_band:
        assert freqs is not None, "Frequency list is required for multiband simulation"
        obs_multiband = []
        beam_multiband = []
        for f_, track_f in zip(freqs, track):
            # Apply beam to the sky
            if beam is not None:
                beam.set_fov(fov_x)
                beam.set_f(f_ / 1e9)
                sky_obs = sky * beam.get_beam()
            else:
                sky_obs = sky
            obs_c, beam_c = _simulate_single(sky_obs, track_f)
            obs_multiband.append(obs_c)
            beam_multiband.append(beam_c)

        obs = np.array(obs_multiband)
        dirty_beam = np.array(beam_multiband)
    else:
        obs, dirty_beam = _simulate_single(sky, track)

    return obs, dirty_beam


def grid_sampling_function(
    track,
    fov_size,
    npix,
    method="kb",
    kernel_support=7,
    beta=None,
    oversampling=2,
    weighting="natural",
):
    """Grid the unit sampling function and return the uv plane and dirty beam.

    GUI helper. Grids unit visibilities (ones) at the uv sample positions to
    build the gridded sampling function, whose inverse Fourier transform is the
    dirty beam (impulse response). Supports the two gridding schemes offered in
    the GUI:

    - ``"nn"``: nearest-neighbour sampling mask (:func:`grid_uv_samples`).
    - ``"kb"``: Kaiser-Bessel convolutional gridding
      (:func:`grid_visibilities_conv`) with image-domain correction and
      oversampling, matching :func:`simulate_dirty_observation`.

    Parameters
    ----------
    track : np.ndarray
        The uv sampling points, shape ``(n_vis, 3)``.
    fov_size : float or tuple
        Field of view in degrees, scalar or ``(fov_y, fov_x)``.
    npix : int
        Number of pixels per side of the native image / uv grid.
    method : {"kb", "nn"}
        Gridding scheme.
    kernel_support : int
        KB kernel support width in pixels (odd). Only used for ``method="kb"``.
    beta : float, optional
        KB shape parameter. If ``None``, uses ``2.34 * kernel_support``.
    oversampling : float
        UV-grid oversampling factor for ``method="kb"``.
    weighting : {"natural", "uniform"}
        Visibility weighting (see :func:`_visibility_weights`). For NN it selects
        the mask (natural -> histogram, uniform -> binary).

    Returns
    -------
    uv_grid : jnp.ndarray
        The gridded sampling function on the uv plane. Shape ``(npix, npix)``
        for NN, or the oversampled ``(n_os, n_os)`` for KB.
    dirty_beam : jnp.ndarray
        The dirty beam (impulse response), shape ``(npix, npix)``.
    """
    fov_y, fov_x = _as_fov_tuple(fov_size)
    if method == "nn":
        mask_type = {"natural": "histogram", "uniform": "binary"}.get(weighting)
        if mask_type is None:
            raise ValueError(
                f"Invalid weighting {weighting!r}. Choose 'natural' or 'uniform'."
            )
        uv_grid, _ = grid_uv_samples(
            track, (npix, npix), (fov_y, fov_x), mask_type=mask_type
        )
        dirty_beam = uv2sky(uv_grid)
        return uv_grid, dirty_beam / dirty_beam[npix // 2, npix // 2]
    if method == "kb":
        W = kernel_support
        beta = 2.34 * W if beta is None else beta
        n_os = int(round(oversampling * npix))
        p0 = (n_os - npix) // 2
        grid_shape = (n_os, n_os)
        fov_os = (fov_y * n_os / npix, fov_x * n_os / npix)
        uv_px = scale_uv_samples_continuous(track, grid_shape, fov_os)
        uv_px_native = scale_uv_samples_continuous(track, (npix, npix), (fov_y, fov_x))
        w_vis = _visibility_weights(uv_px_native, (npix, npix), weighting)
        uv_grid = grid_visibilities_conv(
            w_vis * jnp.ones(uv_px.shape[0], dtype=complex),
            uv_px,
            grid_shape,
            W,
            beta,
        )
        corr = kb_correction(grid_shape, W, beta)
        beam_os = uv2sky(uv_grid) * corr
        dirty_beam = beam_os[p0 : p0 + npix, p0 : p0 + npix]
        return uv_grid, dirty_beam / dirty_beam[npix // 2, npix // 2]
    raise ValueError(f"Invalid method {method!r}. Choose 'kb' or 'nn'.")
