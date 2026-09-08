"""NumPy backend for the profiling comparison.

Pure-NumPy re-implementation of the uv-tracking and the Kaiser-Bessel
convolutional imaging forward model, so the profiling script can compare a
NumPy backend against JAX-CPU and JAX-GPU. These are direct ports of the JAX
functions in ``argosim.antenna_utils`` / ``argosim.imaging_utils`` and are kept
here (script-local) so the library stays pure-JAX. Only the single-band path is
implemented (all that the profiling uses).
"""

import numpy as np
from scipy.special import i0


# ---------------------------------------------------------------------------
# uv tracking
# ---------------------------------------------------------------------------
def ENU_to_XYZ_np(b_ENU, lat):
    """ENU baselines to XYZ (NumPy port of antenna_utils.ENU_to_XYZ)."""
    D = np.sqrt(np.sum(b_ENU**2, axis=1))
    A = np.arctan2(b_ENU[:, 0], b_ENU[:, 1])
    E = np.arcsin(b_ENU[:, 2] / D)
    X = D * (np.cos(lat) * np.sin(E) - np.sin(lat) * np.cos(E) * np.cos(A))
    Y = D * np.cos(E) * np.sin(A)
    Z = D * (np.sin(lat) * np.sin(E) + np.cos(lat) * np.cos(E) * np.cos(A))
    return X, Y, Z


def uv_track_multiband_np(
    b_ENU, lat, dec, track_time, t_0, n_times, f, df, n_freqs
):
    """uv track over times and frequencies (NumPy port; single array output).

    Matches ``antenna_utils.uv_track_multiband(..., multi_band=False)``: the uvw
    points are stacked as ``(n_freqs, n_times, n_baselines, 3)`` and flattened to
    ``(n_vis, 3)``.
    """
    X, Y, Z = ENU_to_XYZ_np(b_ENU, lat)
    h = np.linspace(t_0, t_0 + track_time, n_times) * np.pi / 12
    f_range = np.linspace(f - df / 2, f + df / 2, n_freqs)
    c = 299792458.0
    lam_inv = (f_range / c)[:, None, None]          # (nf, 1, 1)
    sh = np.sin(h)[None, :, None]                   # (1, nt, 1)
    ch = np.cos(h)[None, :, None]
    Xb, Yb, Zb = X[None, None, :], Y[None, None, :], Z[None, None, :]  # (1,1,nb)
    u = lam_inv * (sh * Xb + ch * Yb)
    v = lam_inv * (
        -np.sin(dec) * ch * Xb + np.sin(dec) * sh * Yb + np.cos(dec) * Zb
    )
    w = lam_inv * (
        np.cos(dec) * ch * Xb - np.cos(dec) * sh * Yb + np.sin(dec) * Zb
    )
    track = np.stack([u, v, w], axis=-1).reshape(-1, 3)
    return track


# ---------------------------------------------------------------------------
# Kaiser-Bessel convolutional imaging
# ---------------------------------------------------------------------------
def kaiser_bessel_np(x, W, beta):
    arg = np.clip(1.0 - (2.0 * x / W) ** 2, 0.0, None)
    return i0(beta * np.sqrt(arg)) / i0(beta)


def kb_correction_np(grid_shape, W, beta):
    def correction_1d(n):
        kernel_1d = np.zeros(n)
        x = np.arange(-(W // 2), W // 2 + 1, dtype=float)
        kb_vals = kaiser_bessel_np(x, W, beta)
        center = n // 2
        kernel_1d[center - W // 2 : center + W // 2 + 1] = kb_vals
        kernel_ft = np.abs(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(kernel_1d))))
        kernel_ft = np.where(kernel_ft < 1e-10, 1.0, kernel_ft)
        return 1.0 / kernel_ft

    ny, nx = grid_shape
    return np.outer(correction_1d(ny), correction_1d(nx))


def scale_uv_samples_continuous_np(uv_samples, sky_uv_shape, fov_size):
    fov = (fov_size, fov_size) if np.isscalar(fov_size) else tuple(fov_size)
    max_u = (180 / np.pi) * sky_uv_shape[0] / (2 * fov[0])
    max_v = (180 / np.pi) * sky_uv_shape[1] / (2 * fov[1])
    return (
        uv_samples[:, :2] / np.array([max_u, max_v]) / 2 * np.array(sky_uv_shape)
        + np.array(sky_uv_shape) // 2
    )


def sky2uv_np(sky):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(sky)))


def uv2sky_np(uv):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(uv))).real


def visibility_weights_np(uv_px, grid_shape, weighting):
    if weighting == "natural":
        return np.ones(uv_px.shape[0])
    ny, nx = grid_shape
    iu = np.clip(np.round(uv_px[:, 0]).astype(int), 0, nx - 1)
    iv = np.clip(np.round(uv_px[:, 1]).astype(int), 0, ny - 1)
    flat = iv * nx + iu
    counts = np.bincount(flat, minlength=nx * ny)
    return 1.0 / counts[flat]


def grid_visibilities_conv_np(vis, uv_px, grid_shape, W, beta):
    """Convolutional gridding (NumPy). Out-of-bounds samples are dropped."""
    u_px, v_px = uv_px[:, 0], uv_px[:, 1]
    u0 = np.round(u_px).astype(np.int64)
    v0 = np.round(v_px).astype(np.int64)
    half_W = W // 2
    ny, nx = grid_shape
    grid = np.zeros(grid_shape, dtype=np.complex128)
    for du in range(-half_W, half_W + 1):
        ku = kaiser_bessel_np(u_px - (u0 + du), W, beta)
        iu = u0 + du
        for dv in range(-half_W, half_W + 1):
            kv = kaiser_bessel_np(v_px - (v0 + dv), W, beta)
            iv = v0 + dv
            # Match JAX `.at[].add(mode="drop")`: valid index range is
            # [-size, size) -- indices in [-size, -1] wrap Python-style (which
            # np.add.at also does), while anything >= size or < -size is dropped.
            inb = (iu >= -nx) & (iu < nx) & (iv >= -ny) & (iv < ny)
            np.add.at(
                grid, (iv[inb], iu[inb]), (ku * kv)[inb] * vis[inb]
            )
    return grid


def degrid_visibilities_conv_np(sky_uv, uv_px, grid_shape, W, beta):
    """Convolutional degridding (NumPy). Out-of-bounds clipped to the edge."""
    u_px, v_px = uv_px[:, 0], uv_px[:, 1]
    u0 = np.round(u_px).astype(np.int64)
    v0 = np.round(v_px).astype(np.int64)
    half_W = W // 2
    ny, nx = grid_shape
    vis = np.zeros(uv_px.shape[0], dtype=np.complex128)
    for du in range(-half_W, half_W + 1):
        ku = kaiser_bessel_np(u_px - (u0 + du), W, beta)
        iu = np.clip(u0 + du, 0, nx - 1)
        for dv in range(-half_W, half_W + 1):
            kv = kaiser_bessel_np(v_px - (v0 + dv), W, beta)
            iv = np.clip(v0 + dv, 0, ny - 1)
            vis = vis + (ku * kv) * sky_uv[iv, iu]
    return vis


def simulate_dirty_observation_np(
    sky, track, fov_size, sigma=0.0, seed=None,
    kernel_support=7, beta=None, oversampling=2, weighting="natural",
):
    """Single-band KB dirty observation (NumPy port of the JAX forward model)."""
    W = kernel_support
    beta = 2.34 * W if beta is None else beta
    factor = oversampling
    fov_y = fov_x = float(fov_size) if np.isscalar(fov_size) else fov_size[0]

    sky = np.asarray(sky, dtype=float)
    ny, nx = sky.shape
    ny_p = int(round(factor * ny))
    nx_p = int(round(factor * nx))
    py0 = (ny_p - ny) // 2
    px0 = (nx_p - nx) // 2
    sky_pad = np.zeros((ny_p, nx_p))
    sky_pad[py0 : py0 + ny, px0 : px0 + nx] = sky
    grid_shape = (ny_p, nx_p)
    fov_os = (fov_y * ny_p / ny, fov_x * nx_p / nx)
    corr = kb_correction_np(grid_shape, W, beta)

    sky_uv = sky2uv_np(sky_pad * corr)
    uv_px = scale_uv_samples_continuous_np(track, grid_shape, fov_os)
    uv_px_native = scale_uv_samples_continuous_np(track, (ny, nx), (fov_y, fov_x))
    w_vis = visibility_weights_np(uv_px_native, (ny, nx), weighting)

    vis = degrid_visibilities_conv_np(sky_uv, uv_px, grid_shape, W, beta)
    if sigma != 0.0:
        rng = np.random.default_rng(seed)
        vis = vis + rng.normal(0, sigma / np.sqrt(2), vis.shape) + 1j * rng.normal(
            0, sigma / np.sqrt(2), vis.shape
        )
    gridded = grid_visibilities_conv_np(w_vis * vis, uv_px, grid_shape, W, beta)
    psf_grid = grid_visibilities_conv_np(
        w_vis * np.ones(uv_px.shape[0], dtype=np.complex128), uv_px, grid_shape, W, beta
    )
    crop = (slice(py0, py0 + ny), slice(px0, px0 + nx))
    obs_c = (uv2sky_np(gridded) * corr)[crop]
    beam_c = (uv2sky_np(psf_grid) * corr)[crop]
    w_sum = beam_c[ny // 2, nx // 2]
    return obs_c / w_sum, beam_c / w_sum
