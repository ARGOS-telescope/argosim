import os
import argparse

# --- Arguments & backend selection -------------------------------------------
# Parsed with argparse (stdlib, no JAX import). The backend must be resolved
# BEFORE importing JAX/argosim, since the platform is fixed at initialisation.
# Running both cpu and gpu on the same cluster node makes the comparison fair
# (identical hardware/allocation, only the device changes). The size knobs
# (--npx, --n_times, --n_freqs) let the workload be scaled to fit memory:
# peak memory is dominated by the visibility count n_baselines*n_times*n_freqs
# (~0.18 GB per million visibilities), plus a smaller grid term that grows with
# Npx**2.
parser = argparse.ArgumentParser(description="Profile argosim KB imaging on CPU, GPU or NumPy.")
parser.add_argument("backend", choices=["cpu", "gpu", "numpy"],
                    help="Compute backend: 'cpu'/'gpu' use JAX; 'numpy' uses the pure-NumPy port.")
parser.add_argument("--npx", type=int, default=4096, help="Image size in pixels (Npx x Npx).")
parser.add_argument("--track_time", type=float, default=4.0, help="Observation window in hours (centred on transit).")
parser.add_argument("--n_times", type=int, default=360, help="Number of time samples distributed across the window.")
parser.add_argument("--n_freqs", type=int, default=128, help="Number of frequency channels in the uv track.")
parser.add_argument("--fov", type=float, default=3.0, help="Field of view in degrees.")
parser.add_argument("--outdir", type=str, default="out", help="Directory for all outputs (profiles + figures).")
args = parser.parse_args()

os.environ["JAX_PLATFORMS"] = "cuda" if args.backend == "gpu" else "cpu"
USE_NUMPY = args.backend == "numpy"
BACKEND_LABEL = {"cpu": "jaxCPU", "gpu": "jaxGPU", "numpy": "NumPy"}[args.backend]
OUTDIR = args.outdir
os.makedirs(OUTDIR, exist_ok=True)
# -----------------------------------------------------------------------------

import cProfile
import pstats
import numpy as np
import jax
from io import StringIO
import matplotlib.pyplot as plt

# Import argosim
import argosim
import argosim.antenna_utils
import argosim.data_utils
import argosim.imaging_utils
import argosim.plot_utils 

# Pure-NumPy port of uv-tracking + KB imaging (only used for --backend numpy).
import profiling_numpy as pnp

# Example functions from your package
def test_tracking(antenna=None, track_time=4., n_times=360, n_freqs=128):
    """Profile uv-tracking.

    ``n_times`` time samples are distributed across a ``track_time``-hour window
    centred on meridian transit (``t_0 = -track_time/2``). ``track_time`` sets
    the observation duration (uv-arc length); ``n_times`` only sets the sampling
    density within it, so reducing ``n_times`` (e.g. for memory) coarsens the
    sampling but keeps the full arcs, rather than shortening the observation.
    """
    b_ENU = np.asarray(argosim.antenna_utils.get_baselines(antenna))
    kw = dict(lat=35./180*np.pi, dec=32./180*np.pi, track_time=track_time,
              t_0=-track_time / 2., n_times=n_times, f=2.0e9, df=1.0e9, n_freqs=n_freqs)
    if USE_NUMPY:
        track = pnp.uv_track_multiband_np(b_ENU, **kw)
    else:
        track, freqs = argosim.antenna_utils.uv_track_multiband(b_ENU=b_ENU, multi_band=False, **kw)
    # Force device->host sync so the profiled time captures the actual compute,
    # not just the asynchronous JAX dispatch (critical for CPU vs GPU timings).
    return np.asarray(track)

def test_vis_sampling(Npx=512, fov_deg=1., source_size_deg=0.02, track=None, sigma=0.0):
    """Profile imaging."""
    # Properly flux-normalised sky (n_source_sky, norm='flux' by default).
    sky = np.asarray(argosim.data_utils.n_source_sky(
        (Npx, Npx), fov_deg, [source_size_deg], [1.0], seed=0
    ))
    if USE_NUMPY:
        obs, dirty_beam = pnp.simulate_dirty_observation_np(sky, track, fov_deg, sigma=sigma)
    else:
        obs, dirty_beam = argosim.imaging_utils.simulate_dirty_observation(sky, track, fov_deg, sigma=sigma)
    # Force device->host sync so the profiled time captures the actual compute
    # (JAX is asynchronous; without this the profiler stops before the device
    # finishes, understating CPU and especially GPU timings).
    return np.asarray(sky), np.asarray(dirty_beam), np.asarray(obs)

def profile_function(func, *args, tag="", **kwargs):
    """Profiles a given function and prints the top 20 most time-consuming calls.

    ``tag`` is prepended to the output filename to label the run (e.g. the size
    and backend), so results from different backends do not overwrite each other.
    """
    profiler = cProfile.Profile()

    # Burn-in run
    _ = func(*args, **kwargs)  # Run the function once to warm up

    # Profile the function
    profiler.enable()
    result = func(*args, **kwargs)  # Run the function
    profiler.disable()

    # Print profiling results
    s = StringIO()
    stats = pstats.Stats(profiler, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats(20)  # Show top 20 most expensive function calls
    print(f"\nProfile results for {func.__name__}:\n")
    stats_stream = s.getvalue()
    print(stats_stream)
    # Save stream to text file
    prefix = f"{tag}_" if tag else ""
    with open(os.path.join(OUTDIR, f"{prefix}{func.__name__}_profile.txt"), "w") as f:
        f.write(stats_stream)
    return result  # Return the function output if needed

# Run profiling
if __name__ == "__main__":
    Npx = args.npx
    fov_deg = args.fov
    source_size_deg=0.01
    sigma = 0.0

    # Report the backend (and, for JAX, the device it actually resolved to).
    if USE_NUMPY:
        print(f"Backend: {BACKEND_LABEL} (pure NumPy)")
    else:
        try:
            devices = jax.devices()
        except RuntimeError as err:
            raise SystemExit(
                f"Requested backend '{args.backend}' is not available:\n  {err}\n"
                'For GPU, install a CUDA-enabled jaxlib: pip install "argosim[cuda12]".'
            )
        print(f"Backend: {BACKEND_LABEL} | JAX devices: {devices}")
    print(f"Config: Npx={Npx} track_time={args.track_time}h n_times={args.n_times} n_freqs={args.n_freqs} fov={fov_deg}")
    run_tag = f"test_Npx{Npx}_tt{args.track_time}_nt{args.n_times}_nf{args.n_freqs}_{BACKEND_LABEL}"

    # # warm up (for jit)
    # track_warm = test_tracking()
    # _, _, _ = test_vis_sampling(track=track_warm, Npx=Npx, fov_deg=fov_deg, source_size_deg=source_size_deg)

    antenna = argosim.antenna_utils.load_antenna_enu_txt("../../configs/arrays/vlac.enu.txt")
    track = profile_function(test_tracking, antenna=antenna, track_time=args.track_time, n_times=args.n_times, n_freqs=args.n_freqs, tag=run_tag)
    # KB-gridded uv coverage (native resolution for the plot: oversampling=1).
    uv_grid, _ = argosim.imaging_utils.grid_sampling_function(track, fov_deg, Npx, method='kb', oversampling=1)
    fig, ax = plt.subplots()
    argosim.plot_utils.plot_sky_uv(uv_grid, (fov_deg, fov_deg), ax=ax, fig=fig, scale='log', cbar=True, title='uv coverage')
    # plt.show()
    plt.savefig(os.path.join(OUTDIR, f'{run_tag}_track.pdf'))
    plt.close()
    print("Track shape:", track.shape)

    sky, dirty_beam, obs = profile_function(test_vis_sampling, track=track, Npx=Npx, fov_deg=fov_deg, source_size_deg=source_size_deg, sigma=sigma, tag=run_tag)

    # Crop for plotting: n_source_sky places the source at a random position,
    # crop around the source (sky peak) rather than the image centre.
    crop_scale = .1
    hw = int(Npx * crop_scale / 2)  # half-window in pixels

    def crop_around(img, cy, cx):
        cy = min(max(cy, hw), img.shape[0] - hw)  # keep a full window in bounds
        cx = min(max(cx, hw), img.shape[1] - hw)
        return img[cy - hw:cy + hw, cx - hw:cx + hw]

    sy, sx = np.unravel_index(np.argmax(sky), sky.shape)  # source location
    sky_crop = crop_around(sky, sy, sx)
    obs_crop = crop_around(obs, sy, sx)
    beam_crop = crop_around(dirty_beam, Npx // 2, Npx // 2)  # PSF is centred

    fov_plot = (fov_deg * crop_scale, fov_deg * crop_scale)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    argosim.plot_utils.plot_sky(sky_crop, fov_plot, ax[0], fig, 'Sky')
    argosim.plot_utils.plot_sky(beam_crop, fov_plot, ax[1], fig, 'Dirty Beam (PSF)')
    argosim.plot_utils.plot_sky(obs_crop, fov_plot, ax[2], fig, 'Observation')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, f'{run_tag}_observation.pdf'))
    plt.close()

    print("Finished profiling.")
    print("Bye!")
    