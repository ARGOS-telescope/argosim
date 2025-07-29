import cProfile
import pstats
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt

# Import your radio interferometry package
import argosim
import argosim.antenna_utils
import argosim.beam_utils
import argosim.data_utils
import argosim.imaging_utils
import argosim.metrics_utils
import argosim.plot_utils  # Replace with actual package name

# Example functions from your package
def test_tracking(antenna=None):
    """Profile uv-tracking."""
    b_ENU = argosim.antenna_utils.get_baselines(antenna)
    # print("Max baseline length:", np.max(np.linalg.norm(b_ENU, axis=1)))
    track, freqs = argosim.antenna_utils.uv_track_multiband(
        b_ENU=b_ENU,
        lat=35./180*np.pi,
        dec=32./180*np.pi,
        track_time=1.,
        t_0=1.,
        n_times=360,
        f=2.0e9,
        df=1.0e9,
        n_freqs=512,
        multi_band=False
    )
    return track

def test_vis_sampling(Npx=512, fov_deg=1., source_size_deg=0.02, track=None, sigma=0.0):
    """Profile imaging."""
    # sky = argosim.data_utils.n_source_sky((Npx, Npx), fov_deg, [source_size_deg]*5, [1.]*5) 
    sky = argosim.data_utils.gauss_source(nx=Npx, ny=Npx, mu=np.array([0, 0]), sigma=np.eye(2), fwhm_pix=Npx / fov_deg * source_size_deg)
    p_beam = argosim.beam_utils.CosCubeBeam(n_pix=Npx, fov_deg=fov_deg)
    obs, dirty_beam = argosim.imaging_utils.simulate_dirty_observation(sky, track, fov_deg, beam=p_beam, sigma=sigma)
    return sky, dirty_beam, obs

def profile_function(func, *args, **kwargs):
    """Profiles a given function and prints the top 10 most time-consuming calls."""
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
    stats.print_stats(20)  # Show top 10 most expensive function calls
    print(f"\nProfile results for {func.__name__}:\n")
    stats_stream = s.getvalue()
    print(stats_stream)
    # Save stream to text file
    with open(f"{func.__name__}_profile.txt", "w") as f:
        f.write(stats_stream)
    return result  # Return the function output if needed

# Run profiling
if __name__ == "__main__":
    Npx = 4096
    fov_deg = 3.
    source_size_deg=0.01
    sigma = 0.0
    
    # # warm up (for jit)
    # track_warm = test_tracking()
    # _, _, _ = test_vis_sampling(track=track_warm, Npx=Npx, fov_deg=fov_deg, source_size_deg=source_size_deg)

    antenna = argosim.antenna_utils.load_antenna_enu_txt("../../configs/arrays/vlac.enu.txt")
    track = profile_function(test_tracking, antenna=antenna)
    mask_grid, _ = argosim.imaging_utils.grid_uv_samples(track, sky_uv_shape=(Npx, Npx), fov_size=(fov_deg, fov_deg))
    fig, ax = plt.subplots()
    ax.imshow(np.abs(mask_grid), cmap='gray', origin='lower')
    # argosim.plot_utils.plot_baselines(track, ax=ax[0], fig=fig)
    # plt.show()
    plt.savefig('track.pdf')
    plt.close()
    print("Track shape:", track.shape)

    sky, dirty_beam, obs = profile_function(test_vis_sampling, track=track, Npx=Npx, fov_deg=fov_deg, source_size_deg=source_size_deg, sigma=sigma)
    obs /= np.max(obs)*np.max(sky)  # Normalize the observation to the max of the sky

    # Centre crop the images
    crop_scale = .1
    crop = int(2 / crop_scale)
    sky_crop = sky[Npx//2-Npx//crop:Npx//2+Npx//crop, Npx//2-Npx//crop:Npx//2+Npx//crop]
    dirty_beam_crop = dirty_beam[Npx//2-Npx//crop:Npx//2+Npx//crop, Npx//2-Npx//crop:Npx//2+Npx//crop]
    obs_crop = obs[Npx//2-Npx//crop:Npx//2+Npx//crop, Npx//2-Npx//crop:Npx//2+Npx//crop] / np.max(obs) * np.max(sky)

    metrics = argosim.metrics_utils.compute_metrics(sky_crop, obs_crop)

    print("Metrics:")
    print("mse: {}".format(metrics['mse']))
    print("rel_mse: {}".format(metrics['rel_mse']))

    fov_plot = (fov_deg * crop_scale, fov_deg * crop_scale)
    fig, ax = plt.subplots(1,4, figsize=(15,4))
    argosim.plot_utils.plot_sky(sky_crop, fov_plot, ax[0], fig, 'Sky')
    argosim.plot_utils.plot_sky(dirty_beam_crop, fov_plot, ax[1], fig, 'Dirty Beam')
    argosim.plot_utils.plot_sky(obs_crop, fov_plot, ax[2], fig, 'Observation')
    argosim.plot_utils.plot_sky(np.abs(obs_crop-sky_crop), fov_plot, ax[3], fig, 'Residual RMSE: {:.2f}'.format(metrics['rel_mse']))
    plt.tight_layout()
    plt.savefig('observation.pdf')
    plt.close()

    print("Finished profiling.")
    print("Bye!")
    