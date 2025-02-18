import numpy as np
import matplotlib.pyplot as plt

from argosim.antenna_utils import *
from argosim.plot_utils import *
from argosim.data_utils import *
from argosim.clean import *
from argosim.imaging_utils import *
from argosim.beam_utils import *

# Output products
outputs = {}

# Load the antenna positions & compute the baselines
antenna = load_antenna_enu_txt("/home/configs/arrays/argos_pathfinder.enu.txt")
baselines = get_baselines(antenna)

# Source tracking parameters
array_lat = 35.0 / 180 * np.pi  # Heraklion latitud
source_decl = 90.0 / 180 * np.pi
track_time = 2  # 2 hours
delta_t = 300 / 3600  # 300 segs
t_0 = -1
n_times = int(track_time / delta_t)

# Multifrequency parameters
f = 2000e6  # Central frequency: 2 GHz
bandwidth = 2000e6 # frequency range: 1-3 GHz
n_freqs = 128

# Build the sky model
Npx = 512
fov_size = (1., 1.) # max 3 degrees FOV from 6 m parabolic dishes.
deg_size_list = np.array([.03, .02, .02, .02, .02, .01, .01, .01, .005, .005])
source_intensity_list = [1]*len(deg_size_list)
sky = n_source_sky((Npx, Npx), fov_size[0], deg_size_list, source_intensity_list, seed=39807)

# ------------------------------------#
# Simulate the multiband observations #
# ------------------------------------#
# Compute the uv samplig points
track, freqs = uv_track_multiband(
    baselines,
    array_lat,
    source_decl,
    track_time,
    t_0,
    n_times,
    f,
    bandwidth,
    n_freqs,
    multi_band=True,
)
primary_beam = CosCubeBeam(c=0.2, f=1., n_pix=Npx, fov_deg=fov_size[0])
observations, dirty_beams = simulate_dirty_observation(
    sky, 
    track, 
    fov_size[0], 
    multi_band=True, 
    freqs=freqs, 
    beam=primary_beam, 
    sigma=0.2
)

# Plot
fig, ax = plt.subplots(1, 3, figsize=(18,4))
plot_sky(sky, fov_size, ax[0], fig, 'Sky model')
plot_sky(np.sum(observations, axis=0), fov_size, ax[1], fig, 'Multiband observation')
idx = n_freqs//2
plot_sky(observations[idx], fov_size, ax[2], fig, '{:.1f} GHz observation'.format(freqs[idx]/1e9))
plt.savefig('/workdir/pathfinder_multi_band_obs.pdf')

#--------------------------------------#
# Simulate the single band observation #
#--------------------------------------#
track_single_b, f_single = uv_track_multiband(
    baselines, 
    array_lat, 
    source_decl, 
    track_time, 
    t_0, 
    n_times, 
    f, 
    bandwidth, 
    n_freqs, 
    multi_band=False
)
# Plot the array, baselines and uv_tracks
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
plot_antenna_arr(antenna, ax[0], fig, title="Array")
plot_baselines(baselines, ax[1], fig, ENU=True)
plot_baselines(track_single_b, ax[2], fig)
plt.savefig('/workdir/pathfinder_single_band_tracking.pdf')

observation_single, dirty_beam_single = simulate_dirty_observation(
    sky, 
    track_single_b, 
    fov_size[0], 
    multi_band=False, 
    sigma=0.1
)

# Plot
fig, ax = plt.subplots(1, 3, figsize=(18,4))
plot_sky(sky, fov_size, ax[0], fig, 'Sky model')
plot_sky(observation_single, fov_size, ax[1], fig, 'Single band observation')
idx = 1
plot_sky(dirty_beam_single, fov_size, ax[2], fig, 'Dirty beam')
plt.savefig('/workdir/pathfinder_single_band_obs.pdf')

#--------------------------------------#
# Clean the single band observation    #
#--------------------------------------#

obs_clean, _ = clean_hogbom(observation_single, dirty_beam_single, 0.2, 100, 1e-3 , clean_beam_size_px=10)

# Plot
fig, ax = plt.subplots(1, 3, figsize=(18,4))
plot_sky(sky, fov_size, ax[0], fig, 'Sky model')
plot_sky(observation_single, fov_size, ax[1], fig, 'Single band observation')
plot_sky(obs_clean, fov_size, ax[2], fig, 'Cleaned observation')
plt.savefig('/workdir/pathfinder_single_band_cleaned_obs.pdf')


# Save the outputs
outputs['sky'] = sky
outputs['multiband_observations'] = observations
outputs['multiband_dirty_beams'] = dirty_beams
outputs['single_band_observation'] = observation_single
outputs['single_band_dirty_beam'] = dirty_beam_single
outputs['single_band_observation_clean'] = obs_clean
outputs['antenna'] = antenna
outputs['baselines'] = baselines
outputs['track'] = track
outputs['observation_params'] = {
    'array_lat': array_lat,
    'source_decl': source_decl,
    'track_time': track_time,
    't_0': t_0,
    'n_times': n_times,
    'f': f,
    'bandwidth': bandwidth,
    'n_freqs': n_freqs
}

# Save the outputs
np.save('/workdir/pathfinder_multiband_obs.npy', outputs)
