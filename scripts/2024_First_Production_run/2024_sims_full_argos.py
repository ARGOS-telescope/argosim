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
output_dir = '/workdir/'

# build the antenna with the argosim library
# random_antenna_arr(n_antenna=3, E_lim=1000, N_lim=1000, U_lim=0) -> [[E1, N1, U1], [E2, N2, U2], ...]
E_lim, N_lim, n = 4000., 4000., 176
array_random = random_antenna_arr(n_antenna=n, E_lim=E_lim, N_lim=N_lim)

# uni_antenna_array(n_antenna_E=32, n_antenna_N=32, E_lim=800, N_lim=800, U_lim=0) -> [[E1, N1, U1], [E2, N2, U2], ...]
E_lim_grid, N_lim_grid, n_grid = 800., 800., 32
array_grid = uni_antenna_array(n_antenna_E=n_grid, n_antenna_N=n_grid, E_lim=E_lim_grid, N_lim=N_lim_grid)

# Concatenate the random arrays and the uniform ones
antenna = np.concatenate((array_grid,array_random),axis=0)

baselines = get_baselines(antenna)

# Source tracking parameters
array_lat = 35.0 / 180 * np.pi  # Heraklion latitud
source_decl = 90.0 / 180 * np.pi
track_time = 0.5  # 30 min
delta_t = 10/60  # 5 min
t_0 = -.5
n_times = int(track_time / delta_t)

# Multifrequency parameters
f = 2000e6  # Central frequency: 2 GHz
bandwidth = 2000e6 # frequency range: 1-3 GHz
n_freqs = 11

# Build the sky model
Npx = 4096
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
plt.savefig(output_dir+'full_argos_multi_band_obs.pdf')

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
plt.savefig(output_dir+'full_argos_single_band_tracking.pdf')

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
plt.savefig(output_dir+'full_argos_single_band_obs.pdf')

#--------------------------------------#
# Clean the single band observation    #
#--------------------------------------#

obs_clean, _ = clean_hogbom(observation_single, dirty_beam_single, 0.2, 100, 1e-3 , clean_beam_size_px=10)

# Plot
fig, ax = plt.subplots(1, 3, figsize=(18,4))
plot_sky(sky, fov_size, ax[0], fig, 'Sky model')
plot_sky(observation_single, fov_size, ax[1], fig, 'Single band observation')
plot_sky(obs_clean, fov_size, ax[2], fig, 'Cleaned observation')
plt.savefig(output_dir+'full_argos_single_band_cleaned_obs.pdf')


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
np.save(output_dir+'full_argos_multiband_obs.npy', outputs)
