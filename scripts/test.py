import matplotlib.pyplot as plt
import numpy as np

from argosim.antenna_utils import *
from argosim.beam_utils import *
from argosim.clean import *
from argosim.data_utils import *
from argosim.imaging_utils import *
from argosim.plot_utils import *

antenna = load_antenna_enu_txt("/home/configs/arrays/argos_pathfinder.enu.txt")
# plot_antenna_arr(antenna)
baselines = get_baselines(antenna)
# plot_baselines(baselines, ENU=True)

# Multiband parameters
bandwidth = 2000e6
n_freqs = 11
f = 2000e6  # 2 GHz
# Source tracking parameters
source_decl = 90.0 / 180 * np.pi
array_lat = 35.0 / 180 * np.pi  # Heraklion latitud
track_time = 1
delta_t = 15 / 60  # 900 segs
t_0 = -0.5
n_times = int(track_time / delta_t)

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
    multi_band=False,
)

# Plot the array, baselines and uv_tracks
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
plot_antenna_arr(antenna, ax[0], fig, title="Array")
plot_baselines(baselines, ax[1], fig, ENU=True)
plot_baselines(track, ax[2], fig)
plt.savefig("/workdir/array_baselines.pdf")

Npx = 512
fov_size = (3.0, 3.0)  # 3 degrees FOV from 6 m parabolic dishes.
deg_size_list = np.array([0.1, 0.05, 0.05, 0.03, 0.03, 0.02, 0.02])
source_intensity_list = [1] * len(deg_size_list)
sky = n_source_sky((Npx, Npx), fov_size[0], deg_size_list, source_intensity_list)
# plot_sky(sky, fov_size)
sky_uv = sky2uv(sky)
# plot_sky_uv(sky_uv, fov_size)
uv_mask, uv_sample_indices = grid_uv_samples(track, sky_uv.shape, fov_size)
# plot_sky_uv(uv_mask, fov_size)
vis = compute_visibilities_grid(sky_uv, uv_mask)
# Add noise
vis = add_noise_uv(vis, uv_mask, sigma=0.2)
# plot_sky_uv(vis, fov_size)

obs = uv2sky(vis)
beam = uv2sky(uv_mask)

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
plot_sky(sky, fov_size, ax[0], fig, "Sky")
plot_sky(beam, fov_size, ax[1], fig, "Beam")
plot_sky(obs, fov_size, ax[2], fig, "Observation")
plt.savefig("/workdir/observation.pdf")

# Clean
I_clean, sky_model = clean_hogbom(obs, beam, 0.2, 100, 1e-3, clean_beam_size_px=10)
# Plot clean observation
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
plot_sky(obs, fov_size, ax[0], fig, "Observation")
plot_sky(I_clean, fov_size, ax[1], fig, "Clean observation")
plot_sky(sky, fov_size, ax[2], fig, "Sky model")

plt.savefig("/workdir/clean_observation.pdf")
