import numpy as np
import numpy.testing as npt

import argosim
import argosim.imaging_utils as aiu


class TestImagingUtils:

    sky_model_expected_path = "src/argosim/tests/data/sky_model_exp.npy"
    sky_model_uv_expected_path = "src/argosim/tests/data/sky_model_uv_exp.npy"

    pathfinder_uv_track_path = "src/argosim/tests/data/pathfinder_uv_track.npy"
    grid_uv_samples_params = (
        (256, 256),  # Image shape in pixels
        (3.0, 3.0),  # FOV in degrees
    )
    pathfinder_uv_mask_path = (
        "src/argosim/tests/data/argos_pathfinder_uv_mask_binary.npy"
    )
    pathfinder_uv_mask_hist_path = (
        "src/argosim/tests/data/argos_pathfinder_uv_mask_hist.npy"
    )
    pathfinder_uv_mask_weighted_path = (
        "src/argosim/tests/data/argos_pathfinder_uv_mask_weighted.npy"
    )

    sky_uv_w_masked_path = "src/argosim/tests/data/sky_uv_w_masked.npy"

    obs_sim_single_band_path = "src/argosim/tests/data/obs_sim_single_band.npy"
    dirty_beam_sim_single_band_path = (
        "src/argosim/tests/data/dirty_beam_sim_single_band.npy"
    )
    obs_sim_single_band_params = {
        "fov_size": 1.0,
        "sigma": 0.1,
        "seed": 717,
        "freqs": [1.5e9],
    }
    decimal_uv = 4

    KB_W = 7
    KB_beta = 2.0
    KB_x = 1.5
    KB_result = 0.87684005

    KB_correction_shape = (128, 128)
    KB_correction_result_path = "src/argosim/tests/data/kb_correction_result.npy"

    continuous_scaling_result_path = "src/argosim/tests/data/continuous_scaling_result.npy"

    gridded_visibilities_conv_result_path = "src/argosim/tests/data/conv_gridded_result.npy"

    degridded_visibilities_conv_result_path = "src/argosim/tests/data/conv_degridded_result.npy"

    uniform_w_expected_path = "src/argosim/tests/data/conv_vis_weights_uniform.npy"

    vis_sigma = 0.1
    vis_seed = 42
    noisy_visibilities_path = "src/argosim/tests/data/noisy_visibilities.npy"

    def test_sky2uv(self):
        sky = np.load(self.sky_model_expected_path)
        sky_uv = aiu.sky2uv(sky)
        sky_uv_expected = np.load(self.sky_model_uv_expected_path)
        npt.assert_array_almost_equal(
            sky_uv,
            sky_uv_expected,
            decimal=self.decimal_uv,
            err_msg="Sky to UV conversion failed. The resulting UV image does not match the expected output.",
        )

    def test_uv2sky(self):
        sky_uv = np.load(self.sky_model_uv_expected_path)
        sky = aiu.uv2sky(sky_uv)
        sky_expected = np.load(self.sky_model_expected_path)
        npt.assert_array_almost_equal(
            sky,
            sky_expected,
            err_msg="UV to Sky conversion failed. The resulting Sky image does not match the expected output.",
        )

    def test_grid_uv_samples(self):
        track = np.load(self.pathfinder_uv_track_path)
        # Test binary mask
        mask_uv, _ = aiu.grid_uv_samples(
            track, *self.grid_uv_samples_params, mask_type="binary"
        )
        mask_uv_expected = np.load(self.pathfinder_uv_mask_path)
        npt.assert_array_almost_equal(
            mask_uv,
            mask_uv_expected,
            err_msg="Binary mask UV samples do not match the expected output.",
        )

        # Test histogram mask
        mask_uv_hist, _ = aiu.grid_uv_samples(
            track, *self.grid_uv_samples_params, mask_type="histogram"
        )
        mask_uv_hist_expected = np.load(self.pathfinder_uv_mask_hist_path)
        npt.assert_array_almost_equal(
            mask_uv_hist,
            mask_uv_hist_expected,
            err_msg="Histogram mask UV samples do not match the expected output.",
        )

    def test_grid_uv_samples_out_of_range(self):
        track = np.load(self.pathfinder_uv_track_path)
        # catch ValueError for out of range samples
        with npt.assert_raises(ValueError):
            mask_uv, _ = aiu.grid_uv_samples(track, (128, 128), (3.0, 3.0))
        with npt.assert_raises(ValueError):
            mask_uv, _ = aiu.grid_uv_samples(
                track, (64, 64), (1.0, 1.0), mask_type="histogram"
            )

    def test_grid_uv_samples_invalid_mask_type(self):
        track = np.load(self.pathfinder_uv_track_path)
        # catch ValueError for invalid mask type
        with npt.assert_raises(ValueError):
            mask_uv, _ = aiu.grid_uv_samples(
                track, *self.grid_uv_samples_params, mask_type="invalid_mask_type"
            )

    def test_compute_visibilities_grid(self):
        sky_uv = np.load(self.sky_model_uv_expected_path)
        mask_uv = np.load(self.pathfinder_uv_mask_weighted_path)
        sky_uv_w_masked_out = aiu.compute_visibilities_grid(sky_uv, mask_uv)
        sky_uv_w_masked_exp = np.load(self.sky_uv_w_masked_path)
        npt.assert_array_almost_equal(
            sky_uv_w_masked_out,
            sky_uv_w_masked_exp,
            err_msg="Computed grided visibilities do not match the expected output.",
        )

    def test_simulate_dirty_obs_single_band(self):
        sky = np.load(self.sky_model_expected_path)
        track = np.load(self.pathfinder_uv_track_path)
        params = self.obs_sim_single_band_params
        obs_out, dirty_beam_out = aiu.simulate_dirty_observation(
            sky,
            track,
            fov_size=params["fov_size"],
            sigma=params["sigma"],
            seed=params["seed"],
        )
        obs_exp = np.load(self.obs_sim_single_band_path)
        dirty_beam_exp = np.load(self.dirty_beam_sim_single_band_path)

        npt.assert_array_almost_equal(
            obs_out,
            obs_exp,
            err_msg="Simulated dirty observation does not match the expected output.",
        )
        npt.assert_array_almost_equal(
            dirty_beam_out,
            dirty_beam_exp,
            err_msg="Simulated dirty beam does not match the expected output.",
        )

    def test_simulate_dirty_obs_multi_band(self):
        sky = np.load(self.sky_model_expected_path)
        track = np.load(self.pathfinder_uv_track_path)
        params = self.obs_sim_single_band_params
        obs_out_multi, dirty_beam_out_multi = aiu.simulate_dirty_observation(
            sky,
            np.expand_dims(track, axis=0),
            fov_size=params["fov_size"],
            sigma=params["sigma"],
            seed=params["seed"],
            multi_band=True,
            freqs=params["freqs"],
        )
        obs_exp = np.load(self.obs_sim_single_band_path)
        dirty_beam_exp = np.load(self.dirty_beam_sim_single_band_path)

        npt.assert_array_almost_equal(
            obs_out_multi[0],
            obs_exp,
            err_msg="Simulated multi-band dirty observation does not match the expected output.",
        )
        npt.assert_array_almost_equal(
            dirty_beam_out_multi[0],
            dirty_beam_exp,
            err_msg="Simulated multi-band dirty beam does not match the expected output.",
        )

    def test_as_fov_tuple(self):
        fov_deg = self.obs_sim_single_band_params["fov_size"]
        fov_tuple = aiu._as_fov_tuple(fov_deg)
        npt.assert_array_equal(
            fov_tuple,
            (fov_deg, fov_deg),
            err_msg="FOV tuple conversion failed. The resulting tuple does not match the expected output.",
        )

        npt.assert_almost_equal(
            fov_tuple,
            aiu._as_fov_tuple(fov_tuple),
            err_msg="FOV tuple conversion failed. The resulting tuple does not match the expected output.",
        )

    def test_kaiser_bessel(self):
        result = aiu.kaiser_bessel(self.KB_x, self.KB_W, self.KB_beta)
        npt.assert_almost_equal(
            result,
            self.KB_result,
            decimal=6,
            err_msg="Kaiser-Bessel function computation failed. The resulting value does not match the expected output.",
        )

    def test_kb_correction(self):
        result = aiu.kb_correction(self.KB_correction_shape, self.KB_W, self.KB_beta)
        expected_result = np.load(self.KB_correction_result_path)
        npt.assert_array_almost_equal(
            result,
            expected_result,
            decimal=6,
            err_msg="Kaiser-Bessel correction computation failed. The resulting array does not match the expected output.",
        )

    def test_scale_uv_samples_continuous(self):
        uv_samples = np.load(self.pathfinder_uv_track_path)
        sky_uv_shape = self.KB_correction_shape
        fov_size = self.obs_sim_single_band_params["fov_size"]
        result = aiu.scale_uv_samples_continuous(uv_samples, sky_uv_shape, fov_size)
        result_expected = np.load(self.continuous_scaling_result_path)
        npt.assert_array_almost_equal(
            result,
            result_expected,
            decimal=6,
            err_msg="Continuous scaling of UV samples failed. The resulting array does not match the expected output.",
        )

    def test_grid_visibilities_conv(self):
        uv_px = np.load(self.continuous_scaling_result_path)
        vis = np.ones(uv_px.shape[0], dtype=np.complex64)
        result_grid = aiu.grid_visibilities_conv(vis, uv_px, (128, 128), self.KB_W, self.KB_beta)
        result_expected = np.load(self.gridded_visibilities_conv_result_path)
        npt.assert_array_almost_equal(
            result_grid,
            result_expected,
            decimal=6,
            err_msg="Convolutional gridding of visibilities failed. The resulting grid does not match the expected output.",
        )

    def test_degrid_visibilities_conv(self):
        sky_uv = np.load(self.gridded_visibilities_conv_result_path)
        scaling_result = np.load(self.continuous_scaling_result_path)
        result_degridded = aiu.degrid_visibilities_conv(sky_uv, scaling_result, (128, 128), self.KB_W, self.KB_beta)
        result_expected = np.load(self.degridded_visibilities_conv_result_path)
        npt.assert_array_almost_equal(
            result_degridded,
            result_expected,
            decimal=6,
            err_msg="Convolutional degridding of visibilities failed. The resulting array does not match the expected output.",
        )

    def test_visibility_weights(self):

        scaling_result = np.load(self.continuous_scaling_result_path)

        uniform_w_result = aiu._visibility_weights(scaling_result, (128, 128), "uniform")
        uniform_w_expected = np.load(self.uniform_w_expected_path)
        npt.assert_array_almost_equal(
            uniform_w_result,
            uniform_w_expected,
            decimal=6,
            err_msg="Uniform weighting of visibilities failed. The resulting weights do not match the expected output.",
        )

        # Under natural weighting, all weights should be 1.0
        natural_w_result_true =(aiu._visibility_weights(scaling_result, (128, 128), "natural") == 1.).all()
        npt.assert_equal(
            natural_w_result_true,
            np.array(True),
            err_msg="Natural weighting of visibilities failed. The resulting weights do not match the expected output.",
        )

    def test_add_noise_vis(self):
        vis = np.load(self.degridded_visibilities_conv_result_path)
        noisy_vis_out = aiu.add_noise_vis(vis, sigma=self.vis_sigma, seed=self.vis_seed)
        noisy_vis_expected = np.load(self.noisy_visibilities_path)
        npt.assert_array_almost_equal(
            noisy_vis_out,
            noisy_vis_expected,
            decimal=6,
            err_msg="Adding noise to visibilities failed. The resulting noisy visibilities do not match the expected output.",
        )