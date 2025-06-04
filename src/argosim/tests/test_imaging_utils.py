import numpy as np
import numpy.testing as npt

import argosim.imaging_utils as aiu

class TestImagingUtils:

    sky_model_params = (
        (128, 128), # shape_px
        1.0, # fov
        [0.01, 0.02, 0.03], # deg_size_list
        [0.4, 0.3, 0.3], # source_intensity_list
        332, # seed
    )
    sky_model_expected_path = "src/argosim/tests/data/sky_model_exp.npy"
    sky_model_uv_expected_path = "src/argosim/tests/data/sky_model_uv_exp.npy"

    pathfinder_uv_track_path = "src/argosim/tests/data/pathfinder_uv_track.npy"
    grid_uv_samples_params = (
        (256, 256), # Image shape in pixels
        (3., 3.) # FOV in degrees
    )
    pathfinder_uv_mask_path = "src/argosim/tests/data/argos_pathfinder_uv_mask.npy"
    pathfinder_uv_mask_hist_path = "src/argosim/tests/data/argos_pathfinder_uv_mask_hist.npy"
    pathfinder_uv_mask_weighted_path = "src/argosim/tests/data/argos_pathfinder_uv_mask_weighted.npy"
    uv_weights_path = "src/argosim/tests/data/uv_sampling_weights.npy"

    def test_sky2uv(self):
        sky = np.load(self.sky_model_expected_path)
        sky_uv = aiu.sky2uv(sky)
        sky_uv_expected = np.load(self.sky_model_uv_expected_path)
        npt.assert_array_almost_equal(
            sky_uv, 
            sky_uv_expected,
            err_msg="Sky to UV conversion failed. The resulting UV image does not match the expected output."
        )

    def test_uv2sky(self):
        sky_uv = np.load(self.sky_model_uv_expected_path)
        sky = aiu.uv2sky(sky_uv)
        sky_expected = np.load(self.sky_model_expected_path)
        npt.assert_array_almost_equal(
            sky, 
            sky_expected,
            err_msg="UV to Sky conversion failed. The resulting Sky image does not match the expected output."
        )

    def test_grid_uv_samples(self):
        track = np.load(self.pathfinder_uv_track_path)
        # Test binary mask
        mask_uv, _ = aiu.grid_uv_samples(track, *self.grid_uv_samples_params, mask_type='binary')
        mask_uv_expected = np.load(self.pathfinder_uv_mask_path)
        npt.assert_array_almost_equal(
            mask_uv, 
            mask_uv_expected,
            err_msg="Binary mask UV samples do not match the expected output."
        )

        # Test histogram mask
        mask_uv_hist, _ = aiu.grid_uv_samples(track, *self.grid_uv_samples_params, mask_type='histogram')
        mask_uv_hist_expected = np.load(self.pathfinder_uv_mask_hist_path)
        npt.assert_array_almost_equal(
            mask_uv_hist, 
            mask_uv_hist_expected,
            err_msg="Histogram mask UV samples do not match the expected output."
        )

        # Test weighted mask
        weights = np.load(self.uv_weights_path)
        mask_uv_weighted, _ = aiu.grid_uv_samples(track, *self.grid_uv_samples_params, mask_type='weighted', weights=weights)
        mask_uv_weighted_expected = np.load(self.pathfinder_uv_mask_weighted_path)
        npt.assert_array_almost_equal(
            mask_uv_weighted, 
            mask_uv_weighted_expected,
            err_msg="Weighted mask UV samples do not match the expected output."
        )

    def test_grid_uv_samples_out_of_range(self):
        track = np.load(self.pathfinder_uv_track_path)
        # catch ValueError for out of range samples
        with npt.assert_raises(ValueError):
            mask_uv, _ = aiu.grid_uv_samples(track, (128, 128), (3., 3.))
        with npt.assert_raises(ValueError):
            mask_uv, _ = aiu.grid_uv_samples(track, (64, 64), (1., 1.), mask_type='histogram')

    def test_grid_uv_samples_invalid_mask_type(self):
        track = np.load(self.pathfinder_uv_track_path)
        # catch ValueError for invalid mask type
        with npt.assert_raises(ValueError):
            mask_uv, _ = aiu.grid_uv_samples(track, *self.grid_uv_samples_params, mask_type='invalid_mask_type')

    def test_grid_uv_samples_missing_weights(self):
        track = np.load(self.pathfinder_uv_track_path)
        # catch AssertionError for missing weights when mask_type is 'weighted'
        with npt.assert_raises(AssertionError):
            mask_uv, _ = aiu.grid_uv_samples(track, *self.grid_uv_samples_params, mask_type='weighted')
        
