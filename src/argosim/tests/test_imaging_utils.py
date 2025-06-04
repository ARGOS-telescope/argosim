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

    def test_sky2uv(self):
        sky = np.load(self.sky_model_expected_path)
        sky_uv = aiu.sky2uv(sky)
        sky_uv_expected = np.load(self.sky_model_uv_expected_path)
        npt.assert_array_almost_equal(sky_uv, sky_uv_expected)

    def test_uv2sky(self):
        sky_uv = np.load(self.sky_model_uv_expected_path)
        sky = aiu.uv2sky(sky_uv)
        sky_expected = np.load(self.sky_model_expected_path)
        npt.assert_array_almost_equal(sky, sky_expected)