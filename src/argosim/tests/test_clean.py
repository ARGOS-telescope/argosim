import numpy as np
import numpy.testing as npt

import argosim.clean as ac


class TestClean:

    beam_path = "src/argosim/tests/data/dirty_beam_sim_single_band.npy"
    shifted_beams_path = "src/argosim/tests/data/shifted_beams.npy"

    peaks_exp = [
        (0.00384521484375, 128, 128, 0, 0),
        (0.00384521484375, 228, 238, 100, 110),
        (0.00384521484375, 108, 208, -20, 80),
        (0.00384521484375, 26, 78, -102, -50),
        (0.00384521484375, 160, 108, 32, -20),
    ]

    pad_odd_exp = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    obs_path = "src/argosim/tests/data/obs_sim_single_band.npy"
    clean_obs_exp_path = "src/argosim/tests/data/clean_observation.npy"
    clean_obs_res_exp_path = "src/argosim/tests/data/clean_observation_res.npy"
    sky_model_clean_exp_path = "src/argosim/tests/data/sky_model_clean.npy"
    sky_model_clean_res_exp_path = "src/argosim/tests/data/sky_model_clean_res.npy"
    clean_decimal = 10

    def test_shift_beam(self):
        shifted_beams = np.load(self.shifted_beams_path, allow_pickle=True).item()
        shfits = shifted_beams["shifts"]
        beam_exp = shifted_beams["shifted_beams"]
        beam = np.load(self.beam_path)

        for shift, shifted_beam in zip(shfits, beam_exp):
            npt.assert_array_equal(
                ac.shift_beam(beam, shift[0], shift[1]),
                shifted_beam,
                err_msg=f"Shift {shift} did not match expected beam.",
            )

    def test_find_peak(self):
        beams = np.load(self.shifted_beams_path, allow_pickle=True).item()[
            "shifted_beams"
        ]
        peaks_out = [ac.find_peak(beam) for beam in beams]
        npt.assert_array_equal(
            peaks_out,
            self.peaks_exp,
            err_msg="Peaks found did not match expected values.",
        )

    def test_pad_odd(self):
        # Odd array
        pad_in = np.ones((5, 5))
        pad_out = ac.pad_odd(pad_in)
        npt.assert_array_equal(
            pad_out,
            pad_in,
            err_msg="Padding an odd array did not return the same array.",
        )
        # Even array
        pad_in = np.ones((4, 4))
        pad_out = ac.pad_odd(pad_in)
        pad_exp = self.pad_odd_exp
        npt.assert_array_equal(
            pad_out,
            pad_exp,
            err_msg="Padding an even array did not return the expected padded array.",
        )

    def test_clean_hogbom(self):
        obs = np.load(self.obs_path)
        beam = np.load(self.beam_path)
        # Without residuals
        I_clean, sky_model = ac.clean_hogbom(
            obs, beam, 0.3, 100, 1e-2, clean_beam_size_px=10
        )
        npt.assert_array_almost_equal(
            I_clean,
            np.load(self.clean_obs_exp_path),
            decimal=self.clean_decimal,
            err_msg="Cleaned observation did not match expected values.",
        )

        npt.assert_array_almost_equal(
            sky_model,
            np.load(self.sky_model_clean_exp_path),
            decimal=self.clean_decimal,
            err_msg="Sky model from clean did not match expected values.",
        )

        # With residuals
        I_clean_res, sky_model_res = ac.clean_hogbom(
            obs, beam, 0.3, 100, 1e-2, clean_beam_size_px=10, res=True
        )
        npt.assert_array_almost_equal(
            I_clean_res,
            np.load(self.clean_obs_res_exp_path),
            decimal=self.clean_decimal,
            err_msg="Cleaned observation with res==True did not match expected values.",
        )

        npt.assert_array_almost_equal(
            sky_model_res,
            np.load(self.sky_model_clean_res_exp_path),
            decimal=self.clean_decimal,
            err_msg="Sky model with res==True from clean did not match expected values.",
        )
