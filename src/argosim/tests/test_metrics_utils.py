import numpy as np
import numpy.testing as npt

import argosim.metrics_utils as amu
from argosim.plot_utils import *


class TestMetricsUtils:

    obs_path = "src/argosim/tests/data/obs_sim_single_band.npy"
    sky_path = "src/argosim/tests/data/sky_model_exp.npy"

    mse_expect_same = 0.0
    mse_expect = 0.00010423388742998882
    mse_decimal = 10

    residual_expect_same = np.zeros_like(np.load(obs_path))
    residual_expect_path = "src/argosim/tests/data/residuals_obs_sky.npy"
    residual_decimal = 10

    rel_mse_expect_same = 0.0
    rel_mse_expect = 1.9389156923644497
    rel_mse_decimal = 10

    beam_path = "src/argosim/tests/data/dirty_beam_sim_single_band.npy"
    fit_beam_expect = {
        "center": (128.0, 128.0),
        "width": 17.084378257616556,
        "height": 5.548935593187696,
        "angle_deg": -14.678876771395638,
        "eccentricity": 0.9457841398924928,
    }
    fit_beam_decimal = 10
    fit_beam_decimal_center = 0

    beam_metrics_expect = {
        "fwhm": (17.084378257616556, 5.548935593187696),
        "eccentricity": 0.9457841398924928,
    }
    beam_metrics_decimal = 10

    def test_mse(self):
        obs = np.load(self.obs_path)
        sky = np.load(self.sky_path)
        mse_out = amu.mse(obs, sky)
        npt.assert_almost_equal(
            mse_out,
            self.mse_expect,
            decimal=self.mse_decimal,
            err_msg="MSE does not match expected value.",
        )

        mse_out_same = amu.mse(obs, obs)
        npt.assert_almost_equal(
            mse_out_same,
            self.mse_expect_same,
            decimal=self.mse_decimal,
            err_msg="MSE of identical images should be zero.",
        )

    def test_residuals(self):
        obs = np.load(self.obs_path)
        sky = np.load(self.sky_path)
        residual_expect = np.load(self.residual_expect_path)
        residuals_out = amu.residuals(obs, sky)
        npt.assert_almost_equal(
            residuals_out,
            residual_expect,
            decimal=self.residual_decimal,
            err_msg="Residuals do not match expected value.",
        )

        residuals_out_same = amu.residuals(obs, obs)
        npt.assert_almost_equal(
            residuals_out_same,
            self.residual_expect_same,
            decimal=self.residual_decimal,
            err_msg="Residuals of identical images should be zero.",
        )

    def test_compute_metrics(self):
        obs = np.load(self.obs_path)
        sky = np.load(self.sky_path)
        residual_expect = np.load(self.residual_expect_path)
        compute_metrics_out = amu.compute_metrics(obs, sky)
        compute_metrics_out_same = amu.compute_metrics(obs, obs)

        npt.assert_almost_equal(
            compute_metrics_out["mse"],
            self.mse_expect,
            decimal=self.mse_decimal,
            err_msg="MSE in compute_metrics does not match expected value.",
        )

        npt.assert_almost_equal(
            compute_metrics_out["rel_mse"],
            self.rel_mse_expect,
            decimal=self.rel_mse_decimal,
            err_msg="Relative MSE in compute_metrics does not match expected value.",
        )

        npt.assert_almost_equal(
            compute_metrics_out_same["rel_mse"],
            self.rel_mse_expect_same,
            decimal=self.rel_mse_decimal,
            err_msg="Relative MSE of identical images should be zero.",
        )

        npt.assert_almost_equal(
            compute_metrics_out["residual"],
            residual_expect,
            decimal=self.residual_decimal,
            err_msg="Residuals in compute_metrics does not match expected value.",
        )

    def test_fit_elliptical_beam(self):
        beam = np.load(self.beam_path)
        fit_beam_out = amu.fit_elliptical_beam(beam)

        for key, dec in [
            ("center", self.fit_beam_decimal_center),
            ("width", self.fit_beam_decimal),
            ("height", self.fit_beam_decimal),
            ("angle_deg", self.fit_beam_decimal),
            ("eccentricity", self.fit_beam_decimal),
        ]:
            npt.assert_almost_equal(
                fit_beam_out[key],
                self.fit_beam_expect[key],
                decimal=dec,
                err_msg=f"{key} of fitted beam does not match expected value.",
            )

    def test_beam_psl(self):
        beam = np.load(self.beam_path)
        psl_db = float(amu.beam_psl(beam))
        # Main lobe holds the global peak, so the peak sidelobe level is <= 0 dB.
        assert np.isfinite(psl_db), "PSL should be a finite value."
        assert psl_db <= 0.0, "PSL should be non-positive (main lobe is the peak)."

    def test_beam_isl(self):
        beam = np.load(self.beam_path)
        isl_db = float(amu.beam_isl(beam))
        assert np.isfinite(isl_db), "ISL should be a finite value."

    def test_compute_fwhm(self):
        beam = np.load(self.beam_path)
        beam_fwhm_out = amu.compute_fwhm(beam)
        npt.assert_almost_equal(
            beam_fwhm_out,
            self.beam_metrics_expect["fwhm"],
            decimal=self.beam_metrics_decimal,
            err_msg="FWHM of fitted beam does not match expected value.",
        )

    def test_compute_eccentricity(self):
        beam = np.load(self.beam_path)
        beam_ecc_out = amu.compute_eccentricity(beam)
        npt.assert_almost_equal(
            beam_ecc_out,
            self.beam_metrics_expect["eccentricity"],
            decimal=self.beam_metrics_decimal,
            err_msg="Eccentricity of fitted beam does not match expected value.",
        )

    def test_compute_beam_metrics(self):
        beam = np.load(self.beam_path)
        beam_metrics_out = amu.compute_beam_metrics(beam)

        # FWHM and eccentricity come from the (deterministic) elliptical fit.
        for key in ["fwhm", "eccentricity"]:
            npt.assert_almost_equal(
                beam_metrics_out[key],
                self.beam_metrics_expect[key],
                decimal=self.beam_metrics_decimal,
                err_msg=f"{key} of fitted beam does not match expected value.",
            )

        # PSL / ISL validated by properties (the watershed metrics are compared
        # against the stale .npy fixture only by property, not golden value).
        assert np.isfinite(beam_metrics_out["psl_db"])
        assert np.isfinite(beam_metrics_out["isl_db"])
        assert beam_metrics_out["psl_db"] <= 0.0
