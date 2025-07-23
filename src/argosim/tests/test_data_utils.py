import numpy as np
import numpy.testing as npt

import argosim.data_utils as adu


class TestDataUtils:

    gauss_source_path = "src/argosim/tests/data/test_gauss_source.npy"
    gauss_source_decimal = 10

    seed = 24
    cov_expected = np.array([[5.9600173, 2.91338124], [2.91338124, 5.69951205]])
    cov_decimal = 6

    mu_expected = np.array([0.92003461, 0.3990241])
    mu_decimal = 6

    random_source_path = "src/argosim/tests/data/test_random_source.npy"
    random_source_decimal = 10

    sky_model_path = "src/argosim/tests/data/sky_model_exp.npy"
    sky_model_decimal = 10

    def test_gauss_source(self):
        gauss_source_expected = np.load(self.gauss_source_path)
        gauss_source_out = adu.gauss_source()
        npt.assert_almost_equal(
            gauss_source_out,
            gauss_source_expected,
            decimal=self.gauss_source_decimal,
            err_msg="Gaussian source image does not match expected value.",
        )

    def test_sigma2d(self):
        cov_out = adu.sigma2d(seed=self.seed)
        npt.assert_almost_equal(
            cov_out,
            self.cov_expected,
            decimal=self.cov_decimal,
            err_msg="Covariance matrix does not match expected value.",
        )

    def test_mu2d(self):
        mu_out = adu.mu2d(seed=self.seed)
        npt.assert_almost_equal(
            mu_out,
            self.mu_expected,
            decimal=self.mu_decimal,
            err_msg="Mean vector does not match expected value.",
        )

    def test_random_source(self):
        random_source_expected = np.load(self.random_source_path)
        random_source_out = adu.random_source((256, 256), 5, seed=self.seed)
        npt.assert_almost_equal(
            random_source_out,
            random_source_expected,
            decimal=self.random_source_decimal,
            err_msg="Random source image does not match expected value.",
        )

    def test_n_source_sky(self):
        sky_model_expected = np.load(self.sky_model_path)
        sky_model_out = adu.n_source_sky(
            (256, 256), 1.0, [0.01, 0.02, 0.03], [0.4, 0.3, 0.3], seed=332
        )
        npt.assert_almost_equal(
            sky_model_out,
            sky_model_expected,
            decimal=self.sky_model_decimal,
            err_msg="Number of sources in sky model does not match expected value.",
        )
