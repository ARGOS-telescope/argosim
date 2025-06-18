import numpy as np
import numpy.testing as npt

import argosim.beam_utils as abu


class TestBeamUtils:
    beam_uv_coords = (3.3, -10.5)
    beam_value_exp = -0.6757983092461403
    beam_value_exp_decimal = 10

    r_fov_exp = 1.9644846081521368
    r_fov_exp_decimal = 10

    beam_edge_exp = 6.544984694978735
    beam_edge_exp_decimal = 10

    fov_solid_angle_exp = 12.124033663969739
    fov_solid_angle_exp_decimal = 10

    beam_cos_cube_path = "src/argosim/tests/data/beam_cos_cube.npy"

    def test_call(self):
        beam = abu.CosCubeBeam()
        val_out = beam(*self.beam_uv_coords)
        npt.assert_almost_equal(
            val_out,
            self.beam_value_exp,
            decimal=self.beam_value_exp_decimal,
            err_msg="Beam value does not match expected value.",
        )

    def test_r_fov(self):
        beam = abu.CosCubeBeam()
        r_fov_out = beam.r_fov()
        npt.assert_almost_equal(
            r_fov_out,
            self.r_fov_exp,
            decimal=self.r_fov_exp_decimal,
            err_msg="FOV radius does not match expected value.",
        )

    def test_beam_edge(self):
        beam = abu.CosCubeBeam()
        beam_edge_out = beam.beam_edge()
        npt.assert_almost_equal(
            beam_edge_out,
            self.beam_edge_exp,
            decimal=self.beam_edge_exp_decimal,
            err_msg="Beam edge does not match expected value.",
        )

    def test_fov_solid_angle(self):
        beam = abu.CosCubeBeam()
        fov_solid_angle_out = beam.fov_solid_angle()
        npt.assert_almost_equal(
            fov_solid_angle_out,
            self.fov_solid_angle_exp,
            decimal=self.fov_solid_angle_exp_decimal,
            err_msg="FOV solid angle does not match expected value.",
        )

    def test_get_beam(self):
        beam = abu.CosCubeBeam()
        beam_image_out = beam.get_beam()
        beam_image_exp = np.load(self.beam_cos_cube_path)
        npt.assert_almost_equal(
            beam_image_out,
            beam_image_exp,
            decimal=self.beam_value_exp_decimal,
            err_msg="Beam image does not match expected value.",
        )
