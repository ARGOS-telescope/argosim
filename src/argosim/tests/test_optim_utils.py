import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

import argosim.antenna_utils as aau
import argosim.optim_utils as aou


class TestOptimUtils:

    pathfinder_uv_track_path = "src/argosim/tests/data/pathfinder_uv_track.npy"
    beam_expected_path = "src/argosim/tests/data/antenna_to_beam.npy"
    complementary_beam_path = "src/argosim/tests/data/dirty_beam_sim_single_band.npy"
    decimal_beam = 2
    loss_expected = 0.00516675

    spacing_penalty_d = 50
    spacing_penalty_loss = 60.023014
    radius_penalty_r = 1.5e3
    radius_penalty_loss = 1.5782342
    centroid_penalty_loss = 4.705479e-11

    optim_target_beam_path = "src/argosim/tests/data/optim_target_beam.npy"
    optim_result_path = "src/argosim/tests/data/optimisation_result.npy"

    decimal_array = 0

    def test_antenna_to_beam(self):
        antenna = np.load(self.pathfinder_uv_track_path)
        config = aou.ObservationConfig()
        beam = aou.antenna_to_beam(antenna, config)
        npt.assert_array_almost_equal(
            beam,
            np.load(self.beam_expected_path),
            decimal=self.decimal_beam,
            err_msg="Antenna to beam conversion failed. The resulting beam does not match the expected output.",
        )

    def test_beam_mse_loss(self):
        beam1 = np.load(self.beam_expected_path)
        beam2 = np.load(self.complementary_beam_path)
        loss = aou.beam_mse_loss(beam1, beam2)
        npt.assert_almost_equal(
            loss,
            self.loss_expected,
            err_msg="Beam MSE loss computation failed. The resulting loss does not match the expected output.",
        )

    def test_spacing_penalty(self):
        antenna = np.load(self.pathfinder_uv_track_path)
        d = self.spacing_penalty_d
        penalty = aou.spacing_penalty(antenna, d)
        npt.assert_almost_equal(
            penalty,
            self.spacing_penalty_loss,
            decimal=4,
            err_msg="Spacing penalty computation failed. The resulting penalty does not match the expected output.",
        )

    def test_radius_penalty(self):
        antenna = np.load(self.pathfinder_uv_track_path)
        r = self.radius_penalty_r
        penalty = aou.radius_penalty(antenna, r)
        npt.assert_almost_equal(
            penalty,
            self.radius_penalty_loss,
            err_msg="Radius penalty computation failed. The resulting penalty does not match the expected output.",
        )

    def test_centroid_penalty(self):
        antenna = np.load(self.pathfinder_uv_track_path)
        penalty = aou.centroid_penalty(antenna)
        npt.assert_almost_equal(
            penalty,
            self.centroid_penalty_loss,
            err_msg="Centroid penalty computation failed. The resulting penalty does not match the expected output.",
        )

    def test_optimise_array(self):
        antenna_init = aau.y_antenna_arr(10, r=1.5e3)
        config = aou.ObservationConfig(
            npx=256,
            fov=0.03,
            lat=35.0 / 180 * np.pi,
            dec=32.0 / 180 * np.pi,
            freq=2.0e9,
            bw=1.0e9,
            n_freqs=1,
            n_times=20,
            track_h=1.0,
            t_0=1.0,
            kernel_W=7,
        )
        forward_fn = lambda antenna_: aou.antenna_to_beam(antenna_, config)
        target_beam = jnp.load(self.optim_target_beam_path)
        constraints = [
            aou.Constraint("spacing", lambda a: aou.spacing_penalty(a, 1), weight=1e-4),
            aou.Constraint(
                "radius", lambda a: aou.radius_penalty(a, 10e3), weight=1e-6
            ),
            aou.Constraint("centroid", aou.centroid_penalty, weight=1e-7),
        ]
        loss_fn = aou.make_loss_fn(
            forward_fn=forward_fn,
            target_loss_fn=lambda beam_: aou.beam_mse_loss(beam_, target_beam),
            constraints=constraints,
        )
        schedule = aou.exp_decay_schedule(
            init_lr=8.0, end_lr=4.0, transition_steps=500, decay_rate=0.7
        )
        optimizer = aou.make_optimizer("adam", lr=schedule)
        n_steps = 1000
        snapshot_steps = [5, 10, 20, 50, 100, 200, 500, 990]

        result = aou.optimise_array(
            antenna_init=antenna_init,
            loss_fn=loss_fn,
            optimizer=optimizer,
            n_steps=n_steps,
            snapshot_steps=snapshot_steps,
            freeze_u=True,
            verbose_every=0,
        )

        result_expected = np.load(self.optim_result_path, allow_pickle=True).item()

        npt.assert_array_almost_equal(
            result["antenna"],
            result_expected["antenna"],
            decimal=self.decimal_array,
            err_msg="Optimisation result does not match the expected output.",
        )
