import numpy as np
import numpy.testing as npt

from argosim import antenna_utils as au


class TestAntennaUtils:

    antenna_lims = {"east": (-500.0, 500.0), "north": (-500.0, 500.0), "up": 0.0}

    circ_antenna_exp = np.array(
        [[300.0, 0.0, 0.0], [-150.0, 259.80762114, 0.0], [-150.0, -259.80762114, 0.0]]
    )
    circ_antenna_tolerance = 1e-7

    y_antenna_exp = np.array(
        [
            [100.0, 0.0, 0.0],
            [200.0, 0.0, 0.0],
            [300.0, 0.0, 0.0],
            [400.0, 0.0, 0.0],
            [500.0, 0.0, 0.0],
            [-50.0, 86.60254038, 0.0],
            [-100.0, 173.20508076, 0.0],
            [-150.0, 259.80762114, 0.0],
            [-200.0, 346.41016151, 0.0],
            [-250.0, 433.01270189, 0.0],
            [-50.0, -86.60254038, 0.0],
            [-100.0, -173.20508076, 0.0],
            [-150.0, -259.80762114, 0.0],
            [-200.0, -346.41016151, 0.0],
            [-250.0, -433.01270189, 0.0],
        ]
    )
    y_antenna_tolerance = 1e-7

    random_antenna_seed = 123
    random_antenna_exp = np.array(
        [
            [196.4691856, -213.86066505, 0.0],
            [51.31476908, 219.46896979, 0.0],
            [480.76419838, 184.82973858, 0.0],
        ]
    )
    random_antenna_tolerance = 1e-7

    uni_antenna_span = {"east": 800.0, "north": 800.0, "up": 0.0}
    uni_antenna_test_element = 333
    uni_antenna_333 = np.array([-64.51612903, -141.93548387, 0.0])
    uni_antenna_tolerance = 1e-7

    ENU_antenna_in = np.array([[101.0, 200.5, -33.0]])
    XYZ_antenna_exp = {"X": -142.03409295, "Y": 101.0, "Z": 145.31196248}
    ENU_to_XYZ_decimal_tol = 7
    XYZ_to_uvw_decimal_tol = 7
    uvw_exp = {"u": 478.39762533, "v": 932.45332539, "w": -238.48430193}

    enu_txt_arr_path = "configs/arrays/argos_pathfinder.enu.txt"
    enu_txt_arr_exp = np.array(
        [
            [15.0, 10.0, 0.0],
            [17.0, 35.0, 0.0],
            [7.0, 50.0, 0.0],
            [17.0, 75.0, 0.0],
            [10.0, 183.0, 0.0],
        ]
    )
    latlon_txt_arr_path = "configs/arrays/array_lat_lon.txt"
    latlon_txt_arr_exp = np.array(
        [[-18159.17012419, -5565.97453966, 0.0], [18159.17012419, 5565.97453966, 0.0]]
    )

    random_antenna_baselines_exp = np.array(
        [
            [145.15441652, -433.32963484, 0.0],
            [-284.29501278, -398.69040363, 0.0],
            [-145.15441652, 433.32963484, 0.0],
            [-429.4494293, 34.63923121, 0.0],
            [284.29501278, 398.69040363, 0.0],
            [429.4494293, -34.63923121, 0.0],
        ]
    )

    uv_track_default_exp = np.array(
        [
            [6.87539883e02, -2.05251355e03, -1.34622557e-13],
            [-1.34659464e03, -1.88844101e03, -1.34622557e-13],
            [-6.87539883e02, 2.05251355e03, 1.34622557e-13],
            [-2.03413453e03, 1.64072534e02, 0.00000000e00],
            [1.34659464e03, 1.88844101e03, 0.00000000e00],
            [2.03413453e03, -1.64072534e02, 1.68278197e-14],
        ]
    )
    uv_track_params = (
        35.0 / 180 * np.pi,  # lat
        35.0 / 180 * np.pi,  # dec
        1.5,  # track_time
        -0.5,  # t_0
        3,  # n_times
        1420e6,  # f
        100.0e6,  # df
        2,  # n_freqs
        False,  # multi_band
    )
    uv_track_exp = np.array(
        [
            [509.40157076, -2024.3298598, 62.96406953],
            [-1424.4678981, -1719.55331102, -146.23278783],
            [-509.40157076, 2024.3298598, -62.96406953],
            [-1933.86946886, 304.77654878, -209.19685736],
            [1424.4678981, 1719.55331102, 146.23278783],
            [1933.86946886, -304.77654878, 209.19685736],
            [736.19667995, -1953.96308977, -37.53009284],
            [-1228.04974932, -1869.40026143, 67.77083571],
            [-736.19667995, 1953.96308977, 37.53009284],
            [-1964.24642927, 84.56282834, 105.30092855],
            [1228.04974932, 1869.40026143, -67.77083571],
            [1964.24642927, -84.56282834, -105.30092855],
            [934.70016361, -1859.57020356, -172.33710512],
            [-984.43833736, -1994.38890869, 246.27312317],
            [-934.70016361, 1859.57020356, 172.33710512],
            [-1919.13850097, -134.81870513, 418.6102283],
            [984.43833736, 1994.38890869, -246.27312317],
            [1919.13850097, 134.81870513, -418.6102283],
            [546.58416716, -2172.09116343, 67.55998701],
            [-1528.44365709, -1845.06815124, -156.90671395],
            [-546.58416716, 2172.09116343, -67.55998701],
            [-2075.02782425, 327.0230122, -224.46670096],
            [1528.44365709, 1845.06815124, 156.90671395],
            [2075.02782425, -327.0230122, 224.46670096],
            [789.93366389, -2096.58813282, -40.26951567],
            [-1317.68841715, -2005.85283525, 72.71761204],
            [-789.93366389, 2096.58813282, 40.26951567],
            [-2107.62208104, 90.73529756, 112.98712771],
            [1317.68841715, 2005.85283525, -72.71761204],
            [2107.62208104, -90.73529756, -112.98712771],
            [1002.92645292, -1995.30525491, -184.91645586],
            [-1056.29515031, -2139.96474145, 264.24926355],
            [-1002.92645292, 1995.30525491, 184.91645586],
            [-2059.22160323, -144.65948653, 449.16571942],
            [1056.29515031, 2139.96474145, -264.24926355],
            [2059.22160323, 144.65948653, -449.16571942],
        ]
    )
    multiband_track_shape_exp = (2, 18, 3)

    track_atol = 1e-7

    def test_random_antenna_pos_default(self):

        antenna_pos = au.random_antenna_pos()

        assert np.logical_and(
            antenna_pos[0] >= self.antenna_lims["east"][0],
            antenna_pos[0] <= self.antenna_lims["east"][1],
        )
        assert np.logical_and(
            antenna_pos[1] >= self.antenna_lims["north"][0],
            antenna_pos[1] <= self.antenna_lims["north"][1],
        )
        assert antenna_pos[2] == self.antenna_lims["up"]

    def test_circular_antenna_arr(self):

        circ_antenna_out = au.circular_antenna_arr()

        npt.assert_allclose(
            circ_antenna_out,
            self.circ_antenna_exp,
            atol=self.circ_antenna_tolerance,
            err_msg="Circular antenna outputs do not match.",
        )

    def test_y_antenna_arr(self):

        y_antenna_out = au.y_antenna_arr()

        npt.assert_allclose(
            y_antenna_out,
            self.y_antenna_exp,
            atol=self.y_antenna_tolerance,
            err_msg="Y antenna array outputs do not match.",
        )

    def test_random_antenna_arr(self):

        random_antenna_out = au.random_antenna_arr(seed=self.random_antenna_seed)

        npt.assert_allclose(
            random_antenna_out,
            self.random_antenna_exp,
            atol=self.random_antenna_tolerance,
            err_msg="Random antenna array outputs do not match.",
        )

    def test_uni_antenna_arr(self):

        uni_antenna_out = au.uni_antenna_array()
        uni_antenna_span_out = uni_antenna_out[-1] - uni_antenna_out[0]

        assert uni_antenna_span_out[0] == self.uni_antenna_span["east"]
        assert uni_antenna_span_out[1] == self.uni_antenna_span["north"]
        assert uni_antenna_span_out[2] == self.uni_antenna_span["up"]

        npt.assert_allclose(
            self.uni_antenna_333,
            uni_antenna_out[self.uni_antenna_test_element],
            atol=self.uni_antenna_tolerance,
            err_msg="Uniform antenna output does not match at position {}.".format(
                self.uni_antenna_test_element
            ),
        )

    def test_enu_to_xyz(self):
        x, y, z = au.ENU_to_XYZ(self.ENU_antenna_in)
        npt.assert_almost_equal(
            x,
            self.XYZ_antenna_exp["X"],
            decimal=self.ENU_to_XYZ_decimal_tol,
            err_msg="X coordinate conversion from ENU to XYZ does not match.",
        )
        npt.assert_almost_equal(
            y,
            self.XYZ_antenna_exp["Y"],
            decimal=self.ENU_to_XYZ_decimal_tol,
            err_msg="Y coordinate conversion from ENU to XYZ does not match.",
        )
        npt.assert_almost_equal(
            z,
            self.XYZ_antenna_exp["Z"],
            decimal=self.ENU_to_XYZ_decimal_tol,
            err_msg="Z coordinate conversion from ENU to XYZ does not match.",
        )

    def test_xyz_to_uvw(self):
        dict = self.XYZ_antenna_exp
        x, y, z = dict["X"], dict["Y"], dict["Z"]
        u, v, w = au.XYZ_to_uvw(x, y, z)

        npt.assert_almost_equal(
            u,
            self.uvw_exp["u"],
            decimal=self.XYZ_to_uvw_decimal_tol,
            err_msg="U coordinate conversion from XYZ to UVW does not match.",
        )
        npt.assert_almost_equal(
            v,
            self.uvw_exp["v"],
            decimal=self.XYZ_to_uvw_decimal_tol,
            err_msg="V coordinate conversion from XYZ to UVW does not match.",
        )
        npt.assert_almost_equal(
            w,
            self.uvw_exp["w"],
            decimal=self.XYZ_to_uvw_decimal_tol,
            err_msg="W coordinate conversion from XYZ to UVW does not match.",
        )

    def test_combine_antenna_arr(self):
        arr1 = au.random_antenna_arr()
        arr2 = au.random_antenna_arr()
        arr = au.combine_antenna_arr(arr1, arr2)
        n_antenna_exp = au.random_antenna_arr.__defaults__[0] * 2

        assert (
            len(arr) == n_antenna_exp
        ), "Combined antenna array length does not match expected."

    def test_load_antenna_enu_txt(self):
        arr = au.load_antenna_enu_txt(self.enu_txt_arr_path)
        npt.assert_allclose(
            arr,
            self.enu_txt_arr_exp,
            err_msg="Antenna array loaded from ENU text file does not match expected output.",
        )

    def test_load_antenna_latlon_txt(self):
        arr = au.load_antenna_latlon_txt(self.latlon_txt_arr_path)
        npt.assert_allclose(
            arr,
            self.latlon_txt_arr_exp,
            err_msg="Antenna array loaded from latitude/longitude text file does not match expected output.",
        )

    def test_get_baselines(self):
        baselines_out = au.get_baselines(self.random_antenna_exp)
        npt.assert_allclose(
            baselines_out,
            self.random_antenna_baselines_exp,
            err_msg="Baselines computed from random antenna array do not match expected output.",
        )

    def test_uv_track_default(self):
        track, _ = au.uv_track_multiband(self.random_antenna_baselines_exp)
        npt.assert_allclose(
            track,
            self.uv_track_default_exp,
            atol=self.track_atol,
            err_msg="UV track computed from random antenna baselines does not match expected output.",
        )

    def test_uv_track_params(self):
        track, _ = au.uv_track_multiband(
            self.random_antenna_baselines_exp, *self.uv_track_params
        )
        npt.assert_allclose(
            track,
            self.uv_track_exp,
            atol=self.track_atol,
            err_msg="UV track computed from random antenna baselines does not match expected output.",
        )

    def test_uv_track_multiband(self):
        track, _ = au.uv_track_multiband(
            self.random_antenna_baselines_exp,
            *self.uv_track_params[:-1],
            multi_band=True,
        )
        track_shape_out = track.shape
        npt.assert_equal(
            track_shape_out,
            self.multiband_track_shape_exp,
            err_msg="Multiband UV track shape does not match expected output.",
        )
