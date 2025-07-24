"""Antenna utils.

This module contains functions to generate antenna arrays, compute its baselines,
perform aperture synthesis, obtain uv-coverage and get observations from sky models.

:Authors: Ezequiel Centofanti <ezequiel.centofanti@cea.fr>
          Samuel Gullin <gullin@ia.forth.gr>

"""

import jax.numpy as jnp
import numpy as np
import numpy.random as rnd
from jax import jit, vmap

from argosim.rand_utils import local_seed

########################################
#      Generate antenna positions      #
########################################


def random_antenna_pos(E_lim=1000.0, N_lim=1000.0, U_lim=0.0, seed=None):
    """Random antenna pos.

    Function to generate a random antenna location in ENU coordinates.
    Antenna lies in the range: [-E_lims/2, E_lims/2]x[-N_lims/2, N_lims/2]x[0, U_lims].

    Parameters
    ----------
    E_lim : float
        The east coordinate span width of the antenna position in meters.
    N_lim : float
        The north coordinate span width of the antenna position in meters.
    U_lim : float
        The up coordinate span width of the antenna position in meters.
    seed : int
        Optional seed to set.

    Returns
    -------
    antenna_pos : np.ndarray
        The antenna position in ENU coordinates.
    """
    with local_seed(seed):
        random_coords = rnd.random_sample(3)

    # Return (x,y) random location for single dish
    return (
        random_coords * np.array([E_lim, N_lim, U_lim])
        - np.array([E_lim, N_lim, 0.0]) / 2
    )


def circular_antenna_arr(n_antenna=3, r=300.0):
    """Circular antenna arr.

    Function to generate a circular antenna array. Antennas lie in a circumference
    of radius 'r' from the center [0,0] and are equally spaced.

    Parameters
    ----------
    n_antenna : int
        The number of antennas in the array.
    r : float
        The radius of the antenna array in meters.

    Returns
    -------
    antenna_arr : np.ndarray
        The antenna array positions in ENU coordinates.
    """
    # Return list of 'n' antenna locations (x_i, y_i) equally spaced over a 'r' radius circumference.
    return np.array(
        [
            [np.cos(angle) * r, np.sin(angle) * r, 0.0]
            for angle in [2 * np.pi / n_antenna * i for i in range(n_antenna)]
        ]
    )


def y_antenna_arr(n_antenna=5, r=500.0, alpha=0.0):
    """Y antenna arr.

    Function to generate a Y-shaped antenna array. Antennas lie equispaced in three radial arms
    of 120 degrees each.

    Parameters
    ----------
    n_antenna : int
        The number of antennas per arm.
    r : float
        The radius of the antenna array in meters.
    alpha : float
        The angle of the first arm.

    Returns
    -------
    antenna_arr : np.ndarray
        The antenna array positions in ENU coordinates.
    """
    # Return list of 'n' antenna locations (x_i, y_i) equispaced on three (120 deg) radial arms.
    step = r / n_antenna
    return np.array(
        [
            [
                np.array(
                    [
                        (i + 1) * step * np.cos(angle / 180 * np.pi),
                        (i + 1) * step * np.sin(angle / 180 * np.pi),
                        0.0,
                    ]
                )
                for i in range(n_antenna)
            ]
            for angle in [alpha, alpha + 120, alpha + 240]
        ]
    ).reshape((3 * n_antenna, 3))


def random_antenna_arr(n_antenna=3, E_lim=1000.0, N_lim=1000.0, U_lim=0.0, seed=None):
    """Random antenna arr.

    Function to generate a random antenna array. Antennas lie randomly distributed
    in the range: [-E_lims/2, E_lims/2]x[-N_lims/2, N_lims/2]x[0, U_lims].

    Parameters
    ----------
    n_antenna : int
        The number of antennas in the array.
    E_lim : float
        The east coordinate span width of the antenna positions in meters.
    N_lim : float
        The north coordinate span width of the antenna positions in meters.
    U_lim : float
        The up coordinate span width of the antenna positions in meters.
    seed : int
        Optional seed to set.

    Returns
    -------
    antenna_arr : np.ndarray
        The antenna array positions in ENU coordinates.
    """
    with local_seed(seed):
        # Make a list of 'n' antenna locations (x_i, y_i) randomly distributed.
        positions = [random_antenna_pos(E_lim, N_lim, U_lim) for i in range(n_antenna)]

    return np.array(positions)


def uni_antenna_array(
    n_antenna_E=32, n_antenna_N=32, E_lim=800.0, N_lim=800.0, U_lim=0.0
):
    """Uniform (grid) antenna arr.

    Function to generate a uniform antenna array. Antennas lie uniformely distributed
    in the range, and center on [0,0,U_lim]: [-E_lim/2, E_lim/2]x[-N_lim/2, N_lim/2]x[U_lim].
    Only allow to generate a grid at a fixed height U_lim (fixed Up component).

    Parameters
    ----------
    n_antenna_E : int
        The number of antennas in the North direction (vertical).
    n_antenna_N : int
        The number of antennas in the East direction (horizontal).
    E_lim : float
        The east coordinate span width of the antenna positions in meters.
    N_lim : float
        The north coordinate span width of the antenna positions in meters.
    U_lim : float
        The up coordinate span width of the antenna positions in meters.

    Returns
    -------
    antenna_arr : np.ndarray
        The antenna array positions in ENU coordinates.
    """
    # Return list of 'n_antenna_E x n_antenna_N' antenna locations (x_i, y_i) uniformly distributed, with Up component = U_lim.
    e = np.linspace(-E_lim / 2, E_lim / 2, n_antenna_E)
    n = np.linspace(-N_lim / 2, N_lim / 2, n_antenna_N)
    E, N = np.meshgrid(e, n)
    E_flat = E.flatten()
    N_flat = N.flatten()
    U_flat = np.zeros_like(E_flat) + U_lim
    array_grid = np.column_stack((E_flat, N_flat, U_flat))
    return array_grid


########################################
#  Compute baselines and uv sampling   #
########################################


def get_baselines(array):
    """Get baselines (JAX version).

    Function to compute the baselines of an antenna array.

    Parameters
    ----------
    array : np.ndarray
        The antenna array positions.

    Returns
    -------
    baselines : np.ndarray
        The baselines of the antenna array.
    """
    # Get the baseline for every combination of antennas except for i=j baselines.
    array = jnp.asarray(array)
    diffs = array[:, None, :] - array[None, :, :]  # Shape: (n, n, 3)
    mask = ~jnp.eye(array.shape[0], dtype=bool)  # Shape: (n, n), True where i ≠ j
    return diffs[mask].reshape(-1, 3)


@jit
def ENU_to_XYZ(b_ENU, lat=35.0 / 180 * jnp.pi):
    """ENU to XYZ (JAX version).

    Function to convert the baselines from East-North-Up (ENU) to XYZ coordinates.

    Parameters
    ----------
    b_ENU : np.ndarray
        The baselines in ENU coordinates.
    lat : float
        The latitude of the antenna array in radians.

    Returns
    -------
    X : jnp.ndarray
        The X coordinate of the baselines in XYZ coordinates.
    Y : jnp.ndarray
        The Y coordinate of the baselines in XYZ coordinates.
    Z : jnp.ndarray
        The Z coordinate of the baselines in XYZ coordinates.
    """
    # Compute baseline length, Azimuth and Elevation angles
    D = jnp.sqrt(jnp.sum(b_ENU**2, axis=1))
    A = jnp.arctan2(b_ENU[:, 0], b_ENU[:, 1])
    E = jnp.arcsin(b_ENU[:, 2] / D)
    # Compute the baseline in XYZ coordinates
    X = D * (jnp.cos(lat) * jnp.sin(E) - jnp.sin(lat) * jnp.cos(E) * jnp.cos(A))
    Y = D * jnp.cos(E) * jnp.sin(A)
    Z = D * (jnp.sin(lat) * jnp.sin(E) + jnp.cos(lat) * jnp.cos(E) * jnp.cos(A))

    return X, Y, Z


@jit
def XYZ_to_uvw(X, Y, Z, dec=30.0 / 180 * jnp.pi, ha=0.0, f=1420e6):
    """XYZ to uvw (JAX version).

    Get the uvw sampling points from the XYZ coordinates given a
    source declination, hour angle and frequency.

    Parameters
    ----------
    X : np.ndarray
        The X coordinate of the baselines in XYZ coordinates.
    Y : np.ndarray
        The Y coordinate of the baselines in XYZ coordinates.
    Z : np.ndarray
        The Z coordinate of the baselines in XYZ coordinates.
    dec : float
        The declination of the source in radians.
    ha : float
        The hour angle of the source in radians.
    f : float
        The frequency of the observation in Hz.

    Returns
    -------
    u : np.ndarray
        The u coordinate of the baselines in uvw coordinates.
    v : np.ndarray
        The v coordinate of the baselines in uvw coordinates.
    w : np.ndarray
        The w coordinate of the baselines in uvw coordinates.
    """
    c = 299792458.0
    lam_inv = f / c
    u = lam_inv * (jnp.sin(ha) * X + jnp.cos(ha) * Y)
    v = lam_inv * (
        -jnp.sin(dec) * jnp.cos(ha) * X
        + jnp.sin(dec) * jnp.sin(ha) * Y
        + jnp.cos(dec) * Z
    )
    w = lam_inv * (
        jnp.cos(dec) * jnp.cos(ha) * X
        - jnp.cos(dec) * jnp.sin(ha) * Y
        + jnp.sin(dec) * Z
    )
    return u, v, w


def uv_track_multiband(
    b_ENU,
    lat=35.0 / 180 * jnp.pi,
    dec=35.0 / 180 * jnp.pi,
    track_time=0.0,
    t_0=0.0,
    n_times=1,
    f=1420e6,
    df=0.0,
    n_freqs=1,
    multi_band=False,
):
    """Uv track multiband (JAX version).

    Function to compute the uv sampling baselines for a given observation time and frequency range.

    Parameters
    ----------
    b_ENU : np.ndarray
        The baselines in ENU coordinates.
    lat : float
        The latitude of the antenna array in radians.
    dec : float
        The declination of the source in radians.
    track_time : float
        The duration of the tracking in hours.
    t_0 : float
        The initial tracking time in hours.
    n_times : int
        The number of time steps.
    f : float
        The central frequency of the observation in Hz.
    df : float
        The frequency range of the observation in Hz.
    n_freqs : int
        The number of frequency samples.
    multi_band : bool
        If True separate the uv samples per frequency bands.

    Returns
    -------
    track : jnp.ndarray
        The uv sampling baselines listed for each time step and frequency.
    f_range : jnp.ndarray
        The list of frequency bands used in the simulation.
    """
    # Compute the baselines in XYZ coordinates
    X, Y, Z = ENU_to_XYZ(b_ENU, lat)
    # Compute the time steps
    h = jnp.linspace(t_0, t_0 + track_time, n_times) * jnp.pi / 12
    # Compute the frequency range
    f_range = jnp.linspace(f - df / 2, f + df / 2, n_freqs)

    # Vectorise the function XYZ_to_uvw over the time steps h (5th argument of XYZ_to_uvw)
    vmap_ha = vmap(XYZ_to_uvw, in_axes=(None, None, None, None, 0, None))

    # Vectorise the function vmap_ha over the frequency bands f_range (6th argument of vmap_ha)
    vmap_f = vmap(vmap_ha, in_axes=(None, None, None, None, None, 0))

    # Compute the uvw coordinates for all baselines, time steps and frequency bands
    u, v, w = vmap_f(X, Y, Z, dec, h, f_range)

    # Stack the uvw coordinates in a single array
    track_f = jnp.stack([u, v, w], axis=-1)

    if multi_band:
        # Separate the uv samples per frequency bands
        track = track_f.reshape(n_freqs, -1, 3)
    else:
        # Concatenate all uv samples in a single array
        track = track_f.reshape(-1, 3)
    return track, f_range


def combine_antenna_arr(arr1, arr2):
    """Combine antenna arr.

    Function to combine two antenna arrays.

    Parameters
    ----------
    arr1 : np.ndarray
        The first antenna array positions.
    arr2 : np.ndarray
        The second antenna array positions.

    Returns
    -------
    arr : np.ndarray
        The combined antenna array positions.
    """
    return np.concatenate((arr1, arr2), axis=0)


def load_antenna_enu_txt(path, noise=False):
    """Load antenna txt.

    Function to load antenna name, ENU positions and noise from a txt file.

    Parameters
    ----------
    path : str
        The path to the txt file.
    noise : bool
        Specify if the file contain the noise level of each antenna.

    Returns
    -------
    antenna_arr : np.ndarray
        The antenna array positions.
    noise_level : np.ndarray
        If noise==True, return antenna noise level.
    """
    antenna_info = np.genfromtxt(path)
    name = antenna_info[:, 0]

    if noise == True:
        noise_level = antenna_info[:, -1]
        return antenna_info[:, 1:4], noise_level
    else:
        return antenna_info[:, 1:4]


def load_antenna_latlon_txt(path, noise=False):
    """(lat,long,altitude) to ENU.

    Function to load the antenna name, latitude (degree), longitude (degree), altitude (metre) and noise from a txt file,
    and return the information with ENU positions.
    The reference is taken as the middle of all the antennas.
    Waring: the computation of the upp coordinate is valid for antennas that are close enough.

    Parameters
    ----------
    path : str
        The path to the txt file.
    noise : bool
        Specify if the file contain the noise level of each antenna.

    Returns
    -------
    antenna_arr : np.ndarray
        The antenna array positions.
    noise_level : np.ndarray
        If noise==True, return antenna noise level.
    """
    R = 6378137.0  # Mean radius of the Earth (m)

    antenna_info = np.genfromtxt(path)
    name = antenna_info[:, 0]
    antenna_pos = antenna_info[:, 1:4]

    # Compute the reference
    ref = np.mean(antenna_pos, axis=0)
    lat0, lon0, h0 = np.radians(ref[0]), np.radians(ref[1]), ref[2]

    enu_list = []
    for lat, lon, h in antenna_pos:
        # Compute the difference between the antenna and the reference (deg, deg, m)
        d_lat = np.radians(lat - ref[0])
        d_lon = np.radians(lon - ref[1])
        d_h = h - h0

        # Convert into ENU
        d_north = d_lat * R
        d_east = d_lon * R * np.cos(lat0)
        d_up = d_h
        enu_list.append((d_east, d_north, d_up))
    enu_array = np.array(enu_list)

    if noise == True:
        noise_level = antenna_info[:, -1]
        return enu_array, noise_level
    else:
        return enu_array


def save_antenna_enu_txt(antenna, path, noise=None):  # pragma: no cover
    """Save the antenna information into a txt file.

    Function to save the antenna name, ENU positions and optionally noise into a txt file.

    Parameters
    ----------
    antenna : np.ndarray
        The numpy array which contain the antenna information.
    path : str
        The path to the txt file.
    noise  np.ndarray
        The numpy array which contain the noise information, if available.

    Returns
    -------
    None
        The function only saves the file.
    """
    nb_antennas = len(antenna)
    antenna_name = np.linspace(1, nb_antennas, nb_antennas, dtype=int).reshape(-1, 1)

    if noise is not None:
        antenna_info = np.concatenate(
            (antenna_name, antenna, noise.reshape(-1, 1)), axis=1
        )
        fmt = "%d %.6f %.6f %.6f %.6f"
        header = "Name | E | N | U | Noise level"
    else:
        antenna_info = np.concatenate((antenna_name, antenna), axis=1)
        fmt = "%d %.6f %.6f %.6f"
        header = "Name | E | N | U"

    np.savetxt(
        path,
        antenna_info,
        fmt=fmt,
        header=header,
    )
