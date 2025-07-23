"""Plot utils.

This module contains functions to plot the antenna array, beam, baselines,
uv-coverage and sky models.

:Authors: Ezequiel Centofanti <ezequiel.centofanti@cea.fr>

"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from argosim import metrics_utils


def plot_beam(beam, pRng=(-0.1, 0.5), ax=None, fig=None):  # pragma: no cover
    """Plot beam.

    Function to plot the synthesised beam image.

    Parameters
    ----------
    beam : np.ndarray
        The beam image.
    pRng : tuple
        The range of the colorbar.
    ax : matplotlib.axes.Axes
        The axis to plot the beam. For plotting on a specific subplot axis.
    fig : matplotlib.figure.Figure
        The figure to plot the beam. For plotting on a specific subplot axis.

    Returns
    -------
    None
    """
    # Imshow min and max values.
    zMin = np.nanmin(beam)
    zMax = np.nanmax(beam)
    zRng = zMin - zMax
    zMin -= zRng * pRng[0]
    zMax += zRng * pRng[1]
    if ax == None or fig == None:
        fig, ax = plt.subplots(1, 1)
    im = ax.imshow(np.fft.ifftshift(beam), vmin=zMin, vmax=zMax)
    fig.colorbar(im, ax=ax)
    if ax == None or fig == None:
        plt.show()


def plot_antenna_arr(array, ax=None, fig=None, title="Array"):  # pragma: no cover
    """Plot antenna array.

    Function to plot the antenna array in ground coordinates.

    Parameters
    ----------
    array : np.ndarray
        The antenna array positions in the ground.
    ax : matplotlib.axes.Axes
        The axis to plot the antenna array. For plotting on a specific subplot axis.
    fig : matplotlib.figure.Figure
        The figure to plot the antenna array. For plotting on a specific subplot axis.
    title : str
        The title of the plot.

    Returns
    -------
    None
    """
    # Center the array
    array = array - np.mean(array, axis=0)
    if ax == None or fig == None:
        fig, ax = plt.subplots(1, 1)
    ax.scatter(array[:, 0], array[:, 1], s=20, c="gray")
    for i, txt in enumerate(range(1, len(array) + 1, 1)):
        ax.annotate(txt, (array[i, 0], array[i, 1]))
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        x_lim = np.max(np.abs(array)) * 1.1
        y_lim = x_lim
        ax.set_xlim(-x_lim, x_lim)
        ax.set_ylim(-y_lim, y_lim)
        ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    if ax == None or fig == None:
        plt.show()


def plot_baselines(baselines, ax=None, fig=None, ENU=False):  # pragma: no cover
    """Plot baselines.

    Function to plot the baselines in uv-space.

    Parameters
    ----------
    baselines : np.ndarray
        The uv-space sampling positions.
    ax : matplotlib.axes.Axes
        The axis to plot the baselines. For plotting on a specific subplot axis.
    fig : matplotlib.figure.Figure
        The figure to plot the baselines. For plotting on a specific subplot axis.
    ENU : bool
        If True, plot the baselines in East-North-Up coordinates. Otherwise, plot in uv-space.

    Returns
    -------
    None
    """
    if ax == None or fig == None:
        fig, ax = plt.subplots(1, 1)
    if ENU:
        ax.set_xlabel("East [m]")
        ax.set_ylabel("North [m]")
        ax.set_title("Baselines")
        ax.scatter(baselines[:, 0], baselines[:, 1], s=0.4, c="gray")
        ax.set_xlim([np.min(baselines), np.max(baselines)])
        ax.set_ylim([np.min(baselines), np.max(baselines)])
    else:
        ax.set_xlabel(r"u(k$\lambda$)")
        ax.set_ylabel(r"v(k$\lambda$)")
        ax.set_title(r"uv-plane")
        ax.scatter(baselines[:, 0] / 1000, baselines[:, 1] / 1000, s=0.4, c="gray")
        ax.set_xlim([np.min(baselines) / 1000, np.max(baselines) / 1000])
        ax.set_ylim([np.min(baselines) / 1000, np.max(baselines) / 1000])
    ax.set_aspect("equal", adjustable="box")
    if ax == None or fig == None:
        plt.show()


def plot_sky(
    image, fov_size=(1.0, 1.0), ax=None, fig=None, title="Sky"
):  # pragma: no cover
    """Plot sky.

    Function to plot the sky model.

    Parameters
    ----------
    image : np.ndarray
        The sky model image.
    fov_size : tuple
        The field of view size in degrees.

    Returns
    -------
    None
    """
    if ax == None or fig == None:
        fig, ax = plt.subplots(1, 1)
    im = ax.imshow(
        image,
        extent=[-fov_size[0] / 2, fov_size[0] / 2, -fov_size[1] / 2, fov_size[1] / 2],
        origin="lower",
    )
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("x [deg]")
    ax.set_ylabel("y [deg]")
    ax.set_title("{} ({}x{})".format(title, image.shape[0], image.shape[1]))
    if ax == None or fig == None:
        plt.show()


def plot_sky_uv(sky_uv, fov_size):  # pragma: no cover
    """Plot sky uv.

    Function to plot the sky model in uv-space in logarithmic amplitud sale.

    Parameters
    ----------
    sky_uv : np.ndarray
        The sky model in uv-space.The inverse fourier transform of the sky model.

    Returns
    -------
    None
    """
    max_u = (180 / np.pi) * sky_uv.shape[0] / (2 * fov_size[0]) / 1000
    max_v = (180 / np.pi) * sky_uv.shape[1] / (2 * fov_size[1]) / 1000

    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(
        np.abs(sky_uv / 1000), extent=[-max_u, max_u, -max_v, max_v], origin="lower"
    )
    plt.xlabel(r"$u(k\lambda$)")
    plt.ylabel(r"$v(k\lambda$)")
    plt.title("Amplitude")
    plt.subplot(122)
    plt.imshow(
        np.angle(sky_uv / 1000), extent=[-max_u, max_u, -max_v, max_v], origin="lower"
    )
    plt.xlabel(r"$u(k\lambda$)")
    plt.ylabel(r"$v(k\lambda$)")
    plt.title("Phase")
    plt.show()


# def plot_sampled_sky(sky_uv):
#     """Plot sampled sky.

#     Function to plot the sky model samples in uv-space according to the uv-coverage.

#     Parameters
#     ----------
#     sky_uv : np.ndarray
#         The sampled sky model in uv-space.

#     Returns
#     -------
#     None
#     """
#     plt.imshow(np.abs(sky_uv) + 1e-3, norm=matplotlib.colors.LogNorm())
#     plt.colorbar()
#     plt.show()


def plot_uv_hist(baselines, bins=20, output_folder=None):  # pragma: no cover
    """Plot uv histogram.

    Function to plot the histogram of the uv-sampling distribution.

    Parameters
    ----------
    baselines : np.ndarray
        The uv-space sampling positions.
    bins : int
        The number of bins for the histogram.
    output_folder : str
        The output folder to save the plot.

    Returns
    -------
    np.ndarray
        The histogram of the uv-sampling distribution.
    """
    # scale to kilo-lambda
    baselines = baselines / 1000

    cmap = matplotlib.colormaps["bone"]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    D = np.sqrt(np.sum(baselines[:, :2] ** 2, axis=1))
    baseline_hist = ax[0].hist(D, range=(0, np.max(D) * 1.1), bins=bins)

    n = baseline_hist[0]
    patches = baseline_hist[2]
    col = (n - n.min()) / (n.max() - n.min())
    for c, p in zip(col, patches):
        plt.setp(p, "facecolor", cmap(c))
    ax[0].set_title("Baselines histogram")
    ax[0].set_xlabel(r"UV distance $(k\lambda)$")
    ax[0].set_ylabel("Counts")
    ax[0].set_facecolor("palegoldenrod")
    ax[0].set_box_aspect(1)

    counts = np.flip(baseline_hist[0])
    r_list = baseline_hist[1]

    colors = cmap((counts / max(counts)))

    draw_back = plt.Circle((0.0, 0.0), 10 * r_list[-1], color="black", fill=True)
    ax[1].add_artist(draw_back)
    for color, r in zip(colors, np.flip(r_list[1:])):
        draw1 = plt.Circle((0.0, 0.0), r, color=color, fill=True)
        ax[1].add_artist(draw1)

    ax[1].scatter(baselines[:, 0], baselines[:, 1], color="yellow", s=1, alpha=0.3)

    fig.colorbar(
        plt.cm.ScalarMappable(
            cmap=cmap, norm=matplotlib.colors.Normalize(vmin=0, vmax=max(counts))
        ),
        ax=ax[1],
        orientation="vertical",
        label="Counts",
    )

    ax[1].set_aspect("equal")
    ax[1].set_xlim(-r_list[-1] * 1.1, r_list[-1] * 1.1)
    ax[1].set_ylim(-r_list[-1] * 1.1, r_list[-1] * 1.1)
    ax[1].set_title("Radial distribution")
    ax[1].set_xlabel(r"u $(k\lambda)$")
    ax[1].set_ylabel(r"v $(k\lambda)$")

    plt.suptitle(
        "UV sampling distribution",
        horizontalalignment="center",
        verticalalignment="top",
    )

    if output_folder is not None:
        plt.savefig(output_folder + "uv_hist.pdf")
    else:
        plt.show()

    return baseline_hist


def plot_beam_fit(beam, fit_result=None):  # pragma: no cover
    """Plot beam fit.

    Plot the beam and overlay the fitted ellipse.

    Parameters
    ----------
    beam : np.ndarray
        2D beam image.
    fit_result : dict
        Dictionary containing the ellipse parameters (center, width, height, angle_deg, eccentricity). If None, it is computed from the beam.

    Returns
    -------
    None
    """
    if fit_result is None:
        fit_result = metrics_utils.fit_elliptical_beam(beam)

    plt.imshow(beam, origin="lower")
    plt.colorbar()

    ellipse = Ellipse(
        xy=fit_result["center"],
        width=fit_result["width"],
        height=fit_result["height"],
        angle=fit_result["angle_deg"],
        edgecolor="red",
        facecolor="none",
        lw=2,
        label="Fitted Ellipse",
    )

    ax = plt.gca()
    ax.add_patch(ellipse)
    ax.set_aspect("equal")
    plt.title(f"Elliptical Fit (e = {fit_result['eccentricity']:.4f})")
    plt.legend()
    plt.show()
