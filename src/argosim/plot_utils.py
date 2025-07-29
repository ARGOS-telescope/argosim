"""Plot utils.

This module contains functions to plot the antenna array, beam, baselines,
uv-coverage and sky models.

:Authors: Ezequiel Centofanti <ezequiel.centofanti@cea.fr>

"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from skimage.transform import rotate

from argosim import metrics_utils


def plot_antenna_arr(
    array, ax=None, fig=None, title="Array", antenna_idx=True, s=20, c="mediumblue"
):  # pragma: no cover
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
    antenna_idx : bool
        If True, annotate the antenna indices on the plot.
    s : int
        The size of the scatter points. Default is 20.
    c : str
        The color of the scatter points. Default is "mediumblue".

    Returns
    -------
    None
    """
    # Center the array
    array = array - np.mean(array, axis=0)
    # Convert to km
    array = array / 1000.0
    if ax == None or fig == None:
        fig, ax = plt.subplots(1, 1)
    ax.scatter(array[:, 0], array[:, 1], s=s, c=c)
    if antenna_idx:
        for i, txt in enumerate(range(1, len(array) + 1, 1)):
            ax.annotate(txt, (array[i, 0], array[i, 1]))
    ax.set_xlabel("E [km]")
    ax.set_ylabel("N [km]")
    x_lim = np.max(np.abs(array)) * 1.1
    y_lim = x_lim
    ax.set_xlim(-x_lim, x_lim)
    ax.set_ylim(-y_lim, y_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    if ax is None or fig is None:
        plt.show()


def plot_baselines(
    baselines, ax=None, fig=None, title=None, ENU=False, s=5, c="darkred"
):  # pragma: no cover
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
    title : str
        The title of the plot. If None, a default title is used.
    ENU : bool
        If True, plot the baselines in East-North-Up coordinates. Otherwise, plot in uv-space.
    s : int
        The size of the scatter points. Default is 5.
    c : str
        The color of the scatter points. Default is "darkred".

    Returns
    -------
    None
    """
    if ax == None or fig == None:
        fig, ax = plt.subplots(1, 1)
    # Convert to km
    baselines = baselines / 1000.0
    if ENU:
        ax.set_xlabel("East [km]")
        ax.set_ylabel("North [km]")
        if title is None:
            ax.set_title("Baselines")
        else:
            ax.set_title(title)
        ax.scatter(baselines[:, 0], baselines[:, 1], s=s, c=c)
    else:
        ax.set_xlabel(r"$u$(k$\lambda$)")
        ax.set_ylabel(r"$v$(k$\lambda$)")
        if title is None:
            ax.set_title(r"uv-plane")
        else:
            ax.set_title(title)
        ax.scatter(baselines[:, 0], baselines[:, 1], s=s, c=c)
    ax.set_xlim([np.min(baselines), np.max(baselines)])
    ax.set_ylim([np.min(baselines), np.max(baselines)])
    ax.set_aspect("equal", adjustable="box")
    if ax is None or fig is None:
        plt.show()


def plot_sky(
    image, fov_size, ax=None, fig=None, title=None, cbar=True
):  # pragma: no cover
    """Plot sky.

    Function to plot the sky model.

    Parameters
    ----------
    image : np.ndarray
        The image to be plotted in real space.
    fov_size : tuple
        The field of view size in degrees.
    ax : matplotlib.axes.Axes
        The axis to plot the sky model. For plotting on a specific subplot axis.
    fig : matplotlib.figure.Figure
        The figure to plot the sky model. For plotting on a specific subplot axis.
    title : str
        The title of the plot. Default is "Sky".
    cbar : bool
        If True, display a colorbar. Default is True.

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
    if cbar:
        fig.colorbar(im, ax=ax)
    ax.set_xlabel("l [deg]")
    ax.set_ylabel("m [deg]")
    if title is None:
        ax.set_title("{} ({}x{})".format("Sky", image.shape[0], image.shape[1]))
    else:
        ax.set_title(title)
    if ax is None or fig is None:
        plt.show()


def plot_sky_uv(
    image_uv, fov_size, ax=None, fig=None, title="Sky uv", cbar=False, scale="linear"
):  # pragma: no cover
    """Plot sky uv.

    Function to plot the absolute value of an image in uv-space.

    Parameters
    ----------
    image_uv : np.ndarray
        The image to be plotted in uv-space.
    fov_size : tuple
        The field of view size in degrees.
    ax : matplotlib.axes.Axes
        The axis to plot the sky model in uv-space. For plotting on a specific subplot axis.
    fig : matplotlib.figure.Figure
        The figure to plot the sky model in uv-space. For plotting on a specific subplot axis.
    title : str
        The title of the plot. Default is "Sky uv".
    cbar : bool
        If True, display a colorbar. Default is False.
    scale : str
        The scale of the uv-image. Default is 'linear'. Options are 'linear' or 'log'.

    Returns
    -------
    None
    """
    max_u = (180 / np.pi) * image_uv.shape[0] / (2 * fov_size[0]) / 1000
    max_v = (180 / np.pi) * image_uv.shape[1] / (2 * fov_size[1]) / 1000

    if ax is None or fig is None:
        fig, ax = plt.subplots(1, 1)
    if scale == "log":
        image_uv = np.log10(np.abs(image_uv) + 1e-10)
    elif scale == "linear":
        image_uv = np.abs(image_uv)
    im = ax.imshow(image_uv, extent=[-max_u, max_u, -max_v, max_v], origin="lower")
    if cbar:
        fig.colorbar(im, ax=ax)
    ax.set_xlabel(r"$u$(k$\lambda$)")
    ax.set_ylabel(r"$v$(k$\lambda$)")
    ax.set_title("Amplitude")
    if ax is None or fig is None:
        plt.show()


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


def plot_beam_and_fit(beam, fov_size, fit_result=None):  # pragma: no cover
    """Plot beam and fit.

    Plot the beam and overlay the fitted ellipse.

    Parameters
    ----------
    beam : np.ndarray
        2D beam image.
    fov_size : tuple
        The field of view size in degrees.
    fit_result : dict
        Dictionary containing the ellipse parameters (center, width, height, angle_deg, eccentricity). If None, it is computed from the beam.

    Returns
    -------
    None
    """
    if fit_result is None:
        fit_result = metrics_utils.fit_elliptical_beam(beam)
    Npx = beam.shape[0]
    # Beam fit ellipse
    ellipse = Ellipse(
        xy=fit_result["center"],
        width=fit_result["width"],
        height=fit_result["height"],
        angle=fit_result["angle_deg"],
        edgecolor="k",
        facecolor="none",
        lw=2,
        label="Elliptical fit",
    )
    # Semi-major and semi-minor beam gain
    beam_rotated = rotate(beam, fit_result["angle_deg"], resize=True)
    semi_major_beam = beam_rotated[
        beam_rotated.shape[0] // 2,
        beam_rotated.shape[0] // 2 - Npx // 2 : beam_rotated.shape[0] // 2 + Npx // 2,
    ]
    semi_minor_beam = beam_rotated[
        beam_rotated.shape[0] // 2 - Npx // 2 : beam_rotated.shape[0] // 2 + Npx // 2,
        beam_rotated.shape[1] // 2,
    ]
    # First plot: beam and fit, second plot: semi-major beam gain, third plot: semi-minor beam gain.
    fig, ax = plt.subplots(1, 3, figsize=(25, 10))
    ax[0].imshow(beam, origin="lower")
    ax[0].set_yticks(
        [Npx // 10, Npx // 2, Npx - Npx // 10],
        [-fov_size[0] * 2 / 5, 0, fov_size[0] * 2 / 5],
    )
    ax[0].set_xticks(
        [Npx // 10, Npx // 2, Npx - Npx // 10],
        [-fov_size[0] * 2 / 5, 0, fov_size[0] * 2 / 5],
    )
    ax[0].set_xlabel("l (deg)")
    ax[0].set_ylabel("m (deg)")
    # Plot the ellipse
    ax[0].add_patch(ellipse)
    # Plot the semi-major and semi-minor axes as lines through the center of the ellipse
    ax[0].plot(
        [0, Npx - 1],
        [
            Npx // 2 - Npx // 2 * np.tan(np.radians(fit_result["angle_deg"])),
            Npx // 2 + Npx // 2 * np.tan(np.radians(fit_result["angle_deg"])),
        ],
        color="orange",
        alpha=0.7,
        lw=3,
        label="Major axis",
        linestyle="--",
    )
    ax[0].plot(
        [
            Npx // 2 - Npx // 2 * np.tan(np.radians(fit_result["angle_deg"])),
            Npx // 2 + Npx // 2 * np.tan(np.radians(fit_result["angle_deg"])),
        ],
        [Npx - 1, 0],
        color="royalblue",
        alpha=0.7,
        lw=4,
        label="Minor axis",
        linestyle="--",
    )
    ax[0].set_title("Dirty beam")
    ax[0].legend(loc="upper right", fontsize=32)

    # Plot the semi-major and semi-minor beam gain
    ax[1].plot(semi_major_beam, color="orange", lw=4)
    ax[1].set_title("Beam gain major axis")
    ax[1].set_yticks([])
    ax[1].set_box_aspect(1)
    ax[1].set_xticks(
        [Npx // 10, Npx // 2, Npx - Npx // 10],
        [-fov_size[0] * 2 / 5, 0, fov_size[0] * 2 / 5],
    )
    ax[1].set_xlabel("(deg)")
    ax[1].plot(
        [int(Npx // 2 - fit_result["width"]), int(Npx // 2 + fit_result["width"])],
        [np.max(semi_major_beam) / 2, np.max(semi_major_beam) / 2],
        color="black",
        linestyle="--",
        lw=4,
        label="FWHM",
    )
    ax[1].scatter(
        [
            int(Npx // 2 - fit_result["width"] + 1),
            int(Npx // 2 + fit_result["width"] + 1),
        ],
        [np.max(semi_major_beam) / 2, np.max(semi_major_beam) / 2],
        color="black",
        marker="x",
        s=200,
    )
    ax[1].legend(loc="upper right", fontsize=30)

    ax[2].plot(semi_minor_beam, color="royalblue", lw=4)
    ax[2].set_title("Beam gain minor axis")
    ax[2].set_yticks([])
    ax[2].set_box_aspect(1)
    ax[2].set_xticks(
        [Npx // 10, Npx // 2, Npx - Npx // 10],
        [-fov_size[0] * 2 / 5, 0, fov_size[0] * 2 / 5],
    )
    ax[2].set_xlabel("(deg)")
    ax[2].plot(
        [int(Npx // 2 - fit_result["height"]), int(Npx // 2 + fit_result["height"])],
        [np.max(semi_minor_beam) / 2, np.max(semi_minor_beam) / 2],
        color="black",
        linestyle="--",
        lw=4,
        label="FWHM",
    )
    ax[2].scatter(
        [
            int(Npx // 2 - fit_result["height"] + 1),
            int(Npx // 2 + fit_result["height"] + 1),
        ],
        [np.max(semi_minor_beam) / 2, np.max(semi_minor_beam) / 2],
        color="black",
        marker="x",
        s=200,
    )
    ax[2].legend(loc="upper right", fontsize=30)

    plt.tight_layout()
    plt.show()
