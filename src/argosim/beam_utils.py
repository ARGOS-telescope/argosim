"""Beam utils.

This module contains the beam class to generate the primary beam of the antennas.

:Authors: Ezequiel Centofanti <ezequiel.centofanti@cea.fr>

"""

import matplotlib.pyplot as plt
import numpy as np


class CosCubeBeam:
    """Cosine cubed beam.

    Class to model the primary beam of the antennas using a cosine cubed function.

    Attributes
    ----------
    c : float
        The multiplicative constant of the cosine argument.
    f : float
        The frequency in GHz at which the beam is evaluated.
    grid_size : int
        The number of pixels of the simulated field of view.
    fov_deg : float
        The simulated field of view in degrees.

    """

    def __init__(self, c=0.2, f=1.0, n_pix=100, fov_deg=1.0):
        """Initialize the cosine cubed beam.

        Initialize the cosine cubed beam with the given parameters and check if the desired FOV is in agreement with the beam size.

        Parameters
        ----------
        c : float
            The multiplicative constant of the cosine argument.
        f : float
            The frequency in GHz at which the beam is evaluated.
        n_pix : int
            The number of pixels of the simulated field of view.
        fov_deg : float
            The simulated field of view in degrees.

        """
        self.c = c
        self.f = f
        self.grid_size = n_pix
        self.fov_deg = fov_deg

        # Check if the desired FOV is too large for the beam size
        self.check_fov()

    def __call__(self, l, m):
        """Cosine cubed beam.

        Function to compute the cosine cubed beam. The cosine cubed beam is modeled as:

        z = cos(Cf * sqrt(l^2 + m^2))^3,

        where Cf = c*f + c^2.

        Parameters
        ----------
        l : np.ndarray
            The l coordinate of the uv-plane.
        m : np.ndarray
            The m coordinate of the uv-plane.

        Returns
        -------
        z : np.ndarray
            The cosine cubed beam.

        """
        return np.cos(self.Cf() * np.sqrt(l**2 + m**2)) ** 3

    def check_fov(self):
        """Check FOV.

        Function to check if the desired FOV is too large for the beam size.

        """
        if np.sqrt(2) * self.fov_deg > self.beam_edge():
            print(
                "Warning: FOV diagonal lenght ({:.2f} deg) is larger than beam edge ({:.2f} deg).".format(
                    np.sqrt(2) * self.fov_deg, self.beam_edge()
                )
            )
            print(
                "Suggested FOV: {:.2f} deg".format(
                    np.floor(100 * self.beam_edge() / np.sqrt(2)) / 100
                )
            )

    def set_c(self, c):
        """Set c.

        Parameters
        ----------
        c : float
            The multiplicative constant of the cosine argument.

        """
        self.c = c
        self.check_fov()

    def set_f(self, f):
        """Set f.

        Parameters
        ----------
        f : float
            The frequency in GHz at which the beam is evaluated.

        """
        self.f = f
        self.check_fov()

    def set_fov(self, fov_deg):
        """Set FOV.

        Parameters
        ----------
        fov_deg : float
            The simulated field of view in degrees.

        """
        self.fov_deg = fov_deg
        self.check_fov()

    def Cf(self):
        """Cf.

        Function to compute the Cf parameter.

        Returns
        -------
        Cf : float
            The Cf parameter.

        """
        return self.c * self.f + self.c**2

    def get_mesh(self):
        """Get mesh.

        Function to compute the meshgrid of the lm image plane.

        Returns
        -------
        l : np.ndarray
            The l coordinate of the image plane.
        m : np.ndarray
            The m coordinate of the image plane.

        """
        l, m = np.meshgrid(
            np.linspace(-self.fov_deg, self.fov_deg, self.grid_size),
            np.linspace(-self.fov_deg, self.fov_deg, self.grid_size),
        )
        return l, m

    def get_beam(self):
        """Get beam.

        Function to compute the beam amplitude.

        Returns
        -------
        z : np.ndarray
            The beam amplitude.

        """
        l, m = self.get_mesh()
        z = self(l, m)
        return z

    def r_fov(self):
        """FOV radius.

        Function to compute the radius of the beam at -3 dB.

        Returns
        -------
        r_fov : float
            The radius of the beam at -3 dB.

        """
        return np.arccos((1 / np.sqrt(2)) ** (1 / 3)) / self.Cf()

    # def max_fov(self):
    #     """Max FOV.

    #     Compute the maximum FOF: radius at which the beam reaches 1/2 (max_FOV >> fov_3db).

    #     Returns
    #     -------
    #     max_fov : float
    #         The maximum FOV in degrees.

    #     """
    #     return np.arccos((.5)**(1/3))/self.Cf()

    def beam_edge(self):
        """Beam edge.

        Function to compute the beam edge (where the beam reaches zero).

        Returns
        -------
        beam_edge : float
            The beam edge in degrees.

        """
        return 0.5 * np.pi / self.Cf()

    def fov_solid_angle(self, r_fov=None):
        """FOV solid angle.

        Compute the solid angle of the FOV.

        Parameters
        ----------
        r_fov : float
            The beam width at -3 dB in degrees.

        Returns
        -------
        solid_angle : float
            The solid angle of the FOV in square degrees.

        """
        if r_fov is None:
            r_fov = self.r_fov()
        return np.pi * r_fov**2

    def plot_beam_1d(self, freqs):  # pragma: no cover
        """Plot beam 1D.

        Function to plot the simulated beam in 1D as a function of the angle.

        Parameters
        ----------
        freqs : list
            The list of frequencies in GHz at which the beam is evaluated.

        """
        f_old = self.f
        if freqs is None:
            freqs = [self.f]
        plt.figure()
        plt.hlines(
            1 / np.sqrt(2),
            -self.fov_deg,
            self.fov_deg,
            linestyle="--",
            color="k",
            alpha=0.5,
            label="-3dB",
        )
        cmap = plt.get_cmap("viridis")
        for i, f in enumerate(freqs):
            self.set_f(f)
            l = np.linspace(-self.fov_deg, self.fov_deg, self.grid_size)
            plt.plot(
                l, self(l, 0), color=cmap(i / len(freqs)), label="{} GHz".format(self.f)
            )
            plt.xlabel("Angle (deg)")
            plt.ylabel("Antenna Gain")
            plt.title("Cosine cubed beam")
            plt.legend()
            plt.grid()
        plt.show()
        self.set_f(f_old)

    def plot_beam_2d(self):  # pragma: no cover
        """Plot beam 2D.

        Function to plot the simulated beam in 2D over the field of view.

        """
        z = self.get_beam()
        fig, ax = plt.subplots()
        ax.imshow(
            z,
            extent=(-self.fov_deg, self.fov_deg, -self.fov_deg, self.fov_deg),
            vmin=0,
            vmax=1,
        )
        r_fov = self.r_fov()
        fov_contour = plt.Circle(
            (0, 0), r_fov, color="gray", alpha=0.7, linestyle="--", fill=False
        )
        fov_crop = plt.Rectangle(
            (-r_fov, -r_fov),
            2 * r_fov,
            2 * r_fov,
            color="k",
            alpha=0.5,
            linestyle="--",
            fill=False,
        )
        ax.add_artist(fov_contour)
        ax.add_artist(fov_crop)
        ax.set_xlabel("Angle (deg)")
        ax.set_ylabel("Angle (deg)")
        ax.set_title(
            r"Cosine cubed beam - FOV: {:.1f} $\deg^2$ @ {} GHz".format(
                self.fov_solid_angle(r_fov), self.f
            )
        )
        plt.show()

    def evaluate_beam_fit(self, freqs, norm_beam_list):  # pragma: no cover
        """Evaluate beam fit.

        Function to evaluate the beam fit at the given frequencies and compare it with the beam measurements.

        Parameters
        ----------
        freqs : list
            The list of frequencies in GHz at which the beam is evaluated.
        norm_beam_list : list
            The list of normalized beam measurements (amplitude not dB).

        """
        fig, ax = plt.subplots(1, 5, figsize=(4 * len(freqs), 4))
        cmap = plt.get_cmap("viridis")
        for i, norm_beam in enumerate(norm_beam_list):
            ax[i].plot(
                norm_beam[:, 0],
                norm_beam[:, 2],
                label="Main lobe",
                color=cmap(i / len(freqs)),
            )

            self.set_f(freqs[i])
            C_f = self.Cf()
            x = np.linspace(-1 / C_f, 1 / C_f, self.grid_size)

            ax[i].plot(x, self(0, x), color="k", linestyle="--", label="Fit")

            ax[i].hlines(
                1 / np.sqrt(2),
                -self.beam_edge(),
                self.beam_edge(),
                linestyle="--",
                color="gray",
                alpha=0.7,
                label="-3dB",
            )

            ax[i].set_ylim(0, 1)
            ax[i].set_xlim(-self.beam_edge(), self.beam_edge())
            ax[i].set_title(
                r"{:.1f} GHz - FOV: {:.1f} $\deg^2$".format(
                    self.f, self.fov_solid_angle()
                )
            )
            ax[i].legend()
            ax[i].set_xlabel("Angle (deg)")
            if i == 0:
                ax[i].set_ylabel("Gain (dB)")

        plt.show()
