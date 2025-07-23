import sys
from PyQt6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton
)
from PyQt6.QtCore import Qt

from matplotlib.figure import Figure

import argosim
import argosim.imaging_utils
import argosim.data_utils
import argosim.plot_utils
import numpy as np

from utils import ScrollableFigureCanvas

class ImagingWidget(QWidget):
    def __init__(self, aperture_widget=None):
        super().__init__()
        self.aperture_widget = aperture_widget
        layout = QVBoxLayout()
        title = QLabel("Imaging")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 16px;")
        layout.addWidget(title)

        # Image simulation parameters
        param_row = QHBoxLayout()
        self.n_sources_label = QLabel("Number of sources:")
        self.n_sources_input = QLineEdit()
        self.n_sources_input.setText("3")
        param_row.addWidget(self.n_sources_label)
        param_row.addWidget(self.n_sources_input)
        self.min_size_label = QLabel("Min source size (''):")
        self.min_size_input = QLineEdit()
        self.min_size_input.setText("5")
        param_row.addWidget(self.min_size_label)
        param_row.addWidget(self.min_size_input)
        self.max_size_label = QLabel("Max source size (''):")
        self.max_size_input = QLineEdit()
        self.max_size_input.setText("10")
        param_row.addWidget(self.max_size_label)
        param_row.addWidget(self.max_size_input)
        self.noise_label = QLabel("Noise Level:")
        self.noise_input = QLineEdit()
        self.noise_input.setText("0.1")
        param_row.addWidget(self.noise_label)
        param_row.addWidget(self.noise_input)
        self.seed_label = QLabel("Seed:")
        self.seed_input = QLineEdit()
        self.seed_input.setText("None")
        param_row.addWidget(self.seed_label)
        param_row.addWidget(self.seed_input)
        layout.addLayout(param_row)

        # Buttons row
        button_row = QHBoxLayout()
        self.sim_button = QPushButton("Simulate Imaging")
        self.sim_button.clicked.connect(self._simulate_imaging)
        button_row.addWidget(self.sim_button, 4)
        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self._reset_defaults)
        button_row.addWidget(self.reset_button, 1)
        layout.addLayout(button_row)

        # Matplotlib FigureCanvas for sky model and observation
        self.fig = Figure(figsize=(6, 3))
        self.canvas = ScrollableFigureCanvas(self.fig)
        self.canvas.setMinimumHeight(400)
        layout.addWidget(self.canvas, stretch=1)

        self.setLayout(layout)

    def _reset_defaults(self):
        self.noise_input.setText("0.1")
        self.n_sources_input.setText("3")
        self.min_size_input.setText("5")
        self.max_size_input.setText("10")
        self.seed_input.setText("None")

    def _simulate_imaging(self):
        try:
            noise_level = float(self.noise_input.text())
            n_sources = int(self.n_sources_input.text())
            # Convert arcseconds to degrees
            min_source_size = float(self.min_size_input.text()) /3600
            max_source_size = float(self.max_size_input.text()) /3600
            # Assert valid values
            if noise_level < 0.:
                raise ValueError("Noise level can't be negative.")
            if n_sources <= 0:
                raise ValueError("Number of sources must be positive.")
            if n_sources > 50:
                raise ValueError("The maximum number of sources is set to 50.")
            if min_source_size > max_source_size:
                raise ValueError("Minimum source size should be smaller than maximum source size.")
            if min_source_size <= 0 or max_source_size<= 0:
                raise ValueError("Source sizes must be positive.")
            # Seed handling
            if self.seed_input.text() == "None":
                seed = None
            else:
                seed = int(self.seed_input.text())
        except Exception as e:
            self.fig.clear()
            ax = self.fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f"Error:\n{str(e)}", ha='center', va='center', fontsize=12, color='red')
            ax.axis('off')
            self.canvas.draw()
            return
        
        # Get current uv points ApertureSynthesisWidget
        uv_points = None
        fov_size = None
        Npx = None
        if self.aperture_widget is not None:
            try:
                uv_points = self.aperture_widget.get_current_uv_points()
                fov_size = float(self.aperture_widget.get_current_fov_size())
                Npx = int(self.aperture_widget.get_current_Npx())
            except Exception as e:
                self.fig.clear()
                ax = self.fig.add_subplot(1, 1, 1)
                ax.text(0.5, 0.5, f"Error:\n{str(e)}", ha='center', va='center', fontsize=12, color='red')
                ax.axis('off')
                self.canvas.draw()
                return
        if uv_points is None:
            self.fig.clear()
            ax = self.fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, "Missing aperture.", ha='center', va='center', fontsize=12, color='red')
            ax.axis('off')
            self.canvas.draw()
            return
    
        # Simulate the sky model: n_sources with sizes betweeen (min_source_size, max_source_size)
        if seed is not None:
            np.random.seed(seed)
        rand_sizes = np.random.rand(n_sources)
        source_sizes = rand_sizes * (max_source_size-min_source_size) + min_source_size
        sky_model = argosim.data_utils.n_source_sky((Npx, Npx), fov_size, deg_size_list=source_sizes, source_intensity_list=[1.]*n_sources, seed=seed, norm='max')
        obs, _ = argosim.imaging_utils.simulate_dirty_observation(sky_model, uv_points, fov_size, sigma=noise_level)

        self.fig.clear()
        ax1 = self.fig.add_subplot(1, 2, 1)
        # ax1.imshow(sky_model, origin='lower', cmap='inferno')
        argosim.plot_utils.plot_sky(sky_model, fov_size=(fov_size, fov_size), ax=ax1, fig=self.fig)
        ax1.set_title('Ground Truth Sky Model')
        ax2 = self.fig.add_subplot(1, 2, 2)
        # ax2.imshow(obs, origin='lower', cmap='inferno')
        argosim.plot_utils.plot_sky(obs, fov_size=(fov_size, fov_size), ax=ax2, fig=self.fig)
        ax2.set_title('Observation')
        self.canvas.draw()
