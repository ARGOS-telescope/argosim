# widget_apsyn.py
import sys
from PyQt6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton
)
from PyQt6.QtCore import Qt

from matplotlib.figure import Figure

import argosim
import argosim.antenna_utils
import argosim.imaging_utils
import argosim.plot_utils
import numpy as np

from utils import ScrollableFigureCanvas

class ApertureSynthesisWidget(QWidget):
    def __init__(self, array_widget=None):
        super().__init__()
        self.array_widget = array_widget  # Reference to InterferometricArrayWidget
        layout = QVBoxLayout()
        title = QLabel("Aperture Synthesis")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 16px;")
        layout.addWidget(title)

        self.param_widgets = {}
        self.param_labels = {}

        # Group 1: (Array latitude, Source declination)
        group1 = QHBoxLayout()
        self.param_widgets['latitude'] = QLineEdit()
        self.param_widgets['latitude'].setText("35")
        self.param_labels['latitude'] = QLabel("Array Latitude (°):")
        group1.addWidget(self.param_labels['latitude'])
        group1.addWidget(self.param_widgets['latitude'])
        self.param_widgets['declination'] = QLineEdit()
        self.param_widgets['declination'].setText("35")
        self.param_labels['declination'] = QLabel("Source Declination (°):")
        group1.addWidget(self.param_labels['declination'])
        group1.addWidget(self.param_widgets['declination'])
        self.param_widgets['fov'] = QLineEdit()
        self.param_widgets['fov'].setText("0.1")
        self.param_labels['fov'] = QLabel("FOV size (°):")
        group1.addWidget(self.param_labels['fov'])
        group1.addWidget(self.param_widgets['fov'])
        self.param_widgets['Npx'] = QLineEdit()
        self.param_widgets['Npx'].setText("256")
        self.param_labels['Npx'] = QLabel("Npx:")
        group1.addWidget(self.param_labels['Npx'])
        group1.addWidget(self.param_widgets['Npx'])
        layout.addLayout(group1)

        # Group 2: (Start time, Duration, Timestep)
        group2 = QHBoxLayout()
        self.param_widgets['start_time'] = QLineEdit()
        self.param_widgets['start_time'].setText("-1")
        self.param_labels['start_time'] = QLabel("Start Time (hr, rel. to transit):")
        group2.addWidget(self.param_labels['start_time'])
        group2.addWidget(self.param_widgets['start_time'])
        self.param_widgets['duration'] = QLineEdit()
        self.param_widgets['duration'].setText("2")
        self.param_labels['duration'] = QLabel("Duration (hr):")
        group2.addWidget(self.param_labels['duration'])
        group2.addWidget(self.param_widgets['duration'])
        self.param_widgets['timestep'] = QLineEdit()
        self.param_widgets['timestep'].setText("15")
        self.param_labels['timestep'] = QLabel("Timestep (min):")
        group2.addWidget(self.param_labels['timestep'])
        group2.addWidget(self.param_widgets['timestep'])
        layout.addLayout(group2)

        # Group 3: (Central frequency, Bandwidth, Number of channels)
        group3 = QHBoxLayout()
        self.param_widgets['central_freq'] = QLineEdit()
        self.param_widgets['central_freq'].setText("2")
        self.param_labels['central_freq'] = QLabel("Central Frequency (GHz):")
        group3.addWidget(self.param_labels['central_freq'])
        group3.addWidget(self.param_widgets['central_freq'])
        self.param_widgets['bandwidth'] = QLineEdit()
        self.param_widgets['bandwidth'].setText("0.2")
        self.param_labels['bandwidth'] = QLabel("Bandwidth (GHz):")
        group3.addWidget(self.param_labels['bandwidth'])
        group3.addWidget(self.param_widgets['bandwidth'])
        self.param_widgets['nchan'] = QLineEdit()
        self.param_widgets['nchan'].setText("5")
        self.param_labels['nchan'] = QLabel("Number of Channels:")
        group3.addWidget(self.param_labels['nchan'])
        group3.addWidget(self.param_widgets['nchan'])
        layout.addLayout(group3)

        # Buttons row
        button_row = QHBoxLayout()
        self.sim_button = QPushButton("Simulate")
        self.sim_button.clicked.connect(self._simulate)
        button_row.addWidget(self.sim_button, 4)
        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self._reset_defaults)
        self.reset_button.clicked.connect(self._simulate)
        button_row.addWidget(self.reset_button, 1)
        layout.addLayout(button_row)

        # Matplotlib FigureCanvas for uv and dirty beam
        self.fig = Figure(figsize=(6, 3))
        self.canvas = ScrollableFigureCanvas(self.fig)
        self.canvas.setMinimumHeight(400)
        layout.addWidget(self.canvas, stretch=1)

        self.setLayout(layout)
        self.current_uv_points = None

    def _reset_defaults(self):
        self.param_widgets['latitude'].setText("35")
        self.param_widgets['declination'].setText("35")
        self.param_widgets['start_time'].setText("-1")
        self.param_widgets['duration'].setText("2")
        self.param_widgets['timestep'].setText("15")
        self.param_widgets['central_freq'].setText("2")
        self.param_widgets['bandwidth'].setText("0.2")
        self.param_widgets['nchan'].setText("5")
        self.param_widgets['fov'].setText("0.1")
        self.param_widgets['Npx'].setText("256")

    def _simulate(self):
        # Gather parameters
        try:
            latitude = float(self.param_widgets['latitude'].text())
            declination = float(self.param_widgets['declination'].text())
            start_time = float(self.param_widgets['start_time'].text())
            duration = float(self.param_widgets['duration'].text())
            timestep = float(self.param_widgets['timestep'].text())
            central_freq = float(self.param_widgets['central_freq'].text())
            bandwidth = float(self.param_widgets['bandwidth'].text())
            nchan = int(self.param_widgets['nchan'].text())
            fov_size = float(self.param_widgets['fov'].text())
            Npx = int(self.param_widgets['Npx'].text())
        except Exception as e:
            self.fig.clear()
            ax = self.fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f"Error:\n{str(e)}", ha='center', va='center', fontsize=12, color='red')
            ax.axis('off')
            self.canvas.draw()
            self.current_uv_points = None
            return

        # Get current antenna array from InterferometricArrayWidget
        antenna = None
        if self.array_widget is not None:
            antenna = self.array_widget.get_current_antenna()
        if antenna is None:
            self.fig.clear()
            ax = self.fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, "No antenna array available.", ha='center', va='center', fontsize=12, color='red')
            ax.axis('off')
            self.canvas.draw()
            return

        # Compute the uv points and dirty beam
        baselines = argosim.antenna_utils.get_baselines(antenna)
        uv_points, _ = argosim.antenna_utils.uv_track_multiband(
            b_ENU=baselines, lat=latitude/180*np.pi, dec=declination/180*np.pi, 
            track_time=duration, t_0=start_time, n_times=int(duration*60/timestep),
            f=central_freq*1e9, df=bandwidth*1e9, n_freqs=nchan)
        self.current_uv_points = uv_points
        try:
            uv_mask, _ = argosim.imaging_utils.grid_uv_samples(uv_samples=uv_points, sky_uv_shape=(Npx, Npx), fov_size=(fov_size, fov_size))
        except ValueError as e:
            self.fig.clear()
            ax = self.fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f"Error:\n{str(e)}\nReduce FOV size.", ha='center', va='center', fontsize=12, color='red')
            ax.axis('off')
            self.canvas.draw()
            return
        dirty_beam = argosim.imaging_utils.uv2sky(uv_mask)

        # Plot
        self.fig.clear()
        ax1 = self.fig.add_subplot(1, 2, 1)
        # ax1.imshow(np.abs(uv_mask), origin='lower', cmap='viridis')
        # argosim.plot_utils.plot_sky_uv(np.abs(uv_mask), fov_size=(0.1,0.1), )
        argosim.plot_utils.plot_baselines(uv_points, ax=ax1, fig=self.fig)
        max_uv = (180/np.pi) * Npx / (2*fov_size) / 1e3
        ax1.set_xlim((-max_uv, max_uv))
        ax1.set_ylim((-max_uv, max_uv))
        ax1.set_title('uv Coverage')
        ax2 = self.fig.add_subplot(1, 2, 2)
        # ax2.imshow(dirty_beam, origin='lower', cmap='viridis')
        argosim.plot_utils.plot_sky(dirty_beam, fov_size=(fov_size, fov_size), ax=ax2, fig=self.fig, title='Dirty Beam')
        ax2.set_title("Dirty Beam")
        self.canvas.draw()

    def get_current_uv_points(self):
        return self.current_uv_points
    
    def get_current_fov_size(self):
        return self.param_widgets['fov'].text()
    
    def get_current_Npx(self):
        return self.param_widgets['Npx'].text()
