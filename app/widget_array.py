# widget_array.py
from PyQt6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QComboBox, QLineEdit, QPushButton
)
from PyQt6.QtCore import Qt

from matplotlib.figure import Figure

import argosim.antenna_utils
import argosim.plot_utils

from utils import ScrollableFigureCanvas

class InterferometricArrayWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        title = QLabel("Interferometric Array")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 16px;")
        layout.addWidget(title)

        # Array type selection
        type_row = QHBoxLayout()
        type_label = QLabel("Array Type:")
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Y-shaped", "Circular", "Uniform", "Templates"])
        self.type_combo.setCurrentText("Y-shaped")  # Set default
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        type_row.addWidget(type_label)
        type_row.addWidget(self.type_combo)
        layout.addLayout(type_row)

        # Parameter input fields (create all, show/hide as needed)
        self.param_widgets = {}
        self.param_labels = {}
        self.param_fields = {}

        # Y-shaped and Circular: n_antenna, radius, alpha
        self.param_widgets['n_antenna'] = QLineEdit()
        self.param_labels['n_antenna'] = QLabel("Number of Antennas:")
        self.param_widgets['radius'] = QLineEdit()
        self.param_labels['radius'] = QLabel("Array Radius (m):")
        self.param_widgets['alpha'] = QLineEdit()
        self.param_labels['alpha'] = QLabel("Rotation Angle (°):")

        # Uniform: n_x, n_y, span_width, span_height
        self.param_widgets['n_x'] = QLineEdit()
        self.param_labels['n_x'] = QLabel("Number of antennas in east-west direction:")
        self.param_widgets['n_y'] = QLineEdit()
        self.param_labels['n_y'] = QLabel("Number of antennas in north-south direction:")
        self.param_widgets['span_width'] = QLineEdit()
        self.param_labels['span_width'] = QLabel("East-west span (m):")
        self.param_widgets['span_height'] = QLineEdit()
        self.param_labels['span_height'] = QLabel("North-south span (m):")

        # Template arrays: select array
        self.template_combo = QComboBox()
        self.template_combo.addItems(["Argos", 
                                  "Kat-7",
                                  "Meerkat",
                                  "SKA-Mid_197"])
        self.template_combo.setCurrentText("Meerkat")
        self.template_combo.currentTextChanged.connect(self._plot_array_and_baselines)

        # Horizontal parameter layouts
        self.param_row = QHBoxLayout()
        self.param_row2 = QHBoxLayout()  # Only used for Uniform
        layout.addLayout(self.param_row)
        layout.addLayout(self.param_row2)

        # Buttons row
        button_row = QHBoxLayout()
        self.plot_button = QPushButton("Plot Array and Baselines")
        self.plot_button.clicked.connect(self._plot_array_and_baselines)
        button_row.addWidget(self.plot_button, 4)
        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self._reset_defaults)
        self.reset_button.clicked.connect(self._plot_array_and_baselines)
        button_row.addWidget(self.reset_button, 1)
        layout.addLayout(button_row)

        # Set default values for Y-shaped
        self._reset_defaults()

        # Matplotlib FigureCanvas
        self.fig = Figure()
        self.canvas = ScrollableFigureCanvas(self.fig)
        self.canvas.setMinimumHeight(400)
        self.canvas.setMinimumWidth(1000)
        layout.addWidget(self.canvas, stretch=1)

        self.setLayout(layout)
        self._on_type_changed(self.type_combo.currentText())
        self.current_antenna = None
        self._plot_array_and_baselines()

    def _on_type_changed(self, text):
        # Clear the parameter rows
        for row in [self.param_row, self.param_row2]:
            while row.count():
                item = row.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.setParent(None)
        # Add relevant widgets
        if text == "Y-shaped":
            for key in ["n_antenna", "radius", "alpha"]:
                self.param_row.addWidget(self.param_labels[key])
                self.param_row.addWidget(self.param_widgets[key])
        elif text == "Circular":
            for key in ["n_antenna", "radius"]:
                self.param_row.addWidget(self.param_labels[key])
                self.param_row.addWidget(self.param_widgets[key])
        elif text == "Uniform":
            # First row: n_x, n_y
            for key in ["n_x", "n_y"]:
                self.param_row.addWidget(self.param_labels[key])
                self.param_row.addWidget(self.param_widgets[key])
            # Second row: span_width, span_height
            for key in ["span_width", "span_height"]:
                self.param_row2.addWidget(self.param_labels[key])
                self.param_row2.addWidget(self.param_widgets[key])
        elif text == "Templates":
            self.param_row.addWidget(self.template_combo)
        self._plot_array_and_baselines()

    def _plot_array_and_baselines(self):
        array_type = self.type_combo.currentText()
        try:
            if array_type == "Y-shaped":
                n_antenna = int(self.param_widgets['n_antenna'].text())
                radius = float(self.param_widgets['radius'].text())
                alpha = float(self.param_widgets['alpha'].text())
                antenna = argosim.antenna_utils.y_antenna_arr(n_antenna=n_antenna, r=radius, alpha=alpha)
            elif array_type == "Circular":
                n_antenna = int(self.param_widgets['n_antenna'].text())
                radius = float(self.param_widgets['radius'].text())
                antenna = argosim.antenna_utils.circular_antenna_arr(n_antenna=n_antenna, r=radius)
            elif array_type == "Uniform":
                n_x = int(self.param_widgets['n_x'].text())
                n_y = int(self.param_widgets['n_y'].text())
                span_width = float(self.param_widgets['span_width'].text())
                span_height = float(self.param_widgets['span_height'].text())
                antenna = argosim.antenna_utils.uni_antenna_array(n_antenna_E=n_x, n_antenna_N=n_y, E_lim=span_width, N_lim=span_height)
            elif array_type == "Templates":
                antenna_file = 'configs/arrays/' + self.template_combo.currentText() + '.enu.txt'
                antenna = argosim.antenna_utils.load_antenna_enu_txt(antenna_file)
            else:
                return
            baselines = argosim.antenna_utils.get_baselines(antenna)
            self.current_antenna = antenna
        except Exception as e:
            self.fig.clear()
            ax = self.fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f"Error:\n{str(e)}", ha='center', va='center', fontsize=12, color='red')
            ax.axis('off')
            self.canvas.draw()
            self.current_antenna = None
            return

        self.fig.clear()
        ax1 = self.fig.add_subplot(1, 2, 1)
        argosim.plot_utils.plot_antenna_arr(antenna, fig=self.fig, ax=ax1)
        ax2 = self.fig.add_subplot(1, 2, 2)
        argosim.plot_utils.plot_baselines(baselines, fig=self.fig, ax=ax2)
        self.canvas.draw()

    def get_current_antenna(self):
        return self.current_antenna

    def _reset_defaults(self):
        self.type_combo.setCurrentText("Y-shaped")
        self.param_widgets['n_antenna'].setText("7")
        self.param_widgets['radius'].setText("1e3")
        self.param_widgets['alpha'].setText("12")
        self.param_widgets['n_x'].setText("10")
        self.param_widgets['n_y'].setText("6")
        self.param_widgets['span_width'].setText("2e3")
        self.param_widgets['span_height'].setText("1e3")

