# main_window.py
import sys
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea
)

from widget_array import InterferometricArrayWidget
from widget_apsyn import ApertureSynthesisWidget
from widget_imag import ImagingWidget

class SimulationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("argosim: radio interferometric simulator")

        # Main content widget and layout
        content_widget = QWidget()
        content_layout = QVBoxLayout()
        self.array_widget = InterferometricArrayWidget()
        content_layout.addWidget(self.array_widget)
        self.aperture_widget = ApertureSynthesisWidget(array_widget=self.array_widget)
        content_layout.addWidget(self.aperture_widget)
        self.imaging_widget = ImagingWidget(aperture_widget=self.aperture_widget)
        content_layout.addWidget(self.imaging_widget)
        content_widget.setLayout(content_layout)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content_widget)

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)
