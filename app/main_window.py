# main_window.py
import sys
import matplotlib
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QComboBox, QLabel
)

from widget_array import InterferometricArrayWidget
from widget_apsyn import ApertureSynthesisWidget
from widget_imag import ImagingWidget

class SimulationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("argosim: radio interferometric simulator")
        self.current_theme = "Light"

        # Main layout
        main_layout = QVBoxLayout()

        # Theme selector
        toggle_row = QHBoxLayout()
        toggle_row.addStretch()
        theme_label = QLabel("Theme:")
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark", "Terminal"])
        self.theme_combo.currentTextChanged.connect(self._apply_theme)
        toggle_row.addWidget(theme_label)
        toggle_row.addWidget(self.theme_combo)
        main_layout.addLayout(toggle_row)

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

        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

    def _apply_theme(self, theme):
        self.current_theme = theme
        styles = _THEMES[theme]
        self.setStyleSheet(styles["stylesheet"])
        matplotlib.rcParams.update(styles["mpl"])

        # Set scatter colors per theme
        if theme == "Light":
            self.array_widget.scatter_colors = {"antenna": "mediumblue", "baseline": "darkred"}
            self.aperture_widget.scatter_color = "darkred"
        else:
            self.array_widget.scatter_colors = {"antenna": "gray", "baseline": "gray"}
            self.aperture_widget.scatter_color = "gray"

        # Re-plot all figures so they fully pick up new rcParams
        self.array_widget._plot_array_and_baselines()
        if self.aperture_widget.current_uv_points is not None:
            self.aperture_widget._simulate()
            self.imaging_widget._simulate_imaging()

        # fig.clear() resets facecolor, so re-apply after re-plotting
        mpl = styles["mpl"]
        for widget in (self.array_widget, self.aperture_widget, self.imaging_widget):
            widget.fig.set_facecolor(mpl["figure.facecolor"])
            widget.canvas.draw()



# ---------------------------------------------------------------------------
# Theme definitions
# ---------------------------------------------------------------------------

_SECTION_TITLE_RULE = """
QLabel#section-title {
    font-weight: bold;
    font-size: 16px;
}
"""

_BUTTON_3D = """
QPushButton {{
    background-color: {bg};
    color: {fg};
    border: 1px solid {border_dark};
    border-bottom: 2px solid {border_dark};
    border-right: 2px solid {border_dark};
    border-top: 1px solid {border_light};
    border-left: 1px solid {border_light};
    padding: 6px 12px;
    border-radius: 3px;
}}
QPushButton:hover {{
    background-color: {hover};
    color: {hover_fg};
}}
QPushButton:pressed {{
    border-top: 2px solid {border_dark};
    border-left: 2px solid {border_dark};
    border-bottom: 1px solid {border_light};
    border-right: 1px solid {border_light};
}}
"""

_SCROLLBAR_TEMPLATE = """
QScrollBar:vertical {{
    background-color: {bg};
    width: 12px;
}}
QScrollBar::handle:vertical {{
    background-color: {handle};
    border-radius: 6px;
    min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar:horizontal {{
    background-color: {bg};
    height: 12px;
}}
QScrollBar::handle:horizontal {{
    background-color: {handle};
    border-radius: 6px;
    min-width: 20px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}
"""

# ---- Light theme ----

_LIGHT_STYLESHEET = _SECTION_TITLE_RULE

_LIGHT_MPL = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.edgecolor": "black",
}

# ---- Dark theme (traditional) ----

_DARK_STYLESHEET = (
    """
QWidget {
    background-color: #2b2b2b;
    color: #e0e0e0;
}
"""
    + _SECTION_TITLE_RULE
    + """
QLineEdit {
    background-color: #3c3c3c;
    color: #e0e0e0;
    border: 1px solid #555555;
    padding: 4px;
}
QComboBox {
    background-color: #3c3c3c;
    color: #e0e0e0;
    border: 1px solid #555555;
    padding: 4px;
}
QComboBox QAbstractItemView {
    background-color: #3c3c3c;
    color: #e0e0e0;
    selection-background-color: #555555;
}
"""
    + _BUTTON_3D.format(
        bg="#4a4a4a",
        fg="#e0e0e0",
        border_dark="#333333",
        border_light="#666666",
        hover="#5a5a5a",
        hover_fg="#ffffff",
    )
    + """
QScrollArea {
    background-color: #2b2b2b;
    border: none;
}
"""
    + _SCROLLBAR_TEMPLATE.format(bg="#2b2b2b", handle="#555555")
)

_DARK_MPL = {
    "figure.facecolor": "#2b2b2b",
    "axes.facecolor": "#3c3c3c",
    "text.color": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0",
    "axes.edgecolor": "#555555",
}

# ---- Terminal theme ----

_TERMINAL_STYLESHEET = (
    """
QWidget {
    background-color: #1a1a2e;
    color: #00ff41;
}
"""
    + _SECTION_TITLE_RULE
    + """
QLineEdit {
    background-color: #16213e;
    color: #00ff41;
    border: 1px solid #0f3460;
    padding: 4px;
}
QComboBox {
    background-color: #16213e;
    color: #00ff41;
    border: 1px solid #0f3460;
    padding: 4px;
}
QComboBox QAbstractItemView {
    background-color: #16213e;
    color: #00ff41;
    selection-background-color: #0f3460;
}
"""
    + _BUTTON_3D.format(
        bg="#0f3460",
        fg="#00ff41",
        border_dark="#091b36",
        border_light="#1a4a8a",
        hover="#e94560",
        hover_fg="#ffffff",
    )
    + """
QScrollArea {
    background-color: #1a1a2e;
    border: none;
}
"""
    + _SCROLLBAR_TEMPLATE.format(bg="#1a1a2e", handle="#0f3460")
)

_TERMINAL_MPL = {
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "text.color": "#00ff41",
    "axes.labelcolor": "#00ff41",
    "xtick.color": "#00ff41",
    "ytick.color": "#00ff41",
    "axes.edgecolor": "#0f3460",
}

# ---- Theme registry ----

_THEMES = {
    "Light": {"stylesheet": _LIGHT_STYLESHEET, "mpl": _LIGHT_MPL},
    "Dark": {"stylesheet": _DARK_STYLESHEET, "mpl": _DARK_MPL},
    "Terminal": {"stylesheet": _TERMINAL_STYLESHEET, "mpl": _TERMINAL_MPL},
}
