# utils.py
from PyQt6.QtWidgets import QApplication
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class ScrollableFigureCanvas(FigureCanvas):
    def wheelEvent(self, event):
        # Forward wheel event to parent scroll area
        parent = self.parent()
        if parent:
            QApplication.sendEvent(parent, event)
        else:
            super().wheelEvent(event)
