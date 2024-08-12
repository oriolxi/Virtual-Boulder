from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QCursor

from dialogs.generic import ImageWindow

class AreaSelectionWindow(ImageWindow):
    pen = QPen(Qt.GlobalColor.green, 2)
    signal_done = pyqtSignal(int, int, list)
    signal_click = pyqtSignal()

    def __init__(self, scrn, fs, n=4):
        super().__init__(scrn, fs)

        self.stored_points = [] #stored points as a list of [x,y]
        self.point_n = n

        self.last_position = None
        self.last_canvas = None

        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        self.setMouseTracking(True)
        self.label.setMouseTracking(True)

    def mousePressEvent(self, event):
        if (len(self.stored_points) == 0): self.signal_click.emit()
        if (len(self.stored_points) == self.point_n): self.close()

        self.last_position = event.position()
        self.stored_points.append(self.last_position)
        self.last_canvas = self.label.pixmap()

    def mouseMoveEvent(self, event):
        if self.last_position is None: return

        canvas = self.last_canvas.copy()
        painter = QPainter(canvas)
        painter.setPen(self.pen);
        painter.drawLine(self.last_position, event.position())
        painter.end()
        self.label.setPixmap(canvas)

    def closeEvent(self, event):
        super(AreaSelectionWindow, self).closeEvent(event)
        
        w = self.image.size().width()
        h = self.image.size().height()
        points = [[p.x()/self.scaling,p.y()/self.scaling] for p in self.stored_points]

        if (len(points) == self.point_n): self.signal_done.emit(w,h,points)