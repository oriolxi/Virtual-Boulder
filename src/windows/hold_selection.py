import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QRectF
from PyQt6.QtGui import QPixmap, QPainter, QPen, QCursor

import algebra
from windows.generic import ImageWindow

class HoldSelectionWindow(ImageWindow):
    pen = QPen(Qt.GlobalColor.green, 2)
    signal_done = pyqtSignal(list)

    def __init__(self, scrn, fs): 
        super().__init__(scrn, fs)
        self.stored_points = [] #stored rectangles as a list of [x,y,w,h] (top_left corner, width, height)
        self.moving = False
        self.drawing = False

        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        self.setMouseTracking(True)
        self.label.setMouseTracking(True)

    def _paintRectangles(self, canvas, rectangles):
        painter = QPainter(canvas)
        painter.setPen(self.pen)
        for rec in rectangles: 
            painter.drawRect(QRectF(rec[0], rec[1], rec[2], rec[3]))
        painter.end()
        return canvas

    def _isPointInsideAnyRectangle(self, point): 
        for p in self.stored_points: 
            if algebra.isPointInsideRectangle(point, p): return p
        return None

    def updateImage(self): 
        super().updateImage()
        self.overlay = QPixmap(self.screen_w, self.screen_h)
        self.overlay.fill(Qt.GlobalColor.transparent)
        self.new_overlay = self.overlay.copy()

    def setPoints(self, points):
        self.stored_points = np.multiply(points, self.scaling).tolist()
        self._paintRectangles(self.overlay, self.stored_points)
        self.updateOverlay(self.overlay)

    def updateOverlay(self, overlay):
        canvas = overlay.copy()
        painter = QPainter(canvas)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOver)
        painter.drawImage(0,0,self.image.scaled(self.screen_w, self.screen_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        painter.end()
        self.label.setPixmap(canvas)

    def mousePressEvent(self, event): 
        point = [event.position().x(), event.position().y()]
        if point[0] is None or point[1] is None: return
        collision = self._isPointInsideAnyRectangle(point)

        if event.button() == Qt.MouseButton.LeftButton:
            if not collision:
                self.drawing = True
                self.rect_origin = point
                self.rect_w, self.rect_h = -1, -1

            if collision:
                self.moving = True
                self.moving_origin = point
                self.rect_origin = [collision[0], collision[1]]
                self.rect_w, self.rect_h = collision[2], collision[3]
                self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))

        if event.button() == Qt.MouseButton.RightButton and collision:
            self.stored_points.remove(collision)
            self.overlay.fill(Qt.GlobalColor.transparent)
            self._paintRectangles(self.overlay, self.stored_points)

    def mouseMoveEvent(self, event):
        if self.drawing or self.moving:
            point_b = [event.position().x(), event.position().y()]
            
            if self.drawing:
                self.rect_w = int(point_b[0] - self.rect_origin[0])
                self.rect_h = int(point_b[1] - self.rect_origin[1])

            if self.moving:
                dist_x = int(point_b[0] - self.moving_origin[0])
                dist_y = int(point_b[1] - self.moving_origin[1])
                self.rect_origin = [self.rect_origin[0] + dist_x, self.rect_origin[1] + dist_y]
                self.moving_origin = point_b

            self.new_overlay = self.overlay.copy()
            self._paintRectangles(self.new_overlay, [[self.rect_origin[0], self.rect_origin[1], self.rect_w, self.rect_h]])
            self.updateOverlay(self.new_overlay)

    def mouseReleaseEvent(self, event): 
        if event.button() == Qt.MouseButton.LeftButton:
            self.overlay = self.new_overlay
            if self.drawing:
                self.drawing = False
                if self.rect_w and self.rect_h:
                    if self.rect_w < 0:
                        self.rect_origin[0] += self.rect_w
                        self.rect_w = -self.rect_w
                    if self.rect_h < 0:
                        self.rect_origin[1] += self.rect_h
                        self.rect_h = -self.rect_h
            self.stored_points.append([self.rect_origin[0], self.rect_origin[1], self.rect_w, self.rect_h])
            
            if self.moving:
                self.moving = False
                self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
                self.stored_points.append([self.rect_origin[0], self.rect_origin[1], self.rect_w, self.rect_h])
        
        if event.button() == Qt.MouseButton.RightButton:
            self.updateOverlay(self.overlay)

    def closeEvent(self, event): 
        super(HoldSelectionWindow, self).closeEvent(event)
        points = np.divide(self.stored_points, self.scaling).astype(int).tolist()
        self.signal_done.emit(points)