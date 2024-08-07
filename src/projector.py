import math
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QRectF
from PyQt6.QtGui import QPixmap, QScreen, QImage, QPainter, QPen, QBrush, QCursor
from PyQt6.QtWidgets import QLabel, QWidget, QHBoxLayout

import util
import algebra
from boulder import Boulder, Placement, renderBoulderPreview

class Projection(QWidget):
    screen = None #QScreen
    screen_w = None
    screen_h = None
    is_full_screen = True

    label = None
    image = None #QImage
    scaling = 0.85
    scaling_limit = 0.85 #maximum size of window when not full screen
    
    signal_close = pyqtSignal()
    
    def __init__(self, scrn, fs, img=None, parent=None):
        super().__init__(parent)

        self.label = QLabel()
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.screen = scrn
        self.is_full_screen = fs
        self.setImage(img)

        self.__update()

    def __update(self):
        self.updateImage()
        self.setScreen(self.screen)
        self.move(self.screen.geometry().x(), self.screen.geometry().y())

    def updateImage(self):
        self.screen_w = self.screen.size().width() if self.is_full_screen else math.floor(self.screen.size().width()*self.scaling_limit)
        self.screen_h = self.screen.size().height() if self.is_full_screen else math.floor(self.screen.size().height()*self.scaling_limit)
        
        if self.image is None:
            canvas = QPixmap(self.screen_w, self.screen_h)
            canvas.fill(Qt.GlobalColor.white)
            self.image = canvas.toImage()
        else:
            canvas = QPixmap.fromImage(self.image)
            if self.is_full_screen: canvas = canvas.scaled(self.screen_w, self.screen_h, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation) #KeepAspectRatioByExpanding is also an option
            if not self.is_full_screen: canvas = canvas.scaled(self.screen_w, self.screen_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        
        if not self.is_full_screen: self.resize(canvas.size())

        self.screen_w = canvas.size().width()
        self.screen_h = canvas.size().height()
        self.scaling = self.screen_w/self.image.size().width()
        self.label.setPixmap(canvas)
        self.update() #update() does not cause an immediate repaint; instead it schedules a paint event for processing when Qt returns to the main event loop. This permits Qt to optimize for more speed and less flicker than a call to repaint() does.

    def setImageWithoutResize(self, img):
        if isinstance(img, QImage) or img is  None: self.image = img
        if isinstance(img, np.ndarray): self.image = util.QimageFromCVimage(img)

        canvas = QPixmap.fromImage(self.image)
        self.label.setPixmap(canvas)
        self.update() #update() does not cause an immediate repaint; instead it schedules a paint event for processing when Qt returns to the main event loop. This permits Qt to optimize for more speed and less flicker than a call to repaint() does.

    def setImageWithResize(self, img):
        if isinstance(img, QImage) or img is  None: self.image = img
        if isinstance(img, np.ndarray): self.image = util.QimageFromCVimage(img)

        canvas = QPixmap.fromImage(self.image)
        canvas = canvas.scaled(self.screen_w, self.screen_h, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation) #KeepAspectRatioByExpanding is also an option

        self.label.setPixmap(canvas)
        self.update() #update() does not cause an immediate repaint; instead it schedules a paint event for processing when Qt returns to the main event loop. This permits Qt to optimize for more speed and less flicker than a call to repaint() does.

    def setImage(self, img):
        if isinstance(img, QImage) or img is  None: 
            self.image = img
            self.updateImage()
        if isinstance(img, np.ndarray):
            self.image = util.QimageFromCVimage(img)
            self.updateImage()

    def setScreenObj(self, scrn):
        if isinstance(scrn, QScreen): 
            self.screen = scrn
            self.__update()

    def getSize(self):
        return self.screen.size()

    def start(self):
        if self.is_full_screen: self.showFullScreen()
        if not self.is_full_screen: 
            self.show() #show() equivalent to .setVisible(True)
            self.showNormal()

    def stop(self):
        self.close() #hide() equivalent to .setVisible(False)

    def closeEvent(self, event):
        self.signal_close.emit()

class ProjectionAreaSelection(Projection):
    stored_points = None #stored points as a list of [x,y]
    last_position = None
    last_canvas = None
    point_n = None
    pen = QPen(Qt.GlobalColor.green, 2)

    signal_done = pyqtSignal(int, int, list)
    signal_click = pyqtSignal()

    def __init__(self, scrn, fs, n=4):
        super().__init__(scrn, fs)

        self.stored_points = []
        self.point_n = n

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
        super(ProjectionAreaSelection, self).closeEvent(event)
        
        w = self.image.size().width()
        h = self.image.size().height()
        points = [[p.x()/self.scaling,p.y()/self.scaling] for p in self.stored_points]

        if (len(points) == self.point_n): self.signal_done.emit(w,h,points)

class ProjectionPointSelection(Projection):
    stored_points = None #stored rectangles as a list of [x,y,w,h] (top_left corner, width, height)
    overlay = None
    new_overlay = None
    drawing = False
    moving = False
    rect_origin = None
    moving_origin = None
    rect_w = -1
    rect_h = -1
    pen = QPen(Qt.GlobalColor.green, 2)
    
    signal_done = pyqtSignal(list)

    def __init__(self, scrn, fs): 
        super().__init__(scrn, fs)

        self.stored_points = []

        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        self.setMouseTracking(True)
        self.label.setMouseTracking(True)

    def __paintRectangles(self, canvas, rectangles):
        painter = QPainter(canvas)
        painter.setPen(self.pen)
        for rec in rectangles: painter.drawRect(QRectF(rec[0], rec[1], rec[2], rec[3]))
        painter.end()
        
        return canvas

    def __isPointInsideAnyRectangle(self, point): 
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
        self.__paintRectangles(self.overlay, self.stored_points)
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
        collision = self.__isPointInsideAnyRectangle(point)

        if event.button() == Qt.MouseButton.LeftButton:
            if not collision:
                self.drawing = True
                self.rect_origin = point
                self.rect_w = -1
                self.rect_h = -1

            if collision:
                self.moving = True
                self.moving_origin = point
                self.rect_origin = [collision[0], collision[1]]
                self.rect_w = collision[2]
                self.rect_h = collision[3]
                self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))

        if collision:
            self.stored_points.remove(collision)
            self.overlay.fill(Qt.GlobalColor.transparent)
            self.__paintRectangles(self.overlay, self.stored_points)

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
            self.__paintRectangles(self.new_overlay, [[self.rect_origin[0], self.rect_origin[1], self.rect_w, self.rect_h]])
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
        super(ProjectionPointSelection, self).closeEvent(event)
        points = np.divide(self.stored_points, self.scaling).astype(int).tolist()
        self.signal_done.emit(points)

class BoulderCreator(Projection):
    boulder = None
    holds = None

    signal_done = pyqtSignal(Boulder)

    def __init__(self, scrn, fs, img, h, b):
        super().__init__(scrn, fs, img)

        self.holds = h
        self.boulder = b
        self.__paintBoulder()

        self.setMouseTracking(True)
        self.label.setMouseTracking(True)

    def __isPointInsideHold(self, point): 
        for p in self.holds: 
            if algebra.isPointInsideRectangle(point, p): return p
        return None

    def __paintBoulder(self):
        canvas = renderBoulderPreview(self.boulder, self.holds, self.image)
        canvas = canvas.scaled(self.screen_w, self.screen_h, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation) #KeepAspectRatioByExpanding is also an option
        self.label.setPixmap(canvas)

    def mousePressEvent(self, event):
        # get mouse click coordinates
        point = [int(event.position().x()), int(event.position().y())]
        if point[0] is None or point[1] is None: return

        # scale the coordinate to original image
        point[0] = int(point[0] / self.scaling)
        point[1] = int(point[1] / self.scaling)

        # check if point falls inside of a hold
        hold_idx = -1
        for h in self.holds: 
            if algebra.isPointInsideRectangle(point, h):
                hold_idx = self.holds.index(h)
        if hold_idx < 0: return

        if event.button() == Qt.MouseButton.RightButton:
            self.boulder.removeStepHold(hold_idx) # remove last instance of hold from boulder
        
        if event.button() == Qt.MouseButton.LeftButton:
            if self.boulder.holdInBoulder(hold_idx):
                idx, step = self.boulder.getLastStepWithHold(hold_idx)
                if event.modifiers() and Qt.KeyboardModifier.ShiftModifier:
                    self.boulder.addStep(hold_idx, Placement.HAND_RIGHT)
                elif step[1] == Placement.HAND_MATCHING:
                    self.boulder.replaceStep(hold_idx, Placement.HAND_RIGHT, idx)
                elif step[1] == Placement.HAND_RIGHT:
                    self.boulder.replaceStep(hold_idx, Placement.HAND_LEFT, idx)
                elif step[1] == Placement.HAND_LEFT:
                    self.boulder.replaceStep(hold_idx, Placement.HAND_MATCHING, idx)
            else:
                self.boulder.addStep(hold_idx, Placement.HAND_RIGHT)
           
        self.__paintBoulder()

    def mouseMoveEvent(self, event):
        pass

    def mouseReleaseEvent(self, event):
        pass

    def closeEvent(self, event): 
        super(Projection, self).closeEvent(event)
        self.signal_done.emit(self.boulder)
