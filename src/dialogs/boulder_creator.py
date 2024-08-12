import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal

import util
import algebra
from dialogs.generic import ImageWindow
from boulder import Boulder, Placement, renderBoulderPreview

class BoulderCreatorWindow(ImageWindow):
    signal_done = pyqtSignal(Boulder)
    signal_click = pyqtSignal(np.ndarray)

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

        img_black = self.image.copy()
        img_black.fill(Qt.GlobalColor.transparent)
        canvas_projection = renderBoulderPreview(self.boulder, self.holds, img_black)
        self.signal_click.emit(util.CVimageFromQimage(canvas_projection.toImage()))

    def start(self):
        super(BoulderCreatorWindow, self).start()
        self.__paintBoulder()

    def mousePressEvent(self, event):
        # get mouse click coordinates
        point = [int(event.position().x()), int(event.position().y())]
        if point[0] is None or point[1] is None: return

        # scale the coordinate to original image
        point[0] = int(point[0] / self.scaling)
        point[1] = int(point[1] / self.scaling)

        # check if point falls inside of a hold
        hold = self.__isPointInsideHold(point)
        if hold is None: return
        hold_idx = self.holds.index(hold)

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
        super(ImageWindow, self).closeEvent(event)
        self.signal_done.emit(self.boulder)