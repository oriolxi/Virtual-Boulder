import math
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QScreen, QImage
from PyQt6.QtWidgets import QLabel, QWidget, QHBoxLayout

import util

class ImageWindow(QWidget):
    scaling_limit = 0.85 #maximum size of window when not full screen    
    signal_close = pyqtSignal()
    
    def __init__(self, scrn, fs=True, img=None, parent=None):
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

        self.label.setPixmap( QPixmap.fromImage(self.image) )
        self.update() #update() does not cause an immediate repaint; instead it schedules a paint event for processing when Qt returns to the main event loop. This permits Qt to optimize for more speed and less flicker than a call to repaint() does.

    def setImageWithResize(self, img):
        if isinstance(img, QImage) or img is  None: self.image = img
        if isinstance(img, np.ndarray): self.image = util.QimageFromCVimage(img)

        canvas = QPixmap.fromImage(self.image)
        canvas = canvas.scaled(self.screen_w, self.screen_h, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation) #KeepAspectRatioByExpanding is also an option
        self.label.setPixmap(canvas)
        self.update() #update() does not cause an immediate repaint; instead it schedules a paint event for processing when Qt returns to the main event loop. This permits Qt to optimize for more speed and less flicker than a call to repaint() does.

    def setImage(self, img):
        if isinstance(img, QImage) or img is  None: self.image = img
        if isinstance(img, np.ndarray): self.image = util.QimageFromCVimage(img)
        
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