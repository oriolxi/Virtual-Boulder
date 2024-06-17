import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

class PerspectiveWarper(QThread):
    homography = None
    size = None

    signal_done = pyqtSignal(np.ndarray)

    def __init__(self, h, s):
        super().__init__()
        self.homography = h
        self.size = s

    def apply(self, frame):
        warped_frame = cv2.warpPerspective(frame, self.homography, self.size)        
        
        self.signal_done.emit(warped_frame)
        return warped_frame

    def setHomography(self, h):
        self.homography = h

    def setSize(self, s):
        self.size = s