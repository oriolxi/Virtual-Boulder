import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer

class Camera(QThread):
    timer = None
    camera_index = None
    capture_session = None
    vid_rate = 40 # milliseconds (25fps)
    last_frame = None
    signal_frame = pyqtSignal(np.ndarray)

    def __init__(self, idx):
        super().__init__()
        self.timer = QTimer()
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self._processNextFrame)
        self.setCamera(idx)

    def _processNextFrame(self):
        ret, frame = self.capture_session.read()
        if isinstance(frame, np.ndarray):
            self.signal_frame.emit(frame)
            self.last_frame = frame

    def getSize(self):
        self.capture_session = cv2.VideoCapture(self.camera_index)
        while not self.capture_session.isOpened (): pass
        width = int(self.capture_session.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture_session.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.capture_session.release()
        return (width, height)
        
    def start(self):
        self.capture_session = cv2.VideoCapture(self.camera_index)
        while not self.capture_session.isOpened (): pass
        while self.capture_session.read()[1] is None: pass
        self.timer.start(self.vid_rate)

    def stop(self):
        self.timer.stop()
        self.capture_session.release()

    def getLastFrame(self):
        return self.last_frame

    def setCamera(self, idx):
        self.camera_index = idx
