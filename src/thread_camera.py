import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer

class Camera(QThread):
    timer = None
    camera = None # QCamera
    camera_index = None

    capture_session = None
    vid_rate = 40 # milliseconds (25fps)

    last_frame = None
    signal_frame = pyqtSignal(np.ndarray)

    def __init__(self, cam, idx):
        super().__init__()
        self.timer = QTimer()
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.nextFrameSlot)
        self.setCamera(cam, idx)

    def __processNewFrame(self, frame):
        if isinstance(frame, np.ndarray):
            self.signal_frame.emit(frame)
            self.last_frame = frame

    def nextFrameSlot(self):
        ret, frame = self.capture_session.read()
        self.__processNewFrame(frame)

    def getSize(self):
        return self.camera.cameraDevice().videoFormats()[-1].resolution()
        
    def getLastFrame(self):
        return self.last_frame

    def setCamera(self, cam, idx): 
            self.camera = cam
            self.camera_index = idx
        
    def start(self):
        self.capture_session = cv2.VideoCapture(self.camera_index)
        while not self.capture_session.isOpened (): pass
        while self.capture_session.read()[1] is None: pass
        self.timer.start(self.vid_rate)

    def stop(self):
        self.timer.stop()
        self.capture_session.release()