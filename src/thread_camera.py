import cv2
import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer

import util

class Camera(QThread):
    timer = None
    camera = None #QCamera
    camera_index = None

    cap = None
    vid_rate = 40 #milliseconds

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
        ret, frame = self.cap.read()
        self.__processNewFrame(frame)

    def getSize(self):
        ''' #get capture property open cv
        self.cap = cv2.VideoCapture(self.camera_index)
        width  = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
        self.cap.release()
        '''
        return self.camera.cameraDevice().videoFormats()[-1].resolution()
        
    def getLastFrame(self):
        return self.last_frame

    def setCamera(self, cam, idx): 
            self.camera = cam
            self.camera_index = idx
        
    def start(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        while not self.cap.isOpened (): pass
        while self.cap.read()[1] is None: pass
        self.timer.start(self.vid_rate)

    def stop(self):
        self.timer.stop()
        self.cap.release()