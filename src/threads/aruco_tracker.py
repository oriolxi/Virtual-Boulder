import cv2
import cv2.aruco as aruco
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

class ArucoTrack(QThread):
    signal_preview = pyqtSignal(np.ndarray)
    signal_detection = pyqtSignal(np.ndarray)
    signal_data = pyqtSignal(list)

    render_preview = True

    def __init__(self):
        super().__init__()
        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50) # DICT_APRILTAG_16h5, DICT_4X4_50, 
        self.aruco_parameters = cv2.aruco.DetectorParameters()
        self.aruco_parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX #Possible values are CORNER_REFINE_NONE, CORNER_REFINE_SUBPIX, CORNER_REFINE_CONTOUR, CORNER_REFINE_APRILTAG
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dictionary, self.aruco_parameters) 

    def setRenderPreview(self, b):
        self.render_preview = b

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (corners, ids, rejected) = self.aruco_detector.detectMarkers(gray)

        markers = []
        if len(corners) > 0:
            ids = ids.flatten() # Flatten the ArUco IDs list
            for (marker_corners, marker_id) in zip(corners, ids):
                markers.append( np.array([[[x[0],x[1]] for x in marker_corners.reshape((4, 2))]], dtype=np.float32) )

        img = np.zeros_like(frame)
        img = aruco.drawDetectedMarkers(img, corners, ids)
        self.signal_detection.emit(img)

        preview = None
        if self.render_preview:
            preview = frame.copy()
            aruco.drawDetectedMarkers(preview, corners, ids)
            self.signal_preview.emit(preview)
        
        self.signal_data.emit([ids, markers, preview])
        return [ids, markers, preview]
