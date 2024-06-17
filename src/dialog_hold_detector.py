import cv2
from PyQt6 import uic
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QDialog

import util
import algorithm_hold_detection as hold_detection 

class HoldDetectorDialog(QDialog):
    holds = []

    image = None
    threshImg = None
    closingImg = None
    contoursImg = None
    boundingBoxesImg = None


    signal_done = pyqtSignal(list)

    def __init__(self, img):
        QDialog.__init__(self)
        uic.loadUi("gui_hold_detection.ui", self)

        self.image = img

        Qimg = QPixmap(util.QimageFromCVimage(self.image))
        w = self.label_imgOriginal.size().width()
        h = self.label_imgOriginal.size().height()
        self.label_imgOriginal.setPixmap(Qimg.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.label_imgOriginal.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.__setUpGui()
        self.__computeDetection()

    def __setUpGui(self):
        self.slider_minH.valueChanged.connect(self.__computeDetection)
        self.slider_minS.valueChanged.connect(self.__computeDetection)
        self.slider_minV.valueChanged.connect(self.__computeDetection)
        self.slider_maxH.valueChanged.connect(self.__computeDetection)
        self.slider_maxS.valueChanged.connect(self.__computeDetection)
        self.slider_maxV.valueChanged.connect(self.__computeDetection)

        self.slider_minH.valueChanged.connect(self.__updateLabels)
        self.slider_minS.valueChanged.connect(self.__updateLabels)
        self.slider_minV.valueChanged.connect(self.__updateLabels)
        self.slider_maxH.valueChanged.connect(self.__updateLabels)
        self.slider_maxS.valueChanged.connect(self.__updateLabels)
        self.slider_maxV.valueChanged.connect(self.__updateLabels)

        self.checkBox_removeAruco.stateChanged.connect(self.__computeDetection)

        self.comboBox_kernelType.addItems(["Rectangle", "Ellipse", "Cross"])
        self.comboBox_kernelType.currentIndexChanged.connect(self.__computeDetection)
        self.spinBox_kernelSize.valueChanged.connect(self.__computeDetection)
        self.spinBox_openingIters.valueChanged.connect(self.__computeDetection)
        self.spinBox_closingIters.valueChanged.connect(self.__computeDetection)

        self.spinBox_minArea.valueChanged.connect(self.__computeDetection)
        self.spinBox_maxArea.valueChanged.connect(self.__computeDetection)
        self.doubleSpinBox_minwhRatio.valueChanged.connect(self.__computeDetection)

        self.buttonBox.rejected.connect(self.close)
        self.buttonBox.accepted.connect(self.accept)

    def __updateLabels(self):
        self.label_minH.setText(str(self.slider_minH.value()))
        self.label_minS.setText(str(self.slider_minS.value()))
        self.label_minV.setText(str(self.slider_minV.value()))
        self.label_maxH.setText(str(179 - self.slider_maxH.value()))
        self.label_maxS.setText(str(255 - self.slider_maxS.value()))
        self.label_maxV.setText(str(255 - self.slider_maxV.value()))

    def __computeDetection(self):
        params = hold_detection.defaultParametres()

        params["min_HSV"] = (self.slider_minH.value(), self.slider_minS.value(), self.slider_minV.value())
        params["max_HSV"] = (179 - self.slider_maxH.value(),  255 - self.slider_maxS.value(), 255 - self.slider_maxV.value())

        params["remove_aruco"] = self.checkBox_removeAruco.isChecked()

        params["kernel_type"] = self.comboBox_kernelType.currentIndex()
        params["kernel_size"] = self.spinBox_kernelSize.value()
        params["open_iters"] = self.spinBox_openingIters.value()
        params["close_iters"] = self.spinBox_closingIters.value()

        params["min_area"] = self.spinBox_minArea.value()
        params["max_area"] = self.spinBox_maxArea.value()
        params["min_wh_ratio"] = self.doubleSpinBox_minwhRatio.value()

        self.holds, images = hold_detection.detectHolds(self.image, params)
        self.threshImg = images[0]
        self.closingImg = images[1]
        self.contoursImg = images[2]
        self.boundingBoxesImg = images[3]

        self.__updatePreviews()

    def __updatePreviews(self):
        self.closingImg = cv2.cvtColor(self.closingImg, cv2.COLOR_GRAY2BGR)
        Qimg = QPixmap(util.QimageFromCVimage(self.closingImg))
        w = self.label_imgThreshold.size().width()
        h = self.label_imgThreshold.size().height()
        self.label_imgThreshold.setPixmap(Qimg.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.label_imgThreshold.setAlignment(Qt.AlignmentFlag.AlignCenter)

        Qimg = QPixmap(util.QimageFromCVimage(self.contoursImg))
        w = self.label_imgContours.size().width()
        h = self.label_imgContours.size().height()
        self.label_imgContours.setPixmap(Qimg.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.label_imgContours.setAlignment(Qt.AlignmentFlag.AlignCenter)

        Qimg = QPixmap(util.QimageFromCVimage(self.boundingBoxesImg))
        w = self.label_imgDetectedHolds.size().width()
        h = self.label_imgDetectedHolds.size().height()
        self.label_imgDetectedHolds.setPixmap(Qimg.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.label_imgDetectedHolds.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def start(self):
        self.show()

    def accept(self):
        self.signal_done.emit(self.holds)
        self.close()