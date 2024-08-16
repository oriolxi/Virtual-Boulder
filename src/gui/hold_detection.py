import cv2
import numpy as np
from PyQt6 import uic
from PyQt6.QtCore import Qt, QRectF, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QPen
from PyQt6.QtWidgets import QDialog

import util
import algorithms.hold_detection as hold_detection 

class HoldDetectionDialog(QDialog):
    signal_close = pyqtSignal()
    signal_done = pyqtSignal(list)
    signal_click = pyqtSignal(np.ndarray)

    def __init__(self, img):
        QDialog.__init__(self)
        uic.loadUi("gui/hold_detection.ui", self)

        self.image = img

        Qimg = QPixmap(util.QimageFromCVimage(self.image))
        w, h = self.label_imgOriginal.size().width(), self.label_imgOriginal.size().height()
        self.label_imgOriginal.setPixmap(Qimg.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.label_imgOriginal.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.__setUpGui()

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
        self.closingImg = images[1]
        self.contoursImg = images[2]
        self.boundingBoxesImg = images[3]

        self.__updatePreviews()

    def __updatePreviews(self):
        self.closingImg = cv2.cvtColor(self.closingImg, cv2.COLOR_GRAY2BGR)
        Qimg = QPixmap(util.QimageFromCVimage(self.closingImg))
        w, h = self.label_imgThreshold.size().width(), self.label_imgThreshold.size().height()
        self.label_imgThreshold.setPixmap(Qimg.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.label_imgThreshold.setAlignment(Qt.AlignmentFlag.AlignCenter)

        Qimg = QPixmap(util.QimageFromCVimage(self.contoursImg))
        w, h = self.label_imgContours.size().width(), self.label_imgContours.size().height()
        self.label_imgContours.setPixmap(Qimg.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.label_imgContours.setAlignment(Qt.AlignmentFlag.AlignCenter)

        Qimg = QPixmap(util.QimageFromCVimage(self.boundingBoxesImg))
        w, h = self.label_imgDetectedHolds.size().width(), self.label_imgDetectedHolds.size().height()
        self.label_imgDetectedHolds.setPixmap(Qimg.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.label_imgDetectedHolds.setAlignment(Qt.AlignmentFlag.AlignCenter)

        img_black = Qimg.copy()
        img_black.fill(Qt.GlobalColor.transparent)
        painter = QPainter(img_black)
        painter.setPen(QPen(Qt.GlobalColor.green, 2))
        for rec in self.holds:
            painter.drawRect(QRectF(rec[0], rec[1], rec[2], rec[3]))
        painter.end()
        self.signal_click.emit(util.CVimageFromQimage(img_black.toImage()))

    def start(self):
        self.show()
        self.__computeDetection()

    def accept(self):
        self.signal_done.emit(self.holds)
        self.close()

    def closeEvent(self, event):
        super(HoldDetectionDialog, self).closeEvent(event)
        self.signal_close.emit()