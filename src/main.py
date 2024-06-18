import cv2
import sys
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from PyQt6 import uic
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QScreen, QImage, QPixmap
from PyQt6.QtMultimedia import QCamera, QMediaDevices
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem, QWidget

import util
import algebra
from projector import Projection, ProjectionAreaSelection, ProjectionPointSelection, BoulderCreator
from surface import Surface
import algorithm_feature_match as feature_match
import algorithm_homography_rank as homography_rank
from thread_camera import Camera
from thread_aruco_tracker import ArucoTrack
from thread_mmpose_tracker import MMposeTrack as PoseTrack
#from thread_mediapipe_tracker import BlazePoseTrack as PoseTrack
from dialog_hold_detector import HoldDetectorDialog
from thread_hold_interaction import HoldInteractionTrack, InteractiveBoulderTrack
from thread_perspective_warper import PerspectiveWarper

class MainWindow(QMainWindow):
    projector_calibrator = None #created and destroyed as needed
    camera_calibrator = None #created and destroyed as needed
    hold_selector = None #created and destroyed as needed
    dialog = None #created and destroyed as needed
    perspective_warper = None #created and destroyed as needed
    boulder_creator = None #created and destroyed as needed

    file_test_image = "img/via1.jpg"
    file_feature_match_pattern = "img/calibration_rainbow.png"

    def __init__(self):
        super().__init__()
        
        uic.loadUi("gui_main_window.ui", self)
        self.setWindowTitle("TFG projecció roco")
        self.__loadResources()
        plt.ion() # force pyplot to use it's own thread for figures (QCoreApplication::exec: The event loop is already running)   

    def __loadResources(self): 
        self.available_screens = QScreen.virtualSiblings(self.screen())
        self.available_cameras = QMediaDevices.videoInputs()

        self.camera = Camera(QCamera(self.available_cameras[0]),0)
        self.surface = Surface()
        self.projector = Projection(self.available_screens[0], True)
        self.camera_preview = Projection(self.available_screens[0], False)
        self.boulder = []

        self.setImagePreview(self.file_test_image, self.label_testImage)
        self.setImagePreview(self.file_feature_match_pattern, self.label_calibrationImage)

        self.render_previews = False
        self.tracker_aruco = ArucoTrack()
        self.tracker_mmpose = PoseTrack()

        self.feature_match_pattern = None #CVimage
        self.feature_match_frame = None #CVimage
        self.reference_image = None #CVimage
        self.reference_image_reg = None #CVimage
        self.feature_matches_image = None #CVimage

        self.label_holdLabels.clear()
        self.updateTableRegularizedSize(self.surface.getRegularizedSize())
        self.updateCalibrationTable(self.table_surface_calibration, ["-","-","-","-"])
        self.updateCalibrationTable(self.table_camera_calibration, ["-","-","-","-"])
        self.updateCalibrationTable(self.table_screen1_calibration, ["-","-","-","-"])

        self.__setUpGui()

    def __setUpGui(self): 
        self.btn_surface_mCal.clicked.connect(self.startCameraSurfaceDetection)

        self.btn_screen1_mCal.clicked.connect(self.startManualProjectorSurfaceDetection)
        self.btn_screen1_aCal.clicked.connect(self.startAutoProjectorSurfaceDetection)

        self.btn_camera_aCal.clicked.connect(self.startFrontViewSurfaceDetection)

        self.btn_camera_preview.clicked.connect(self.startCameraPreview)
        self.cBox_camera.addItems([i.description() for i in self.available_cameras])
        self.cBox_camera.currentIndexChanged.connect(self.__updateCamera)
        self.cBox_camera.setCurrentIndex(0)
        self.__updateCamera()

        self.cBox_screen1.addItems([i.name() for i in self.available_screens])
        self.cBox_screen1.currentIndexChanged.connect(self.__updateProjectorScreen)
        self.cBox_screen1.setCurrentIndex(0)
        self.__updateProjectorScreen()

        self.cBox_controlScreen.addItems([i.name() for i in self.available_screens])
        self.cBox_controlScreen.currentIndexChanged.connect(self.__updateMainScreen)
        self.cBox_controlScreen.setCurrentIndex(0)

        self.btn_testImage_openFile.clicked.connect(self.openTestImage)
        self.btn_calibrationImage_openFile.clicked.connect(self.openCalibrationImage)

        self.btn_testImage_project.clicked.connect(self.projectTestImage)
        self.btn_testImage_projectWarped.clicked.connect(self.projectTestImageDistorted)

        self.btn_surface_showSelection.clicked.connect(self.showSelectedSurface)
        self.btn_screen1_showFMatch.clicked.connect(self.showFeatureMatch)

        self.btn_preview_aruco_tracker.clicked.connect(self.previewArucoTracker)
        self.btn_show_aruco_tracker.clicked.connect(self.startArucoTracker)
        self.btn_preview_mmpose_tracker.clicked.connect(self.previewPoseTracker)
        self.btn_show_mmpose_tracker.clicked.connect(self.startPoseTracker)
        
        self.btn_holdSelector.clicked.connect(self.startHoldManualSelection)
        self.btn_holdDetection.clicked.connect(self.startHoldDetection)
        self.btn_holdsDelete.clicked.connect(self.clearHolds)
        self.btn_projectHolds.clicked.connect(self.projectHolds)
        self.btn_show_hold_interaction.clicked.connect(self.startHoldInteraction)
        
        self.btn_regularized_rotate.clicked.connect(self.startRegularizedHorizontal)

        self.action_render_preview.toggled.connect(self.setRenderLivePreviews)
        self.setRenderLivePreviews()
        self.action_clearAll.triggered.connect(self.__loadResources)

        self.action_saveRefImages.triggered.connect(self.saveReferenceImages)
        self.action_saveRefVideo.triggered.connect(self.saveReferenceVideo)

        self.action_create_boulder.triggered.connect(self.createBoulder)
        self.action_start_boulder.triggered.connect(self.startBoulder)
        self.action_delete_boulder.triggered.connect(self.deleteBoulder)

    def __updateCamera(self): 
        self.camera.setCamera(QCamera(self.available_cameras[self.cBox_camera.currentIndex()]),self.cBox_camera.currentIndex())
        self.table_camera_size.setItem(0,1,QTableWidgetItem(str(self.camera.getSize().width())))
        self.table_camera_size.setItem(1,1,QTableWidgetItem(str(self.camera.getSize().height())))

    def __updateProjectorScreen(self): 
        screen = self.available_screens[self.cBox_screen1.currentIndex()]
        self.table_screen1_size.setItem(0,1,QTableWidgetItem(str(screen.size().width())))
        self.table_screen1_size.setItem(1,1,QTableWidgetItem(str(screen.size().height())))
        self.projector.setScreenObj(screen)

    def __updateMainScreen(self): 
        self.camera_preview.setScreenObj(self.available_screens[self.cBox_controlScreen.currentIndex()])

    def closeEvent(self, event):
        plt.close('all')
        QApplication.quit()

#   ---------------------------------------------------- WORK AREA ----------------------------------------------------  #


#   ---------------------------------------------------- WORK AREA ----------------------------------------------------  #

#  SIGNALS & SLOTS ----------------------------------------------------  #
    def startWindowThread(self, thread, close_slots=[]):
        try: thread.disconnect()
        except Exception: pass
        for slot in close_slots: thread.signal_close.connect(slot)
        self.btn_stop.clicked.connect(thread.close)
        thread.start()

    def startSelectionThread(self, thread, close_slots=[], done_slots=[], click_slots=[]):
        try: thread.disconnect()
        except Exception: pass
        for slot in close_slots: thread.signal_close.connect(slot)
        for slot in done_slots: thread.signal_done.connect(slot)
        for slot in click_slots: thread.signal_click.connect(slot)
        self.btn_stop.clicked.connect(thread.close)
        thread.start()

    def startTrackerThread(self, thread, preview_slots=[], detection_slots=[], data_slots=[]):
        try: thread.disconnect()
        except Exception: pass
        for slot in preview_slots: thread.signal_preview.connect(slot)
        for slot in detection_slots: thread.signal_detection.connect(slot)
        for slot in data_slots: thread.signal_data.connect(slot)

    def startGenericThread(self, thread, signal, slots=[]):
        try: thread.disconnect()
        except Exception: pass
        for slot in slots: signal.connect(slot)
        thread.start()

#  GUI HELPERS  ----------------------------------------------------  #
    def setImagePreview(self, i, qLabel):  #accepts file path, Qimage or CVimage
        if isinstance(i, QImage) or isinstance(i, str) : 
            img = QPixmap(i)
        elif isinstance(i, np.ndarray):
            img = QPixmap(util.QimageFromCVimage(i))
        else: return

        qLabel.setPixmap(img.scaled(qLabel.size().width(), qLabel.size().height(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        qLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def openImage(self): 
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter("Images (*.png *.jpg)")
        dialog.setDirectory('./')
        dialog.setViewMode(QFileDialog.ViewMode.Detail)
        if dialog.exec():
            fileNames = dialog.selectedFiles()
            return str(Path(fileNames[0]))

    def openTestImage(self): 
        path = self.openImage()
        if path is None: return
        self.file_test_image = path
        self.setImagePreview(self.file_test_image, self.label_testImage)

    def openCalibrationImage(self):
        path = self.openImage()
        if path is None: return 
        self.file_feature_match_pattern = path
        self.setImagePreview(self.file_feature_match_pattern, self.label_calibrationImage)

    def updateSurfacePreview(self):
        if self.surface.getCameraHomography() is None:
            reference_image_masked = cv2.bitwise_and(self.reference_image, self.surface.getCameraMask())
            self.setImagePreview(reference_image_masked, self.label_holdLabels)
            return

        self.reference_image_reg = cv2.warpPerspective(self.reference_image, self.surface.getCameraHomography(), self.surface.getRegularizedSize())
        self.reference_image_reg = cv2.bitwise_and(self.reference_image_reg, self.surface.getRegularizedMask())
        self.updateTableRegularizedSize(self.surface.getRegularizedSize())
        
        paintedHolds = util.paintRectangles(self.reference_image_reg.copy(), self.surface.getHolds())
        self.setImagePreview(paintedHolds, self.label_holdLabels)

    def updateTableRegularizedSize(self,size): 
        self.table_surface1_size.setItem(0,1,QTableWidgetItem(str(size[0])))
        self.table_surface1_size.setItem(1,1,QTableWidgetItem(str(size[1])))

    def updateCalibrationTable(self, table, points): 
        table.setItem(0,0,QTableWidgetItem(str(points[0])))
        table.setItem(1,0,QTableWidgetItem(str(points[1])))
        table.setItem(1,1,QTableWidgetItem(str(points[2])))
        table.setItem(0,1,QTableWidgetItem(str(points[3])))

    def setRenderLivePreviews(self):
        self.render_previews = self.action_render_preview.isChecked()

#  DATA PREVIEWS  ----------------------------------------------------  #
    def startCameraPreview(self):
        self.startGenericThread(self.camera, self.camera.signal_frame, slots=[self.camera_preview.setImage])
        self.startWindowThread(thread=self.camera_preview, close_slots=[self.camera.stop])

    def showSelectedSurface(self): 
        if self.reference_image is None: return

        painted = util.paintSelection(self.reference_image.copy(), self.surface.getCameraMap())
        rgb = cv2.cvtColor(painted, cv2.COLOR_BGR2RGB) #matlplotlob uses RGB and opencv BGR
        plt.figure()
        plt.imshow(rgb, 'gray')
        plt.show()

    def showFeatureMatch(self): 
        if self.feature_matches_image is None: return

        plt.figure()
        plt.imshow(self.feature_matches_image, 'gray')
        plt.show()

    def projectHolds(self):
            if self.surface.getCameraMap() == []: return
            if self.surface.getProjectorMap() == []: return

            img = np.zeros_like(self.reference_image_reg)
            util.paintRectangles(img, self.surface.getHolds(), (255,255,255), -1)
            img = cv2.bitwise_and(img, self.surface.getRegularizedMask())
            projection = cv2.warpPerspective(img, self.surface.getProjectorHomography(), self.surface.getProjectorSize())
            qImg = util.QimageFromCVimage(projection)

            self.projector.setImage(qImg)
            self.projector.start()

    def projectTestImage(self): 
        qImg = QImage(self.file_test_image)
        self.projector.setImage(qImg)
        self.projector.start()

    def projectTestImageDistorted(self): 
        if self.surface.getProjectorMap() == []: return

        img = cv2.imread(self.file_test_image)
        img_distorted = cv2.warpPerspective(img, self.surface.getProjectorHomographyFromSize(img.shape[1],img.shape[0]), self.surface.getProjectorSize())
        qImg = util.QimageFromCVimage(img_distorted)

        self.projector.setImage(qImg)
        self.projector.start()

#  SURFACE CALIBRATION  ----------------------------------------------------  #
    def startCameraSurfaceDetection(self): 
        self.projector.setImage(None)
        self.camera_calibrator = ProjectionAreaSelection(self.available_screens[self.cBox_controlScreen.currentIndex()], False)
        self.startGenericThread(self.camera, self.camera.signal_frame, slots=[self.camera_calibrator.setImage])
        self.startSelectionThread(self.camera_calibrator, close_slots=[self.projector.stop], done_slots=[self.updateCameraSurfaceSelection], click_slots=[self.camera.stop])
        self.startWindowThread(thread=self.projector, close_slots=[])

    def updateCameraSurfaceSelection(self, w, h, p): 
        self.surface.setCameraParametres(p,w,h)
        self.updateCalibrationTable(self.table_surface_calibration, self.surface.getCameraMap())
        self.reference_image = self.camera.getLastFrame()
        self.updateSurfacePreview()
        if self.camera_calibrator:  self.camera_calibrator.deleteLater()
        self.camera_calibrator = None

    def startManualProjectorSurfaceDetection(self): 
        self.projector_calibrator = ProjectionAreaSelection(self.available_screens[self.cBox_screen1.currentIndex()], True)
        self.startSelectionThread(self.projector_calibrator, close_slots=[], done_slots=[self.updateProjectorSurfaceSelection], click_slots=[])

    def startAutoProjectorSurfaceDetection(self): 
        if self.surface.getCameraMap() == []: return

        img = cv2.imread(self.file_feature_match_pattern, cv2.IMREAD_COLOR)
        self.feature_match_pattern = img[0:self.projector.getSize().height(),0:self.projector.getSize().width()].copy()
        self.projector.setImage(util.QimageFromCVimage(self.feature_match_pattern))

        self.startWindowThread(thread=self.projector, close_slots=[])
        self.startGenericThread(self.camera, self.camera.signal_frame, slots=[self.camera_preview.setImage])
        self.startWindowThread(thread=self.camera_preview, close_slots=[self.camera.stop, self.projector.stop, self.helperAutoProjectorSurfaceSelection])

    def helperAutoProjectorSurfaceSelection(self): 
        self.feature_match_frame = cv2.bitwise_and(self.camera.getLastFrame(), self.surface.getCameraMask())

        H, self.feature_matches_image = feature_match.featureMatching(self.feature_match_frame, self.feature_match_pattern)
        
        if H is None: return
        pts = cv2.perspectiveTransform(np.array([self.surface.getCameraMap()], dtype=np.float32), H)

        self.updateProjectorSurfaceSelection(self.feature_match_pattern.shape[1], self.feature_match_pattern.shape[0], pts[0])

    def updateProjectorSurfaceSelection(self, w, h, p): 
        self.surface.setProjectorParametres(p,w,h)
        self.updateCalibrationTable(self.table_screen1_calibration, self.surface.getProjectorMap())
        if self.projector_calibrator: self.projector_calibrator.deleteLater()
        self.projector_calibrator = None

    def startFrontViewSurfaceDetection(self):
        if self.surface.getCameraMap() == []: return
        if self.surface.getProjectorMap() == []: return

        mask = self.surface.getProjectorMask()
        qImg = util.QimageFromCVimage(mask)
        self.projector.setImage(qImg)

        self.startWindowThread(thread=self.projector, close_slots=[])
        self.startGenericThread(self.camera, self.camera.signal_frame, slots=[self.tracker_aruco.detect])
        self.startTrackerThread(self.tracker_aruco, preview_slots=[self.camera_preview.setImage], detection_slots=[], data_slots=[])
        self.startWindowThread(thread=self.camera_preview, close_slots=[self.camera.stop, self.projector.stop, self.updateFrontViewSelection])

    def updateFrontViewSelection(self):
        W = self.tracker_aruco.detect(self.camera.getLastFrame())[1]
        if W == []: return

        h = homography_rank.getBestArucoHomography(W)

        new_pts = cv2.perspectiveTransform(np.array([self.surface.getCameraMap()], dtype=np.float32), h)
        new_pts = algebra.rotatePtsToHorizontalLine(new_pts, new_pts[0][0], new_pts[0][3])
        new_pts = algebra.translatePtsPositive(new_pts)
        new_pts = algebra.scalePtsToLimits(new_pts,self.surface.getMaxRegularizedsize())
        bb = algebra.get2DBoundingBox(new_pts[0])

        self.surface.setRegularizedParametres(new_pts[0], int(bb[0]), int(bb[1]))
        
        self.updateCalibrationTable(self.table_camera_calibration, self.surface.getRegularizedMap())
        self.updateSurfacePreview()

        if self.camera_calibrator:  self.camera_calibrator.deleteLater()
        self.camera_calibrator = None

#  SURFACE HORIZONATAL CORRECTION  ----------------------------------------------------  #
    def startRegularizedHorizontal(self): 
        if self.surface.getCameraMap() == []: return
        if self.surface.getProjectorMap() == []: return

        self.camera_calibrator = ProjectionAreaSelection(self.available_screens[self.cBox_controlScreen.currentIndex()], False, 2)
        self.camera_calibrator.setImage(util.QimageFromCVimage(self.reference_image_reg))
        self.startSelectionThread(self.camera_calibrator, close_slots=[], done_slots=[self.updateRegularizedHorizontal], click_slots=[])

    def updateRegularizedHorizontal(self, w, h, p):
        new_pts = algebra.rotatePtsToHorizontalLine(np.array([self.surface.getRegularizedMap()], dtype=np.float32), p[0], p[1])
        new_pts = algebra.translatePtsPositive(new_pts)
        new_pts = algebra.scalePtsToLimits(new_pts, self.surface.getMaxRegularizedsize())
        bb = algebra.get2DBoundingBox(new_pts[0])

        self.surface.setRegularizedParametres(new_pts[0], int(bb[0]), int(bb[1]))
        self.updateCalibrationTable(self.table_camera_calibration, self.surface.getRegularizedMap())
        self.updateSurfacePreview()

#  HOLD SELECTION  ----------------------------------------------------  #
    def clearHolds(self):
        self.surface.setHolds([])
        self.updateSurfacePreview()

    def setHolds(self, holds):
        self.surface.setHolds(holds)
        self.updateSurfacePreview()

    def startHoldManualSelection(self): 
        if self.reference_image_reg is None: return

        self.hold_selector = ProjectionPointSelection(self.available_screens[self.cBox_controlScreen.currentIndex()], False)
        self.hold_selector.setImage(self.reference_image_reg)
        self.hold_selector.setPoints(self.surface.getHolds())
        self.startSelectionThread(self.hold_selector, close_slots=[], done_slots=[self.setHolds], click_slots=[])

    def startHoldDetection(self):
        if self.reference_image_reg is None: return

        self.dialog = HoldDetectorDialog(self.reference_image_reg)
        self.startGenericThread(self.dialog, self.dialog.signal_done, slots=[self.setHolds])

#  LIVE VIDEO ANALISIS  ----------------------------------------------------  #
    def previewArucoTracker(self):
        self.camera_preview.setImage(self.reference_image_reg)
        self.startGenericThread(self.camera, self.camera.signal_frame, slots=[self.tracker_aruco.detect])
        self.startTrackerThread(self.tracker_aruco, preview_slots=[self.camera_preview.setImageWithResize], detection_slots=[], data_slots=[])
        self.startWindowThread(thread=self.camera_preview, close_slots=[self.camera.stop])

    def startArucoTracker(self):
        if self.surface.getCameraMap() == []: return
        if self.surface.getProjectorMap() == []: return

        self.perspective_warper = PerspectiveWarper(self.surface.getCameraProjectorHomography(), self.surface.getProjectorSize())

        self.tracker_aruco.setRenderPreview(self.render_previews)
        if self.render_previews: 
            self.startWindowThread(thread=self.camera_preview, close_slots=[self.projector.close])

        self.camera_preview.setImage(self.reference_image_reg)
        self.startGenericThread(self.camera, self.camera.signal_frame, slots=[self.tracker_aruco.detect])
        self.startTrackerThread(self.tracker_aruco, preview_slots=[self.camera_preview.setImageWithResize], detection_slots=[self.perspective_warper.apply], data_slots=[])
        self.perspective_warper.signal_done.connect(self.projector.setImageWithoutResize)
        self.startWindowThread(thread=self.projector, close_slots=[self.camera.stop])

    def previewPoseTracker(self):
        self.tracker_mmpose.setRenderPreview(True)
        self.camera_preview.setImage(self.reference_image_reg)
        self.startGenericThread(self.camera, self.camera.signal_frame, slots=[self.tracker_mmpose.detect])
        self.startTrackerThread(self.tracker_mmpose, preview_slots=[self.camera_preview.setImageWithResize], detection_slots=[], data_slots=[])
        self.startWindowThread(thread=self.camera_preview, close_slots=[self.camera.stop])

    def startPoseTracker(self):
        if self.surface.getCameraMap() == []: return
        if self.surface.getProjectorMap() == []: return

        self.perspective_warper = PerspectiveWarper(self.surface.getCameraProjectorHomography(), self.surface.getProjectorSize())

        self.tracker_mmpose.setRenderPreview(self.render_previews)
        if self.render_previews: 
            self.startWindowThread(thread=self.camera_preview, close_slots=[self.projector.close])

        self.camera_preview.setImage(self.reference_image_reg)
        self.startGenericThread(self.camera, self.camera.signal_frame, slots=[self.tracker_mmpose.detect])
        self.startTrackerThread(self.tracker_mmpose, preview_slots=[self.camera_preview.setImageWithResize], detection_slots=[self.perspective_warper.apply], data_slots=[])
        self.perspective_warper.signal_done.connect(self.projector.setImageWithoutResize)
        self.startWindowThread(thread=self.projector, close_slots=[self.camera.stop])

# INTERACTIVE EXPERIENCES  ----------------------------------------------------  #

    def startHoldInteraction(self):
        if self.surface.getCameraMap() == []: return
        if self.surface.getProjectorMap() == []: return

        self.hold_interaction_traker = HoldInteractionTrack(self.surface)

        self.tracker_mmpose.setRenderPreview(self.render_previews)
        self.hold_interaction_traker.setRenderPreview(self.render_previews)
        if self.render_previews:
            self.startWindowThread(thread=self.camera_preview, close_slots=[self.projector.close])
        
        self.camera_preview.setImage(self.reference_image_reg)
        self.startGenericThread(self.camera, self.camera.signal_frame, slots=[self.tracker_mmpose.detect])
        self.startTrackerThread(self.tracker_mmpose, preview_slots=[], detection_slots=[], data_slots=[self.hold_interaction_traker.detect])
        self.startTrackerThread(self.hold_interaction_traker, preview_slots=[self.camera_preview.setImageWithResize], detection_slots=[self.projector.setImageWithoutResize], data_slots=[])
        self.startWindowThread(thread=self.projector, close_slots=[self.camera.stop])

    def createBoulder(self):
        if self.reference_image_reg is None: return
        self.boulder_creator = BoulderCreator(self.available_screens[self.cBox_controlScreen.currentIndex()], False, self.reference_image_reg, self.surface.getHolds(), self.boulder)
        self.startSelectionThread(self.boulder_creator, close_slots=[], done_slots=[self.updateBoulder], click_slots=[])

    def updateBoulder(self, b):
        self.boulder = b
        if self.boulder_creator:  self.boulder_creator.deleteLater()

    def startBoulder(self):
        if self.surface.getCameraMap() == []: return
        if self.surface.getProjectorMap() == []: return
        if self.boulder == []: return

        self.boulder_traker = InteractiveBoulderTrack(self.surface, self.boulder)
        self.boulder_traker.startBoulder()

        self.tracker_mmpose.setRenderPreview(self.render_previews)
        self.boulder_traker.setRenderPreview(self.render_previews)
        if self.render_previews:
            self.startWindowThread(thread=self.camera_preview, close_slots=[self.projector.close])
        
        self.camera_preview.setImage(self.reference_image_reg)
        self.startGenericThread(self.camera, self.camera.signal_frame, slots=[self.tracker_mmpose.detect])
        self.startTrackerThread(self.tracker_mmpose, preview_slots=[], detection_slots=[], data_slots=[self.boulder_traker.detect])
        self.startTrackerThread(self.boulder_traker, preview_slots=[self.camera_preview.setImageWithResize], detection_slots=[self.projector.setImageWithoutResize], data_slots=[])
        self.startWindowThread(thread=self.projector, close_slots=[self.camera.stop])

    def deleteBoulder(self):
        self.boulder = []

#  SAVE IMAGES FOR BENCHMARKING  ----------------------------------------------------  #
    def saveReferenceImages(self): 
        directory = QFileDialog.getExistingDirectory(self, "Open Directory","./")
        if directory != '':
            if self.reference_image is not None: 
                cv2.imwrite(directory + "/reference_img.jpg", self.reference_image)
                if self.surface.getCameraMap() != []:
                    regularized = cv2.warpPerspective(self.reference_image, self.surface.getCameraHomography(), self.surface.getRegularizedSize())
                    regularized = cv2.bitwise_and(regularized, self.surface.getRegularizedMask())
                    cv2.imwrite(directory + "/reference_img_reg.jpg", regularized)
            
            if self.feature_match_frame is not None: 
                cv2.imwrite(directory + "/feature_match_frame.jpg", self.feature_match_frame)
                if self.surface.getCameraMap() != []:
                    regularized = cv2.warpPerspective(self.feature_match_frame, self.surface.getCameraHomography(), self.surface.getRegularizedSize())
                    cv2.imwrite(directory + "/feature_match_frame_reg.jpg", regularized)

            if self.feature_match_pattern is not None: 
                cv2.imwrite(directory + "/feature_match_pattern.jpg", self.feature_match_pattern)

            if self.surface.getProjectorMap() != []:
                img = np.zeros((self.surface.getRegularizedSize()[1],self.surface.getRegularizedSize()[0],3), dtype=np.uint8)
                img.fill(255)
                regularized = cv2.warpPerspective(img, self.surface.getProjectorHomography(), self.surface.getProjectorSize())
                cv2.imwrite(directory + "/homography_representation.jpg", regularized)
            
            print("Images written to: " + directory)

    def saveReferenceVideo(self): 
        if self.surface.getCameraMap() == []: return

        self.vid_directory = QFileDialog.getExistingDirectory(self, "Open Directory","./")
        self.vid_counter = 0
        if self.vid_directory != '':
            self.startGenericThread(self.camera, self.camera.signal_frame, slots=[self.saveReferenceVideoHelper])

    def saveReferenceVideoHelper(self,frame): 
        cv2.imwrite(self.vid_directory + "/vid_" + str(self.vid_counter) + ".jpg", frame)
        if self.surface.getCameraMap() is not None:
            regularized = cv2.warpPerspective(frame, self.surface.getCameraHomography(), self.surface.getRegularizedSize())
            regularized = cv2.bitwise_and(regularized, self.surface.getRegularizedMask())
            cv2.imwrite(self.vid_directory + "/vid_reg_" + str(self.vid_counter) + ".jpg", regularized)
        self.vid_counter += 1

app = QApplication(sys.argv)
mainWindow = MainWindow() # Create a Qt widget, which will be our window.
mainWindow.show() # show, showFullSreen, showMazimized
app.exec() # Start the event loop.