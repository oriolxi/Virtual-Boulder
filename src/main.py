import os
import cv2
import sys
import pickle
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
from thread_mmpose_tracker import MMposeTracker as PoseTrack
from dialog_hold_detector import HoldDetectorDialog
from thread_hold_interaction import FreeClimbingTracker, InteractiveBoulderTrack
from thread_perspective_warper import PerspectiveWarper

class MainWindow(QMainWindow):
    wdw_projected_area_selector = None #created and destroyed as needed
    wdw_windowed_area_selector = None #created and destroyed as needed
    wdw_hold_editor = None #created and destroyed as needed
    wdw_dialog = None #created and destroyed as needed
    wdw_boulder_creator = None  # created and destroyed as needed

    file_test_image = "img/via1.jpg"
    file_fmatch_pattern = "img/calibration_rainbow.png"

    def __init__(self):
        super().__init__()
        
        uic.loadUi("gui_main_window.ui", self)
        self.setWindowTitle("Virtual Boulder")
        self.__loadResources()
        plt.ion() # force pyplot to use it's own thread for figures (QCoreApplication::exec: The event loop is already running)   

    def __loadResources(self): 
        self.available_screens = QScreen.virtualSiblings(self.screen())
        self.available_cameras = QMediaDevices.videoInputs()

        self.camera = Camera(QCamera(self.available_cameras[0]),0)
        self.surface = Surface()
        self.projector = Projection(self.available_screens[0], True)
        self.boulder = []

        self.wdw_preview = Projection(self.available_screens[0], False)

        self.setImageLblPreview(self.file_test_image, self.lbl_preview_test_image)
        self.setImageLblPreview(self.file_fmatch_pattern, self.lbl_preview_fmatch_pattern)

        self.render_previews = False
        self.tracker_aruco = ArucoTrack()
        self.tracker_mmpose = PoseTrack()

        self.img_fmatch_pattern = None #CVimage
        self.img_fmatch_frame = None #CVimage
        self.img_reference = None #CVimage
        self.img_reference_frontview = None #CVimage
        self.img_feature_matches = None #CVimage

        self.lbl_preview_surface.clear()
        self.updateFrontViewSizeTable(self.surface.getSizeSurface())
        self.updateRoiTable(self.tbl_roi_frontview, ["-", "-", "-", "-"])
        self.updateRoiTable(self.tbl_roi_camera, ["-", "-", "-", "-"])
        self.updateRoiTable(self.tbl_roi_projector, ["-", "-", "-", "-"])

        self.__setUpGui()

    def __setUpGui(self): 
        self.btn_srfdet_camera.clicked.connect(self.startCameraSurfaceDetection)
        self.btn_srfdet_projector_m.clicked.connect(self.startManualProjectorSurfaceDetection)
        self.btn_srfdet_projector_a.clicked.connect(self.startAutoProjectorSurfaceDetection)
        self.btn_srfdet_frontview.clicked.connect(self.startFrontViewSurfaceDetection)
        self.btn_rotate_frontview.clicked.connect(self.startFrontViewHorizon)

        self.btn_preview_camera.clicked.connect(self.previewCamera)
        self.cbox_available_cameras.addItems([i.description() for i in self.available_cameras])
        self.cbox_available_cameras.currentIndexChanged.connect(self.__updateCamera)
        self.cbox_available_cameras.setCurrentIndex(0)
        self.__updateCamera()

        self.cbox_available_projector_screens.addItems([i.name() for i in self.available_screens])
        self.cbox_available_projector_screens.currentIndexChanged.connect(self.__updateProjectorScreen)
        self.cbox_available_projector_screens.setCurrentIndex(0)
        self.__updateProjectorScreen()

        self.cbox_available_control_screens.addItems([i.name() for i in self.available_screens])
        self.cbox_available_control_screens.currentIndexChanged.connect(self.__updateMainScreen)
        self.cbox_available_control_screens.setCurrentIndex(0)

        self.btn_openfile_test_image.clicked.connect(self.loadTestImage)
        self.btn_openfile_fmatch_pattern.clicked.connect(self.loadFMatchPattern)

        self.btn_project_test_image.clicked.connect(self.projectTestImage)
        self.btn_project_warped_test_image.clicked.connect(self.projectTestImageWarped)

        self.btn_preview_camera_roi.clicked.connect(self.previewCameraRoi)
        self.btn_preview_fmatch.clicked.connect(self.previewFeatureMatch)

        self.btn_preview_aruco_tracker.clicked.connect(self.previewArucoTracker)
        self.btn_start_aruco_tracker.clicked.connect(self.startArucoTracker)
        self.btn_preview_mmpose_tracker.clicked.connect(self.previewPoseTracker)
        self.btn_start_mmpose_tracker.clicked.connect(self.startPoseTracker)
        
        self.btn_open_hold_editor.clicked.connect(self.showHoldEditor)
        self.btn_open_hold_detection.clicked.connect(self.showHoldAutoDetection)
        self.btn_delete_holds.clicked.connect(self.clearHolds)
        self.btn_project_holds.clicked.connect(self.projectHolds)
        
        self.act_set_render_previews.toggled.connect(self.setRenderLivePreviews)
        self.setRenderLivePreviews()
        self.act_clear_all_data.triggered.connect(self.__loadResources)

        self.act_savefile_reference_images.triggered.connect(self.saveReferenceImages)
        self.act_savefile_reference_video.triggered.connect(self.saveReferenceVideo)

        self.act_open_boulder_creator.triggered.connect(self.showBoulderCreator)
        self.act_delete_boulder.triggered.connect(self.deleteBoulder)
        self.act_start_boulder.triggered.connect(self.startBoulder)
        self.act_start_free_climbing.triggered.connect(self.startFreeClimbing)

        self.act_savefile_sroi_and_holds.triggered.connect(self.createSaveFile)
        self.act_openfile_sroi_and_holds.triggered.connect(self.loadSaveFile)

    def __updateCamera(self): 
        self.camera.setCamera(QCamera(self.available_cameras[self.cbox_available_cameras.currentIndex()]),self.cbox_available_cameras.currentIndex())
        self.tbl_size_camera.setItem(0,1,QTableWidgetItem(str(self.camera.getSize().width())))
        self.tbl_size_camera.setItem(1,1,QTableWidgetItem(str(self.camera.getSize().height())))

    def __updateProjectorScreen(self): 
        screen = self.available_screens[self.cbox_available_projector_screens.currentIndex()]
        self.tbl_size_projector.setItem(0,1,QTableWidgetItem(str(screen.size().width())))
        self.tbl_size_projector.setItem(1,1,QTableWidgetItem(str(screen.size().height())))
        self.projector.setScreenObj(screen)

    def __updateMainScreen(self): 
        self.wdw_preview.setScreenObj(self.available_screens[self.cbox_available_control_screens.currentIndex()])

    def closeEvent(self, event):
        plt.close('all')
        QApplication.quit()

#   ---------------------------------------------------- WORK AREA ----------------------------------------------------  #

    def loadSaveFile(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter("Images (*.pkl)")
        dialog.setDirectory('./saves/')
        dialog.setViewMode(QFileDialog.ViewMode.Detail)
        if dialog.exec():
            fileNames = dialog.selectedFiles()
        if os.path.isfile(fileNames[0]):
            with open(fileNames[0], 'rb') as inp:
                self.surface.setHolds(pickle.load(inp))
                roi = pickle.load(inp)
                size = pickle.load(inp)
                self.surface.setSurfaceParametres(roi,size[0],size[1])
            self.updateRoiTable(self.tbl_roi_camera, self.surface.getWallRoiSurface())
            self.updateFrontViewPreview()

    def createSaveFile(self):
        directory = QFileDialog.getExistingDirectory(self, "Open Directory","./saves/")
        if directory != '':
            with open(directory + '/save.pkl', 'wb') as output:
                pickle.dump(self.surface.getHolds(), output, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.surface.getWallRoiSurface(), output, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.surface.getSizeSurface(), output, pickle.HIGHEST_PROTOCOL)

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
    def setImageLblPreview(self, i, qLabel):  #accepts file path, Qimage or CVimage
        if isinstance(i, QImage) or isinstance(i, str) : 
            img = QPixmap(i)
        elif isinstance(i, np.ndarray):
            img = QPixmap(util.QimageFromCVimage(i))
        else: return

        qLabel.setPixmap(img.scaled(qLabel.size().width(), qLabel.size().height(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        qLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def openImageFile(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter("Images (*.png *.jpg)")
        dialog.setDirectory('./')
        dialog.setViewMode(QFileDialog.ViewMode.Detail)
        if dialog.exec():
            fileNames = dialog.selectedFiles()
            return str(Path(fileNames[0]))

    def loadTestImage(self):
        path = self.openImageFile()
        if path is None: return
        self.file_test_image = path
        self.setImageLblPreview(self.file_test_image, self.lbl_preview_test_image)

    def loadFMatchPattern(self):
        path = self.openImageFile()
        if path is None: return 
        self.file_fmatch_pattern = path
        self.setImageLblPreview(self.file_fmatch_pattern, self.lbl_preview_fmatch_pattern)

    def updateFrontViewPreview(self):
        if self.surface.getHomographyCS() is None:
            reference_image_masked = cv2.bitwise_and(self.img_reference, self.surface.getMaskCamera())
            self.setImageLblPreview(reference_image_masked, self.lbl_preview_surface)
            return

        self.img_reference_frontview = cv2.warpPerspective(self.img_reference, self.surface.getHomographyCS(),
                                                           self.surface.getSizeSurface())
        self.img_reference_frontview = cv2.bitwise_and(self.img_reference_frontview, self.surface.getMaskSurface())
        self.updateFrontViewSizeTable(self.surface.getSizeSurface())
        
        painted_holds = util.paintRectangles(self.img_reference_frontview.copy(), self.surface.getHolds())
        self.setImageLblPreview(painted_holds, self.lbl_preview_surface)

    def updateFrontViewSizeTable(self, size):
        self.table_surface1_size.setItem(0,1,QTableWidgetItem(str(size[0])))
        self.table_surface1_size.setItem(1,1,QTableWidgetItem(str(size[1])))

    def updateRoiTable(self, table, roi):
        table.setItem(0, 0, QTableWidgetItem(str(roi[0])))
        table.setItem(1, 0, QTableWidgetItem(str(roi[1])))
        table.setItem(1, 1, QTableWidgetItem(str(roi[2])))
        table.setItem(0, 1, QTableWidgetItem(str(roi[3])))

    def setRenderLivePreviews(self):
        self.render_previews = self.act_set_render_previews.isChecked()

#  DATA PREVIEWS  ----------------------------------------------------  #
    def previewCamera(self):
        self.startGenericThread(self.camera, self.camera.signal_frame, slots=[self.wdw_preview.setImage])
        self.startWindowThread(thread=self.wdw_preview, close_slots=[self.camera.stop])

    def previewCameraRoi(self):
        if self.img_reference is None: return

        painted = util.paintSelection(self.img_reference.copy(), self.surface.getWallRoiCamera())
        rgb = cv2.cvtColor(painted, cv2.COLOR_BGR2RGB) #matlplotlob uses RGB and opencv BGR
        plt.figure()
        plt.imshow(rgb, 'gray')
        plt.show()

    def previewFeatureMatch(self):
        if self.img_feature_matches is None: return

        plt.figure()
        plt.imshow(self.img_feature_matches, 'gray')
        plt.show()

    def projectHolds(self):
            if self.surface.getWallRoiCamera() == []: return
            if self.surface.getWallRoiProjector() == []: return

            img = np.zeros_like(self.img_reference_frontview)
            util.paintRectangles(img, self.surface.getHolds(), (255,255,255), -1)
            img = cv2.bitwise_and(img, self.surface.getMaskSurface())
            projection = cv2.warpPerspective(img, self.surface.getHomographySP(), self.surface.getSizeProjector())
            qImg = util.QimageFromCVimage(projection)

            self.projector.setImage(qImg)
            self.projector.start()

    def projectTestImage(self): 
        qImg = QImage(self.file_test_image)
        self.projector.setImage(qImg)
        self.projector.start()

    def projectTestImageWarped(self):
        if self.surface.getWallRoiProjector() == []: return

        img = cv2.imread(self.file_test_image)
        w = img.shape[1]
        h = img.shape[0]
        pts_src = np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]], dtype=np.float32)
        pts_dst = np.array(self.surface.getWallRoiProjector(), dtype=np.float32)
        H = cv2.getPerspectiveTransform(pts_src, pts_dst)
        img_distorted = cv2.warpPerspective(img, H, self.surface.getSizeProjector())
        qImg = util.QimageFromCVimage(img_distorted)

        self.projector.setImage(qImg)
        self.projector.start()

#  SURFACE CALIBRATION  ----------------------------------------------------  #
    def startCameraSurfaceDetection(self): 
        self.projector.setImage(None)
        self.wdw_windowed_area_selector = ProjectionAreaSelection(self.available_screens[self.cbox_available_control_screens.currentIndex()], False)
        self.startGenericThread(self.camera, self.camera.signal_frame, slots=[self.wdw_windowed_area_selector.setImage])
        self.startSelectionThread(self.wdw_windowed_area_selector, close_slots=[self.projector.stop], done_slots=[self.updateCameraSurfaceSelection], click_slots=[self.camera.stop])
        self.startWindowThread(thread=self.projector, close_slots=[])

    def updateCameraSurfaceSelection(self, w, h, p): 
        self.surface.setCameraParametres(p, w, h)
        self.updateRoiTable(self.tbl_roi_frontview, self.surface.getWallRoiCamera())
        self.img_reference = self.camera.getLastFrame()
        self.updateFrontViewPreview()
        if self.wdw_windowed_area_selector:  self.wdw_windowed_area_selector.deleteLater()
        self.wdw_windowed_area_selector = None

    def startManualProjectorSurfaceDetection(self): 
        self.wdw_projected_area_selector = ProjectionAreaSelection(self.available_screens[self.cbox_available_projector_screens.currentIndex()], True)
        self.startSelectionThread(self.wdw_projected_area_selector, close_slots=[], done_slots=[self.updateProjectorSurfaceSelection], click_slots=[])

    def startAutoProjectorSurfaceDetection(self): 
        if self.surface.getWallRoiCamera() == []: return

        img = cv2.imread(self.file_fmatch_pattern, cv2.IMREAD_COLOR)
        self.feature_match_pattern = img[0:self.projector.getSize().height(),0:self.projector.getSize().width()].copy()
        self.projector.setImage(util.QimageFromCVimage(self.feature_match_pattern))

        self.startWindowThread(thread=self.projector, close_slots=[])
        self.startGenericThread(self.camera, self.camera.signal_frame, slots=[self.wdw_preview.setImage])
        self.startWindowThread(thread=self.wdw_preview, close_slots=[self.camera.stop, self.projector.stop, self.helperAutoProjectorSurfaceSelection])

    def helperAutoProjectorSurfaceSelection(self): 
        self.img_fmatch_frame = cv2.bitwise_and(self.camera.getLastFrame(), self.surface.getMaskCamera())

        H, self.img_feature_matches = feature_match.featureMatching(self.img_fmatch_frame, self.feature_match_pattern)
        
        if H is None: return
        pts = cv2.perspectiveTransform(np.array([self.surface.getWallRoiCamera()], dtype=np.float32), H)

        self.updateProjectorSurfaceSelection(self.feature_match_pattern.shape[1], self.feature_match_pattern.shape[0], pts[0])

    def updateProjectorSurfaceSelection(self, w, h, p): 
        self.surface.setProjectorParametres(p, w, h)
        self.updateRoiTable(self.tbl_roi_projector, self.surface.getWallRoiProjector())
        if self.wdw_projected_area_selector: self.wdw_projected_area_selector.deleteLater()
        self.wdw_projected_area_selector = None

    def startFrontViewSurfaceDetection(self):
        if self.surface.getWallRoiCamera() == []: return
        if self.surface.getWallRoiProjector() == []: return

        mask = self.surface.getMaskProjector()
        qImg = util.QimageFromCVimage(mask)
        self.projector.setImage(qImg)

        self.startWindowThread(thread=self.projector, close_slots=[])
        self.startGenericThread(self.camera, self.camera.signal_frame, slots=[self.tracker_aruco.detect])
        self.startTrackerThread(self.tracker_aruco, preview_slots=[self.wdw_preview.setImage], detection_slots=[], data_slots=[])
        self.startWindowThread(thread=self.wdw_preview, close_slots=[self.camera.stop, self.projector.stop, self.updateFrontViewRoi])

    def updateFrontViewRoi(self):
        W = self.tracker_aruco.detect(self.camera.getLastFrame())[1]
        if W == []: return

        h = homography_rank.getBestArucoHomography(W)

        new_pts = cv2.perspectiveTransform(np.array([self.surface.getWallRoiCamera()], dtype=np.float32), h)
        new_pts = algebra.rotatePtsToHorizontalLine(new_pts, new_pts[0][0], new_pts[0][3])
        new_pts = algebra.translatePtsPositive(new_pts)
        new_pts = algebra.scalePtsToLimits(new_pts, self.surface.getMaxSizeSurface())
        bb = algebra.get2DBoundingBox(new_pts[0])

        self.surface.setSurfaceParametres(new_pts[0], int(bb[0]), int(bb[1]))

        self.updateRoiTable(self.tbl_roi_camera, self.surface.getWallRoiSurface())
        self.updateFrontViewPreview()

        if self.wdw_windowed_area_selector:  self.wdw_windowed_area_selector.deleteLater()
        self.wdw_windowed_area_selector = None

#  FrontView HORIZON CORRECTION  ----------------------------------------------------  #
    def startFrontViewHorizon(self):
        if self.surface.getWallRoiCamera() == []: return
        if self.surface.getWallRoiProjector() == []: return

        self.wdw_windowed_area_selector = ProjectionAreaSelection(self.available_screens[self.cbox_available_control_screens.currentIndex()], False, 2)
        self.wdw_windowed_area_selector.setImage(util.QimageFromCVimage(self.img_reference_frontview))
        self.startSelectionThread(self.wdw_windowed_area_selector, close_slots=[], done_slots=[self.updateFrontViewHorizon], click_slots=[])

    def updateFrontViewHorizon(self, w, h, p):
        new_pts = algebra.rotatePtsToHorizontalLine(np.array([self.surface.getWallRoiSurface()], dtype=np.float32), p[0], p[1])
        new_pts = algebra.translatePtsPositive(new_pts)
        new_pts = algebra.scalePtsToLimits(new_pts, self.surface.getMaxSizeSurface())
        bb = algebra.get2DBoundingBox(new_pts[0])

        self.surface.setSurfaceParametres(new_pts[0], int(bb[0]), int(bb[1]))
        self.updateRoiTable(self.tbl_roi_camera, self.surface.getWallRoiSurface())
        self.updateFrontViewPreview()

#  HOLD SELECTION  ----------------------------------------------------  #
    def clearHolds(self):
        self.surface.setHolds([])
        self.updateFrontViewPreview()

    def setHolds(self, holds):
        self.surface.setHolds(holds)
        self.updateFrontViewPreview()

    def showHoldEditor(self):
        if self.img_reference_frontview is None: return

        self.wdw_hold_editor = ProjectionPointSelection(self.available_screens[self.cbox_available_control_screens.currentIndex()], False)
        self.wdw_hold_editor.setImage(self.img_reference_frontview)
        self.wdw_hold_editor.setPoints(self.surface.getHolds())
        self.startSelectionThread(self.wdw_hold_editor, close_slots=[], done_slots=[self.setHolds], click_slots=[])

    def showHoldAutoDetection(self):
        if self.img_reference_frontview is None: return

        self.wdw_dialog = HoldDetectorDialog(self.img_reference_frontview)
        self.startGenericThread(self.wdw_dialog, self.wdw_dialog.signal_done, slots=[self.setHolds])

#  LIVE VIDEO ANALISIS  ----------------------------------------------------  #
    def previewArucoTracker(self):
        self.wdw_preview.setImage(self.img_reference_frontview)
        self.startGenericThread(self.camera, self.camera.signal_frame, slots=[self.tracker_aruco.detect])
        self.startTrackerThread(self.tracker_aruco, preview_slots=[self.wdw_preview.setImageWithResize], detection_slots=[], data_slots=[])
        self.startWindowThread(thread=self.wdw_preview, close_slots=[self.camera.stop])

    def startArucoTracker(self):
        if self.surface.getWallRoiCamera() == []: return
        if self.surface.getWallRoiProjector() == []: return

        self.perspective_warper = PerspectiveWarper(self.surface.getHomographyCP(), self.surface.getSizeProjector())

        self.tracker_aruco.setRenderPreview(self.render_previews)
        if self.render_previews: 
            self.startWindowThread(thread=self.wdw_preview, close_slots=[self.projector.close])

        self.wdw_preview.setImage(self.img_reference_frontview)
        self.startGenericThread(self.camera, self.camera.signal_frame, slots=[self.tracker_aruco.detect])
        self.startTrackerThread(self.tracker_aruco, preview_slots=[self.wdw_preview.setImageWithResize], detection_slots=[self.perspective_warper.apply], data_slots=[])
        self.perspective_warper.signal_done.connect(self.projector.setImageWithoutResize)
        self.startWindowThread(thread=self.projector, close_slots=[self.camera.stop])

    def previewPoseTracker(self):
        self.tracker_mmpose.setRenderPreview(True)
        self.wdw_preview.setImage(self.img_reference_frontview)
        self.startGenericThread(self.camera, self.camera.signal_frame, slots=[self.tracker_mmpose.detect])
        self.startTrackerThread(self.tracker_mmpose, preview_slots=[self.wdw_preview.setImageWithResize], detection_slots=[], data_slots=[])
        self.startWindowThread(thread=self.wdw_preview, close_slots=[self.camera.stop])

    def startPoseTracker(self):
        if self.surface.getWallRoiCamera() == []: return
        if self.surface.getWallRoiProjector() == []: return

        self.perspective_warper = PerspectiveWarper(self.surface.getHomographyCP(), self.surface.getSizeProjector())

        self.tracker_mmpose.setRenderPreview(self.render_previews)
        if self.render_previews: 
            self.startWindowThread(thread=self.wdw_preview, close_slots=[self.projector.close])

        self.wdw_preview.setImage(self.img_reference_frontview)
        self.startGenericThread(self.camera, self.camera.signal_frame, slots=[self.tracker_mmpose.detect])
        self.startTrackerThread(self.tracker_mmpose, preview_slots=[self.wdw_preview.setImageWithResize], detection_slots=[self.perspective_warper.apply], data_slots=[])
        self.perspective_warper.signal_done.connect(self.projector.setImageWithoutResize)
        self.startWindowThread(thread=self.projector, close_slots=[self.camera.stop])

# INTERACTIVE EXPERIENCES  ----------------------------------------------------  #

    def startFreeClimbing(self):
        if self.surface.getWallRoiCamera() == []: return
        if self.surface.getWallRoiProjector() == []: return

        self.tracker_free_climbing = FreeClimbingTracker(self.surface)

        self.tracker_mmpose.setRenderPreview(self.render_previews)
        self.tracker_free_climbing.setRenderPreview(self.render_previews)
        if self.render_previews:
            self.startWindowThread(thread=self.wdw_preview, close_slots=[self.projector.close])
        
        self.wdw_preview.setImage(self.img_reference_frontview)
        self.startGenericThread(self.camera, self.camera.signal_frame, slots=[self.tracker_mmpose.detect])
        self.startTrackerThread(self.tracker_mmpose, preview_slots=[], detection_slots=[], data_slots=[self.tracker_free_climbing.detect])
        self.startTrackerThread(self.tracker_free_climbing, preview_slots=[self.wdw_preview.setImageWithResize], detection_slots=[self.projector.setImageWithoutResize], data_slots=[])
        self.startWindowThread(thread=self.projector, close_slots=[self.camera.stop])

    def showBoulderCreator(self):
        if self.img_reference_frontview is None: return
        self.wdw_boulder_creator = BoulderCreator(self.available_screens[self.cbox_available_control_screens.currentIndex()], False, self.img_reference_frontview, self.surface.getHolds(), self.boulder)
        self.startSelectionThread(self.wdw_boulder_creator, close_slots=[], done_slots=[self.updateBoulder], click_slots=[])

    def updateBoulder(self, b):
        self.boulder = b
        if self.wdw_boulder_creator:  self.wdw_boulder_creator.deleteLater()

    def startBoulder(self):
        if self.surface.getWallRoiCamera() == []: return
        if self.surface.getWallRoiProjector() == []: return
        if self.boulder == []: return

        self.boulder_traker = InteractiveBoulderTrack(self.surface, self.boulder)
        self.boulder_traker.startBoulder()

        self.tracker_mmpose.setRenderPreview(self.render_previews)
        self.boulder_traker.setRenderPreview(self.render_previews)
        if self.render_previews:
            self.startWindowThread(thread=self.wdw_preview, close_slots=[self.projector.close])
        
        self.wdw_preview.setImage(self.img_reference_frontview)
        self.startGenericThread(self.camera, self.camera.signal_frame, slots=[self.tracker_mmpose.detect])
        self.startTrackerThread(self.tracker_mmpose, preview_slots=[], detection_slots=[], data_slots=[self.boulder_traker.detect])
        self.startTrackerThread(self.boulder_traker, preview_slots=[self.wdw_preview.setImageWithResize], detection_slots=[self.projector.setImageWithoutResize], data_slots=[])
        self.startWindowThread(thread=self.projector, close_slots=[self.camera.stop])

    def deleteBoulder(self):
        self.boulder = []

#  SAVE IMAGES FOR BENCHMARKING  ----------------------------------------------------  #
    def saveReferenceImages(self): 
        directory = QFileDialog.getExistingDirectory(self, "Open Directory","./")
        if directory != '':
            if self.img_reference is not None:
                cv2.imwrite(directory + "/img_reference.jpg", self.img_reference)
                if self.surface.getWallRoiCamera() != []:
                    frontview = cv2.warpPerspective(self.img_reference, self.surface.getHomographyCS(),
                                                    self.surface.getSizeSurface())
                    frontview = cv2.bitwise_and(frontview, self.surface.getMaskSurface())
                    cv2.imwrite(directory + "/img_reference_frontview.jpg", frontview)
            
            if self.img_fmatch_frame is not None:
                cv2.imwrite(directory + "/img_fmatch_frame.jpg", self.img_fmatch_frame)
                if self.surface.getWallRoiCamera() != []:
                    frontview = cv2.warpPerspective(self.img_fmatch_frame, self.surface.getHomographyCS(),
                                                    self.surface.getSizeSurface())
                    cv2.imwrite(directory + "/img_fmatch_frame_frontview.jpg", frontview)

            if self.feature_match_pattern is not None: 
                cv2.imwrite(directory + "/img_fmatch_pattern.jpg", self.feature_match_pattern)

            if self.surface.getWallRoiProjector() != []:
                img = np.zeros((self.surface.getSizeSurface()[1], self.surface.getSizeSurface()[0], 3), dtype=np.uint8)
                img.fill(255)
                frontview = cv2.warpPerspective(img, self.surface.getHomographySP(), self.surface.getSizeProjector())
                cv2.imwrite(directory + "/img_projector_roi.jpg", frontview)
            
            print("Images written to: " + directory)

    def saveReferenceVideo(self): 
        if self.surface.getWallRoiCamera() == []: return

        self.vid_directory = QFileDialog.getExistingDirectory(self, "Open Directory","./")
        self.vid_counter = 0
        if self.vid_directory != '':
            self.startGenericThread(self.camera, self.camera.signal_frame, slots=[self.saveReferenceVideoHelper])

    def saveReferenceVideoHelper(self,frame): 
        cv2.imwrite(self.vid_directory + "/vid_" + str(self.vid_counter) + ".jpg", frame)
        if self.surface.getWallRoiCamera() is not None:
            frontview = cv2.warpPerspective(frame, self.surface.getHomographyCS(), self.surface.getSizeSurface())
            frontview = cv2.bitwise_and(frontview, self.surface.getMaskSurface())
            cv2.imwrite(self.vid_directory + "/vid_frontview_" + str(self.vid_counter) + ".jpg", frontview)
        self.vid_counter += 1

app = QApplication(sys.argv)
mainWindow = MainWindow() # Create a Qt widget, which will be our window.
mainWindow.show() # show, showFullSreen, showMazimized
app.exec() # Start the event loop.