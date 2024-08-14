import os
import cv2
import sys
import pickle
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from PyQt6 import uic
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QScreen, QImage, QPixmap
from PyQt6.QtMultimedia import QCamera, QMediaDevices
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem

import util
import algebra
from surface import Surface
from boulder import Boulder
from dialogs.generic import ImageWindow
from dialogs.area_selection import AreaSelectionWindow
from dialogs.hold_selection import HoldSelectionWindow
from dialogs.boulder_creator import BoulderCreatorWindow
from gui.hold_detection import HoldDetectionDialog
from gui.interactive_boulder import InteractiveBoulderDialog
from threads.camera import Camera
from threads.aruco_tracker import ArucoTrack
from threads.pose_tracker import  PoseTrack
from threads.hold_interaction import FreeClimbingTrack, InteractiveBoulderTrack, RandomBoulderTrack
from threads.perspective_warper import PerspectiveWarper
import algorithms.feature_match as feature_match
import algorithms.homography_rank as homography_rank

class MainWindow(QMainWindow):
    wdw_projected_area_selector = None #created and destroyed as needed
    wdw_windowed_area_selector = None #created and destroyed as needed
    wdw_hold_editor = None #created and destroyed as needed
    wdw_dialog = None #created and destroyed as needed
    wdw_boulder_editor = None  # created and destroyed as needed
    wdw_boulder_selector = None # created and destroyed as needed

    file_test_image = "img/via1.jpg"
    file_fmatch_pattern = "img/calibration_rainbow.png"

    def __init__(self):
        super().__init__()
        
        uic.loadUi("gui/main_window.ui", self)
        self.setWindowTitle("Virtual Boulder")
        self.__loadResources()
        plt.ion() # force pyplot to use it's own thread for figures (QCoreApplication::exec: The event loop is already running)   

    def __loadResources(self): 
        self.available_screens = QScreen.virtualSiblings(self.screen())
        self.available_cameras = QMediaDevices.videoInputs()

        self.camera = Camera(QCamera(self.available_cameras[0]),0)
        self.surface = Surface()
        self.projector = ImageWindow(self.available_screens[0], True)
        self.boulder_list = list()
        self.boulder_list.append(Boulder())

        self.wdw_preview = ImageWindow(self.available_screens[0], False)

        self.setImageLblPreview(self.file_test_image, self.lbl_preview_test_image)
        self.setImageLblPreview(self.file_fmatch_pattern, self.lbl_preview_fmatch_pattern)

        self.tracker_aruco = ArucoTrack()
        self.tracker_pose = PoseTrack()

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
        
        self.act_set_cp_max_area.triggered.connect(self.setCamProjToMaxArea)
        self.act_set_render_previews.toggled.connect(self.setRenderLivePreviews)
        self.setRenderLivePreviews()
        self.act_clear_all_data.triggered.connect(self.__loadResources)

        self.act_savefile_reference_images.triggered.connect(self.saveReferenceImages)
        self.act_savefile_reference_video.triggered.connect(self.saveReferenceVideo)

        self.btn_start_boulder.clicked.connect(self.openBoulderSelectorDialog)
        self.btn_start_free_climbing.clicked.connect(self.startFreeClimbing)
        self.btn_start_random.clicked.connect(self.startRandomBoulder)

        self.act_savefile_sroi_and_holds.triggered.connect(self.createRoiSaveFile)
        self.act_openfile_sroi_and_holds.triggered.connect(self.loadRoiSaveFile)

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

    def loadRoiSaveFile(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter("Pickle file (*.pkl)")
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

    def createRoiSaveFile(self):
        directory = QFileDialog.getExistingDirectory(self, "Open Directory","./saves/")
        if directory != '':
            with open(directory + '/save.pkl', 'wb') as output:
                pickle.dump(self.surface.getHolds(), output, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.surface.getWallRoiSurface(), output, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.surface.getSizeSurface(), output, pickle.HIGHEST_PROTOCOL)

    def setCamProjToMaxArea(self):
        # experimental feature used for debugging
        # set camera surface selection to max available area
        w = self.camera.getSize().width()
        h = self.camera.getSize().height()
        self.surface.setCameraParametres([[0,0], [0,h], [w,0], [w,h]], w, h)
        self.updateRoiTable(self.tbl_roi_frontview, self.surface.getWallRoiCamera())
        self.img_reference =  np.zeros(shape=(h, w, 3), dtype=np.uint8)
        self.updateFrontViewPreview()
        
        # set projector surface selection to max available area
        w = self.projector.getSize().width()
        h = self.projector.getSize().height()
        self.surface.setProjectorParametres([[0,0], [0,h], [w,0], [w,h]], w, h)
        self.updateRoiTable(self.tbl_roi_projector, self.surface.getWallRoiProjector())

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
        self.wdw_windowed_area_selector = AreaSelectionWindow(self.available_screens[self.cbox_available_control_screens.currentIndex()], False)
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
        self.wdw_projected_area_selector = AreaSelectionWindow(self.available_screens[self.cbox_available_projector_screens.currentIndex()], True)
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

#  FRONTVIEW HORIZON CORRECTION  ----------------------------------------------------  #
    def startFrontViewHorizon(self):
        if self.surface.getWallRoiCamera() == []: return
        if self.surface.getWallRoiProjector() == []: return

        self.wdw_windowed_area_selector = AreaSelectionWindow(self.available_screens[self.cbox_available_control_screens.currentIndex()], False, 2)
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

    def addHolds(self, holds):
        self.surface.addHolds(holds)
        self.updateFrontViewPreview()

    def showHoldEditor(self):
        if self.img_reference_frontview is None: return

        self.wdw_hold_editor = HoldSelectionWindow(self.available_screens[self.cbox_available_control_screens.currentIndex()], False)
        self.wdw_hold_editor.setImage(self.img_reference_frontview)
        self.wdw_hold_editor.setPoints(self.surface.getHolds())
        self.startSelectionThread(self.wdw_hold_editor, close_slots=[], done_slots=[self.setHolds], click_slots=[])

    def showHoldAutoDetection(self):
        if self.img_reference_frontview is None: return

        self.wdw_dialog = HoldDetectionDialog(self.img_reference_frontview)
        self.startGenericThread(self.wdw_dialog, self.wdw_dialog.signal_done, slots=[self.addHolds])

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
        self.tracker_pose.setRenderPreview(True)
        self.tracker_pose.setRenderReprojection(False)
        self.wdw_preview.setImage(self.img_reference_frontview)
        self.startGenericThread(self.camera, self.camera.signal_frame, slots=[self.tracker_pose.detect])
        self.startTrackerThread(self.tracker_pose, preview_slots=[self.wdw_preview.setImageWithResize], detection_slots=[], data_slots=[])
        self.startWindowThread(thread=self.wdw_preview, close_slots=[self.camera.stop])

    def startPoseTracker(self):
        if self.surface.getWallRoiCamera() == []: return
        if self.surface.getWallRoiProjector() == []: return

        self.perspective_warper = PerspectiveWarper(self.surface.getHomographyCP(), self.surface.getSizeProjector())

        self.tracker_pose.setRenderPreview(self.render_previews)
        self.tracker_pose.setRenderReprojection(True)

        if self.render_previews: 
            self.startWindowThread(thread=self.wdw_preview, close_slots=[self.projector.close])

        self.wdw_preview.setImage(self.img_reference_frontview)
        self.startGenericThread(self.camera, self.camera.signal_frame, slots=[self.tracker_pose.detect])
        self.startTrackerThread(self.tracker_pose, preview_slots=[self.wdw_preview.setImageWithResize], detection_slots=[self.perspective_warper.apply], data_slots=[])
        self.perspective_warper.signal_done.connect(self.projector.setImageWithoutResize)
        self.startWindowThread(thread=self.projector, close_slots=[self.camera.stop])

# INTERACTIVE EXPERIENCES  ----------------------------------------------------  #

    def startFreeClimbing(self):
        if self.surface.getWallRoiCamera() == []: return
        if self.surface.getWallRoiProjector() == []: return

        self.tracker_free_climbing = FreeClimbingTrack(self.surface)

        self.tracker_pose.setRenderPreview(self.render_previews)
        self.tracker_pose.setRenderReprojection(False)
        self.tracker_free_climbing.setRenderPreview(self.render_previews)
        if self.render_previews:
            self.startWindowThread(thread=self.wdw_preview, close_slots=[self.projector.close])
        
        self.wdw_preview.setImage(self.img_reference_frontview)
        self.startGenericThread(self.camera, self.camera.signal_frame, slots=[self.tracker_pose.detect])
        self.startTrackerThread(self.tracker_pose, preview_slots=[], detection_slots=[], data_slots=[self.tracker_free_climbing.detect])
        self.startTrackerThread(self.tracker_free_climbing, preview_slots=[self.wdw_preview.setImageWithResize], detection_slots=[self.projector.setImageWithoutResize], data_slots=[])
        self.startWindowThread(thread=self.projector, close_slots=[self.camera.stop])

    def openBoulderSelectorDialog(self):
        if self.img_reference_frontview is None: return

        self.wdw_boulder_selector = InteractiveBoulderDialog(self, self.boulder_list, self.surface.getHolds(), self.img_reference_frontview)
        self.wdw_boulder_selector.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)

        self.wdw_boulder_selector.signal_start.connect(self.startBoulder)
        self.wdw_boulder_selector.signal_edit.connect(self.editBoulder)

        self.wdw_boulder_selector.show()

    def editBoulder(self, idx):
        self.wdw_boulder_editor = BoulderCreatorWindow(self.available_screens[self.cbox_available_control_screens.currentIndex()], False, self.img_reference_frontview, self.surface.getHolds(), self.boulder_list[idx])
        self.wdw_boulder_editor.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        
        self.perspective_warper = PerspectiveWarper(self.surface.getHomographySP(), self.surface.getSizeProjector())
        self.perspective_warper.signal_done.connect(self.projector.setImageWithoutResize)

        self.startSelectionThread(self.wdw_boulder_editor, close_slots=[], done_slots=[self.wdw_boulder_selector.updateBoulderPreview, self.projector.stop], click_slots=[self.perspective_warper.apply])
        self.startWindowThread(thread=self.projector, close_slots=[])


    def startBoulder(self, idx, start_step):
        self.traker_boulder = InteractiveBoulderTrack(self.surface, self.boulder_list[idx], start_step)

        self.tracker_pose.setRenderPreview(self.render_previews)
        self.tracker_pose.setRenderReprojection(False)
        self.traker_boulder.setRenderPreview(self.render_previews)
        if self.render_previews:
            self.startWindowThread(thread=self.wdw_preview, close_slots=[self.projector.close])
        
        self.wdw_preview.setImage(self.img_reference_frontview)
        self.wdw_boulder_selector.hide()
        self.startGenericThread(self.camera, self.camera.signal_frame, slots=[self.tracker_pose.detect])
        self.startTrackerThread(self.tracker_pose, preview_slots=[], detection_slots=[], data_slots=[self.traker_boulder.detect])
        self.startTrackerThread(self.traker_boulder, preview_slots=[self.wdw_preview.setImageWithResize], detection_slots=[self.projector.setImageWithoutResize], data_slots=[])
        self.startWindowThread(thread=self.projector, close_slots=[self.camera.stop, self.wdw_boulder_selector.show])              

    def startRandomBoulder(self):
        if self.surface.getWallRoiCamera() == []: return
        if self.surface.getWallRoiProjector() == []: return

        self.tracker_random_bulder = RandomBoulderTrack(self.surface)

        self.tracker_pose.setRenderPreview(self.render_previews)
        self.tracker_pose.setRenderReprojection(False)
        self.tracker_random_bulder.setRenderPreview(self.render_previews)
        if self.render_previews:
            self.startWindowThread(thread=self.wdw_preview, close_slots=[self.projector.close])
        
        self.wdw_preview.setImage(self.img_reference_frontview)
        self.startGenericThread(self.camera, self.camera.signal_frame, slots=[self.tracker_pose.detect])
        self.startTrackerThread(self.tracker_pose, preview_slots=[], detection_slots=[], data_slots=[self.tracker_random_bulder.detect])
        self.startTrackerThread(self.tracker_random_bulder, preview_slots=[self.wdw_preview.setImageWithResize], detection_slots=[self.projector.setImageWithoutResize], data_slots=[])
        self.startWindowThread(thread=self.projector, close_slots=[self.camera.stop])

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
                cv2.imwrite(directory + "/img_fmatch_pattern.jpg", self.img_fmatch_pattern)

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