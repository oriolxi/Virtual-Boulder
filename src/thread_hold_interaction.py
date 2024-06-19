import cv2
import numpy as np
from collections import deque
from PyQt6.QtCore import QThread, pyqtSignal

import util
import algebra

COLOR_RIGHT = (0, 0, 255)
COLOR_LEFT = (255, 0, 0)

class ClimbrTrack(QThread):
    signal_preview = pyqtSignal(np.ndarray)
    signal_detection = pyqtSignal(np.ndarray)
    signal_data = pyqtSignal(list)

    surface = None
    accept_score = 0.6
    overlap_score = 0.20
    shoulder_hand_ratio = 0.35
    elbow_hand_ratio = 0.35
    min_hand_std = 40
    smoothing_len = 3

    def __init__(self, s):
        super().__init__()
        self.surface = s
        self.smooth_hand_l = deque(maxlen = self.smoothing_len)
        self.smooth_hand_l.append([0, 0])
        self.smooth_hand_r = deque(maxlen = self.smoothing_len)
        self.smooth_hand_r.append([0, 0])

    def _handFromPoints(self, wrist, elbow, wrist_score, elbow_score, smooth):
        if wrist_score > self.accept_score:
            hand = wrist
            if elbow_score > self.accept_score:
                elbow_wrist_vec = np.subtract(wrist, elbow)
                hand_vec = np.multiply(elbow_wrist_vec, self.elbow_hand_ratio)
                hand = np.add(wrist, hand_vec)
            smooth.append(hand)
        hand = np.array(np.mean(smooth, axis = 0))
        std = np.linalg.norm(np.std(smooth, axis = 0))

        return hand.astype(int), std

    def _getHands(self, keypoints):
        keypoint_array = [keypoints["shoulder_R"], 
                        keypoints["shoulder_L"], 
                        keypoints["elbow_R"], 
                        keypoints["elbow_L"], 
                        keypoints["wrist_R"], 
                        keypoints["wrist_L"]]
        warped_keypoints = cv2.perspectiveTransform(np.array([keypoint_array], dtype=np.float32),
                                                    self.surface.getHomographyCS())[0]
        
        right_shoulder = warped_keypoints[0]
        left_shoulder = warped_keypoints[1]
        hand_radious = (self.shoulder_hand_ratio) * np.linalg.norm(np.subtract(right_shoulder, left_shoulder))

        hand_r, std_hand_r = self._handFromPoints(warped_keypoints[4], warped_keypoints[2], keypoints["wrist_R_score"], keypoints["elbow_R_score"], self.smooth_hand_r)
        hand_l, std_hand_l = self._handFromPoints(warped_keypoints[5], warped_keypoints[3], keypoints["wrist_L_score"], keypoints["elbow_L_score"], self.smooth_hand_l)
        
        return hand_r, hand_l, std_hand_r, std_hand_l, int(hand_radious)

    def detect(self, pose_data):
        keypoints = pose_data[0]
        if keypoints["detection"]:
            return self._getHands(keypoints)

class HoldInteractionTrack(ClimbrTrack):
    render_preview = True

    def __init__(self, s):
        super().__init__(s)

    def setRenderPreview(self, b):
        self.render_preview = b

    def __getBestInteraction(self, point, radious, std):
        interaction = []
        if std < self.min_hand_std:
            holds = np.array(self.surface.getHolds())
            collisions = np.apply_along_axis(algebra.isCircleTouchingRectangle, 1, holds, circle_p=point, circle_r=radious)
            if np.count_nonzero(collisions) > 0:
                overlaps = np.apply_along_axis(algebra.overlapCircleRectangle, 1, holds[collisions], circle_p=point, circle_r=radious)
                interaction = [ holds[collisions][np.argmax(overlaps)] ]
        return interaction

    def detect(self, pose_data):
        keypoints = pose_data[0]
        frame_skeletons = pose_data[2]

        (surface_width, surface_height) = self.surface.getSizeSurface()
        img = np.zeros(shape=(surface_height, surface_width, 3), dtype=np.uint8)
        if keypoints["detection"]:
            hand_r, hand_l, std_hand_r, std_hand_l, hand_radious = self._getHands(keypoints)

            interactions_r = self.__getBestInteraction(hand_r, hand_radious, std_hand_r)
            interactions_l = self.__getBestInteraction(hand_l, hand_radious, std_hand_l)

            util.paintBoundingCircles(img, interactions_r + interactions_l, (255, 255, 255), -1, 0.2)
            util.paintBoundingCircles(img, interactions_r, COLOR_RIGHT, 5, 0.2)
            util.paintBoundingCircles(img, interactions_l, COLOR_LEFT, 5, 0.2)
            projector_img = cv2.warpPerspective(img, self.surface.getHomographySP(), self.surface.getSizeProjector())
            self.signal_detection.emit(projector_img)

            if self.render_preview:
                regularized_preview = cv2.warpPerspective(frame_skeletons, self.surface.getHomographyCS(),
                                                          self.surface.getSizeSurface())
                regularized_preview = cv2.bitwise_and(regularized_preview, self.surface.getMaskSurface())
                util.paintRectangles(regularized_preview, self.surface.getHolds(), (0,255,0), 2) #holds bounding boxes in green
                util.paintRectangles(regularized_preview, interactions_r, COLOR_RIGHT, -1) #hold right interaction filled in red
                util.paintRectangles(regularized_preview, interactions_l, COLOR_LEFT, -1) #hold left interaction filled in blue
                #hand with detection radious on preview
                util.paintCircles(regularized_preview, [[hand_r, 10]], COLOR_RIGHT, -1)
                util.paintCircles(regularized_preview, [[hand_r, hand_radious]], COLOR_RIGHT, 2)
                util.paintCircles(regularized_preview, [[hand_l, 10]], COLOR_LEFT, -1)
                util.paintCircles(regularized_preview, [[hand_l, hand_radious]], COLOR_LEFT, 2)
                self.signal_preview.emit(regularized_preview)
        else:
            projector_img = cv2.warpPerspective(img, self.surface.getHomographySP(), self.surface.getSizeProjector())
            self.signal_detection.emit(projector_img)
            if self.render_preview: 
                regularized_preview = cv2.warpPerspective(frame_skeletons, self.surface.getHomographyCS(),
                                                          self.surface.getSizeSurface())
                regularized_preview = cv2.bitwise_and(regularized_preview, self.surface.getMaskSurface())
                util.paintRectangles(regularized_preview, self.surface.getHolds(), (0,255,0), 2) #holds bounding boxes in green
                self.signal_preview.emit(regularized_preview)
    

class InteractiveBoulderTrack(ClimbrTrack):
    render_preview = True
    boulder = []
    progression = []
    current_hand_l = []
    current_hand_r = []

    wc = util.Wildcard()

    def __init__(self, s, b):
        super().__init__(s)
        self.boulder = b

    def setRenderPreview(self, b):
        self.render_preview = b

    def startBoulder(self):
        self.progression = self.boulder.copy()
        self.current_hand_r = self.__getNextHold("R")
        self.current_hand_l = self.__getNextHold("L")

    def __getNextHold(self, hand):
        move = [self.wc, self.wc, self.wc, self.wc, hand]
        try: 
            idx = self.progression.index(move)
            move = self.progression[idx]
            self.progression.remove(move)
            return move[0:4]
        except Exception: return []

    def __checkInteraction(self, point, radious, std, rectangle):
        if rectangle == []: return -1
        if std < self.min_hand_std:
            return algebra.overlapCircleRectangle(rectangle, point, radious)
        return 0

    def detect(self, pose_data):
        keypoints = pose_data[0]
        frame_skeletons = pose_data[2]

        (w_regularized, h_regularized) = self.surface.getSizeSurface()
        img = np.zeros(shape=(h_regularized, w_regularized, 3), dtype=np.uint8)
        if keypoints["detection"]:

            hand_r, hand_l, std_hand_r, std_hand_l, hand_radious = self._getHands(keypoints)

            overlap_hand_r = self.__checkInteraction(hand_r, hand_radious, std_hand_r, self.current_hand_r)
            if (overlap_hand_r >= self.overlap_score): self.current_hand_r = self.__getNextHold("R")
            overlap_hand_l = self.__checkInteraction(hand_l, hand_radious, std_hand_l, self.current_hand_l)
            if (overlap_hand_l >= self.overlap_score): self.current_hand_l = self.__getNextHold("L")

            if self.current_hand_r == [] and self.current_hand_l == []:
                img[:] = (96, 168, 48)
            if self.current_hand_r != []:
                util.paintBoundingCircles(img, [self.current_hand_r], (255, 255, 255), -1, 0.2)
                util.paintBoundingCircles(img, [self.current_hand_r], COLOR_RIGHT, 5, 0.2)
            if self.current_hand_l != []:
                util.paintBoundingCircles(img, [self.current_hand_l], (255, 255, 255), -1, 0.2)
                util.paintBoundingCircles(img, [self.current_hand_l], COLOR_LEFT, 5, 0.2)
            projector_img = cv2.warpPerspective(img, self.surface.getHomographySP(), self.surface.getSizeProjector())
            self.signal_detection.emit(projector_img)

            regularized_preview = None
            if self.render_preview:
                regularized_preview = cv2.warpPerspective(frame_skeletons, self.surface.getHomographyCS(),
                                                          self.surface.getSizeSurface())
                regularized_preview = cv2.bitwise_and(regularized_preview, self.surface.getMaskSurface())
                if self.current_hand_r != []:
                    util.paintRectangles(regularized_preview, [self.current_hand_r], COLOR_RIGHT, 2)
                    if (overlap_hand_r >= self.overlap_score):
                        util.paintRectangles(regularized_preview, [self.current_hand_r], COLOR_RIGHT, -1) #hold right interaction filled in red
                if self.current_hand_l != []:
                    util.paintRectangles(regularized_preview, [self.current_hand_l], COLOR_LEFT, 2)
                    if (overlap_hand_l >= self.overlap_score):
                        util.paintRectangles(regularized_preview, [self.current_hand_l], COLOR_LEFT, -1) #hold left interaction filled in blue
                #hand with detection radious on preview
                util.paintCircles(regularized_preview, [[hand_r, 10]], COLOR_RIGHT, -1)
                util.paintCircles(regularized_preview, [[hand_r, hand_radious]], COLOR_RIGHT, 2)
                util.paintCircles(regularized_preview, [[hand_l, 10]], COLOR_LEFT, -1)
                util.paintCircles(regularized_preview, [[hand_l, hand_radious]], COLOR_LEFT, 2)
                self.signal_preview.emit(regularized_preview)

        else:
            if self.current_hand_r == [] and self.current_hand_l == []:
                img[:] = (96, 168, 48)
            if self.current_hand_r != []:
                util.paintBoundingCircles(img, [self.current_hand_r], (255, 255, 255), -1, 0.2)
                util.paintBoundingCircles(img, [self.current_hand_r], COLOR_RIGHT, 5, 0.2)
            if self.current_hand_l != []:
                util.paintBoundingCircles(img, [self.current_hand_l], (255, 255, 255), -1, 0.2)
                util.paintBoundingCircles(img, [self.current_hand_l], COLOR_LEFT, 5, 0.2)
            projector_img = cv2.warpPerspective(img, self.surface.getHomographySP(), self.surface.getSizeProjector())
            self.signal_detection.emit(projector_img)
            if self.render_preview: 
                regularized_preview = cv2.warpPerspective(frame_skeletons, self.surface.getHomographyCS(),
                                                          self.surface.getSizeSurface())
                regularized_preview = cv2.bitwise_and(regularized_preview, self.surface.getMaskSurface())
                if self.current_hand_r != []:
                    util.paintRectangles(regularized_preview, [self.current_hand_r], COLOR_RIGHT, 2)
                if self.current_hand_l != []:
                    util.paintRectangles(regularized_preview, [self.current_hand_l], COLOR_LEFT, 2)
                self.signal_preview.emit(regularized_preview)