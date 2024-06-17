import cv2
import numpy as np
from collections import deque
from PyQt6.QtCore import QThread, pyqtSignal

import util
import algebra

RIGHT_COLOR = (0,0,255)
LEFT_COLOR = (255,0,0)

class ClimbrTrack(QThread):
    signal_preview = pyqtSignal(np.ndarray)
    signal_detection = pyqtSignal(np.ndarray)
    signal_data = pyqtSignal(list)

    surface = None
    accept_score = 0.6
    overap_score = 0.20
    shoulder_hand_ratio = 0.3
    shoulder_foot_ratio = 0.4
    elbow_hand_ratio = 0.35
    knee_foot_ratio = 0.35
    min_std = 40
    smoothing_len = 3
    extra_radious = 0.05

    def __init__(self, s):
        super().__init__()
        self.surface = s
        self.smooth_leftH = deque(maxlen = self.smoothing_len)
        self.smooth_leftH.append([0,0])
        self.smooth_rightH = deque(maxlen = self.smoothing_len)
        self.smooth_rightH.append([0,0])
        self.smooth_leftF = deque(maxlen = self.smoothing_len)
        self.smooth_leftF.append([0,0])
        self.smooth_rightF = deque(maxlen = self.smoothing_len)
        self.smooth_rightF.append([0,0])

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

    def _footFromPoints(self, ankle, knee, ankle_score, knee_score, smooth):
        if ankle_score > self.accept_score:
            foot = ankle
            if knee_score > self.accept_score:
                knee_ankle_vec = np.subtract(ankle, knee)
                knee_ankle_len = np.linalg.norm(knee_ankle_vec)
                unit_vec = knee_ankle_vec / knee_ankle_len
                foot_dist = self.knee_foot_ratio * knee_ankle_len
                foot_vec = np.multiply(unit_vec, foot_dist)
                foot = np.add(ankle, foot_vec)
            smooth.append(foot)
        foot = np.array(np.mean(smooth, axis = 0))
        std = np.linalg.norm(np.std(smooth, axis = 0))

        return foot.astype(int), std

    def _getHands(self, keypoints):
        keypoint_array = [keypoints["shoulder_R"], 
                        keypoints["shoulder_L"], 
                        keypoints["elbow_R"], 
                        keypoints["elbow_L"], 
                        keypoints["wrist_R"], 
                        keypoints["wrist_L"]]
        warped_keypoints = cv2.perspectiveTransform(np.array([keypoint_array], dtype=np.float32), self.surface.getCameraHomography())[0]
        
        right_shoulder = warped_keypoints[0]
        left_shoulder = warped_keypoints[1]
        hand_radious = (self.shoulder_hand_ratio + self.extra_radious) * np.linalg.norm(np.subtract(right_shoulder, left_shoulder))

        right_hand, right_std = self._handFromPoints(warped_keypoints[4], warped_keypoints[2], keypoints["wrist_R_score"], keypoints["elbow_R_score"], self.smooth_rightH)
        left_hand, left_std = self._handFromPoints(warped_keypoints[5], warped_keypoints[3], keypoints["wrist_L_score"], keypoints["elbow_L_score"], self.smooth_leftH)
        
        return right_hand, left_hand, right_std, left_std, int(hand_radious)

    def _getFeet(self, keypoints):
        keypoint_array = [keypoints["shoulder_R"], 
                        keypoints["shoulder_L"], 
                        keypoints["knee_R"], 
                        keypoints["knee_L"], 
                        keypoints["ankle_R"], 
                        keypoints["ankle_L"]]
        warped_keypoints = cv2.perspectiveTransform(np.array([keypoint_array], dtype=np.float32), self.surface.getCameraHomography())[0]

        right_shoulder = warped_keypoints[0]
        left_shoulder = warped_keypoints[1]
        foot_radious = self.shoulder_foot_ratio * np.linalg.norm(np.subtract(right_shoulder, left_shoulder))

        right_foot, right_std = self._footFromPoints(warped_keypoints[4], warped_keypoints[2], keypoints["ankle_R_score"], keypoints["knee_R_score"], self.smooth_rightF)
        left_foot, left_std = self._footFromPoints(warped_keypoints[5], warped_keypoints[3], keypoints["ankle_L_score"], keypoints["knee_L_score"], self.smooth_leftF)
        
        return right_foot, left_foot, right_std, left_std, int(foot_radious)

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
        if std < self.min_std:
            holds = np.array(self.surface.getHolds())
            collisions = np.apply_along_axis(algebra.isCircleTouchingRectangle, 1, holds, circle_p=point, circle_r=radious)
            if np.count_nonzero(collisions) > 0:
                overlaps = np.apply_along_axis(algebra.overlapCircleRectangle, 1, holds[collisions], circle_p=point, circle_r=radious)
                interaction = [ holds[collisions][np.argmax(overlaps)] ]
        return interaction

    def detect(self, pose_data):
        keypoints = pose_data[0]
        frame_skeletons = pose_data[2]

        (w_regularized, h_regularized) = self.surface.getRegularizedSize()
        img = np.zeros(shape=(h_regularized, w_regularized, 3), dtype=np.uint8)
        if keypoints["detection"]:
            right_hand, left_hand, right_std, left_std, hand_radious = self._getHands(keypoints)
            #right_foot, left_foot, right_stdF, left_stdF, foot_radious = self._getFeet(keypoints)

            right_interactions = self.__getBestInteraction(right_hand, hand_radious, right_std)
            left_interactions = self.__getBestInteraction(left_hand, hand_radious, left_std)
            #right_interactionsF = self.__getBestInteraction(right_foot, foot_radious, right_stdF)
            #left_interactionsF = self.__getBestInteraction(left_foot, foot_radious, left_stdF)


            util.paintBoundingCircles(img, right_interactions + left_interactions, (255,255,255), -1, 0.2)
            util.paintBoundingCircles(img, right_interactions, RIGHT_COLOR, 5, 0.2)
            util.paintBoundingCircles(img, left_interactions, LEFT_COLOR, 5, 0.2)
            projector_img = cv2.warpPerspective(img, self.surface.getProjectorHomography(), self.surface.getProjectorSize())
            self.signal_detection.emit(projector_img)

            if self.render_preview:
                regularized_preview = cv2.warpPerspective(frame_skeletons, self.surface.getCameraHomography(), self.surface.getRegularizedSize())
                regularized_preview = cv2.bitwise_and(regularized_preview, self.surface.getRegularizedMask())
                util.paintRectangles(regularized_preview, self.surface.getHolds(), (0,255,0), 2) #holds bounding boxes in green
                util.paintRectangles(regularized_preview, right_interactions, RIGHT_COLOR, -1) #hold right interaction filled in red
                util.paintRectangles(regularized_preview, left_interactions, LEFT_COLOR, -1) #hold left interaction filled in blue
                #hand with detection radious on preview
                util.paintCircles(regularized_preview, [[right_hand, 10]], RIGHT_COLOR, -1)
                util.paintCircles(regularized_preview, [[right_hand, hand_radious]], RIGHT_COLOR, 2)
                util.paintCircles(regularized_preview, [[left_hand, 10]], LEFT_COLOR, -1)
                util.paintCircles(regularized_preview, [[left_hand, hand_radious]], LEFT_COLOR, 2)
                self.signal_preview.emit(regularized_preview)
        else:
            projector_img = cv2.warpPerspective(img, self.surface.getProjectorHomography(), self.surface.getProjectorSize())
            self.signal_detection.emit(projector_img)
            if self.render_preview: 
                regularized_preview = cv2.warpPerspective(frame_skeletons, self.surface.getCameraHomography(), self.surface.getRegularizedSize())
                regularized_preview = cv2.bitwise_and(regularized_preview, self.surface.getRegularizedMask())
                util.paintRectangles(regularized_preview, self.surface.getHolds(), (0,255,0), 2) #holds bounding boxes in green
                self.signal_preview.emit(regularized_preview)
    

class InteractiveBoulderTrack(ClimbrTrack):
    render_preview = True
    boulder = []
    progression = []
    current_left = []
    current_right = []
    left_double = False
    right_double = False

    wc = util.Wildcard()

    def __init__(self, s, b):
        super().__init__(s)
        self.boulder = b

    def setRenderPreview(self, b):
        self.render_preview = b

    def startBoulder(self):
        self.progression = self.boulder.copy()
        self.current_right = self.__getNextHold("R")
        self.current_left = self.__getNextHold("L")

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
        if std < self.min_std:
            return algebra.overlapCircleRectangle(rectangle, point, radious)
        return 0

    def detect(self, pose_data):
        keypoints = pose_data[0]
        frame_skeletons = pose_data[2]

        (w_regularized, h_regularized) = self.surface.getRegularizedSize()
        img = np.zeros(shape=(h_regularized, w_regularized, 3), dtype=np.uint8)
        if keypoints["detection"]:

            right_hand, left_hand, right_std, left_std, hand_radious = self._getHands(keypoints)

            right_overlap = self.__checkInteraction(right_hand, hand_radious, right_std, self.current_right)
            if (right_overlap >= self.overap_score): self.current_right = self.__getNextHold("R")
            left_overlap = self.__checkInteraction(left_hand, hand_radious, left_std, self.current_left)
            if (left_overlap >= self.overap_score): self.current_left = self.__getNextHold("L")

            if self.current_right == [] and self.current_left == []:
                img[:] = (96, 168, 48)
            if self.current_right != []:
                util.paintBoundingCircles(img, [self.current_right], (255,255,255), -1, 0.2)
                util.paintBoundingCircles(img, [self.current_right], RIGHT_COLOR, 5, 0.2)
            if self.current_left != []:
                util.paintBoundingCircles(img, [self.current_left], (255,255,255), -1, 0.2)
                util.paintBoundingCircles(img, [self.current_left], LEFT_COLOR, 5, 0.2)
            projector_img = cv2.warpPerspective(img, self.surface.getProjectorHomography(), self.surface.getProjectorSize())
            self.signal_detection.emit(projector_img)

            regularized_preview = None
            if self.render_preview:
                regularized_preview = cv2.warpPerspective(frame_skeletons, self.surface.getCameraHomography(), self.surface.getRegularizedSize())
                regularized_preview = cv2.bitwise_and(regularized_preview, self.surface.getRegularizedMask())
                if self.current_right != []: 
                    util.paintRectangles(regularized_preview, [self.current_right], RIGHT_COLOR, 2) 
                    if (right_overlap >= self.overap_score):
                        util.paintRectangles(regularized_preview, [self.current_right], RIGHT_COLOR, -1) #hold right interaction filled in red
                if self.current_left != []: 
                    util.paintRectangles(regularized_preview, [self.current_left], LEFT_COLOR, 2) 
                    if (left_overlap >= self.overap_score):
                        util.paintRectangles(regularized_preview, [self.current_left], LEFT_COLOR, -1) #hold left interaction filled in blue
                #hand with detection radious on preview
                util.paintCircles(regularized_preview, [[right_hand, 10]], RIGHT_COLOR, -1)
                util.paintCircles(regularized_preview, [[right_hand, hand_radious]], RIGHT_COLOR, 2)
                util.paintCircles(regularized_preview, [[left_hand, 10]], LEFT_COLOR, -1)
                util.paintCircles(regularized_preview, [[left_hand, hand_radious]], LEFT_COLOR, 2)
                self.signal_preview.emit(regularized_preview)

        else:
            if self.current_right == [] and self.current_left == []:
                img[:] = (96, 168, 48)
            if self.current_right != []:
                util.paintBoundingCircles(img, [self.current_right], (255,255,255), -1, 0.2)
                util.paintBoundingCircles(img, [self.current_right], RIGHT_COLOR, 5, 0.2)
            if self.current_left != []:
                util.paintBoundingCircles(img, [self.current_left], (255,255,255), -1, 0.2)
                util.paintBoundingCircles(img, [self.current_left], LEFT_COLOR, 5, 0.2)
            projector_img = cv2.warpPerspective(img, self.surface.getProjectorHomography(), self.surface.getProjectorSize())
            self.signal_detection.emit(projector_img)
            if self.render_preview: 
                regularized_preview = cv2.warpPerspective(frame_skeletons, self.surface.getCameraHomography(), self.surface.getRegularizedSize())
                regularized_preview = cv2.bitwise_and(regularized_preview, self.surface.getRegularizedMask())
                if self.current_right != []: 
                    util.paintRectangles(regularized_preview, [self.current_right], RIGHT_COLOR, 2) 
                if self.current_left != []: 
                    util.paintRectangles(regularized_preview, [self.current_left], LEFT_COLOR, 2)
                self.signal_preview.emit(regularized_preview)
