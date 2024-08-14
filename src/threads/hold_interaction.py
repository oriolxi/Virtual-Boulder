import cv2
import numpy as np
from random import choice
from collections import deque
from PyQt6.QtCore import QThread, pyqtSignal

import util
import algebra
from boulder import Placement

class ClimbrTrack(QThread):
    elbow_hand_ratio = 0.35
    shoulder_hand_ratio = 0.35
    smoothing_len = 3
    accept_score = 0.6

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
        warped_keypoints = cv2.perspectiveTransform(np.array([keypoint_array], dtype=np.float32), self.surface.getHomographyCS())[0]
        
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

class FreeClimbingTrack(ClimbrTrack):
    render_preview = True
    min_hand_std = 40

    signal_preview = pyqtSignal(np.ndarray)
    signal_detection = pyqtSignal(np.ndarray)

    def __init__(self, s):
        super().__init__(s)
        self.holds = np.array(self.surface.getHolds())
        self.holds_idx = np.array(range(len(self.holds)))
        (self.surface_width, self.surface_height) = self.surface.getSizeSurface()
        
        self.holds_overlay = np.zeros(shape=(self.surface.getSizeSurface()[1], self.surface.getSizeSurface()[0], 3), dtype=np.uint8)
        util.paintRectangles(self.holds_overlay, self.surface.getHolds(), (0,255,0), 2) #holds bounding boxes in green
        self.holds_overlay_mask = np.full(fill_value=255, shape=(self.surface.getSizeSurface()[1], self.surface.getSizeSurface()[0], 3), dtype=np.uint8)
        util.paintRectangles(self.holds_overlay_mask, self.surface.getHolds(), (0,0,0), 2) #holds bounding inverse mask

    def setRenderPreview(self, b):
        self.render_preview = b

    def __getBestInteraction(self, point, radious, std):
        interaction = []
        interaction_idx = -1
        if std < self.min_hand_std:
            collisions = np.apply_along_axis(algebra.isCircleTouchingRectangle, 1, self.holds, circle_point=point, radious=radious)
            if np.count_nonzero(collisions) > 0:
                overlaps = np.apply_along_axis(algebra.overlapCircleRectangle, 1, self.holds[collisions], circle_point=point, radious=radious)
                interaction = [ self.holds[collisions][np.argmax(overlaps)] ]
                interaction_idx = self.holds_idx[collisions][np.argmax(overlaps)]
        return interaction, interaction_idx

    def detect(self, pose_data):
        keypoints = pose_data[0]
        img = np.zeros(shape=(self.surface_height, self.surface_width, 3), dtype=np.uint8)
        if keypoints["detection"]:
            hand_r, hand_l, std_hand_r, std_hand_l, hand_radious = self._getHands(keypoints)
            interaction_r, idx_r = self.__getBestInteraction(hand_r, hand_radious, std_hand_r)
            interaction_l, idx_l = self.__getBestInteraction(hand_l, hand_radious, std_hand_l)

            util.paintBoundingCircles(img, interaction_l + interaction_r, (255, 255, 255), -1, 0.2)
            if idx_r == idx_l:
                util.paintBoundingCircles(img, interaction_l, Placement.COLOR_BGR[Placement.HAND_MATCHING], 5, 0.2)
            else:
                util.paintBoundingCircles(img, interaction_l, Placement.COLOR_BGR[Placement.HAND_LEFT], 5, 0.2)
                util.paintBoundingCircles(img, interaction_r, Placement.COLOR_BGR[Placement.HAND_RIGHT], 5, 0.2)
        
        projector_img = cv2.warpPerspective(img, self.surface.getHomographySP(), self.surface.getSizeProjector())
        self.signal_detection.emit(projector_img)

        if self.render_preview:
            frame_skeletons = pose_data[2]
            regularized_preview = cv2.warpPerspective(frame_skeletons, self.surface.getHomographyCS(),self.surface.getSizeSurface())
            regularized_preview = cv2.bitwise_and(regularized_preview, self.surface.getMaskSurface())
            regularized_preview_masked = cv2.bitwise_and(regularized_preview, self.holds_overlay_mask)
            regularized_preview = cv2.add(regularized_preview_masked, self.holds_overlay)

            if keypoints["detection"]:
                if idx_r == idx_l:
                    util.paintRectangles(regularized_preview, interaction_r, Placement.COLOR_BGR[Placement.HAND_MATCHING], -1) #fill the hold with matched hand interaction
                else:
                    util.paintRectangles(regularized_preview, interaction_r, Placement.COLOR_BGR[Placement.HAND_RIGHT], -1) #fill the hold with right interaction
                    util.paintRectangles(regularized_preview, interaction_l, Placement.COLOR_BGR[Placement.HAND_LEFT], -1) #fill the hold with left interaction
                
                #hand with detection radious on preview
                util.paintCircles(regularized_preview, [[hand_r, 10]], Placement.COLOR_BGR[Placement.HAND_RIGHT], -1)
                util.paintCircles(regularized_preview, [[hand_r, hand_radious]], Placement.COLOR_BGR[Placement.HAND_RIGHT], 2)
                util.paintCircles(regularized_preview, [[hand_l, 10]], Placement.COLOR_BGR[Placement.HAND_LEFT], -1)
                util.paintCircles(regularized_preview, [[hand_l, hand_radious]], Placement.COLOR_BGR[Placement.HAND_LEFT], 2)
            
            self.signal_preview.emit(regularized_preview)

class InteractiveBoulderTrack(ClimbrTrack):
    render_preview = True
    min_hand_std = 150
    overlap_score = 0.12

    signal_preview = pyqtSignal(np.ndarray)
    signal_detection = pyqtSignal(np.ndarray)

    def __init__(self, s, b):
        super().__init__(s)
        self.boulder = b
        self.holds = self.surface.getHolds()
        (self.surface_width, self.surface_height) = self.surface.getSizeSurface()

        self.boulder.start()
        self.current_step, self.next_step = self.boulder.getNext()
        
    def setRenderPreview(self, b):
        self.render_preview = b

    def __checkInteraction(self, point, radious, std, rectangle):
        if std < self.min_hand_std:
            return algebra.overlapCircleRectangle(rectangle, point, radious)
        return 0

    def detect(self, pose_data):
        keypoints = pose_data[0]
        if keypoints["detection"]: 
            hand_r, hand_l, std_hand_r, std_hand_l, hand_radious = self._getHands(keypoints)
            if self.current_step is not None:
                hold, placement = self.holds[self.current_step[0]], self.current_step[1]
                if placement == Placement.HAND_RIGHT:
                    overlap_hand_r = self.__checkInteraction(hand_r, hand_radious, std_hand_r, hold)
                    if (overlap_hand_r >= self.overlap_score): self.current_step, self.next_step = self.boulder.getNext()
                elif placement == Placement.HAND_LEFT:
                    overlap_hand_l = self.__checkInteraction(hand_l, hand_radious, std_hand_l, hold)
                    if (overlap_hand_l >= self.overlap_score): self.current_step, self.next_step = self.boulder.getNext()
                elif placement == Placement.HAND_MATCHING:
                    overlap_hand_r = self.__checkInteraction(hand_r, hand_radious, std_hand_r, hold)
                    overlap_hand_l = self.__checkInteraction(hand_l, hand_radious, std_hand_l, hold)
                    if (overlap_hand_r >= self.overlap_score) and (overlap_hand_l >= self.overlap_score): self.current_step, self.next_step = self.boulder.getNext()

        img = np.zeros(shape=(self.surface_height, self.surface_width, 3), dtype=np.uint8)
        if self.current_step is not None:
            hold, placement = self.holds[self.current_step[0]], self.current_step[1]
            util.paintBoundingCircles(img, [hold], (255, 255, 255), -1, 0.2)
            util.paintBoundingCircles(img, [hold], Placement.COLOR_BGR[placement], 6, 0.2)
            if self.next_step is not None:
                next_hold, next_placement = self.holds[self.next_step[0]], self.next_step[1]
                util.paintBoundingCircles(img, [next_hold], (255, 255, 255), -1, 0.2)
                util.paintBoundingCircles(img, [next_hold], Placement.COLOR_BGR[next_placement], 2, 0.2)
        else:
            img[:] = (96, 168, 48)
        projector_img = cv2.warpPerspective(img, self.surface.getHomographySP(), self.surface.getSizeProjector())
        self.signal_detection.emit(projector_img)

        if self.render_preview: 
            frame_skeletons = pose_data[2]
            regularized_preview = cv2.warpPerspective(frame_skeletons, self.surface.getHomographyCS(),self.surface.getSizeSurface())
            regularized_preview = cv2.bitwise_and(regularized_preview, self.surface.getMaskSurface())

            if keypoints["detection"]: #hand with detection radious on preview
                util.paintCircles(regularized_preview, [[hand_r, 10]], Placement.COLOR_BGR[Placement.HAND_RIGHT], -1)
                util.paintCircles(regularized_preview, [[hand_r, hand_radious]], Placement.COLOR_BGR[Placement.HAND_RIGHT], 2)
                util.paintCircles(regularized_preview, [[hand_l, 10]], Placement.COLOR_BGR[Placement.HAND_LEFT], -1)
                util.paintCircles(regularized_preview, [[hand_l, hand_radious]], Placement.COLOR_BGR[Placement.HAND_LEFT], 2)
            
            if self.current_step is not None:
                util.paintRectangles(regularized_preview, [hold], Placement.COLOR_BGR[placement], 6)
                if self.next_step is not None:
                    util.paintRectangles(regularized_preview, [next_hold], Placement.COLOR_BGR[next_placement], 2)
                
            self.signal_preview.emit(regularized_preview)

class RandomBoulderTrack(ClimbrTrack):
    render_preview = True
    min_hand_std = 150
    overlap_score = 0.12

    signal_preview = pyqtSignal(np.ndarray)
    signal_detection = pyqtSignal(np.ndarray)

    def __init__(self, s):
        super().__init__(s)
        self.holds = self.surface.getHolds()
        self.holds_idx = np.array(range(len(self.holds)))
        (self.surface_width, self.surface_height) = self.surface.getSizeSurface()

        self.current_step, self.next_step = None, None
        self.__getNext()
        
    def setRenderPreview(self, b):
        self.render_preview = b

    def __checkInteraction(self, point, radious, std, rectangle):
        if std < self.min_hand_std:
            return algebra.overlapCircleRectangle(rectangle, point, radious)
        return 0

    def __getNext(self):
        if self.current_step is None:
            idx = choice(self.holds_idx)
            placement = choice([Placement.HAND_RIGHT, Placement.HAND_LEFT])
            self.next_step = (idx, placement)
        
        self.current_step = self.next_step
        hold = self.holds[self.current_step[0]]
        center = [hold[0] + hold[2]/2, hold[1] + hold[3]/2]
        radious = (hold[2] + hold[3]) * 7
        collisions = np.apply_along_axis(algebra.isCircleTouchingRectangle, 1, self.holds, circle_point=center, radious=radious)
        if np.count_nonzero(collisions) > 0:
            idx = choice(self.holds_idx[collisions])
        else:
            idx = choice(self.holds_idx)
        placement = Placement.HAND_RIGHT if self.current_step[1] == Placement.HAND_LEFT else Placement.HAND_LEFT
        self.next_step = (idx, placement)

    def detect(self, pose_data):
        keypoints = pose_data[0]
        if keypoints["detection"]:
            hand_r, hand_l, std_hand_r, std_hand_l, hand_radious = self._getHands(keypoints)
            hold, placement = self.holds[self.current_step[0]], self.current_step[1]
            if placement == Placement.HAND_RIGHT:
                overlap_hand_r = self.__checkInteraction(hand_r, hand_radious, std_hand_r, hold)
                if (overlap_hand_r >= self.overlap_score): self.__getNext()
            elif placement == Placement.HAND_LEFT:
                overlap_hand_l = self.__checkInteraction(hand_l, hand_radious, std_hand_l, hold)
                if (overlap_hand_l >= self.overlap_score): self.__getNext()
            elif placement == Placement.HAND_MATCHING:
                overlap_hand_r = self.__checkInteraction(hand_r, hand_radious, std_hand_r, hold)
                overlap_hand_l = self.__checkInteraction(hand_l, hand_radious, std_hand_l, hold)
                if (overlap_hand_r >= self.overlap_score) and (overlap_hand_l >= self.overlap_score): self.__getNext()

        img = np.zeros(shape=(self.surface_height, self.surface_width, 3), dtype=np.uint8)
        hold, placement = self.holds[self.current_step[0]], self.current_step[1]
        util.paintBoundingCircles(img, [hold], (255, 255, 255), -1, 0.2)
        util.paintBoundingCircles(img, [hold], Placement.COLOR_BGR[placement], 6, 0.2)
        next_hold, next_placement = self.holds[self.next_step[0]], self.next_step[1]
        util.paintBoundingCircles(img, [next_hold], (255, 255, 255), -1, 0.2)
        util.paintBoundingCircles(img, [next_hold], Placement.COLOR_BGR[next_placement], 2, 0.2)
        projector_img = cv2.warpPerspective(img, self.surface.getHomographySP(), self.surface.getSizeProjector())
        self.signal_detection.emit(projector_img)

        if self.render_preview:
            frame_skeletons = pose_data[2] 
            regularized_preview = cv2.warpPerspective(frame_skeletons, self.surface.getHomographyCS(),self.surface.getSizeSurface())
            regularized_preview = cv2.bitwise_and(regularized_preview, self.surface.getMaskSurface())
            
            util.paintRectangles(regularized_preview, [hold], Placement.COLOR_BGR[placement], 6)
            util.paintRectangles(regularized_preview, [next_hold], Placement.COLOR_BGR[next_placement], 2)
            
            if keypoints["detection"]: #hand with detection radious on preview
                util.paintCircles(regularized_preview, [[hand_r, 10]], Placement.COLOR_BGR[Placement.HAND_RIGHT], -1)
                util.paintCircles(regularized_preview, [[hand_r, hand_radious]], Placement.COLOR_BGR[Placement.HAND_RIGHT], 2)
                util.paintCircles(regularized_preview, [[hand_l, 10]], Placement.COLOR_BGR[Placement.HAND_LEFT], -1)
                util.paintCircles(regularized_preview, [[hand_l, hand_radious]], Placement.COLOR_BGR[Placement.HAND_LEFT], 2)
            
            self.signal_preview.emit(regularized_preview)