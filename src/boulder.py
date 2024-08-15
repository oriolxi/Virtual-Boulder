import cv2
import numpy as np
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPixmap, QPen, QBrush, QPainter, QColor

import algebra
from util import Wildcard

class Placement(): # placement type global definitions
    HAND_RIGHT = "Rh" # left hand
    HAND_LEFT = "Lh" # left hand
    HAND_MATCHING = "Mh" # both hands on the same hold

    COLOR_RGB = {   HAND_RIGHT:(255,0,0),
                HAND_LEFT:(0,0,255),
                HAND_MATCHING:(170,0,255)}

    COLOR_BGR = {   HAND_RIGHT:(0,0,255),
                HAND_LEFT:(255,0,0),
                HAND_MATCHING:(170,0,255)}

class Boulder():
    # boulders are stored as an ordered list of pairs (hold_idx, Placement)
    # hold_idx is used to find the corresoponding hold in set of hold bounding boxes stored in the surface
    # this means boulders are not tied to a particular set of holds

    def __init__(self):
        self.clear()
        self.name = "default"

    def clear(self):
        self.steps = list()

    def addStep(self, hold_idx, placement, idx=-1):
        step = (hold_idx, placement)
        if idx < 0: 
            self.steps.append(step)
        else:
            self.steps.insert(idx, step)

    def replaceStep(self, hold_idx, placement, idx):
        step = (hold_idx, placement)
        self.steps[idx] = step

    def removeStepHold(self, hold_idx): # removes the last ocurrence of the setp that uses the hold
        self.steps.reverse()
        step = (hold_idx, Wildcard())
        self.steps.remove(step)
        self.steps.reverse()

    def holdInBoulder(self, hold_idx):
        return (hold_idx, Wildcard()) in self.steps

    def getLastStepWithHold(self, hold_idx):
        self.steps.reverse()
        step = (hold_idx, Wildcard())
        idx = self.steps.index(step)
        step = self.steps[idx]
        self.steps.reverse()

        return len(self.steps) - 1 - idx, step

    def getSteps(self):
        return self.steps

    def setSteps(self, s):
        self.steps = s

    def getName(self):
        return self.name

    def setName(self, n):
        self.name = n

    def getNumSteps(self):
        return len(self.steps)

    def start(self, start_step=0):
        self.current = start_step -1

    def getNext(self):
        self.current += 1
        first = None
        if self.current < len(self.steps):
            first = self.steps[self.current]
        second = None
        if self.current + 1 < len(self.steps):
            second = self.steps[self.current + 1]
        return first, second

def renderBoulderPreview(boulder, hold_boundboxes, ref_img, draw_lines=False):
    # define color parametres
    pen_size = 2
    label_w = 18
    label_h = 25
    label_padding = 3
    left_brush = QBrush(QColor.fromRgb(*Placement.COLOR_RGB[Placement.HAND_LEFT]))
    match_brush = QBrush(QColor.fromRgb(*Placement.COLOR_RGB[Placement.HAND_MATCHING]))
    right_brush = QBrush(QColor.fromRgb(*Placement.COLOR_RGB[Placement.HAND_RIGHT]))
    green_pen = QPen(Qt.GlobalColor.green, pen_size)
    transparent_pen = QPen(Qt.GlobalColor.transparent, pen_size)
    white_pen = QPen(Qt.GlobalColor.white, pen_size)

    # create an empty overlay
    overlay = QPixmap(ref_img.size().width(), ref_img.size().height())
    overlay.fill(Qt.GlobalColor.transparent)

    # draw hold bounding boxes
    painter = QPainter(overlay)
    painter.setPen(green_pen)
    for rec in hold_boundboxes:
        painter.drawRect(QRectF(rec[0], rec[1], rec[2], rec[3]))
    
    #set font size
    font = painter.font()
    font.setPointSize(14)
    painter.setFont(font)

    # draw white lines joining steps
    if draw_lines:
        prev_point = []
        painter.setPen(white_pen)
        for step in boulder.getSteps():
            hold = hold_boundboxes[step[0]]
            current_point = [hold[0] + label_w/2, hold[1] + label_h/2]
            if prev_point != []:
                painter.drawLine(int(current_point[0]), int(current_point[1]), int(prev_point[0]), int(prev_point[1]))
            prev_point = current_point

    # draw boulder steps as "tape labels"
    step_idx = 0
    hold_repeats = [0] * len(hold_boundboxes) 
    for step in boulder.getSteps():
        hold = hold_boundboxes[step[0]]
        if step[1] == Placement.HAND_LEFT: painter.setBrush(left_brush)
        if step[1] == Placement.HAND_RIGHT: painter.setBrush(right_brush)
        if step[1] == Placement.HAND_MATCHING: painter.setBrush(match_brush)
        painter.setPen(transparent_pen)
        painter.drawRect(QRectF(hold[0] + (label_w + label_padding) * hold_repeats[step[0]] , hold[1], label_w, label_h))
        painter.setPen(white_pen)
        painter.drawText(int(hold[0] + (label_w + label_padding) * hold_repeats[step[0]] + label_padding), int(hold[1]+label_h - label_padding), str(step_idx))
        step_idx = step_idx + 1
        hold_repeats[step[0]] += 1

    # add overlay over the reference image 
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOver)
    painter.drawImage(0,0,ref_img)
    painter.end()
    
    # return the resulting QPixmap
    return overlay

def mirrorBoulder(boulder, hold_boundboxes, surface_width):
    mirror_boulder = Boulder()
    holds_idx = np.array(range(len(hold_boundboxes)))

    for step in boulder.getSteps():
        # 1 Find mirrored placement
        placement = step[1]
        if placement == Placement.HAND_LEFT: new_placement = Placement.HAND_RIGHT
        elif placement == Placement.HAND_RIGHT: new_placement = Placement.HAND_LEFT
        elif placement == Placement.HAND_MATCHING: new_placement = Placement.HAND_MATCHING
        else:
            print("Could not mirror boulder: no match found for hand placement " + placement)
            return None

        # 2. Find mirrored hold
        hold = hold_boundboxes[step[0]]
        # hold bounding circle
        pts = np.array([[hold[0], hold[1]], [hold[0]+hold[2], hold[1]+hold[3]]])
        (x,y), radious = cv2.minEnclosingCircle(pts)
        # translate bounding circle to other side of vertical center line
        new_x = surface_width - x
        # find hold with gratest overlap with the translated bounding circle
        collisions = np.apply_along_axis(algebra.isCircleTouchingRectangle, 1, np.array(hold_boundboxes), circle_point=[new_x,y], radious=radious)
        if np.count_nonzero(collisions) > 0:
            overlaps = np.apply_along_axis(algebra.overlapCircleRectangle, 1, np.array(hold_boundboxes)[collisions], circle_point=[new_x,y], radious=radious)
            new_hold_idx = holds_idx[collisions][np.argmax(overlaps)]
        
            # 3. Add mirrored step to boulder
            mirror_boulder.addStep(new_hold_idx, new_placement)
        else: 
            print("Step omited: no match found for hold with idx " + str(step[0]))

    # 4. Add "_mirror" to the name of the new boulder
    mirror_boulder.setName(boulder.getName() + "_mirror")
    return mirror_boulder