from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPixmap, QPen, QBrush, QPainter, QColor

from util import Wildcard

class Placement(): # placement type global definitions
    HAND_RIGHT = "Rh" # left hand
    HAND_LEFT = "Lh" # left hand
    HAND_MATCHING = "Mh" # both hands on the same hold

    COLOR_HAND_RIGHT = (255,0,0)
    COLOR_HAND_LEFT = (0,0,255)
    COLOR_HAND_MATCHING = (170,0,255)


class Boulder():
    # boulders are stored as an ordered list of pairs (hold_idx, Placement)
    # hold_idx is used to find the corresoponding hold in set of hold bounding boxes stored in the surface
    # this means boulders are not tied to a particular set of holds

    def __init__(self):
        self.clear()

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

    def removeStepIdx(self, idx):
        self.steps.pop(idx)

    def removeStepHold(self, hold_idx): # removes the last ocurrence of the setp including the hold
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

    def getNumSteps(self):
        return len(self.steps)

    def start(self):
        self.current = -1

    def getNext(self):
        self.current += 1
        first = None
        if self.current < len(self.steps):
            first = self.steps[self.current]
        second = None
        if self.current + 1 < len(self.steps):
            second = self.steps[self.current + 1]
        return first, second

def renderBoulderPreview(boulder, hold_boundboxes, ref_img):
    # define color parametres
    pen_size = 2
    label_w = 18
    label_h = 25
    label_padding = 3
    left_brush = QBrush(QColor.fromRgb(*Placement.COLOR_HAND_LEFT))
    match_brush = QBrush(QColor.fromRgb(*Placement.COLOR_HAND_MATCHING))
    right_brush = QBrush(QColor.fromRgb(*Placement.COLOR_HAND_RIGHT))
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

    # draw boulder steps as "tape labels"
    step_idx = 0
    hold_repeats = [0] * len(hold_boundboxes) 
    for step in boulder.getSteps():
        if step[1] == Placement.HAND_LEFT: painter.setBrush(left_brush)
        if step[1] == Placement.HAND_RIGHT: painter.setBrush(right_brush)
        if step[1] == Placement.HAND_MATCHING: painter.setBrush(match_brush)
        hold = hold_boundboxes[step[0]]
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