import cv2
import numpy as np
from PyQt6.QtGui import QImage

def QimageFromCVimage(cvImg):
    if cvImg is None: return None
    return QImage(cvImg.data, cvImg.shape[1], cvImg.shape[0], cvImg.strides[0], QImage.Format.Format_BGR888)

def CVimageFromQimage(qImg):
    if qImg is None: return None
    incomingImage = qImg.convertToFormat(QImage.Format.Format_BGR888)
    
    width = incomingImage.width()
    height = incomingImage.height()

    ptr = incomingImage.bits()
    ptr.setsize(height * width * 3)
    arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3)).copy()
    return arr

def paintSelection(img, points):
    cv2.line(img,points[0],points[1],(0,255,0),2)
    cv2.line(img,points[1],points[2],(0,255,0),2)
    cv2.line(img,points[2],points[3],(0,255,0),2)
    cv2.line(img,points[3],points[0],(0,255,0),2)
    return img

def paintCircles(img, circles, color=(0,255,0), thickness=2):
    for circle in circles:
        cv2.circle(img, circle[0], circle[1], color, thickness)
    return img

def paintRectangles(img, rectangles, color=(0,255,0), thickness=2):
    for rect in rectangles:
        cv2.rectangle(img, [rect[0], rect[1]], (rect[0]+rect[2], rect[1]+rect[3]), color, thickness)
    return img

def paintBoundingCircles(img, rectangles, color=(0,255,0), thickness=2, margin=0):
    for rect in rectangles:
        pts = np.array([[rect[0], rect[1]], [rect[0]+rect[2], rect[1]+rect[3]]])
        (x,y), radius = cv2.minEnclosingCircle(pts)
        center = (int(x),int(y))
        radius = int(radius*(1+margin))
        cv2.circle(img, center, radius, color, thickness)
    return img

class Wildcard:
    def __eq__(self, anything):
        return True