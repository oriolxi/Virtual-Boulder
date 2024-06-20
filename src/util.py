import cv2
import numpy as np
from PyQt6.QtGui import QImage

# INPUTS:
#   img = image as np.array as used in OpenCv
# OUTPUTS:
#   img = image as a QImgage
def QimageFromCVimage(cvImg):
    if cvImg is None: return None
    return QImage(cvImg.data, cvImg.shape[1], cvImg.shape[0], cvImg.strides[0], QImage.Format.Format_BGR888)

# INPUTS:
#   img = image as QImgage
# OUTPUTS:
#   img = image as np.array as used in OpenCv
def CVimageFromQimage(qImg):
    if qImg is None: return None
    incomingImage = qImg.convertToFormat(QImage.Format.Format_BGR888)
    
    width = incomingImage.width()
    height = incomingImage.height()

    ptr = incomingImage.bits()
    ptr.setsize(height * width * 3)
    arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3)).copy()
    return arr

# INPUTS:
#   img = image as  np.array
#   points = list of 4 [x,y] points that correspond to the corners of a 4 sided polygon
# OUTPUTS:
#   img = input img with the 4 sided polygon painted on top
def paintSelection(img, points):
    cv2.line(img,points[0],points[1],(0,255,0),2)
    cv2.line(img,points[1],points[2],(0,255,0),2)
    cv2.line(img,points[2],points[3],(0,255,0),2)
    cv2.line(img,points[3],points[0],(0,255,0),2)
    return img

# INPUTS:
#   img = image as  np.array
#   rectangle = list of circles [x,y,radious] where (x,y) is the circle's center
#   color = rectangle color
#   thickness = circle border thickness (-1 corresponds to filled circle)
# OUTPUTS:
#   img = input img with circles painted on top
def paintCircles(img, circles, color=(0,255,0), thickness=2):
    for circle in circles:
        cv2.circle(img, circle[0], circle[1], color, thickness)
    return img

# INPUTS:
#   img = image as  np.array
#   rectangle = list of rectangles [x,y,width,height] where (x,y) is the rectangle's top-left corner
#   color = rectangle color
#   thickness = rectangle border thickness (-1 corresponds to filled rectangle)
# OUTPUTS:
#   img = input img with rectangles painted on top
def paintRectangles(img, rectangles, color=(0,255,0), thickness=2):
    for rect in rectangles:
        cv2.rectangle(img, [rect[0], rect[1]], (rect[0]+rect[2], rect[1]+rect[3]), color, thickness)
    return img

# INPUTS:
#   img = image as  np.array
#   rectangle = list of rectangles [x,y,width,height] where (x,y) is the rectangle's top-left corner
#   color = circle color
#   thickness = cricle border thickness (-1 corresponds to filled circle)
#   margin = amount that the bounding circle of each rectangle will be enlarged (5% -> 0.05)
# OUTPUTS:
#   img = input img with the bounding circle of each rectangle painted on top
def paintBoundingCircles(img, rectangles, color=(0,255,0), thickness=2, margin=0):
    for rect in rectangles:
        pts = np.array([[rect[0], rect[1]], [rect[0]+rect[2], rect[1]+rect[3]]])
        (x,y), radius = cv2.minEnclosingCircle(pts)
        center = (int(x),int(y))
        radius = int(radius*(1+margin))
        cv2.circle(img, center, radius, color, thickness)
    return img

# Wildcard used for comparision when compating list of multiple elements where only some of the elements need to be matched
# An instance of it always returns true when compared
class Wildcard:
    def __eq__(self, anything):
        return True