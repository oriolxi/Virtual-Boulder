import cv2
import math
import numpy as np

def isPointInsideRectangle(p, rectangle): #returns True if poiny p is inside of rectangle
    return (rectangle[0] <= p[0] <= rectangle[0] + rectangle[2]) and (rectangle[1] <= p[1] <= rectangle[1] + rectangle[3])

def isCircleTouchingRectangle(rectangle, circle_p, circle_r): #returns True if circle touches rectangle
    nearest_x = max(rectangle[0], min(circle_p[0], rectangle[0] + rectangle[2]))
    nearest_y = max(rectangle[1], min(circle_p[1], rectangle[1] + rectangle[3]))

    distance = np.linalg.norm(np.subtract(circle_p, [nearest_x, nearest_y]))
    return distance <= circle_r

def overlapCircleRectangle(rectangle, circle_p, circle_r, resolution=8): #returns the amount circle overlaped by the rectangle
    #create grid points within the circle using polar coordinates, this way proximity to the center is favored
    radious_samples = np.linspace(0, circle_r, resolution)
    angle_samples = np.linspace(0, np.pi*2, resolution)
    polar_pts = np.array(np.meshgrid(radious_samples, angle_samples)).T.reshape(-1, 2)
    pts_circle = np.add(circle_p, polarToCartesian(polar_pts))

    pts_in_rectangle = np.apply_along_axis(isPointInsideRectangle, 1, pts_circle, rectangle=rectangle)
    return np.count_nonzero(pts_in_rectangle) / len(pts_circle)

def polarToCartesian(points): #gets array of polar coordinate pairs [r, w] and returns corresponding array of cartesian pairs[x,y]
    x = points[:,0] * np.cos(points[:,1])
    y = points[:,0] * np.sin(points[:,1])

    return np.stack((x,y), axis=1)

def get2DBoundingBox(points):
    max_idx = np.argmax(points, axis=0)
    min_idx = np.argmin(points, axis=0)
    max_x, max_y = points[max_idx]
    min_x, min_y = points[min_idx]
    
    return [max_x[0], max_y[1], min_x[0], min_y[1]]

def translatePtsPositive(points):
    bb = get2DBoundingBox(points[0])

    hT = np.identity(3)
    hT[0,2] = -bb[2]
    hT[1,2] = -bb[3]

    return cv2.perspectiveTransform(points, hT)

def scalePtsToLimits(points, limits):
    bb = get2DBoundingBox(points[0])
    
    s = limits[0]/bb[0] if bb[0] > bb[1] else limits[1]/bb[1]
    hS = np.identity(3)
    hS[0,0] = s
    hS[1,1] = s
    
    return cv2.perspectiveTransform(points, hS)

def rotatePtsToHorizontalLine(points, pt1, pt2):
    ang = np.pi - np.arctan2(pt1[1] - pt2[1], pt1[0] - pt2[0])

    hR = np.identity(3)
    hR[0,0] = np.cos(ang)
    hR[1,1] = np.cos(ang)
    hR[0,1] = -np.sin(ang)
    hR[1,0] = np.sin(ang)

    return cv2.perspectiveTransform(points, hR)

def polarSort(points): #counteclockwise polar sort from vertical y axis beeing 0
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    cx, cy = (sum(x) / len(points), sum(y) / len(points))

    angles = np.arctan2(x-cx, y-cy)
    indices = np.argsort(angles)
    sorted = np.array(points)

    return sorted[indices].tolist()