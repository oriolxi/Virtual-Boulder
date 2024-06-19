import cv2
import numpy as np

# INPUTS:
#   point = [x,y] coordinates
#   rectangle = [x,y,width,height] where (x,y) is the rectangle's top-left corner
# OUTPUTS:
#   True / False = weather the point is inside or outside the rectangle
def isPointInsideRectangle(point, rectangle):
    return (rectangle[0] <= point[0] <= rectangle[0] + rectangle[2]) and (rectangle[1] <= point[1] <= rectangle[1] + rectangle[3])

# INPUTS:
#   rectangle = [x,y,width,height] where (x,y) is the rectangle's top-left corner
#   circle_point = [x,y] coordinates of the circle's center
#   radious = circle's radious
# OUTPUTS:
#   True / False = weather the point touching the rectangle or not. 
#   -> This includes the cricle fully enclosed inside the rectangle
def isCircleTouchingRectangle(rectangle, circle_point, radious):
    nearest_x = max(rectangle[0], min(circle_point[0], rectangle[0] + rectangle[2]))
    nearest_y = max(rectangle[1], min(circle_point[1], rectangle[1] + rectangle[3]))

    distance = np.linalg.norm(np.subtract(circle_point, [nearest_x, nearest_y]))
    return distance <= radious

# INPUTS:
#   rectangle = [x,y,width,height] where (x,y) is the rectangle's top-left corner
#   circle_point = [x,y] coordinates of the circle's center
#   radious = circle's radious
#   resolution = number of samples generated in each coordinate degree of freedom
# OUTPUTS:
#   overlap = the amount of the circle that overlapes with the rectangle
#   -> Estimated using point sampling using polar coordinates
def overlapCircleRectangle(rectangle, circle_point, radious, resolution=8):
    radious_samples = np.linspace(0, radious, resolution)
    angle_samples = np.linspace(0, np.pi*2, resolution)
    polar_pts = np.array(np.meshgrid(radious_samples, angle_samples)).T.reshape(-1, 2)
    pts_circle = np.add(circle_point, polarToCartesian(polar_pts))

    pts_in_rectangle = np.apply_along_axis(isPointInsideRectangle, 1, pts_circle, rectangle=rectangle)
    return np.count_nonzero(pts_in_rectangle) / len(pts_circle)

# INPUTS:
#   points = list of polar coordinates [radious, angle]
# OUTPUTS:
#   List of corresponding cartesin coordinates [x,y]
def polarToCartesian(points):
    x = points[:,0] * np.cos(points[:,1])
    y = points[:,0] * np.sin(points[:,1])

    return np.stack((x,y), axis=1)

def get2DBoundingBox(points):
    max_idx = np.argmax(points, axis=0)
    min_idx = np.argmin(points, axis=0)
    max_x, max_y = points[max_idx]
    min_x, min_y = points[min_idx]
    
    return [max_x[0], max_y[1], min_x[0], min_y[1]]

# INPUTS:
#   points = list of [x,y] points
# OUTPUTS:
#   List of [x,y] points translated to the positive x-y axis
def translatePtsPositive(points):
    bounding_box = get2DBoundingBox(points[0])

    hT = np.identity(3)
    hT[0,2] = -bounding_box[2]
    hT[1,2] = -bounding_box[3]

    return cv2.perspectiveTransform(points, hT)

# INPUTS:
#   points = list of [x,y] points
#   limits = [width, height]
# OUTPUTS:
#   List of [x,y] points scaled fit inside the limits specified by [width, height]
def scalePtsToLimits(points, limits):
    bounding_box = get2DBoundingBox(points[0])
    
    s = limits[0] / bounding_box[0] if bounding_box[0] > bounding_box[1] else limits[1] / bounding_box[1]
    hS = np.identity(3)
    hS[0,0] = s
    hS[1,1] = s
    
    return cv2.perspectiveTransform(points, hS)

# INPUTS:
#   points = list of [x,y] points
#   pt1 = [x,y] point
#   pt2 = [x,y] point
# OUTPUTS:
#   List of [x,y] points rotated by the angle formed by the line pt1 - pt2 and the x axis
def rotatePtsToHorizontalLine(points, pt1, pt2):
    ang = np.pi - np.arctan2(pt1[1] - pt2[1], pt1[0] - pt2[0])

    hR = np.identity(3)
    hR[0,0] = np.cos(ang)
    hR[1,1] = np.cos(ang)
    hR[0,1] = -np.sin(ang)
    hR[1,0] = np.sin(ang)

    return cv2.perspectiveTransform(points, hR)

# INPUTS:
#   points = list of [x,y] points
# OUTPUTS:
#   List of [x,y] points sorted counteclockwise from the positive y axis by the angle each point forms with the list's centroid.
def polarSort(points):
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    centroid_x, centroid_y = (sum(x) / len(points), sum(y) / len(points))

    angles = np.arctan2(x - centroid_x, y - centroid_y)
    indices = np.argsort(angles)
    sorted = np.array(points)

    return sorted[indices].tolist()