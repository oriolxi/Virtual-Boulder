import cv2
import numpy as np

import algebra

class Surface():
    max_surface_size = [1400, 1400] # maximum size the fronto-parallel view image can take
    
    def __init__(self):
        self.clear()

    def clear(self):
        self.wall_roi_camera = [] # list of 4 points that delimit the climbing wall on the camera image plane
        self.width_camera = None # size of the camera image
        self.height_camera = None

        self.wall_roi_projector = [] # list of 4 points that delimit the climbing wall on the projector image plane
        self.width_projector = None # size of the projector image
        self.height_projector = None
        
        self.wall_roi_surface = [] # list of 4 points that delimit the climbing wall on the surface plane
        self.width_surface = None # size of the surface image (fronto-parallel view)
        self.height_surface = None

        self.homography_CS = None # homography that maps the camera plane to the surface plane
        self.homography_SP = None # homography that maps the surface plane to the projector plane
        self.homography_CP = None # homography that maps the camera plane to the projector plane

        self.holds = [] # set of hold bounding boxes stored as rectangles b = [x,y,w,h]

    def setCameraParametres(self, roi, width, height):
        self.width_camera = width
        self.height_camera = height

        self.wall_roi_camera = [[int(x), int(y)] for x, y in roi]
        self.wall_roi_camera = algebra.polarSort(self.wall_roi_camera)

    def setProjectorParametres(self, roi, width, height):
        self.width_projector = width
        self.height_projector = height

        self.wall_roi_projector = [[int(x), int(y)] for x, y in roi]
        self.wall_roi_projector = algebra.polarSort(self.wall_roi_projector)

        pts_src = np.array(self.wall_roi_camera, dtype=np.float32)
        pts_dst = np.array(self.wall_roi_projector, dtype=np.float32)
        self.homography_CP = cv2.getPerspectiveTransform(pts_src, pts_dst)

    def setSurfaceParametres(self, roi, width, height):
        self.width_surface = width
        self.height_surface = height

        self.wall_roi_surface = [[int(x), int(y)] for x, y in roi]
        self.wall_roi_surface = algebra.polarSort(self.wall_roi_surface)


        pts_src = np.array(self.wall_roi_camera, dtype=np.float32)
        pts_dst = np.array(self.wall_roi_surface, dtype=np.float32)
        self.homography_CS = cv2.getPerspectiveTransform(pts_src, pts_dst)

        pts_src = np.array(self.wall_roi_surface, dtype=np.float32)
        pts_dst = np.array(self.wall_roi_projector, dtype=np.float32)
        self.homography_SP = cv2.getPerspectiveTransform(pts_src, pts_dst)

    def setHolds(self,h):
        self.holds = h

    def addHolds(self,h):
        self.holds += h
        
    def getWallRoiCamera(self):
        return self.wall_roi_camera

    def getWallRoiProjector(self):
        return self.wall_roi_projector

    def getWallRoiSurface(self):
        return self.wall_roi_surface

    def getSizeCamera(self):
        return (self.width_camera, self.height_camera)

    def getSizeProjector(self):
        return (self.width_projector, self.height_projector)

    def getSizeSurface(self):
        return (self.width_surface, self.height_surface)

    def getMaxSizeSurface(self):
        return self.max_surface_size

    def getMaskCamera(self):
        mask = np.zeros(shape=(self.height_camera, self.width_camera, 3), dtype=np.uint8)
        polygon = np.array(self.wall_roi_camera, dtype=np.int32)
        cv2.fillPoly(mask, pts=[polygon], color=(255, 255, 255))
        return mask

    def getMaskProjector(self):
        mask = np.zeros(shape=(self.height_projector, self.width_projector, 3), dtype=np.uint8)
        polygon = np.array(self.wall_roi_projector, dtype=np.int32)
        cv2.fillPoly(mask, pts=[polygon], color=(255, 255, 255))
        return mask

    def getMaskSurface(self):
        mask = np.zeros(shape=(self.height_surface, self.width_surface, 3), dtype=np.uint8)
        polygon = np.array(self.wall_roi_surface, dtype=np.int32)
        cv2.fillPoly(mask, pts=[polygon], color=(255, 255, 255))
        return mask

    def getHolds(self):
        return self.holds

    def getHomographyCS(self):
        return self.homography_CS

    def getHomographySP(self):
        return self.homography_SP

    def getHomographyCP(self):
        return self.homography_CP