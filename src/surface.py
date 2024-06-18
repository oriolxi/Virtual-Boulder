import cv2
import numpy as np

import algebra

class Surface():
    max_regularized_size = [1400, 1400]
    
    def __init__(self):
        self.clear()

    def clear(self):
        self.map_camera = [] #list of 4 points that delimit the surface in the camera image
        self.w_camera = None #size of the camera image
        self.h_camera = None

        self.map_projector = [] #list of 4 points that delimit the surface on the projection image
        self.w_projector = None #size of the projection image
        self.h_projector = None
        
        self.map_regularized = [] #list of 4 points that delimit the surface in the regularized image
        self.w_regularized = None #size of the regularized surface image
        self.h_regularized = None

        self.homography_camera = None #homography that goes from camera image -> regularized image
        self.homography_projector = None #homography that goes from regularized image -> projected image
        self.homography_camproj = None #homography that goes from camera image -> projected image

        self.holds = [] #list of hold labels as dircles in the format [[x,y],radious]

    def setCameraParametres(self, m, w, h):
        self.w_camera = w
        self.h_camera = h

        self.map_camera = [[int(x), int(y)] for x, y in m]
        self.map_camera = algebra.polarSort(self.map_camera)

    def setProjectorParametres(self, m, w, h):
        self.w_projector = w
        self.h_projector = h

        self.map_projector = [[int(x), int(y)] for x, y in m]
        self.map_projector = algebra.polarSort(self.map_projector)

        pts_src = np.array(self.map_camera, dtype=np.float32)
        pts_dst = np.array(self.map_projector, dtype=np.float32)
        
        self.homography_camproj = cv2.getPerspectiveTransform(pts_src, pts_dst)

    def setRegularizedParametres(self, m, w, h):
        self.w_regularized = w
        self.h_regularized = h 

        self.map_regularized = [[int(x), int(y)] for x, y in m]
        self.map_regularized = algebra.polarSort(self.map_regularized)

        #find the homography that transforms surface plane in camera image to regularized image
        pts_src = np.array(self.map_camera, dtype=np.float32)
        pts_dst = np.array(self.map_regularized, dtype=np.float32)

        self.homography_camera = cv2.getPerspectiveTransform(pts_src, pts_dst)

        #find the homography that transforms regularized image into projector plane        
        pts_src = np.array(self.map_regularized, dtype=np.float32)
        pts_dst = np.array(self.map_projector, dtype=np.float32)

        self.homography_projector = cv2.getPerspectiveTransform(pts_src, pts_dst)

    def setHolds(self,h):
        self.holds = h

    def addHolds(self,h):
        self.holds += h
        
    def getCameraMap(self):
        return self.map_camera

    def getProjectorMap(self):
        return self.map_projector

    def getRegularizedMap(self):
        return self.map_regularized

    def getCameraSize(self):
        return (self.w_camera, self.h_camera)

    def getProjectorSize(self):
        return (self.w_projector, self.h_projector)

    def getRegularizedSize(self):
        return (self.w_regularized, self.h_regularized)

    def getMaxRegularizedsize(self):
        return self.max_regularized_size

    def getCameraMask(self):
        mask = np.zeros(shape=(self.h_camera, self.w_camera, 3), dtype=np.uint8)
        polygon = np.array(self.map_camera, dtype=np.int32)
        cv2.fillPoly(mask, pts=[polygon], color=(255, 255, 255))
        return mask

    def getProjectorMask(self):
        mask = np.zeros(shape=(self.h_projector, self.w_projector, 3), dtype=np.uint8)
        polygon = np.array(self.map_projector, dtype=np.int32)
        cv2.fillPoly(mask, pts=[polygon], color=(255, 255, 255))
        return mask

    def getRegularizedMask(self):
        mask = np.zeros(shape=(self.h_regularized, self.w_regularized, 3), dtype=np.uint8)
        polygon = np.array(self.map_regularized, dtype=np.int32)
        cv2.fillPoly(mask, pts=[polygon], color=(255, 255, 255))
        return mask

    def getHolds(self):
        return self.holds

    def getCameraHomography(self):
        return self.homography_camera

    def getProjectorHomography(self):
        return self.homography_projector

    def getCameraProjectorHomography(self):
        return self.homography_camproj

    def getProjectorHomographyFromSize(self,w,h):
        pts_src = np.array([[0,0], [0,h-1], [w-1,h-1], [w-1,0]], dtype=np.float32)
        pts_dst = np.array(self.map_projector, dtype=np.float32)

        return cv2.getPerspectiveTransform(pts_src, pts_dst)
