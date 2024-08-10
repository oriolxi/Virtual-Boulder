import cv2
import cv2.aruco as aruco
import numpy as np

def defaultParametres():
    params = {  "min_HSV":(0,0,0), 
                "max_HSV":(179,0,180),
                "remove_aruco":True,
                "kernel_type":1,
                "kernel_size":3,
                "open_iters":1,
                "close_iters":1,
                "min_area":200,
                "max_area":9000, 
                "min_wh_ratio":0.1}

    return params 

def detectHolds(img, params):
    # HSV range thresholding
    frame_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(frame_HSV, params["min_HSV"], params["max_HSV"])

    #remove any aruco present in the img
    if params["remove_aruco"]:
        aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_parameters = cv2.aruco.DetectorParameters()
        aruco_parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        aruco_detector = cv2.aruco.ArucoDetector(aruco_dictionary, aruco_parameters) 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (corners, ids, rejected) = aruco_detector.detectMarkers(gray)

        if len(corners) > 0:
            ids = ids.flatten()
            for (marker_corners, marker_id) in zip(corners, ids):
                pts = np.array([[[x[0], x[1]]] for x in marker_corners.reshape((4, 2))], dtype=np.int32)
                cv2.fillConvexPoly(thresh, pts, (0))

    # perform morphological opening (erosion followed by dilation) to remove noise
    k_type = cv2.MORPH_RECT
    if params["kernel_type"] == 1: k_type = cv2.MORPH_ELLIPSE
    if params["kernel_type"] == 2: k_type = cv2.MORPH_CROSS

    kernel = cv2.getStructuringElement(k_type, (params["kernel_size"], params["kernel_size"]))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=params["open_iters"])
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=params["close_iters"])

    # find contours in the resulting img
    # the elements hierarchy[0][i][0], hierarchy[0][i][1] , hierarchy[0][i][2] , and hierarchy[0][i][3] are the next and previous contours at the same hierarchical level, the first child contour and the parent contour
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

    holds = []
    contoursImg = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    boundingBoxesImg = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    if contours is not None and hierarchy is not None:
        # eliminate contours based on area and shape
        for cont, cont_h in zip(contours, hierarchy[0]):
            area = cv2.contourArea(cont)
            rect = cv2.boundingRect(cont)
            if area < params["min_area"] or area > params["max_area"] or min(rect[2] / rect[3], rect[3] / rect[2]) < params["min_wh_ratio"]:
                next_c = cont_h[2] #get first child contour from the eliminated contour
                while next_c > 0: #eliminate the parent contour from all nested contours
                    hierarchy[0, next_c ,3] = -1
                    next_c = hierarchy[0, next_c ,0]
                cont_h[2] = -2 # set first child contour to -2 to signal elimination by size

        # eliminate nested contours
        for cont, cont_h in zip(contours, hierarchy[0]): 
            if cont_h[2] != -2 and cont_h[3] != -1: #discard nested countours
                cont_h[2] = -3 # set first child contour to -3 to signal elimination by nesting

        # generate preview image with all contours
        for cont, cont_h in zip(contours, hierarchy[0]): 
            if cont_h[2] == -2:
                color = (0, 0, 255) # RED for discarded for size
            elif cont_h[2] == -3:
                color = (0, 165, 255) # ORANGE for discarded for nesting
            else:
                color = (0, 255, 0) # GREEN for good contour
            
            cv2.drawContours(contoursImg, [cont], 0, color, 2)

        # draw bounding boxes and store holds
        for cont, cont_h in zip(contours, hierarchy[0]): 
            if cont_h[2] != -2 and cont_h[2] != -3:
                x,y,w,h = cv2.boundingRect(cont)
                cv2.rectangle(boundingBoxesImg, (x,y), (x+w,y+h), (255,255,0),2)
                holds.append([x,y,w,h])

    return holds, [thresh, closing, contoursImg, boundingBoxesImg]
