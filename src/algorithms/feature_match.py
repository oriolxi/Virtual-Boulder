import cv2
import numpy as np

# Code from https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
# INPUTS:
#   img_from
#   img_to
# OUTPUTS:
#   H = homography that maps img_from to img_to using the matches found between them
#   img_feature_match = side by side of img_from and img_to with matches drawn in between the two
def featureMatching(img_from, img_to):
    MIN_MATCH_COUNT = 10 # minimum matches to atempt estimating H
    img_from = cv2.cvtColor(img_from, cv2.COLOR_BGR2GRAY)
    img_to = cv2.cvtColor(img_to, cv2.COLOR_BGR2GRAY)
    
    # initiate SIFT detector
    sift = cv2.SIFT_create()
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img_from,None)
    kp2, des2 = sift.detectAndCompute(img_to,None)
    
    # find mathces with FLANN based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # store all the good matches as per Lowe's ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    # estimate H if sufficient good mathces are available
    H = None
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
 
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matches_mask = None

    # draw matches in green
    draw_params = dict(matchColor = (0,255,0), singlePointColor = None, matchesMask = matches_mask, flags = 2)
    img_feature_match = cv2.drawMatches(img_from, kp1, img_to, kp2, good, None, **draw_params)

    return H, img_feature_match