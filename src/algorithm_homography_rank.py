import cv2
import numpy as np

def getBestArucoHomography(W):
    T = np.array([[[0,0], [10,0], [10,10], [0,10]]], dtype=np.float32) # target marker shape
    m = len(W) # number of markers
    H = [0] * m # homography list, contains the homography that rectifies each marker to the original shape T
    S = [0] * m # similarity list, contains the similarities that map each H*W on top of T
    s = [0] * m # scores list, contains the homography ranking score for each corresponding H[i]

    for i in range(0, m):
        H[i] = cv2.getPerspectiveTransform(W[i], T)

        for j in range(0, m):
            S[j] = np.identity(3)
            if j != i:
                S[j][0:2,0:3], inliers = cv2.estimateAffinePartial2D( cv2.perspectiveTransform(W[j], H[i]), T)

            t = np.subtract(cv2.perspectiveTransform(W[j], np.dot(S[j], H[i])), T)
            s[i] += np.linalg.norm(t[0], ord='fro') / m # compute the average of the Frobenious norms

    best_H = H[np.argmin(s)]
    return best_H