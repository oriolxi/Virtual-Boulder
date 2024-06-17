import cv2
import numpy as np

def getBestArucoHomography(W):
    T = np.array([[[0,0], [10,0], [10,10], [0,10]]], dtype=np.float32)
    m = len(W)
    H = [0] * m
    S = [0] * m
    s = [0] * m

    for i in range(0, m):
        H[i] = cv2.getPerspectiveTransform(W[i], T)

        for j in range(0, m):
            S[j] = np.identity(3)
            if j != i:
                S[j][0:2,0:3], inliers = cv2.estimateAffinePartial2D( cv2.perspectiveTransform(W[j], H[i]), T) # find similarity transform from unwarped points using H[i] to target T

            t = np.subtract(cv2.perspectiveTransform(W[j], np.dot(S[j], H[i])), T) # compute de difference between the rectified points and the target
            s[i] += np.linalg.norm(t[0], ord='fro') / m # compute the avarage of the frobenious norms

    #print(f'Scores -> {s}')
    best_H = H[np.argmin(s)]

    return best_H