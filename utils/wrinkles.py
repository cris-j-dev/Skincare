import numpy as np
import cv2
import src.face_type.utils as utils
import src.face_type.facemesh as facemesh

class Wrinkles:
    def __init__(self):
        return


    def detect_shape(img, size, size2, size3):
        res = cv2.fastNlMeansDenoising(img, None, size, size2, size3)
        res = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX)
        ret, res = cv2.threshold(res, 127, 255, cv2.THRESH_BINARY_INV)
        # res = cv2.distanceTransform(res, cv2.DIST_L2, 5)
        # res = (res/(res.max()-res.min()) * 255).astype(np.uint8)
        # res = cv2.adaptiveThreshold(res, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, -3)

        return res

    def non_max_suppression(img, D):
        M, N = img.shape
        Z = np.zeros((M,N), dtype=np.int32)
        angle = D * 180. / np.pi
        angle[angle < 0] += 180

        
        for i in range(1,M-1):
            for j in range(1,N-1):
                try:
                    q = 255
                    r = 255
                    
                    #angle 0
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        q = img[i, j+1]
                        r = img[i, j-1]
                    #angle 45
                    elif (22.5 <= angle[i,j] < 67.5):
                        q = img[i+1, j-1]
                        r = img[i-1, j+1]
                    #angle 90
                    elif (67.5 <= angle[i,j] < 112.5):
                        q = img[i+1, j]
                        r = img[i-1, j]
                    #angle 135
                    elif (112.5 <= angle[i,j] < 157.5):
                        q = img[i-1, j-1]
                        r = img[i+1, j+1]

                    if (img[i,j] >= q) and (img[i,j] >= r):
                        Z[i,j] = img[i,j]
                    else:
                        Z[i,j] = 0

                except IndexError as e:
                    pass
        
        return Z

    def run(self, fm, image):
        return image

