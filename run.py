import os
import numpy as np
import cv2
import sys

sys.path.append('src')
sys.path.append('src/face_type')

import type as face_type

face_type = face_type.FaceType()

filelist = os.listdir("testdata")

for filename in filelist:
    image = cv2.imread("testdata/"+filename)
    # res = face_type.run(image)
    img_flushings, img_acnes, img_lesions, img_wrinkles, img_pores = face_type.run(image)

    # cv2.imshow("res", res)
    # cv2.waitKey(0)

    merged = np.hstack((image, img_flushings))
    merged = np.hstack((merged, img_acnes))
    merged = np.hstack((merged, img_lesions))
    merged = np.hstack((merged, img_pores))
    merged = np.hstack((merged, img_wrinkles))

    cv2.imwrite("result/"+filename.split('.')[0]+"_result.png", merged)

    # merged_flushings = np.hstack((image, img_flushings))
    # merged_acnes = np.hstack((image, img_acnes))
    # merged_lesions = np.hstack((image, img_lesions))
    # merged_pores = np.hstack((image, img_pores))
    # merged_wrinkles = np.hstack((image, img_wrinkles))

    # cv2.imwrite("result/"+filename.split('.')[0]+"_flushings.png", merged_flushings)
    # cv2.imwrite("result/"+filename.split('.')[0]+"_acnes.png", merged_acnes)
    # cv2.imwrite("result/"+filename.split('.')[0]+"_lesions.png", merged_lesions)
    # cv2.imwrite("result/"+filename.split('.')[0]+"_pores.png", merged_pores)
    # cv2.imwrite("result/"+filename.split('.')[0]+"_pores.png", merged_wrinkles)