import os
import numpy as np
import cv2
import src.type as face_type

face_type = face_type.FaceType()

filelist = os.listdir("data")

for filename in filelist:
    image = cv2.imread("data/"+filename)
    # res = face_type.run(image)
    img_flushings, img_acnes, img_lesions, img_wrinkles, img_pores = face_type.run(image)

    # cv2.imshow("res", res)
    # cv2.waitKey(0)

    merged_flushings = np.hstack((image, img_flushings))
    merged_acnes = np.hstack((image, img_acnes))
    merged_lesions = np.hstack((image, img_lesions))
    merged_pores = np.hstack((image, img_pores))

    cv2.imwrite("result/"+filename.split('.')[0]+"_flushings.png", merged_flushings)
    # cv2.imwrite("result/"+filename.split('.')[0]+"_acnes.png", merged_acnes)
    # cv2.imwrite("result/"+filename.split('.')[0]+"_lesions.png", merged_lesions)
    # cv2.imwrite("result/"+filename.split('.')[0]+"_pores.png", merged_pores)