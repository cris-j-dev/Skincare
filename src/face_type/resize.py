
import os
import sys
import numpy as np
import cv2
import utils as utils
import facemesh as facemesh
# import src.face_type.utils as utils
# import src.face_type.facemesh as facemesh
sys.path.append('src')
sys.path.append('src/face_type')


def run(fm, image, size):

    multi_face_landmarks = fm.detect_face_point(image)
    h, w, c = image.shape
    fm.set_points_loc(w=w, h=h)
    fm.set_lines()

    res = image.copy()

#########################################################################################################
    eye_point = np.array(fm.points_loc["eye_point"], dtype=np.int)

    x = utils.get_distance_point(eye_point[0], eye_point[1])
    y = utils.get_distance_point(eye_point[2], eye_point[3])
    eye_distance = utils.get_distance(x, y)
    
    W = int(size * w / eye_distance)
    H = int(W * h / w)
    resize_img = cv2.resize(res, (W, H), cv2.INTER_LINEAR)

    w_ratio = W/w
    h_ratio = H/h
    return resize_img, w, h

if __name__ == "__main__":

    faceMesh = facemesh.FaceMesh(thickness=5)
    faceMesh.set_label(
        [
            "face_flushing_right_point",
            "face_flushing_left_point",
            "face_cheek_right_point",
            "face_cheek_left_point",
            "face_forehead_point",
            "face_chin_point",
            "face_nose_point",
        ]
    )

    path = "Test/"
    filelist = os.listdir(path)
    # print(filelist)
    for filename in filelist[:]:
        if filename.split(".")[-1] == "jpg":
     #       print(path+filename)
            image = cv2.imread(path + filename)

            res = run(faceMesh, image)
            # merged = np.hstack((image, res))
            # merged = np.hstack((merged, res2))
            # cv2.imshow("result", merged)
            # cv2.waitKey(0)
            # cv2.imwrite(path+filename.split(".")[0]+"_acnes.png", merged)
