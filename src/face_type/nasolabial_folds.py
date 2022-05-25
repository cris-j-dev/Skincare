import os
import sys
import numpy as np
import cv2
import skimage 
# import src.face_type.utils as utils
# import src.face_type.facemesh as facemesh

sys.path.append('src')
sys.path.append('src/face_type')

import utils as utils
import facemesh as facemesh

class NasolabialFolds:
    def __init__(self):
        return

    def nasolabial_folds(self, image):


        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        median = cv2.medianBlur(hsv[:,:,2], 7)
        meijering = skimage.filters.meijering(median)
        meijering *= 255
        meijering = meijering.astype(np.uint8)

        ret, res = cv2.threshold(meijering, 10, 255, cv2.THRESH_BINARY_INV)
        # ret, res = cv2.threshold(meijering, 10, 255, cv2.THRESH_BINARY)
        res = cv2.GaussianBlur(res, (25, 25), 0, borderType=cv2.BORDER_ISOLATED)
        ret, res = cv2.threshold(res, 10, 255, cv2.THRESH_BINARY)
        # kernel = np.ones((3, 3), np.uint8)
        # blur = cv2.dilate(blur, kernel)
        # blur = cv2.dilate(blur, kernel)
        # res = cv2.adaptiveThreshold(
        #     blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
        # )

        return res

    def draw(self, image, lesions, points):

        rect = cv2.boundingRect(points)
        x, y , w, h = rect

        res = image.copy()
        index_list = np.array(list(np.where(lesions==255)))
        for index in zip(index_list[0], index_list[1]):
            res[y+index[0], x+index[1], 0] = 255
            res[y+index[0], x+index[1], 1] = 0 
            res[y+index[0], x+index[1], 2] = 255

        return res 

    def crop(self, image, points):
        res = utils.crop_image(image, points, "white")
        # bg_img = np.ones_like(image, np.uint8)
        # bg_img = utils.crop_image(bg_img, points)
        # res = res - bg_img
        return 255 - res

    def run(self, fm, image):

        multi_face_landmarks = fm.detect_face_point(image)
        h, w, c = image.shape
        fm.set_points_loc(w=w, h=h)
        fm.set_lines()

        copyed = image.copy()
        nasolabial_folds_image = self.nasolabial_folds(copyed)

        # face_cheek_right_point = np.array(fm.points_loc["face_cheek_right_point"], dtype=uint8)
        # face_cheek_left_point  = np.array(fm.points_loc["face_cheek_left_point"], dtype=uint8)
        face_cheek_right_point = np.array(fm.points_loc["face_nasolabial_folds_right_point"], dtype=np.int)
        face_cheek_left_point  = np.array(fm.points_loc["face_nasolabial_folds_left_point"], dtype=np.int)
        # face_forehead_point    = np.array(fm.points_loc["face_forehead_point"], dtype='uint8')
        # face_chin_point        = np.array(fm.points_loc["face_chin_point"], dtype='uint8')
        # face_nose_point        = np.array(fm.points_loc["face_nose_point"], dtype='uint8')

        # face_cheek_right_point = fm.points_loc["face_smile_line_right_point3"]
        # face_cheek_left_point  = fm.points_loc["face_smile_line_left_point3"]
        # face_forehead_point    = fm.points_loc["face_forehead_point"]
        # face_chin_point        = fm.points_loc["face_chin_point"]
        # face_nose_point        = fm.points_loc["face_nose_point"]

        cheek_right = self.crop(nasolabial_folds_image, face_cheek_right_point)
        cheek_left  = self.crop(nasolabial_folds_image, face_cheek_left_point)
        # forehead    = self.crop(res, face_forehead_point)
        # chin        = self.crop(res, face_chin_point)
        # nose        = self.crop(res, face_nose_point)

        res = self.draw(image, cheek_right, face_cheek_right_point)
        res = self.draw(res,   cheek_left , face_cheek_left_point)
        # res = self.draw(res,   forehead   , face_forehead_point)
        # res = self.draw(res,   chin       , face_chin_point)
        # res = self.draw(res,   nose       , face_nose_point)

        return res


if __name__ == "__main__":

    faceMesh = facemesh.FaceMesh(thickness=5)
    faceMesh.set_label(
        [
            "face_flushing_right_point",
            "face_flushing_left_point",
            "face_cheek_right_point3",
            "face_cheek_left_point3",
            "face_forehead_point",
            "face_chin_point",
            "face_nose_point",
        ]
    )

    path = "data/"
    filelist = os.listdir(path)

    nasolabial_folds = NasolabialFolds()

    print(filelist)
    for filename in filelist[:]:
        if filename.split(".")[-1] == "jpg":
            print(path+filename)
            image = cv2.imread(path + filename)

            res = nasolabial_folds.run(faceMesh, image)
            merged = np.hstack((image, res))

            # cv2.imshow("original", image)
            cv2.imshow("result", merged)
            cv2.imwrite(path+filename.split(".")[0]+"_nasolabial_folds.png", merged)
            # cv2.waitKey(0)
            # exit()
