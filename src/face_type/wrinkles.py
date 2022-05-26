import os
import sys
import numpy as np
import cv2
import skimage 
import resize as resize_img
# import src.face_type.utils as utils
# import src.face_type.facemesh as facemesh

sys.path.append('src')
sys.path.append('src/face_type')

import utils as utils
import facemesh as facemesh

class Wrinkles:
    def __init__(self):
        return

    def wrinkles(self, image):


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

    def draw(self, image, re_image, wrinkles, points, re_points):

        rect = cv2.boundingRect(points)
        re_rect = cv2.boundingRect(re_points)

        x, y , w, h = rect
        _x, _y , _w, _h = re_rect

        H, W, C = image.shape
        _H, _W, _C = re_image.shape

        res = image.copy()
        index_list = np.array(list(np.where(wrinkles==255)))
        for index in zip(index_list[0], index_list[1]):
            x_point = int(index[0]/_h * h + y)
            y_point = int(index[1]/_w * w + x)
            res[x_point, y_point, 0] = 0
            res[x_point, y_point, 1] = 255
            res[x_point, y_point, 2] = 255

        return res 

    def crop(self, image, points):
        res = utils.crop_image(image, points, "white")
        # bg_img = np.ones_like(image, np.uint8)
        # bg_img = utils.crop_image(bg_img, points)
        # res = res - bg_img
        return 255 - res

    def run(self, fm, image):

        multi_face_landmarks = fm.detect_face_point(image)
        re_image, _, _ = resize_img.run(fm, image, 100)
        H, W, C = image.shape
        h, w, c = re_image.shape

        # fm.set_lines()

        copyed = re_image.copy()
        wrinklesed_image = self.wrinkles(copyed)

        fm.set_points_loc(w=W, h=H)
        face_cheek_right_point = np.array(fm.points_loc["face_smile_line_right_point"], dtype=np.int)
        face_cheek_left_point  = np.array(fm.points_loc["face_smile_line_left_point"], dtype=np.int)

        fm.set_points_loc(w=w, h=h)
        re_face_cheek_right_point = np.array(fm.points_loc["face_smile_line_right_point"], dtype=np.int)
        re_face_cheek_left_point  = np.array(fm.points_loc["face_smile_line_left_point"], dtype=np.int)

        # cheek_right = self.crop(wrinklesed_image, face_cheek_right_point)
        # cheek_left  = self.crop(wrinklesed_image, face_cheek_left_point)

        # res = self.draw(image, cheek_right, face_cheek_right_point)
        # res = self.draw(res,   cheek_left , face_cheek_left_point)

        cheek_right = self.crop(wrinklesed_image, re_face_cheek_right_point)
        cheek_left  = self.crop(wrinklesed_image, re_face_cheek_left_point)

        res = self.draw(image, re_image, cheek_right, face_cheek_right_point, re_face_cheek_right_point)
        res = self.draw(res,   re_image, cheek_left , face_cheek_left_point, re_face_cheek_left_point)


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

    path = "Test/"
    filelist = os.listdir(path)

    wrinkles = Wrinkles()

    print(filelist)
    for filename in filelist[:]:
        if filename.split(".")[-1] == "jpg":
            print(path+filename)
            image = cv2.imread(path + filename)

            import time
            start = time.time()
            res = wrinkles.run(faceMesh, image)
            print(time.time() - start)
            merged = np.hstack((image, res))

            # cv2.imshow("original", image)
            # cv2.imshow("result", merged)
            cv2.imwrite(path+filename.split(".")[0]+"_wrinkles3.png", merged)
            # cv2.waitKey(0)
            # exit()
