import os
import numpy as np
import cv2
import math
import utils.utils as utils
import utils.facemesh as facemesh

class Pores:
    def __init__(self):
        return

    def pores(self, image):

        res3 = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        l_space = res3[:, :, 0]
        #cv2.imshow("l_space", l_space)

        median_img = cv2.medianBlur(l_space, 7)
        #cv2.imshow("median", median_img)

        norm_img = cv2.normalize(median_img, None, 0, 255, cv2.NORM_MINMAX)

        k = 15
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        blackhat_img = cv2.morphologyEx(norm_img, cv2.MORPH_BLACKHAT, kernel)
        #cv2.imshow("black", blackhat_img)
        ret, res = cv2.threshold(blackhat_img, 10, 255, cv2.THRESH_BINARY)

        return res


    def draw(self, image, pores, points):

        rect = cv2.boundingRect(points)
        x, y, w, h = rect
        # croped = image[y : y + h, x : x + w].copy()

        H, W, C = image.shape

        mask = np.zeros((H, W, 3), dtype=np.uint8)
        # mask[y:y+h,x:x+w,0] = pores
        mask[y:y+h,x:x+w,1] = pores
        mask[y:y+h,x:x+w,2] = pores

        alpha = 0.1 

        # cv2.imshow("image", image)
        # cv2.imshow("mask", mask)
        # cv2.ellipse(mask, (cx, cy), (lx, ly), deg, 0, 360, (0, 0, 255), -1)
        res = cv2.addWeighted(image, 1, mask, (1 - alpha), 0)  # 방식2
        # cv2.circle(blended2, (cx, cy), 10, (0,0,255), -1)

        return res 

    def crop(self, image, points):
        res = utils.crop_image(image, points)
        bg_img = np.zeros_like(image, np.uint8)
        bg_img = utils.crop_image(bg_img, points)
        res = res - bg_img
        return res


    def run(self, fm, image):

        multi_face_landmarks = fm.detect_face_point(image)
        h, w, c = image.shape
        fm.set_points_loc(w=w, h=h)
        fm.set_lines()

        res = image.copy()
        face_cheek_right_point = np.array(fm.points_loc["face_cheek_right_point"], dtype=np.int)
        face_cheek_left_point  = np.array(fm.points_loc["face_cheek_left_point"], dtype=np.int)
        face_chin_point        = np.array(fm.points_loc["face_chin_point"], dtype=np.int)
        face_nose_point        = np.array(fm.points_loc["face_nose_point"], dtype=np.int)

        pores_img = self.pores(image)

        cheek_right = self.crop(pores_img, face_cheek_right_point)
        cheek_left  = self.crop(pores_img, face_cheek_left_point)
        chin        = self.crop(pores_img, face_chin_point)
        nose        = self.crop(pores_img, face_nose_point)

        res = self.draw(image, cheek_right, face_cheek_right_point)
        res = self.draw(res, cheek_left , face_cheek_left_point)
        res = self.draw(res, chin       , face_chin_point)
        res = self.draw(res, nose       , face_nose_point)

        return res


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

    path = "../Test/pore/"
    filelist = os.listdir(path)

    print(filelist)
    for filename in filelist[:]:
        if filename.split(".")[-1] == "jpg":
            print(path+filename)
            image = cv2.imread(path + filename)

            res = run(faceMesh, image)
            merged = np.hstack((image, res))
            cv2.imshow("result", merged)
            cv2.imwrite(path+filename.split(".")[0]+"_pores.png", merged)
            cv2.waitKey(0)
