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

class Lesions:
    def __init__(self):
        return

    def lesions(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (25, 25), 0, borderType=cv2.BORDER_ISOLATED)
        kernel = np.ones((3, 3), np.uint8)
        blur = cv2.dilate(blur, kernel)
        blur = cv2.dilate(blur, kernel)
        res = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
        )


        return res

    def lesions2(self, image, points):

        # image = fm.crop(image, label)
        temp = np.zeros((image.shape), dtype=np.uint8)
        bg = utils.crop_image(temp, points, "white")
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
        res3 = utils.crop_image(image, points, "black")
        gray = cv2.cvtColor(res3, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (25, 25), 0, borderType=cv2.BORDER_ISOLATED)

        res = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
        )

        return res

    def draw(self, image, lesions, points):

        # if lesions is None:
        #     return image
        rect = cv2.boundingRect(points)
        x, y , w, h = rect

        H, W, C = image.shape

        # mask = np.zeros((H, W, 3), dtype=np.uint8)

        # mask[y:y+h, x:x+w,1] = lesions
        # mask[y:y+h, x:x+w,2] = lesions

        res = image.copy()
        index_list = np.array(list(np.where(lesions==255)))
        for index in zip(index_list[0], index_list[1]):
            res[y+index[0], x+index[1], 0] = 0
            res[y+index[0], x+index[1], 1] = 255
            res[y+index[0], x+index[1], 2] = 255



        # alpha = 1.5 
        # res = cv2.addWeighted(image, 1, mask, (1-alpha), 0)

        return res 

    def crop(self, image, points):
        res = utils.crop_image(image, points, "white")
        # bg_img = np.zeros_like(image, np.uint8)
        # bg_img = utils.crop_image(bg_img, points)
        # res = res - bg_img
        return 255 - res

    def run(self, fm, image):

        multi_face_landmarks = fm.detect_face_point(image)
        h, w, c = image.shape
        fm.set_points_loc(w=w, h=h)
        fm.set_lines()

        res = image.copy()
        face_cheek_right_point = np.array(fm.points_loc["face_cheek_right_point"], dtype=np.int)
        face_cheek_left_point  = np.array(fm.points_loc["face_cheek_left_point"], dtype=np.int)
        face_forehead_point    = np.array(fm.points_loc["face_forehead_point"], dtype=np.int)
        face_chin_point        = np.array(fm.points_loc["face_chin_point"], dtype=np.int)
        face_nose_point        = np.array(fm.points_loc["face_nose_point"], dtype=np.int)

        lesions_img = self.lesions(image)

        cheek_right = self.crop(lesions_img, face_cheek_right_point)
        cheek_left  = self.crop(lesions_img, face_cheek_left_point)
        forehead    = self.crop(lesions_img, face_forehead_point)
        chin        = self.crop(lesions_img, face_chin_point)
        nose        = self.crop(lesions_img, face_nose_point)

        res = self.draw(image, cheek_right, face_cheek_right_point)
        res = self.draw(res,   cheek_left , face_cheek_left_point)
        res = self.draw(res,   forehead   , face_forehead_point)
        res = self.draw(res,   chin       , face_chin_point)
        res = self.draw(res,   nose       , face_nose_point)

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

    path = "Test/"
    filelist = os.listdir(path)

    lesions = Lesions()

    print(filelist)
    for filename in filelist[:]:
        if filename.split(".")[-1] == "jpg":
            print(path+filename)
            image = cv2.imread(path + filename)

            res = lesions.run(faceMesh, image)
            merged = np.hstack((image, res))
            cv2.imshow("result", merged)
            # cv2.imwrite(path+filename.split(".")[0]+"_lesions.png", merged)
            cv2.waitKey(0)
