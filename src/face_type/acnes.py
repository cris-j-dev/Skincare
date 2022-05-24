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


class Acnes:
    def __init__(self):
        return

    def acnes(self, image, points):

        # 1 Crop and get average, min, max value
        # 2 Color Balancing
        # 3 Normalization of a*
        res2 = utils.color_balancing(image, points)
        res3 = cv2.cvtColor(res2, cv2.COLOR_BGR2Lab)
        _, max_a, avg_a = utils.get_mean_from_masked_image( res3[:, :, 1], points)
        if max_a > 134:
            res3 = utils.crop_image(res3, points)

            alpha = res3[:, :, 1]
            mask=cv2.inRange(alpha,0,10)
            alpha[mask==255]=avg_a

            alpha = cv2.normalize(alpha, None, 0, 255, cv2.NORM_MINMAX)
            res5, ma = utils.estimation_of_AC(alpha, 255)
            res6 = utils.morphology(res5)
            ret, res = cv2.threshold(res6, 230, 255, cv2.THRESH_BINARY)

            return res
        return None


    def draw(self, image, acnes, points):

        if acnes is None:
            return image

        rect = cv2.boundingRect(points)
        x, y, w, h = rect
        H, W, C = image.shape
        mask = np.zeros((H, W, 3), dtype=np.uint8)

        acnes = acnes.astype(np.uint8)

        # temp = np.zeros((h,w), dtype=np.uint8)
        temp = image.copy()
        contours, hirerarchy = cv2.findContours(acnes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            contour += [x, y]
            area = cv2.contourArea(contour)
            # print(area)
            cv2.drawContours(temp, [contour], -1, (43,0,255), -1)

        return temp

    def draw2(image, acnes, points):

        if acnes is None:
            return image

        rect = cv2.boundingRect(points)
        x, y, w, h = rect
        # croped = image[y : y + h, x : x + w].copy()

        H, W, C = image.shape

        mask = np.zeros((H, W, 3), dtype=np.uint8)
        # mask[y:y+h,x:x+w,0] = acnes
        # mask[y:y+h,x:x+w,1] = acnes

        mask[y:y+h,x:x+w,2] = acnes


        alpha = 0.01 

        # cv2.ellipse(mask, (cx, cy), (lx, ly), deg, 0, 360, (0, 0, 255), -1)
        blended2 = cv2.addWeighted(image, 1, mask, (1 - alpha), 0)  # 방식2
        # cv2.circle(blended2, (cx, cy), 10, (0,0,255), -1)

        return blended2

    def draw3(self, image, acnes, points):

        if acnes is None:
            return image

        rect = cv2.boundingRect(points)
        x, y, w, h = rect
        acnes = acnes.astype(np.uint8)

        image[y:y+h,x:x+w] = acnes

        return image

    def crop(self, image, points):
        res = utils.crop_image(image, points)
        bg_img = np.zeros_like(image, np.uint8)
        bg_img = utils.crop_image(bg_img, points, "white")
        res = res - bg_img
        return res



    def run(self, fm, image):

        multi_face_landmarks = fm.detect_face_point(image)
        h, w, c = image.shape
        fm.set_points_loc(w=w, h=h)
        fm.set_lines()

        res = image.copy()
        # print("res.shape : ", res.shape)

        face_cheek_right_point = np.array(fm.points_loc["face_cheek_right_point"], dtype=np.int)
        face_cheek_left_point  = np.array(fm.points_loc["face_cheek_left_point"], dtype=np.int)
        face_forehead_point    = np.array(fm.points_loc["face_forehead_point"], dtype=np.int)
        face_chin_point        = np.array(fm.points_loc["face_chin_point"], dtype=np.int)
        face_nose_point        = np.array(fm.points_loc["face_nose_point"], dtype=np.int)

        cheek_right = self.acnes(image, face_cheek_right_point)
        cheek_left  = self.acnes(image, face_cheek_left_point)
        forehead    = self.acnes(image, face_forehead_point)
        chin        = self.acnes(image, face_chin_point)
        nose        = self.acnes(image, face_nose_point)

        res = self.draw(image, cheek_right, face_cheek_right_point)
        res = self.draw(res,   cheek_left , face_cheek_left_point)
        res = self.draw(res,   forehead   , face_forehead_point)
        res = self.draw(res,   chin       , face_chin_point)
        res = self.draw(res,   nose       , face_nose_point)


        # print("res.shape : ", res.shape)
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
    acens = Acnes()
    # print(filelist)
    for filename in filelist[:]:
        if filename.split(".")[-1] == "jpg":
     #       print(path+filename)
            image = cv2.imread(path + filename)

            res = acens.run(faceMesh, image)
            merged = np.hstack((image, res))
            # merged = np.hstack((merged, res2))
            # cv2.imshow("result", merged)
            # cv2.waitKey(0)
            cv2.imwrite(path+filename.split(".")[0]+"_acnes.png", merged)
