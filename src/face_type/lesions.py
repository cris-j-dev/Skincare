from ctypes import resize
import os
import sys
import numpy as np
import cv2
import utils as utils
import facemesh as facemesh
import resize as resize_img
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

    def draw(self, image, lesions, points):

        rect = cv2.boundingRect(points)

        x, y , w, h = rect

        H, W, C = image.shape

        res = image.copy()
        index_list = np.array(list(np.where(lesions==255)))

        for index in zip(index_list[0], index_list[1]):
            x_point = int(index[0] + y)
            y_point = int(index[1] + x)
            res[x_point, y_point, 0] = 0
            res[x_point, y_point, 1] = 255
            res[x_point, y_point, 2] = 255

        return res


    def lesions2(self, image, points):

        # 1 Crop and get average, min, max value
        # 2 Color Balancing
        # 3 Normalization of a*
        # res2 = utils.color_balancing(image, points)
        res3 = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        min_a, max_a, avg_a = utils.get_mean_from_masked_image( res3[:, :, 1], points)
        _, max_L, avg_L = utils.get_mean_from_masked_image( res3[:, :, 0], points)

        if max_a > 140 and (max_a - min_a) >= 10 and (max_a - avg_a) >= 7:
            kernel = np.ones((3, 3), np.uint8)
            res3 = utils.crop_image(res3, points)

            alpha = res3[:, :, 1]
            mask=cv2.inRange(alpha,0,8)
            alpha[mask==255]=min_a
            # alpha[mask==255]=avg_a

            alpha = cv2.GaussianBlur(alpha, (5,5), 0)
            alpha = cv2.normalize(alpha, None, 0, 255, cv2.NORM_MINMAX)
            # res5, ma = utils.estimation_of_AC(alpha, 255)
            mA = alpha / 255 
            _, res5 = cv2.threshold(mA, 0.6, 255, cv2.THRESH_BINARY)
            # res5 = cv2.blur(res5, (9,9))

            res5 = cv2.dilate(res5, kernel, iterations=2)  #// make dilation image
            res5 = cv2.erode(res5, kernel, iterations=2)

            # res6 = utils.morphology(res5)
            res6 = cv2.GaussianBlur(res5, (5,5), 0)

            ret, res = cv2.threshold(res6, 100, 255, cv2.THRESH_BINARY)

            res = cv2.morphologyEx(res, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9)))
            res = cv2.dilate(res, kernel, iterations=2)  #// make dilation image
            res = cv2.erode(res, kernel, iterations=2)

            return res
        return None

    
    def draw2(self, image, re_image, acnes, points, re_points):

        if acnes is None:
            return image

        rect = cv2.boundingRect(points)
        re_rect = cv2.boundingRect(re_points)
        x, y, w, h = rect
        _x, _y, _w, _h = re_rect
        H, W, C = image.shape
        _H, _W, _C = re_image.shape
        mask = np.zeros((H, W, 3), dtype=np.uint8)

        acnes = acnes.astype(np.uint8)

        temp = image.copy()
        contours, hirerarchy = cv2.findContours(acnes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            contour = contour.astype('float32')
            contour /= [_w, _h]
            contour *= [w, h]
            contour += [x, y]
            contour = contour.astype('int32')
            area = cv2.contourArea(contour)
            # contour *= [x, y]
            if area > 180 and area < 800:
                # print(area)
                cv2.drawContours(temp, [contour], -1, (0,255,255), -1)
            # cv2.drawContours(temp, [contour], -1, (0,255,255), -1)

        return temp


    def crop(self, image, points):
        res = utils.crop_image(image, points, "white")
        return 255 - res

    def run(self, fm, image):

        re_image, _, _ = resize_img.run(fm, image, 1000)

        H, W, C = image.shape
        h, w, c = re_image.shape
        fm.set_points_loc(w=W, h=H)

        res = image.copy()
        face_cheek_right_point = np.array(fm.points_loc["face_cheek_right_point"], dtype=np.int)
        face_cheek_left_point  = np.array(fm.points_loc["face_cheek_left_point"], dtype=np.int)
        face_forehead_point    = np.array(fm.points_loc["face_forehead_point"], dtype=np.int)
        face_chin_point        = np.array(fm.points_loc["face_chin_point"], dtype=np.int)
        # face_nose_point        = np.array(fm.points_loc["face_nose_point"], dtype=np.int)

        fm.set_points_loc(w=w, h=h)
        re_face_cheek_right_point = np.array(fm.points_loc["face_cheek_right_point"], dtype=np.int)
        re_face_cheek_left_point  = np.array(fm.points_loc["face_cheek_left_point"], dtype=np.int)
        re_face_forehead_point    = np.array(fm.points_loc["face_forehead_point"], dtype=np.int)
        re_face_chin_point        = np.array(fm.points_loc["face_chin_point"], dtype=np.int)
        # re_face_nose_point        = np.array(fm.points_loc["face_nose_point"], dtype=np.int)

        lesions_img = self.lesions(image)
        # cheek_right = self.lesions(re_image, re_face_cheek_right_point)
        # cheek_left  = self.lesions(re_image, re_face_cheek_left_point)
        # chin        = self.lesions(re_image, re_face_chin_point)
        forehead    = self.lesions2(re_image, re_face_forehead_point)
        # nose        = self.lesions3(re_image, re_face_nose_point)

        cheek_right = self.crop(lesions_img, face_cheek_right_point)
        cheek_left  = self.crop(lesions_img, face_cheek_left_point)
        chin        = self.crop(lesions_img, face_chin_point)
        # nose        = self.crop(lesions_img, face_nose_point)
        # forehead    = self.crop(lesions_img, face_forehead_point)

        # lesions_img = self.lesions3(re_image)

        # cheek_right = self.crop(lesions_img, re_face_cheek_right_point)
        # cheek_left  = self.crop(lesions_img, re_face_cheek_left_point)
        # forehead    = self.crop(lesions_img, re_face_forehead_point)
        # chin        = self.crop(lesions_img, re_face_chin_point)
        # nose        = self.crop(lesions_img, re_face_nose_point)

        res = self.draw(image,   cheek_right, face_cheek_right_point)
        res = self.draw(res,   cheek_left , face_cheek_left_point)
        res = self.draw(res,   chin       , face_chin_point)
        res = self.draw2(res,   re_image, forehead   , face_forehead_point, re_face_forehead_point)
        # res = self.draw(res,   re_image, nose       , face_nose_point, re_face_nose_point)

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
    # filelist = ['input_46e4f730c0.jpg', 'input_0a51f37a14.jpg', 'input_0a2cff1ee6.jpg', 'input_0b8f0af0dc.jpg']
    for filename in filelist[:]:
        if filename.split(".")[-1] == "jpg":
            print(path+filename)
            image = cv2.imread(path + filename)
            res = lesions.run(faceMesh, image)
            merged = np.hstack((image, res))
            cv2.imwrite(path+filename.split(".")[0]+"_lesions3.png", merged)
            # cv2.imshow("res", merged)
            # cv2.waitKey(0)
