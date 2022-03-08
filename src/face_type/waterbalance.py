import os
import sys
import numpy as np
import cv2
import math
import utils as utils
import facemesh as facemesh
# import src.face_type.utils as utils
# import src.face_type.facemesh as facemesh
sys.path.append('src')
sys.path.append('src/face_type')

class WaterBalance:
    def __init__(self):
        return
    
    def draw(self, image, points):
        # T zone
        # points = np.array(points, dtype=np.uint8)
        rect = cv2.boundingRect(points)
        x, y, w, h = rect

        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [points], -1, (255,255,255), -1, cv2.LINE_AA)

        alpha = 0.85

        # cv2.ellipse(mask, (cx, cy), (lx, ly), 0, 0, 360, (255, 255, 255), -1)
        res = cv2.addWeighted(image, 1, mask, (1 - alpha), 0)  # 방식2
        return res 

    def draw_oil(self, image, points):
        # T zone
        # points = np.array(points, dtype=np.uint8)
        rect = cv2.boundingRect(points)
        x, y, w, h = rect

        mask = np.zeros(image.shape, dtype=np.uint8)

        # cx = int((points[1][0] + points[4][0] )/2)
        # cy = int((points[1][1] + points[4][1] )/2)
        # lx = int(points[1][0] - points[0][0])
        # ly = int(points[4][1] - points[1][1])

        thin = int((points[4][1]-points[1][1]))

        start_x = int((points[0][0] + points[5][0]) / 2)
        start_y = int((points[0][1] + points[5][1]) / 2)
        end_x = int((points[2][0] + points[3][0]) / 2)
        end_y = int((points[2][1] + points[3][1]) / 2)

        cv2.line(mask, (points[5]), (points[3]), (255,255,255), thin, cv2.LINE_AA)
        # cv2.line(mask, (start_x, start_y), (end_x, end_y), (255,255,255), thin, cv2.LINE_AA)
        # cv2.line(image, (start_x, start_y), (end_x, end_y), (255,255,255), thin, cv2.LINE_AA)

        start_x = int((points[0][0] + points[5][0]) / 2)
        start_y = int((points[0][1] + points[5][1]) / 2)

        cv2.line(mask, points[4], points[6], (255,255,255), thin, cv2.LINE_AA)

        alpha = 0.75

        # cv2.ellipse(mask, (cx, cy), (lx, ly), 0, 0, 360, (255, 255, 255), -1)
        res = cv2.addWeighted(image, 1, mask, (1 - alpha), 0)  # 방식2
        return res 

    def draw_water(self, image, points, center, deg):

        h, w, c = image.shape
        # faceMesh.set_points_loc(w=w, h=h)
        # faceMesh.set_lines()

        mask = np.zeros((h, w, 3), dtype=np.uint8)

        # cv2.drawContours(mask, [points], -1, (0, 0, 255), -1)

        # center
        M = cv2.moments(points)

        # cx = int(M["m10"] / M["m00"])
        # cy = int(M["m01"] / M["m00"])

        point = points[0] - points[5]
        c = math.sqrt((point[0] ** 2) + (point[1] ** 2))
        lx = round(c * 0.8)
        ly = round(c * 0.55)

        alpha = 0.75

        # cv2.imshow("image", image)
        # cv2.imshow("mask", mask)
        cx = int((center[0][0]+center[1][0])/2)
        cy = int((center[0][1]+center[1][1])/2)

        cv2.ellipse(mask, (cx, cy), (lx, ly), deg, 0, 360, (255, 255, 255), -1)
        blended2 = cv2.addWeighted(image, 1, mask, (1 - alpha), 0)  # 방식2
        # cv2.circle(blended2, (cx, cy), 10, (0,0,255), -1)

        return blended2


    def run(self, fm, image):

        multi_face_landmarks = fm.detect_face_point(image)
        h, w, c = image.shape
        fm.set_points_loc(w=w, h=h)
        fm.set_lines()

        res_water = image.copy()
        res_oil = image.copy()

        # water
        # points = np.array(fm.points_loc["face_flushings_right_point2"], dtype=np.int)
        points = np.array(fm.points_loc["face_cheek_right_point"], dtype=np.int)
        res_water = self.draw(res_water, points)
        # points = np.array(fm.points_loc["face_flushings_left_point2"], dtype=np.int)
        points = np.array(fm.points_loc["face_cheek_left_point"], dtype=np.int)
        res_water = self.draw(res_water, points)
        # center = np.array(fm.points_loc["face_waterbalance_center_point"], dtype=np.int)
        # points = np.array(fm.points_loc["face_flushings_right_point"], dtype=np.int)
        # res_water = self.draw_water(res_water, points, center[0:2], -110)

        # points = np.array(fm.points_loc["face_flushings_left_point"], dtype=np.int)
        # res_water = self.draw_water(res_water, points, center[2:4], 110)


        # oil
        # points = np.array(fm.points_loc["face_oil_point"], dtype=np.int)
        points = np.array(fm.points_loc["face_forehead_point"], dtype=np.int)
        res_oil = self.draw(res_oil, points)
        points = np.array(fm.points_loc["face_nose_point"], dtype=np.int)
        res_oil = self.draw(res_oil, points)
        # points = np.array(fm.points_loc["face_forehead_point"], dtype=np.int)
        # res_oil = self.draw_oil(res_oil, points)
        # points = np.array(fm.points_loc["face_nose_point"], dtype=np.int)
        # res_oil = self.draw_oil(res_oil, points)


        return res_water, res_oil


if __name__ == "__main__":

    faceMesh = facemesh.FaceMesh(thickness=5)
    faceMesh.set_label(
        [
            "face_waterbalance_center_point",
            "face_flushings_right_point",
            "face_flushings_left_point",
            "face_oil_point"
            # "face_flushing_right_point2",
            # "face_flushing_left_point2",
            # "face_cheek_right_point",
            # "face_cheek_left_point",
            # "face_forehead_point",
            # "face_chin_point",
            # "face_nose_point",
            # "face_smile_line_right_point",
            # "face_smile_line_left_point",
        ]
    )

    waterbalance = WaterBalance()

    path = "data/"
    filelist = os.listdir(path)

    # filelist = [
    #     "/Users/chan/Projects/Skincare/Test/flushing/drg_0000235380.jpg",
    #     "/Users/chan/Projects/Skincare/Test/flushing/drg_0000237290.jpg",
    #     "/Users/chan/Projects/Skincare/Test/flushing/drg_0000237508.jpg",
    #     "/Users/chan/Projects/Skincare/Test/wrinkle/drg_0000237368.jpg",
    #     "/Users/chan/Projects/Skincare/Test/acne/drg_0000235845.jpg",
    #     "/Users/chan/Projects/Skincare/Test/acne/drg_0000237524.jpg",
    # ]

    for filename in filelist[:]:
        # filename = "/Users/chan/Projects/Skincare/Test/flushing/drg_0000237452.jpg"
        if filename.split(".")[1] == "jpg":
            print(path + filename)
            image = cv2.imread(path+filename)

            water, oil = waterbalance.run(faceMesh, image)
            cv2.imshow("water", water)
            cv2.imshow("oil", oil)
            cv2.imwrite("oil_85.png", oil)
            cv2.imwrite("water_85.png", water)
            cv2.waitKey(0)
            exit()