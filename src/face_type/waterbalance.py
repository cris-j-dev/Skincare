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
    
    def draw_oil(self, image, points):
        points = np.array(points, dtype=np.int)

        import pdb;pdb.set_trace()
        mask = np.zeros((h, w, 3), dtype=np.uint8)

        cx = int((points[1][0] + points[4][0] )/2)
        cy = int((points[1][1] + points[4][1] )/2)
        lx = int(points[1][0] - points[0][0])
        ly = int(points[4][1] - points[1][1])

        alpha = 0.75

        cv2.ellipse(mask, (cx, cy), (lx, ly), 0, 0, 360, (255, 255, 255), -1)
        blended2 = cv2.addWeighted(image, 1, mask, (1 - alpha), 0)  # 방식2

    def draw_water(self, image, points, deg):
        points = np.array(points, dtype=np.int)
        h, w, c = image.shape
        # faceMesh.set_points_loc(w=w, h=h)
        # faceMesh.set_lines()

        mask = np.zeros((h, w, 3), dtype=np.uint8)

        # cv2.drawContours(mask, [points], -1, (0, 0, 255), -1)

        # center
        M = cv2.moments(points)

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        point = points[0] - points[5]
        c = math.sqrt((point[0] ** 2) + (point[1] ** 2))
        lx = round(c * 0.8)
        ly = round(c * 0.5)
        # import pdb;pdb.set_trace()

        alpha = 0.75

        # cv2.imshow("image", image)
        # cv2.imshow("mask", mask)
        cv2.ellipse(mask, (cx, cy), (lx, ly), deg, 0, 360, (255, 255, 255), -1)
        blended2 = cv2.addWeighted(image, 1, mask, (1 - alpha), 0)  # 방식2
        # cv2.circle(blended2, (cx, cy), 10, (0,0,255), -1)

        return blended2


    def run(self, fm, image):

        multi_face_landmarks = fm.detect_face_point(image)
        h, w, c = image.shape
        fm.set_points_loc(w=w, h=h)
        fm.set_lines()

        res = image.copy()

        # water
        points = fm.points_loc["face_flushing_right_point"]
        res_water = self.draw_water(res, points, -50)

        points = fm.points_loc["face_flushing_left_point"]
        res_water = self.draw_water(res_water, points, 50)

        # oil
        points = fm.points_loc["face_oil_point"]
        res_oil = self.draw_oil(res, points)


        return res


if __name__ == "__main__":

    faceMesh = facemesh.FaceMesh(thickness=5)
    faceMesh.set_label(
        [
            "face_flushing_right_point",
            "face_flushing_left_point",
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

    path = "Test/"
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

            res = waterbalance.run(faceMesh, image)
            # cv2.imshow("result", res)
            # cv2.waitKey(0)