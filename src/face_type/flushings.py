import os
import numpy as np
import cv2
import math
import src.face_type.utils as utils
import src.face_type.facemesh as facemesh

class Flushings:
    def __init__(self):
        return

    def flushings(self, fm, image, label):

        points = np.array(fm.points_loc[label])
        labImage = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # min_l, max_l, mean_l = utils.get_mean_from_masked_image(labImage[:,:,0], fm.points_loc[label])
        min_a, max_a, mean_a = utils.get_mean_from_masked_image(
            labImage[:, :, 1], fm.points_loc[label]
        )
        # min_b, max_b, mean_b = utils.get_mean_from_masked_image(labImage[:,:,2], fm.points_loc[label])

        print(mean_a)

        return mean_a


    def draw(self, image, points, deg):
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
        lx = round(c / 2)
        ly = round(c / 2 * 0.7)
        # import pdb;pdb.set_trace()

        alpha = 0.5

        # cv2.imshow("image", image)
        # cv2.imshow("mask", mask)
        cv2.ellipse(mask, (cx, cy), (lx, ly), deg, 0, 360, (0, 0, 255), -1)
        blended2 = cv2.addWeighted(image, 1, mask, (1 - alpha), 0)  # 방식2
        # cv2.circle(blended2, (cx, cy), 10, (0,0,255), -1)

        return blended2


    def run(self, fm, image):

        multi_face_landmarks = fm.detect_face_point(image)
        h, w, c = image.shape
        fm.set_points_loc(w=w, h=h)
        fm.set_lines()

        res = image.copy()
        mean_right = self.flushings(fm, image, "face_flushing_right_point")
        mean_left = self.flushings(fm, image, "face_flushing_left_point")

        if (mean_right + mean_left) / 2 > 142:
            print("flushing")
            points = fm.points_loc["face_flushing_right_point"]
            res = self.draw(image, points, -35)

            points = fm.points_loc["face_flushing_left_point"]
            res = self.draw(res, points, 35)

        return res


if __name__ == "__main__":

    faceMesh = facemesh.FaceMesh(thickness=5)
    faceMesh.set_label(
        [
            "face_flushing_right_point",
            "face_flushing_left_point",
            "face_flushing_right_point2",
            "face_flushing_left_point2",
            "face_cheek_right_point",
            "face_cheek_left_point",
            "face_forehead_point",
            "face_chin_point",
            "face_nose_point",
            # "face_smile_line_right_point",
            # "face_smile_line_left_point",
        ]
    )

    path = "../Test/flushing/"
    filelist = os.listdir(path)

    filelist = [
        "/Users/chan/Projects/Skincare/Test/flushing/drg_0000235380.jpg",
        "/Users/chan/Projects/Skincare/Test/flushing/drg_0000237290.jpg",
        "/Users/chan/Projects/Skincare/Test/flushing/drg_0000237508.jpg",
        "/Users/chan/Projects/Skincare/Test/wrinkle/drg_0000237368.jpg",
        "/Users/chan/Projects/Skincare/Test/acne/drg_0000235845.jpg",
        "/Users/chan/Projects/Skincare/Test/acne/drg_0000237524.jpg",
    ]

    for filename in filelist[:]:
        # filename = "/Users/chan/Projects/Skincare/Test/flushing/drg_0000237452.jpg"
        if filename.split(".")[1] == "jpg":
            print(path + filename)
            image = cv2.imread(filename)

            res = run(faceMesh, image)
            cv2.imshow("result", res)
            cv2.waitKey(0)

            # multi_face_landmarks = faceMesh.detect_face_point(image)

            # h, w, c = image.shape
            # faceMesh.set_points_loc(w=w, h=h)
            # faceMesh.set_lines()

            # mask = np.zeros((h, w, 3), dtype=np.uint8)
            # # mask[:, :, 2] = 255

            # res = faceMesh.draw(image)
            # # flushing(faceMesh, image, "face_flushing_right_point2")
            # # flushing(faceMesh, image, "face_flushing_left_point2")
            # flushing(faceMesh, image, "face_flushing_right_point")
            # flushing(faceMesh, image, "face_flushing_left_point")
            # # flushing(faceMesh, image, "face_cheek_right_point")
            # # flushing(faceMesh, image, "face_cheek_left_point")
            # # flushing(faceMesh, image, "face_forehead_point")
            # # flushing(faceMesh, image, "face_chin_point")
            # # flushing(faceMesh, image, "face_nose_point")

            # points = faceMesh.points_loc["face_flushing_right_point"]
            # points = np.array(points, dtype=np.int)

            # # cv2.drawContours(mask, [points], -1, (0, 0, 255), -1)

            # # center
            # M = cv2.moments(points)

            # cx = int(M['m10']/M['m00'])
            # cy = int(M['m01']/M['m00'])

            # point = points[0] - points[5]
            # c = math.sqrt((point[0]**2)+(point[1]**2))
            # lx = round(c/2)
            # ly = round(c/2*0.7)
            # # import pdb;pdb.set_trace()

            # alpha = 0.5

            # cv2.imshow("image", image)
            # cv2.imshow("mask", mask)
            # cv2.ellipse(mask, (cx, cy), (lx, ly), -45, 0, 360, (0,0,255), -1)
            # blended2 = cv2.addWeighted(image, 1, mask, (1-alpha), 0) # 방식2
            # # cv2.circle(blended2, (cx, cy), 10, (0,0,255), -1)

            # cv2.imshow("res", blended2)

            # cv2.imwrite("./"+filename.split("/")[-1], res)
            # # cv2.waitKey(0)
