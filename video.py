import io
import os
import cv2
import numpy as np
import base64
import json
import argparse
import mediapipe as mp
import utils.utils as utils
import utils.acne as acne
import utils.lesion as lesion
import utils.smileline as smileline
import utils.FaceMesh as FaceMesh


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--filename",
        "-f",
        type=str,
        default="/Users/chan/Projects/Skincare/Test/wrinkle/drg_0000237290.jpg",
        help="load file name",
    )
    parser.add_argument("--label", "-l", type=str, help="crop coordinate label")
    args = parser.parse_args()

    facemesh = FaceMesh(thickness=5)
    facemesh.set_label(
        [
            "face_cheek_right_point",
            "face_cheek_left_point",
            "face_forehead_point",
            "face_chin_point",
            "face_nose_point",
            "face_smile_line_right_point",
            "face_smile_line_left_point",
        ]
    )

    filelist = os.listdir("Test/wrinkle")
    for filename in filelist[:]:
        # filename="drg_0000235295.jpg"

        print(filename)

        image = utils.load_image(os.path.join("Test/wrinkle", filename))
        # image = utils.load_image(args.filename)

        multi_face_landmarks = facemesh.detect_face_point(image)

        h, w, c = image.shape
        facemesh.set_points_loc(w=w, h=h)
        facemesh.set_lines()

        res = image.copy()

        # res = facemesh.draw(image)

        # lesion ???
        # face_cheek_right_point = lesion.run(facemesh, image, "face_cheek_right_point")
        # face_cheek_left_point  = lesion.run(facemesh, image, "face_cheek_left_point")
        # face_forehead_point    = lesion.run(facemesh, image, "face_forehead_point")
        # face_chin_point        = lesion.run(facemesh, image, "face_chin_point")
        # face_nose_point        = lesion.run(facemesh, image, "face_nose_point")

        # detect acne 여드름
        # face_cheek_right_point = acne.run(facemesh, image, "face_cheek_right_point")
        # face_cheek_left_point  = acne.run(facemesh, image, "face_cheek_left_point")
        # face_forehead_point    = acne.run(facemesh, image, "face_forehead_point")
        # face_chin_point        = acne.run(facemesh, image, "face_chin_point")
        # face_nose_point        = acne.run(facemesh, image, "face_nose_point")

        # for cnt in face_cheek_left_point:
        #     cv2.drawContours(res, [cnt], -1, (0, 0, 255), 2)
        # for cnt in face_cheek_right_point:
        #     cv2.drawContours(res, [cnt], -1, (0, 0, 255), 2)
        # for cnt in face_forehead_point:
        #     cv2.drawContours(res, [cnt], -1, (0, 0, 255), 2)
        # for cnt in face_chin_point:
        #     cv2.drawContours(res, [cnt], -1, (0, 0, 255), 2)

        face_smile_line_right_point = smileline.run(
            facemesh, image, "face_smile_line_right_point"
        )
        # face_smile_line_left_point = smileline.run(
        #     facemesh, image, "face_smile_line_left_point"
        # )

        # for cnt in face_smile_line_right_point:
        #     import pdb;pdb.set_trace()
        #     # epsilon = 0.04 * cv2.arcLength(cnt, True)
        #     # approx = cv2.approxPolyDP(cnt, epsilon, True)
        #     # print( len(approx))
        #     # cv2.drawContours(res, [approx], -1, (0, 0, 255), 2)
        #     cv2.drawContours(res, [cnt], -1, (0, 0, 255), 1)
        #     cv2.imshow("res", res)
        #     cv2.waitKey(0)

        # for cnt in face_smile_line_left_point:
        #     # epsilon = 0.04 * cv2.arcLength(cnt, True)
        #     # approx = cv2.approxPolyDP(cnt, epsilon, True)
        #     # print( len(approx))
        #     # cv2.drawContours(res, [approx], -1, (0, 0, 255), 2)
        #     cv2.drawContours(res, [cnt], -1, (0, 0, 255), 1)
        #     # cv2.imshow("res", res)
        #     # cv2.waitKey(0)

        # merged = np.hstack((image, res))

        # cv2.imshow("result", merged)
        # cv2.imwrite(f"Result/{filename}", merged)
        temp = cv2.cvtColor(face_smile_line_right_point, cv2.COLOR_GRAY2BGR)
        res = facemesh.draw(temp)
        cv2.imshow("result", res)
        cv2.waitKey(0)
