import cv2
import numpy as np
import json
import mediapipe as mp
import utils


class FaceMesh:
    def __init__(self, filename="point.json", thickness=1, circle_radius=1):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.drawing_spec = self.mp_drawing.DrawingSpec(
            thickness=thickness, circle_radius=circle_radius, color=(48, 48, 255)
        )

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

        with open(filename, "r") as st_json:
            st_python = json.load(st_json)
        self.points = st_python

        # point_left = utils.get_point("face_cheek_left_point")
        # point_right = utils.get_point("face_cheek_right_point")
        # point_forehead = utils.get_point("face_forehead_point")
        # point_chin = utils.get_point("face_chin_point")

        # lines_left = utils.points_to_lines(point_left)
        # lines_right = utils.points_to_lines(point_right)
        # lines_forehead = utils.points_to_lines(point_forehead)
        # lines_chin = utils.points_to_lines(point_chin)

        # self.point = point_left + point_right + point_forehead + point_chin
        # self.cheek_lines = frozenset(lines_left + lines_right)
        # self.forehead_lines = frozenset(lines_forehead)
        # self.chin_lines = frozenset(lines_chin)

        return

    def set_label(self, label):
        self.labels = label

    def set_points_loc(self, w=1, h=1):
        points_loc = {}
        for label in self.points.keys():
            points_loc[label] = []
            for idx in self.points[label]:
                points_loc[label].append(
                    [
                        self.multi_face_landmarks.landmark[idx].x * w,
                        self.multi_face_landmarks.landmark[idx].y * h,
                    ]
                )
        self.points_loc = points_loc
        return points_loc

    def set_lines(self):
        lines = {}
        for label in self.points.keys():
            lines[label] = frozenset(utils.points_to_lines(self.points[label]))

        self.lines = lines
        return lines

    # def run(self, data):
    #     # For static images:
    #     with self.mp_face_mesh.FaceMesh(
    #         static_image_mode=True,
    #         max_num_faces=1,
    #         refine_landmarks=True,
    #         min_detection_confidence=0.5,
    #     ) as face_mesh:

    #         # Test local image
    #         # image = self.load_image("1.jpg")
    #         image = utils.base64_to_image(data)

    #         # Convert the BGR image to RGB before processing.
    #         results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #         height, width, _ = image.shape
    #         # Print and draw face mesh landmarks on the image.
    #         if results.multi_face_landmarks:
    #             annotated_image = image.copy()

    #             for face_landmarks in results.multi_face_landmarks:

    #                 # for i in range(300, 330):
    #                 for i in self.point:
    #                     pt1 = face_landmarks.landmark[i]
    #                     x = int(pt1.x * width)
    #                     y = int(pt1.y * height)
    #                     cv2.circle(image, (x, y), 2, (255, 0, 0), -1)
    #                     cv2.putText(image, str(i), (x, y), 0, 1, (0, 0, 255))

    #                 self.mp_drawing.draw_landmarks(
    #                     image=annotated_image,
    #                     landmark_list=face_landmarks,
    #                     connections=self.cheek_lines,
    #                     landmark_drawing_spec=None,
    #                     connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style(),
    #                 )

    #                 self.mp_drawing.draw_landmarks(
    #                     image=annotated_image,
    #                     landmark_list=face_landmarks,
    #                     connections=self.forehead_lines,
    #                     landmark_drawing_spec=None,
    #                     connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style(),
    #                 )

    #             utils.save_image(annotated_image, "result_mesh")
    #             utils.save_image(image, "result_point")
    #             base64_data = self.image_to_base64(annotated_image)

    #             return base64_data

    #         else:
    #             return "Not find face mesh"

    def detect_face_point(self, image):
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            self.multi_face_landmarks = results.multi_face_landmarks[0]
            return results.multi_face_landmarks[0]
        else:
            return "Not find face mesh"

    def draw(self, image):
        # Convert the BGR image to RGB before processing.
        # Print and draw face mesh landmarks on the image.
        annotated_image = image.copy()

        for label in self.labels:
            self.mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=self.multi_face_landmarks,
                connections=self.lines[label],
                landmark_drawing_spec=None,
                connection_drawing_spec=self.drawing_spec,
            )

        return annotated_image

    def crop(self, image, label):
        points = self.points_loc[label]
        ret = utils.crop_image(image, points)
        return ret
