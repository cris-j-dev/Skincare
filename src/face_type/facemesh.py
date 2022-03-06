import cv2
import numpy as np
import json
import mediapipe as mp
# import src.face_type.utils as utils
import utils as utils


class FaceMesh:
    # def __init__(self, filename="src/face_type/point.json", thickness=1, circle_radius=1):
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

