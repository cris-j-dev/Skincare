import io
import cv2
import numpy as np
import base64
import mediapipe as mp


class FaceMesh:
    def __init__(self, thickness=1, circle_radius=1):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.point = [
            127,
            234,
            93,
            132,
            58,
            136,
            150,
            176,
            152,
            400,
            379,
            365,
            288,
            361,
            323,
            454,
            356,
            70,
            63,
            105,
            66,
            107,
            336,
            296,
            334,
            293,
            300,
            8,
            6,
            5,
            4,
            59,
            60,
            2,
            328,
            305,
            33,
            160,
            158,
            133,
            153,
            144,
            362,
            385,
            387,
            263,
            373,
            380,
            61,
            39,
            37,
            0,
            267,
            269,
            291,
            321,
            314,
            17,
            84,
            91,
            78,
            81,
            13,
            311,
            308,
            402,
            14,
            178,
        ]

        self.lines = frozenset(
            [
                (127, 70),
                (127, 33),
                (127, 234),
                (234, 33),
                (234, 144),
                (234, 93),
                (93, 144),
                (93, 59),
                (93, 132),
                (132, 59),
                (132, 61),
                (132, 58),
                (58, 61),
                (58, 136),
                (136, 61),
                (136, 150),
                (150, 61),
                (150, 91),
                (150, 176),
                (176, 91),
                (176, 84),
                (176, 152),
                (152, 84),
                (152, 17),
                (152, 314),
                (152, 400),
                (400, 314),
                (400, 321),
                (400, 379),
                (379, 321),
                (379, 291),
                (379, 365),
                (365, 291),
                (365, 288),
                (288, 291),
                (288, 361),
                (361, 291),
                (361, 305),
                (361, 323),
                (323, 305),
                (323, 373),
                (323, 454),
                (454, 373),
                (454, 263),
                (454, 356),
                (356, 263),
                (356, 300),
                (70, 33),
                (70, 63),
                (63, 33),
                (63, 160),
                (63, 105),
                (105, 160),
                (105, 158),
                (105, 66),
                (66, 158),
                (66, 133),
                (66, 107),
                (66, 296),
                (107, 133),
                (107, 8),
                (107, 336),
                (107, 296),
                (336, 8),
                (336, 362),
                (336, 296),
                (296, 362),
                (296, 385),
                (296, 334),
                (334, 385),
                (334, 387),
                (334, 293),
                (293, 387),
                (293, 263),
                (293, 300),
                (300, 263),
                (8, 133),
                (8, 6),
                (8, 362),
                (6, 133),
                (6, 362),
                (6, 5),
                (5, 133),
                (5, 362),
                (5, 153),
                (5, 380),
                (5, 59),
                (5, 305),
                (5, 4),
                (4, 59),
                (4, 60),
                (4, 2),
                (4, 328),
                (4, 305),
                (59, 153),
                (59, 144),
                (59, 61),
                (59, 39),
                (59, 37),
                (59, 60),
                (60, 37),
                (60, 0),
                (60, 2),
                (2, 0),
                (2, 328),
                (328, 0),
                (328, 267),
                (328, 305),
                (305, 267),
                (305, 269),
                (305, 291),
                (305, 373),
                (305, 380),
                (33, 144),
                (33, 160),
                (160, 144),
                (160, 153),
                (160, 158),
                (158, 153),
                (158, 133),
                (133, 153),
                (153, 144),
                (362, 380),
                (362, 385),
                (385, 380),
                (385, 387),
                (387, 380),
                (387, 373),
                (387, 263),
                (263, 373),
                (373, 380),
                (61, 91),
                (61, 78),
                (61, 39),
                (39, 78),
                (39, 81),
                (39, 37),
                (37, 81),
                (37, 13),
                (37, 0),
                (0, 13),
                (0, 267),
                (267, 13),
                (267, 311),
                (267, 269),
                (269, 311),
                (269, 308),
                (269, 291),
                (291, 308),
                (291, 321),
                (321, 308),
                (321, 402),
                (321, 314),
                (314, 402),
                (314, 14),
                (314, 17),
                (17, 14),
                (17, 84),
                (84, 14),
                (84, 178),
                (84, 91),
                (91, 178),
                (91, 78),
                (78, 178),
                (78, 81),
                (81, 178),
                (81, 14),
                (81, 13),
                (13, 14),
                (13, 311),
                (311, 14),
                (311, 402),
                (311, 308),
                (308, 402),
                (402, 14),
                (14, 178),
            ]
        )

        return

    def loadImage(self, file_name):
        image = cv2.imread(file_name)
        return image

    def saveImage(self, image, file_name, ext="png"):
        cv2.imwrite(file_name + "." + ext, image)
        return

    def base64ToImage(self, data):
        data = data.encode()
        image_data = base64.decodestring(data)

        image_stream = io.BytesIO()
        image_stream.write(image_data)
        image_stream.seek(0)

        file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        return img

    def imageToBase64(self, image):
        retval, temp = cv2.imencode(".jpg", image)
        data = base64.b64encode(temp)
        return data

    def run(self, data):
        # For static images:
        with self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as face_mesh:

            # Test local image
            # image = self.loadImage("1.jpg")
            image = self.base64ToImage(data)

            # Convert the BGR image to RGB before processing.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            height, width, _ = image.shape
            # Print and draw face mesh landmarks on the image.
            if results.multi_face_landmarks:
                annotated_image = image.copy()

                for face_landmarks in results.multi_face_landmarks:

                    # for i in range(300, 330):
                    for i in self.point:
                        pt1 = face_landmarks.landmark[i]
                        x = int(pt1.x * width)
                        y = int(pt1.y * height)
                        cv2.circle(image, (x, y), 2, (255, 0, 0), -1)
                        cv2.putText(image, str(i), (x, y), 0, 1, (0, 0, 255))

                    self.mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=self.lines,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                    )

                self.saveImage(annotated_image, "result_mesh")
                self.saveImage(image, "result_point")
                base64_data = self.imageToBase64(annotated_image)

                return base64_data

            else:
                return "Not find face mesh"

    def run_video(self, image):
        # For static images:
        with self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as face_mesh:

            # Convert the BGR image to RGB before processing.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            height, width, _ = image.shape
            # Print and draw face mesh landmarks on the image.
            if results.multi_face_landmarks:
                annotated_image = image.copy()

                for face_landmarks in results.multi_face_landmarks:

                    for i in range(0, 468):
                        # for i in self.point:
                        pt1 = face_landmarks.landmark[i]
                        x = int(pt1.x * width)
                        y = int(pt1.y * height)
                        cv2.circle(image, (x, y), 2, (255, 0, 0), -1)
                        cv2.putText(image, str(i), (x, y), 0, 0.3, (0, 0, 255))

                    self.mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=self.lines,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                    )

                self.saveImage(annotated_image, "result_mesh")
                self.saveImage(image, "result_point")

                return image

            else:
                return "Not find face mesh"


if __name__ == "__main__":
    capture = cv2.VideoCapture(0)
    # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    facemesh = FaceMesh()

    i = 0
    while True:
        ret, frame = capture.read()
        frame = facemesh.run_video(frame)
        if ret:
            cv2.imshow("VideoFrame", frame)

            if cv2.waitKey(33) == ord("c"):
                cv2.imwrite(f"{i}.png", frame)
                i += 1
            elif cv2.waitKey(33) == ord("q"):
                capture.release()
                cv2.destroyAllWindows()
