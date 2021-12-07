import io
import cv2
import numpy as np
import base64
import mediapipe as mp
import json


def points_to_lines(points):
    res = []
    print(len(points))
    for i in range(len(points) - 1):
        res.append((points[i], points[i + 1]))
    res.append((points[-1], points[0]))
    return res


def get_point(label):
    with open("point.json", "r") as st_json:
        st_python = json.load(st_json)
        points = st_python[label]

    return points


class FaceMesh:
    def __init__(self, thickness=1, circle_radius=1):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        point_left = get_point("face_cheek_left_point")
        point_right = get_point("face_cheek_right_point")

        lines_left = points_to_lines(point_left)
        lines_right = points_to_lines(point_right)

        self.point = point_left + point_right
        self.lines = frozenset(lines_left + lines_right)

        return

    def load_image(self, file_name):
        image = cv2.imread(file_name)
        return image

    def save_image(self, image, file_name, ext="png"):
        cv2.imwrite(file_name + "." + ext, image)
        return

    def base64_to_image(self, data):
        data = data.encode()
        image_data = base64.decodestring(data)

        image_stream = io.BytesIO()
        image_stream.write(image_data)
        image_stream.seek(0)

        file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        return img

    def image_to_base64(self, image):
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
            # image = self.load_image("1.jpg")
            image = self.base64_to_image(data)

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

                self.save_image(annotated_image, "result_mesh")
                self.save_image(image, "result_point")
                base64_data = self.image_to_base64(annotated_image)

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

                    # for i in range(0, 468):
                    for i in self.point:
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

                self.save_image(annotated_image, "result_mesh")
                self.save_image(image, "result_point")

                return annotated_image
                # return image

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
