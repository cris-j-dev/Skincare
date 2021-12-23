import io
import cv2
import numpy as np
import base64
import mediapipe as mp
import utils.utils as utils


class FaceMesh:
    def __init__(self, thickness=1, circle_radius=1):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.drawing_spec = self.mp_drawing.DrawingSpec(
            thickness=thickness, circle_radius=circle_radius, color=(48, 48, 255)
        )

        point_left = utils.get_point("face_cheek_left_point")
        point_right = utils.get_point("face_cheek_right_point")
        point_forehead = utils.get_point("face_forehead_point")
        point_chin = utils.get_point("face_chin_point")

        lines_left = utils.points_to_lines(point_left)
        lines_right = utils.points_to_lines(point_right)
        lines_forehead = utils.points_to_lines(point_forehead)
        lines_chin = utils.points_to_lines(point_chin)

        self.point = point_left + point_right + point_forehead + point_chin
        self.cheek_lines = frozenset(lines_left + lines_right)
        self.forehead_lines = frozenset(lines_forehead)
        self.chin_lines = frozenset(lines_chin)

        return

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

    def run(self, image):
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
                        connections=self.cheek_lines,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.drawing_spec,
                    )

                    self.mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=self.forehead_lines,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.drawing_spec,
                    )

                    self.mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=self.chin_lines,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.drawing_spec,
                    )

                utils.save_image(annotated_image, "result_mesh")
                utils.save_image(image, "result_point")

                return annotated_image
                # return image

            else:
                return "Not find face mesh"


if __name__ == "__main__":
    # capture = cv2.VideoCapture(0)
    # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    facemesh = FaceMesh(thickness=5)
    image = utils.load_image("Test/drg_0000235164.jpg")
    frame = facemesh.run(image)
    cv2.imshow("result", frame)
    cv2.waitKey(0)

    # i = 0
    # while True:
    #     ret, frame = capture.read()
    #     frame = facemesh.run_video(frame)
    #     if ret:
    #         cv2.imshow("VideoFrame", frame)

    #         if cv2.waitKey(33) == ord("c"):
    #             cv2.imwrite(f"{i}.png", frame)
    #             i += 1
    #         elif cv2.waitKey(33) == ord("q"):
    #             capture.release()
    #             cv2.destroyAllWindows()
