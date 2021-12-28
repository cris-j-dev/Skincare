import cv2
import numpy as np
import mediapipe as mp


class FaceDetection:
    def __init__(self):

        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_image(self):
        # For static images:
        IMAGE_FILES = []
        with self.mp_face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5) as face_detection:
            for idx, file in enumerate(IMAGE_FILES):
                image = cv2.imread(file)
                # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
                results = face_detection.process(
                    cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                # Draw face detections of each face.
                if not results.detections:
                    continue
                annotated_image = image.copy()
                for detection in results.detections:
                    print('Nose tip:')
                    print(self.mp_face_detection.get_key_point(
                        detection, self.mp_face_detection.FaceKeyPoint.NOSE_TIP))
                    self.mp_drawing.draw_detection(annotated_image, detection)
                cv2.imwrite('/tmp/annotated_image' +
                            str(idx) + '.png', annotated_image)

    def detect_video(self):
        # For webcam input:
        cap = cv2.VideoCapture(0)
        with self.mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5) as face_detection:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_detection.process(image)

                # Draw the face detection annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.detections:
                    for detection in results.detections:
                        self.mp_drawing.draw_detection(image, detection)
                # Flip the image horizontally for a selfie-view display.
                cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
                if cv2.waitKey(5) & 0xFF == 27:
                    break
        cap.release()


if __name__ == "__main__":

    fd = FaceDetection()
    fd.detect_video()
