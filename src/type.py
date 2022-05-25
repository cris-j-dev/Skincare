import src.face_type.acnes as acnes
import src.face_type.flushings as flushings
import src.face_type.lesions as lesions
import src.face_type.pores as pores
import src.face_type.wrinkles as wrinkles
import src.face_type.utils as utils
import src.face_type.facemesh as facemesh
import src.face_type.resize as resize_img

import cv2

class FaceType:
    def __init__(self):
        self.facemesh = facemesh.FaceMesh(thickness=5)
        self.facemesh.set_label([
            "face_flushing_right_point",
            "face_flushing_left_point",
            "face_cheek_right_point",
            "face_cheek_left_point",
            "face_forehead_point",
            "face_chin_point",
            "face_nose_point",
            "face_smile_line_right_point",
            "face_smile_line_left_point",
            ])

        self.acnes = acnes.Acnes()
        self.flushings = flushings.Flushings()
        self.lesions = lesions.Lesions()
        self.pores = pores.Pores()
        self.wrinkles = wrinkles.Wrinkles()

        return

    def set_point(self, image):
        multi_face_landmarks = self.facemesh.detect_face_point(image)
        h, w, c = image.shape
        self.facemesh.set_points_loc(w=w, h=h)
        self.facemesh.set_lines()

    def run(self, image):

        re_image, w, h = resize_img.run(self.facemesh, image)

        img_flushings = cv2.resize(self.flushings.run(self.facemesh, image, re_image), (w, h), cv2.INTER_LINEAR)
        img_acnes     = cv2.resize(self.acnes.run(self.facemesh, image, re_image), (w, h), cv2.INTER_LINEAR)
        img_lesions   = cv2.resize(self.lesions.run(self.facemesh, image, re_image), (w, h), cv2.INTER_LINEAR)
        img_wrinkles  = cv2.resize(self.wrinkles.run(self.facemesh, image, re_image), (w, h), cv2.INTER_LINEAR)
        img_pores     = cv2.resize(self.pores.run(self.facemesh, image, re_image), (w, h), cv2.INTER_LINEAR)

        # return img_acnes, temp
        return img_flushings, img_acnes, img_lesions, img_wrinkles, img_pores
