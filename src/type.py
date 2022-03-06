import src.face_type.acnes as acnes
import src.face_type.flushings as flushings
import src.face_type.lesions as lesions
import src.face_type.pores as pores
import src.face_type.wrinkles as wrinkles
import src.face_type.utils as utils
import src.face_type.facemesh as facemesh


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

        img_flushings = self.flushings.run(self.facemesh, image)
        img_acnes, temp = self.acnes.run(self.facemesh, image)
        img_lesions = self.lesions.run(self.facemesh, image)
        img_wrinkles = self.wrinkles.run(self.facemesh, image)
        img_pores = self.pores.run(self.facemesh, image)

        # return img_acnes, temp
        return img_flushings, img_acnes, img_lesions, img_wrinkles, img_pores
