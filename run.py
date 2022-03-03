import cv2
import src.type as face_type

face_type = face_type.FaceType()

image = cv2.imread("test.jpg")
res = face_type.run(image)

cv2.imshow("res", res)
cv2.waitKey(0)
