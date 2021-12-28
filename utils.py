import io
import cv2
import base64
import numpy as np


def loadImage(file_name):
    image = cv2.imread(file_name)
    return image


def saveImage(image, file_name, ext='png'):
    cv2.imwrite(file_name + '.' + ext, image)
    return


def base64ToImage(data):
    data = data.encode()
    image_data = base64.decodestring(data)

    image_stream = io.BytesIO()
    image_stream.write(image_data)
    image_stream.seek(0)

    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    return img


def imageToBase64(image):
    retval, temp = cv2.imencode('.jpg', image)
    data = base64.b64encode(temp)
    return data
