import io
import base64
import json

import numpy as np
import cv2


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


def load_image(file_name):
    image = cv2.imread(file_name)
    return image


def save_image(image, file_name, ext="png"):
    cv2.imwrite(file_name + "." + ext, image)
    return


def base64_to_image(data):
    data = data.encode()
    image_data = base64.decodestring(data)

    image_stream = io.BytesIO()
    image_stream.write(image_data)
    image_stream.seek(0)

    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    return img


def image_to_base64(image):
    retval, temp = cv2.imencode(".jpg", image)
    data = base64.b64encode(temp)
    return data
