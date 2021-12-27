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
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    return image


def image_to_base64(image):
    retval, temp = cv2.imencode(".jpg", image)
    data = base64.b64encode(temp)
    return data


def crop_image(image, points):
    """
    points = [ [x1, y1], [x2, y2], ... , [xn, yn] ]
    """
    ## (1) Crop the bounding rect

    points = np.array(points)
    points = points.astype(np.int)

    rect = cv2.boundingRect(points)
    x, y, w, h = rect
    croped = image[y : y + h, x : x + w].copy()

    ## (2) make mask
    points = points - points.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)

    ## (3) do bit-op
    dst = cv2.bitwise_and(croped, croped, mask=mask)

    ## (4) add the white background
    bg = np.ones_like(croped, np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=mask)
    dst2 = bg + dst

    return dst2
