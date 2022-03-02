import io
import base64
import json

import numpy as np
import cv2


def points_to_lines(points):
    res = []
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


def crop_image(image, points, bg_value="black"):
    """
    points = [ [x1, y1], [x2, y2], ... , [xn, yn] ]
    """
    ## (1) Crop the bounding rect

    points = np.array(points, dtype=np.int)

    rect = cv2.boundingRect(points)
    x, y, w, h = rect
    croped = image[y : y + h, x : x + w].copy()

    ## (2) make mask
    points = points - points.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [points], -1, (255, 255, 255), -1)

    ## (3) do bit-op
    dst = cv2.bitwise_and(croped, croped, mask=mask)

    ## (4) add the white background
    if bg_value == "black":
        bg = np.zeros_like(croped, np.uint8) # black
    elif bg_value == "white":
        bg = np.ones_like(croped, np.uint8) * 255  # withe
    else:
        bg = np.ones_like(croped, np.uint8) * bg_value 

    cv2.bitwise_not(bg, bg, mask=mask)
    dst2 = bg + dst

    return dst2


def color_balancing(image, points):
    ret = image.copy()

    _, _, mean_r = get_mean_from_masked_image(image[:, :, 2], points)
    _, _, mean_g = get_mean_from_masked_image(image[:, :, 1], points)
    _, _, mean_b = get_mean_from_masked_image(image[:, :, 0], points)

    mVr = 1 / mean_r
    mVg = 1 / mean_g
    mVb = 1 / mean_b

    M = max(mVr, mVg, mVb)
    sr = mVr / M
    sg = mVg / M
    sb = mVb / M

    ret[:, :, 2] = image[:, :, 2] * sr
    ret[:, :, 1] = image[:, :, 1] * sg
    ret[:, :, 0] = image[:, :, 0] * sb

    return ret


def estimation_of_AC(alpha, max_value):
    mA = alpha / max_value

    # ret, binary_image = cv2.threshold(mA, 0.18, 255, cv2.THRESH_BINARY_INV)
    # ret, binary_image = cv2.threshold(mA, 0.17, 255, cv2.THRESH_BINARY)
    ret, binary_image = cv2.threshold(mA, 0.8, 255, cv2.THRESH_BINARY)
    return binary_image, mA


def get_mean_from_masked_image(image, points):

    points = np.array(points, dtype=np.int)

    rect = cv2.boundingRect(points)
    x, y, w, h = rect
    croped = image[y : y + h, x : x + w].copy()

    ## (2) make mask
    points = points - points.min(axis=0)

    mask = np.zeros(croped.shape, np.uint8)
    # print("=====================")
    # print(image.shape)
    # print(points.shape)
    # print(croped.shape)
    # print(mask.shape)

    cv2.drawContours(mask, [points], -1, 255, -1)

    # temp = croped[:, :, 1]
    min_value, max_value, min_loc, max_loc = cv2.minMaxLoc(croped, mask=mask)
    mean_value = cv2.mean(croped, mask=mask)

    return min_value, max_value, mean_value[0]


def morphology(image, k_size=3):

    # Erosion
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    erosion = cv2.erode(image, k)

    # Dilation
    kernel = np.ones((k_size, k_size), np.uint8)
    result = cv2.dilate(erosion, kernel, iterations=1)

    return result
