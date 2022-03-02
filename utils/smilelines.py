import numpy as np
import cv2
import utils.utils as utils


def detect_shape(img, size, size2, size3):
    res = cv2.fastNlMeansDenoising(img, None, size, size2, size3)
    res = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX)
    ret, res = cv2.threshold(res, 127, 255, cv2.THRESH_BINARY_INV)
    # res = cv2.distanceTransform(res, cv2.DIST_L2, 5)
    # res = (res/(res.max()-res.min()) * 255).astype(np.uint8)
    # res = cv2.adaptiveThreshold(res, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, -3)

    return res


def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z


def run(fm, image, label):

    # points = np.array(fm.points_loc[label])
    import pdb

    pdb.set_trace()

    # res2 = utils.color_balancing(image, fm.points_loc[label])
    res4 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # _, max_a, _ = utils.get_mean_from_masked_image(
    #     res3[:, :, 1], fm.points_loc[label]
    # )
    # min_value, max_value, mean_value = utils.get_mean_from_masked_image( res3, fm.points_loc[label])
    croped = fm.crop(res4, label)
    min_value, max_value, mean_value = utils.get_mean_from_masked_image(
        croped, fm.points_loc[label]
    )
    res4 = cv2.medianBlur(res4, 9)
    # denoised_img = cv2.fastNlMeansDenoisingMulti(res4, 2, 5, None, 15, 15, 5, 15)
    # res4 = np.where(255 == res4, int(mean_value), res4)
    # res4 = cv2.normalize(res4, None, 0, 255, cv2.NORM_MINMAX)

    # res4 = cv2.fastNlMeansDenoising(res4, None, 17, 9, 21)
    # res4 = cv2.normalize(res4, None, 0, 255, cv2.NORM_MINMAX)
    # ret, res6 = cv2.threshold(res, mean_value, 255, cv2.THRESH_BINARY_INV)

    # res6 = detect_shape(res4, 17, 9, 21)

    # mask2 = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])
    # mask3 = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]])
    mask2 = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])
    mask3 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])

    if label == "face_smile_line_right_point":
        print("right")
        res4 = cv2.filter2D(res4, -1, mask2)
    else:
        res4 = cv2.filter2D(res4, -1, mask3)

    cv2.imshow("1", res4)
    ret, res4 = cv2.threshold(res4, mean_value, 255, cv2.THRESH_BINARY)
    cv2.imshow("2", res4)
    out2 = fm.crop(res4, label)
    # out2 = cv2.medianBlur(out2, 5)

    # gaussian = cv2.GaussianBlur(res4, (5, 5), 0)
    # LoG = cv2.filter2D(gaussian, -1, mask3)

    # canny1 = cv2.Canny(res4, 50, 200)
    # canny2 = cv2.Canny(res4, 100, 200)
    # canny3 = cv2.Canny(res4, 170, 200)

    # cv2.imshow("res4", res4)
    # cv2.imshow("gaussian", gaussian)
    # cv2.imshow("LoG", LoG)
    # cv2.imshow("embossing", out2)
    # cv2.imshow("denosied", denoised_img)
    cv2.imshow("3", out2)
    # cv2.imshow("2", out3)
    # cv2.imshow("3", canny3)

    # cv2.imshow("1", d1)
    # cv2.imshow("2", d2)
    # cv2.imshow("3", d3)
    # cv2.imshow("4", d4)
    # cv2.imshow("5", d5)
    # cv2.imshow("6", d6)
    # cv2.imshow("7", d7)
    # cv2.imshow("8", d8)
    # cv2.imshow("9", res4)

    cv2.waitKey(0)
    # alpha = res4[:, :, 1]

    # alpha = cv2.normalize(alpha, None, 0, 255, cv2.NORM_MINMAX)
    # alpha = np.where(255 == alpha, 0, alpha)
    # res5, ma = utils.estimation_of_AC(alpha, max_a)
    # res6 = utils.morphology(res5)

    # contours
    # temp = res6.copy()
    # temp = temp.astype("uint8")
    # contours, _ = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # res5 = np.zeros((temp.shape[0], temp.shape[1], 3), np.uint8)

    # contours = np.array(contours)

    # for idx in range(len(contours)):
    #     contours[idx] = contours[idx] + points.min(axis=0)
    #     contours[idx] = contours[idx].astype("int")
    #     # cv2.drawContours(res5, [cnt], -1, (0,0,255), 2)
    #     # cv2.drawContours(res, [cnt2], -1, (0,0,255), 2)
    #     # cv2.imshow("result", res)
    #     # cv2.imshow("result5", res5)
    #     # cv2.waitKey(0)

    # print(f"{label} Done")
    return res4
