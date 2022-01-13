import numpy as np
import cv2
import utils.utils as utils

def run(fm, image, label):

    points = np.array(fm.points_loc[label])

    # res2 = utils.color_balancing(image, fm.points_loc[label])
    res3 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # _, max_a, _ = utils.get_mean_from_masked_image(
    #     res3[:, :, 1], fm.points_loc[label]
    # )
    res4 = fm.crop(res3, label)

    canny1 = cv2.Canny(res4, 50, 200)
    canny2 = cv2.Canny(res4, 100, 200)
    canny3 = cv2.Canny(res4, 170, 200)

    cv2.imshow("res4", res4)
    cv2.imshow("1", canny1)
    cv2.imshow("2", canny2)
    cv2.imshow("3", canny3)
    cv2.waitKey(0)
    # alpha = res4[:, :, 1]

    # alpha = cv2.normalize(alpha, None, 0, 255, cv2.NORM_MINMAX)
    # alpha = np.where(255 == alpha, 0, alpha)
    # res5, ma = utils.estimation_of_AC(alpha, max_a)
    # res6 = utils.morphology(res5)

    # # contours
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
    # return contours