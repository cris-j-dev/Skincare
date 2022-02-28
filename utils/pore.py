from audioop import avg
import os
import numpy as np
import cv2
import utils
import math
import facemesh


def pore(image):

    # res2 = utils.color_balancing(image, points)
    res3 = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    # _, max_a, _ = utils.get_mean_from_masked_image(
    #     res3[:, :, 1], fm.points_loc[label]
    # )
    # res4 = fm.crop(res3, label)
    l_space = res3[:, :, 0]
    cv2.imshow("l_space", l_space)

    median_img = cv2.medianBlur(l_space, 7)
    cv2.imshow("median", median_img)

    norm_img = cv2.normalize(median_img, None, 0, 255, cv2.NORM_MINMAX)

    k = 15
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    blackhat_img = cv2.morphologyEx(norm_img, cv2.MORPH_BLACKHAT, kernel)
    cv2.imshow("black", blackhat_img)
    ret, res = cv2.threshold(blackhat_img, 10, 255, cv2.THRESH_BINARY)

    return res


def draw(image, pore, points):

    rect = cv2.boundingRect(points)
    x, y, w, h = rect
    # croped = image[y : y + h, x : x + w].copy()

    H, W, C = image.shape
    # faceMesh.set_points_loc(w=w, h=h)
    # faceMesh.set_lines()

    mask = np.zeros((H, W, 3), dtype=np.uint8)
    # mask[y:y+h,x:x+w,0] = pore
    mask[y:y+h,x:x+w,1] = pore
    mask[y:y+h,x:x+w,2] = pore

    # cv2.drawContours(mask, [points], -1, (0, 0, 255), -1)

    alpha = 0.1 

    # cv2.imshow("image", image)
    # cv2.imshow("mask", mask)
    # cv2.ellipse(mask, (cx, cy), (lx, ly), deg, 0, 360, (0, 0, 255), -1)
    blended2 = cv2.addWeighted(image, 1, mask, (1 - alpha), 0)  # 방식2
    # cv2.circle(blended2, (cx, cy), 10, (0,0,255), -1)

    return blended2

def crop(image, points):
    res = utils.crop_image(image, points)
    bg_img = np.zeros_like(image, np.uint8)
    bg_img = utils.crop_image(bg_img, points)
    res = res - bg_img
    return res


def run(fm, image):

    multi_face_landmarks = faceMesh.detect_face_point(image)
    h, w, c = image.shape
    faceMesh.set_points_loc(w=w, h=h)
    faceMesh.set_lines()
    mesh_img = fm.draw(image)
    cv2.imshow("mesh", mesh_img)


    res = image.copy()
    face_cheek_right_point = np.array(fm.points_loc["face_cheek_right_point"], dtype=np.int)
    face_cheek_left_point  = np.array(fm.points_loc["face_cheek_left_point"], dtype=np.int)
    face_chin_point        = np.array(fm.points_loc["face_chin_point"], dtype=np.int)
    face_nose_point        = np.array(fm.points_loc["face_nose_point"], dtype=np.int)

    pore_img = pore(image)
    # cheek_right = pore(faceMesh, image, face_cheek_right_point)
    # cheek_left = pore(faceMesh, image, face_cheek_left_point)
    # # forehead = pore(faceMesh, image, face_forehead_point)
    # chin = pore(faceMesh, image, face_chin_point)
    # nose = pore(faceMesh, image, face_nose_point)
    cheek_right = crop(pore_img, face_cheek_right_point)
    cheek_left  = crop(pore_img, face_cheek_left_point)
    # forehead  = crop(pore_img, face_forehead_point)
    chin        = crop(pore_img, face_chin_point)
    nose        = crop(pore_img, face_nose_point)

    # cheek_right = 255 - cheek_right  
    # cheek_left  = 255 - cheek_left   
    # # forehead  = 255 - forehead   
    # chin        = 255 - chin         

    res = draw(image, cheek_right, face_cheek_right_point)
    res = draw(res, cheek_left , face_cheek_left_point)
    # res = draw(res,forehead   , face_forehead_point)
    res = draw(res, chin       , face_chin_point)
    res = draw(res, nose       , face_nose_point)

    # cv2.imshow("cheek_right", cheek_right)
    # cv2.imshow("cheek_left", cheek_left)
    # cv2.imshow("forehead", forehead)
    # cv2.imshow("chin", chin)
    # cv2.imshow("nose", nose)

    # cv2.imshow("image", res)
    # cv2.waitKey(0)

    return res


if __name__ == "__main__":

    faceMesh = facemesh.FaceMesh(thickness=5)
    faceMesh.set_label(
        [
            "face_flushing_right_point",
            "face_flushing_left_point",
            # "face_flushing_right_point2",
            # "face_flushing_left_point2",
            "face_cheek_right_point",
            "face_cheek_left_point",
            "face_forehead_point",
            "face_chin_point",
            "face_nose_point",
            # "face_smile_line_right_point",
            # "face_smile_line_left_point",
        ]
    )

    path = "../Test/pore/"
    filelist = os.listdir(path)

    # filelist = [
    #     "../Test/pore/drg_0000237117.jpg"
        # "/Users/chan/Projects/Skincare/Test/flushing/drg_0000235380.jpg",
        # "/Users/chan/Projects/Skincare/Test/flushing/drg_0000237290.jpg",
        # "/Users/chan/Projects/Skincare/Test/flushing/drg_0000237508.jpg",
        # "/Users/chan/Projects/Skincare/Test/wrinkle/drg_0000237368.jpg",
        # "/Users/chan/Projects/Skincare/Test/acne/drg_0000235845.jpg",
        # "/Users/chan/Projects/Skincare/Test/acne/drg_0000237524.jpg",
        # "/Users/chan/Projects/Skincare/Test/pore/drg_0000236701.jpg"
    # ]

    print(filelist)
    for filename in filelist[:]:
        # filename = "/Users/chan/Projects/Skincare/Test/flushing/drg_0000237452.jpg"
        if filename.split(".")[-1] == "jpg":
            print(path+filename)
            image = cv2.imread(path + filename)

            res = run(faceMesh, image)
            merged = np.hstack((image, res))
            cv2.imshow("result", merged)
            cv2.imwrite(path+filename.split(".")[0]+"_pore.png", merged)
            cv2.waitKey(0)