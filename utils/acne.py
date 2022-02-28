import io
import os
import black
import numpy as np
import cv2
import utils
import facemesh

def acne(image, points):

    # 1 Crop and get average, min, max value
    # 2 Color Balancing
    # 3 Normalization of a*
    res2 = utils.color_balancing(image, points)
    res3 = cv2.cvtColor(res2, cv2.COLOR_BGR2Lab)
    _, max_a, avg_a = utils.get_mean_from_masked_image( res3[:, :, 1], points)
    if max_a > 134:
        res3 = utils.crop_image(res3, points)

        alpha = res3[:, :, 1]
        # cv2.imshow("res2", res4)
        mask=cv2.inRange(alpha,0,10)
        alpha[mask==255]=avg_a

        alpha = cv2.normalize(alpha, None, 0, 255, cv2.NORM_MINMAX)
        res5, ma = utils.estimation_of_AC(alpha, 255)
        res6 = utils.morphology(res5)
        ret, res = cv2.threshold(res6, 230, 255, cv2.THRESH_BINARY)

        return res
    return None

    # cv2.imshow("alpha", alpha)
    # cv2.imshow("res5", res5)
    # cv2.imshow("res6", res5)
    # cv2.waitKey(0)

    # # contours
    # temp = res6.copy()
    # temp = temp.astype("uint8")
    # contours, _ = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # res5 = np.zeros((temp.shape[0], temp.shape[1], 3), np.uint8)

    # contours = np.array(contours)

    # print(contours)

    # for idx in range(len(contours)):
    #     # contours[idx] = contours[idx] + points.min(axis=0)
    #     contours[idx] = contours[idx].astype("int")
    #     cv2.drawContours(res5, [contours[idx]], -1, (0,0,255), 2)
    #     # cv2.drawContours(res, [cnt2], -1, (0,0,255), 2)
    #     # cv2.imshow("result", res)
    # cv2.imshow("result5", res5)
    # cv2.waitKey(0)

    # res2 = utils.color_balancing(image, points)
    # res3 = cv2.cvtColor(res2, cv2.COLOR_BGR2Lab)
    # min_a, max_a, avg_a = utils.get_mean_from_masked_image( res3[:, :, 1], points)
    # res3 = utils.crop_image(res3, points)

    # # bg_img = np.zeros_like(res3[:,:,0], np.uint8)

    # alpha = res3[:, :, 1]
    # mask=cv2.inRange(alpha,0,10)
    # alpha[mask==255]=avg_a
    # # res = bg_img - alpha
    # # cv2.imshow("res", res)
    # cv2.imshow("mask", mask)
    # cv2.imshow("alpha", alpha)

    # alpha = cv2.normalize(alpha, None, 0, 255, cv2.NORM_MINMAX)
    # res6 = utils.morphology(alpha)
    # res6 = cv2.normalize(res6, None, 0, 255, cv2.NORM_MINMAX)
    # clahe = cv2.createCLAHE()
    # res6 = clahe.apply(res6)

    # cv2.imshow("norphology", res6)
    # cv2.waitKey(0)
    
    # ret, res = cv2.threshold(res6, 230, 255, cv2.THRESH_BINARY)

    # cv2.imshow("res", res)
    # cv2.waitKey(0)


    return 

def draw(image, acne, points):

    if acne is None:
        return image

    rect = cv2.boundingRect(points)
    x, y, w, h = rect
    H, W, C = image.shape
    mask = np.zeros((H, W, 3), dtype=np.uint8)
    # mask[y:y+h,x:x+w,2] = pore

    acne = acne.astype(np.uint8)
    # temp = np.zeros((h,w), dtype=np.uint8)
    temp = image.copy()
    contours, hirerarchy = cv2.findContours(acne, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        contour += [x, y]
        # cv2.drawContours(temp, [contour], -1, (255), -1)
        area = cv2.contourArea(contour)
        print(area)
        cv2.drawContours(temp, [contour], -1, (0,0,255), -1)
        # cv2.imshow("test", temp)
        # cv2.waitKey(0)

    # mask[y:y+h,x:x+w,2] = temp


    return temp

def draw2(image, acne, points):

    if acne is None:
        return image

    rect = cv2.boundingRect(points)
    x, y, w, h = rect
    # croped = image[y : y + h, x : x + w].copy()

    H, W, C = image.shape
    # faceMesh.set_points_loc(w=w, h=h)
    # faceMesh.set_lines()

    mask = np.zeros((H, W, 3), dtype=np.uint8)
    # mask[y:y+h,x:x+w,0] = acne
    # mask[y:y+h,x:x+w,1] = acne
    mask[y:y+h,x:x+w,2] = acne

    # cv2.drawContours(mask, [points], -1, (0, 0, 255), -1)

    alpha = 0.01 

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
    # mesh_img = fm.draw(image)
    # cv2.imshow("mesh", mesh_img)

    res = image.copy()
    face_cheek_right_point = np.array(fm.points_loc["face_cheek_right_point"], dtype=np.int)
    face_cheek_left_point  = np.array(fm.points_loc["face_cheek_left_point"], dtype=np.int)
    face_forehead_point    = np.array(fm.points_loc["face_forehead_point"], dtype=np.int)
    face_chin_point        = np.array(fm.points_loc["face_chin_point"], dtype=np.int)
    face_nose_point        = np.array(fm.points_loc["face_nose_point"], dtype=np.int)

    cheek_right = acne(image, face_cheek_right_point)
    cheek_left  = acne(image, face_cheek_left_point)
    forehead    = acne(image, face_forehead_point)
    chin        = acne(image, face_chin_point)
    nose        = acne(image, face_nose_point)

    # cheek_right = pore(faceMesh, image, face_cheek_right_point)
    # cheek_left = pore(faceMesh, image, face_cheek_left_point)
    # # forehead = pore(faceMesh, image, face_forehead_point)
    # chin = pore(faceMesh, image, face_chin_point)
    # nose = pore(faceMesh, image, face_nose_point)

    # cheek_right = crop(pore_img, face_cheek_right_point)
    # cheek_left  = crop(pore_img, face_cheek_left_point)
    # forehead  = crop(pore_img, face_forehead_point)
    # chin        = crop(pore_img, face_chin_point)
    # nose        = crop(pore_img, face_nose_point)

    # cheek_right = 255 - cheek_right  
    # cheek_left  = 255 - cheek_left   
    # # forehead  = 255 - forehead   
    # chin        = 255 - chin         

    res = draw(image, cheek_right, face_cheek_right_point)
    res = draw(res,   cheek_left , face_cheek_left_point)
    res = draw(res,   forehead   , face_forehead_point)
    res = draw(res,   chin       , face_chin_point)
    res = draw(res,   nose       , face_nose_point)

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

    path = "../Test/acne/"
    filelist = os.listdir(path)

    # filelist = ["drg_0000235457.jpg"]
        # "../Test/pore/drg_0000237117.jpg"
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