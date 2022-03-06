import os
import numpy as np
import cv2
import utils as utils
import facemesh as facemesh

class Acnes:
    def __init__(self):
        return

    def homomorphic_filter(self, image):
        img_YUV = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)    
        y = img_YUV[:,:,0]    
        
        rows = y.shape[0]    
        cols = y.shape[1]
        
        ### illumination elements와 reflectance elements를 분리하기 위해 log를 취함
        imgLog = np.log1p(np.array(y, dtype='float') / 255) # y값을 0~1사이로 조정한 뒤 log(x+1)
        
        ### frequency를 이미지로 나타내면 4분면에 대칭적으로 나타나므로 
        ### 4분면 중 하나에 이미지를 대응시키기 위해 row와 column을 2배씩 늘려줌
        M = 2*rows + 1
        N = 2*cols + 1
        
        ### gaussian mask 생성 sigma = 10
        sigma = 10
        (X, Y) = np.meshgrid(np.linspace(0, N-1, N), np.linspace(0, M-1, M)) # 0~N-1(and M-1) 까지 1단위로 space를 만듬
        Xc = np.ceil(N/2) # 올림 연산
        Yc = np.ceil(M/2)
        gaussianNumerator = (X - Xc)**2 + (Y - Yc)**2 # 가우시안 분자 생성
        
        ### low pass filter와 high pass filter 생성
        LPF = np.exp(-gaussianNumerator / (2*sigma*sigma))
        HPF = 1 - LPF
        
        ### LPF랑 HPF를 0이 가운데로 오도록iFFT함. 
        ### 사실 이 부분이 잘 이해가 안 가는데 plt로 이미지를 띄워보니 shuffling을 수행한 효과가 났음
        ### 에너지를 각 귀퉁이로 모아 줌
        LPF_shift = np.fft.ifftshift(LPF.copy())
        HPF_shift = np.fft.ifftshift(HPF.copy())
        
        ### Log를 씌운 이미지를 FFT해서 LPF와 HPF를 곱해 LF성분과 HF성분을 나눔
        img_FFT = np.fft.fft2(imgLog.copy(), (M, N))
        img_LF = np.real(np.fft.ifft2(img_FFT.copy() * LPF_shift, (M, N))) # low frequency 성분
        img_HF = np.real(np.fft.ifft2(img_FFT.copy() * HPF_shift, (M, N))) # high frequency 성분
        
        ### 각 LF, HF 성분에 scaling factor를 곱해주어 조명값과 반사값을 조절함
        gamma1 = 0.3
        gamma2 = 1.5
        img_adjusting = gamma1*img_LF[0:rows, 0:cols] + gamma2*img_HF[0:rows, 0:cols]
        
        ### 조정된 데이터를 이제 exp 연산을 통해 이미지로 만들어줌
        img_exp = np.expm1(img_adjusting) # exp(x) + 1
        img_exp = (img_exp - np.min(img_exp)) / (np.max(img_exp) - np.min(img_exp)) # 0~1사이로 정규화
        img_out = np.array(255*img_exp, dtype = 'uint8') # 255를 곱해서 intensity값을 만들어줌
        
        ### 마지막으로 YUV에서 Y space를 filtering된 이미지로 교체해주고 RGB space로 converting
        img_YUV[:,:,0] = img_out
        result = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2BGR)
        return result

    def acnes2(self, image, points):

        # 1. BGR to Gray
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 2. ROI Maximum Intensity
        min_a, max_a, avg_a = utils.get_mean_from_masked_image(gray, points)

        # 3. Gray-scale Normalization
        norm_gray = utils.crop_image(gray, points)
        mask=cv2.inRange(norm_gray,0,10)
        norm_gray[mask==255]=avg_a
        norm_gray = cv2.normalize(norm_gray, None, 0, 255, cv2.NORM_MINMAX)
        bg = np.ones((norm_gray.shape), dtype=np.uint8)
        norm_gray = bg - norm_gray


        # 4. BGR to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 5. Extract V
        hsv_v = hsv[:,:,2]
        hsv_v = utils.crop_image(hsv_v, points, "white")
        bg = np.ones((hsv_v.shape), dtype=np.uint8)
        hsv_v = bg - hsv_v
        # mask=cv2.inRange(hsv_v,0,10)
        # hsv_v[mask==255]=avg_a

        # 6. Image Subtraction for ROI
        sub_ROI = hsv_v - norm_gray

        cv2.imshow("gray", gray)
        cv2.imshow("norm_gray", norm_gray)
        cv2.imshow("hsv", hsv)
        cv2.imshow("hsv_v", hsv_v)
        cv2.imshow("sub_ROI", sub_ROI)
        cv2.waitKey(0)


        # 1 Crop and get average, min, max value
        # 2 Color Balancing
        # 3 Normalization of a*
        # res2 = utils.color_balancing(image, points)
        res3 = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

        # cv2.imshow("image", image)
        # cv2.imshow("L", res3[:,:,0])
        # cv2.imshow("a", res3[:,:,1])
        # cv2.imshow("b", res3[:,:,2])
        # cv2.waitKey(0)
        
        _, max_a, avg_a = utils.get_mean_from_masked_image( res3[:, :, 1], points)
        res4 = utils.crop_image(res3, points)
        image_crop = utils.crop_image(image, points)

        threshold_l = 55
        threshold_a = -5

        threshold_l = int(threshold_l * 255 / 100)
        threshold_a = int(threshold_a + 128) 
        print(max_a)
        print(threshold_l)
        print(threshold_a)

        if max_a > 134:

            CIE_L = res4[:,:,0]
            CIE_a = res4[:,:,1]
            CIE_b = res4[:,:,2]
            

            # res4[:,:,0] = 87 * 255 / 100
            # res4[:,:,1] =123 
            # res4[:,:,2] = 128-15

            mask=cv2.inRange(CIE_a,0,10)
            CIE_a[mask==255]=avg_a

            # cv2.imshow("L", CIE_L)
            # cv2.imshow("a", CIE_a)
            # cv2.imshow("b", CIE_b)
            # cv2.imshow("image", image_crop)
            CIE_a = cv2.normalize(CIE_a, None, 0, 255, cv2.NORM_MINMAX)
            # cv2.imshow("temp", CIE_a)

            ret, CIE_L = cv2.threshold(CIE_L, threshold_l, 255, cv2.THRESH_BINARY)
            ret, CIE_a = cv2.threshold(CIE_a, 200, 255, cv2.THRESH_BINARY)
            res = CIE_a
            # cv2.imshow("temp2", CIE_a)
            # cv2.waitKey(0)

            # res = cv2.bitwise_and(CIE_L, CIE_a)

            # res[res==0]=255
            # res[res==1]=0
            # res[res==1]=255
            # cv2.imshow("image", image)
            
            # res4 = cv2.cvtColor(res4, cv2.COLOR_Lab2BGR)
            # cv2.imshow("res4", res4)
            # cv2.imshow("L", CIE_L)
            # cv2.imshow("a", CIE_a)
            # cv2.imshow("res", res)
            # print(np.unique(res))
            # cv2.waitKey(0)
            # exit()

            # ret, res = cv2.threshold(res, 1, 255, cv2.THRESH_BINARY)




            # kernel = np.ones((3, 3), np.uint8)
            # CIE_a = cv2.dilate(CIE_a, kernel)
            # CIE_a = cv2.dilate(CIE_a, kernel)

            # CIE_a = cv2.normalize(CIE_a, None, 0, 255, cv2.NORM_MINMAX)
            # res5, ma = utils.estimation_of_AC(CIE_a, 255)
            # res6 = utils.morphology(res5)
            # ret, res = cv2.threshold(res6, 230, 255, cv2.THRESH_BINARY)

            return res, res4
        return None, res4

    def acnes(self, image, points):

        # 1 Crop and get average, min, max value
        # 2 Color Balancing
        # 3 Normalization of a*
        res2 = utils.color_balancing(image, points)
        res3 = cv2.cvtColor(res2, cv2.COLOR_BGR2Lab)
        _, max_a, avg_a = utils.get_mean_from_masked_image( res3[:, :, 1], points)
        res4 = utils.crop_image(res3, points)

        print(max_a)
        if max_a > 134:

            alpha = res4[:, :, 1]
            mask=cv2.inRange(alpha,0,10)
            alpha[mask==255]=avg_a

            # kernel = np.ones((3, 3), np.uint8)
            # alpha = cv2.dilate(alpha, kernel)
            # alpha = cv2.dilate(alpha, kernel)

            alpha = cv2.normalize(alpha, None, 0, 255, cv2.NORM_MINMAX)
            res5, ma = utils.estimation_of_AC(alpha, 255)
            res6 = utils.morphology(res5)
            ret, res = cv2.threshold(res6, 230, 255, cv2.THRESH_BINARY)

            return res, res4
        return None, res4

    def draw(self, image, acnes, points):

        if acnes is None:
            return image

        rect = cv2.boundingRect(points)
        x, y, w, h = rect
        H, W, C = image.shape
        mask = np.zeros((H, W, 3), dtype=np.uint8)

        acnes = acnes.astype(np.uint8)

        # temp = np.zeros((h,w), dtype=np.uint8)
        temp = image.copy()
        contours, hirerarchy = cv2.findContours(acnes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            contour += [x, y]
            area = cv2.contourArea(contour)
            # print(area)
            cv2.drawContours(temp, [contour], -1, (0,0,255), -1)

        return temp

    def draw2(image, acnes, points):

        if acnes is None:
            return image

        rect = cv2.boundingRect(points)
        x, y, w, h = rect
        # croped = image[y : y + h, x : x + w].copy()

        H, W, C = image.shape

        mask = np.zeros((H, W, 3), dtype=np.uint8)
        # mask[y:y+h,x:x+w,0] = acnes
        # mask[y:y+h,x:x+w,1] = acnes

        mask[y:y+h,x:x+w,2] = acnes


        alpha = 0.01 

        # cv2.ellipse(mask, (cx, cy), (lx, ly), deg, 0, 360, (0, 0, 255), -1)
        blended2 = cv2.addWeighted(image, 1, mask, (1 - alpha), 0)  # 방식2
        # cv2.circle(blended2, (cx, cy), 10, (0,0,255), -1)

        return blended2

    def draw3(self, image, acnes, points):

        if acnes is None:
            return image

        rect = cv2.boundingRect(points)
        x, y, w, h = rect
        acnes = acnes.astype(np.uint8)

        image[y:y+h,x:x+w] = acnes

        return image

    def crop(self, image, points):
        res = utils.crop_image(image, points)
        bg_img = np.zeros_like(image, np.uint8)
        bg_img = utils.crop_image(bg_img, points, "white")
        res = res - bg_img
        return res


    def run(self, fm, image):

        multi_face_landmarks = fm.detect_face_point(image)
        h, w, c = image.shape
        fm.set_points_loc(w=w, h=h)
        fm.set_lines()

        image = self.homomorphic_filter(image)
        # cv2.imshow("image", image)
        # cv2.imshow("image2", homomorphic_image)
        # cv2.waitKey(0)


        # lab = image.copy()
        lab = np.zeros((h, w, c), dtype=np.uint8)
        # print("res.shape : ", res.shape)
        
        face_cheek_right_point = np.array(fm.points_loc["face_cheek_right_point"], dtype=np.int)
        face_cheek_left_point  = np.array(fm.points_loc["face_cheek_left_point"], dtype=np.int)
        face_forehead_point    = np.array(fm.points_loc["face_forehead_point"], dtype=np.int)
        face_chin_point        = np.array(fm.points_loc["face_chin_point"], dtype=np.int)
        face_nose_point        = np.array(fm.points_loc["face_nose_point"], dtype=np.int)

        cheek_right, lab_right = self.acnes(image, face_cheek_right_point)
        cheek_left , lab_left = self.acnes(image, face_cheek_left_point)
        forehead   , lab_forehead = self.acnes(image, face_forehead_point)
        chin       , lab_chin = self.acnes(image, face_chin_point)
        nose       , lab_nose = self.acnes(image, face_nose_point)

        res = self.draw(image, cheek_right, face_cheek_right_point)
        res = self.draw(res,   cheek_left , face_cheek_left_point)
        res = self.draw(res,   forehead   , face_forehead_point)
        res = self.draw(res,   chin       , face_chin_point)
        res = self.draw(res,   nose       , face_nose_point)

        res2 = self.draw3(lab, lab_right, face_cheek_right_point)
        res2 = self.draw3(res2,   lab_left , face_cheek_left_point)
        res2 = self.draw3(res2,   lab_forehead   , face_forehead_point)
        res2 = self.draw3(res2,   lab_chin       , face_chin_point)
        res2 = self.draw3(res2,   lab_nose       , face_nose_point)



        # print("res.shape : ", res.shape)
        return res, res2


if __name__ == "__main__":

    faceMesh = facemesh.FaceMesh(thickness=5)
    faceMesh.set_label(
        [
            "face_flushing_right_point",
            "face_flushing_left_point",
            "face_cheek_right_point",
            "face_cheek_left_point",
            "face_forehead_point",
            "face_chin_point",
            "face_nose_point",
        ]
    )

    path = "../../data/"
    acens = Acnes()
    filelist = os.listdir(path)
    # print(filelist)
    # filename = "drg_0000235457_acnes.png"
    # filelist = ["drg_0000235457.jpg"]
    for filename in filelist[:]:
        if filename.split(".")[-1] == "jpg" or filename.split(".")[-1] == "png":
        #    print(path+filename)
            image = cv2.imread(path + filename)
            res, res2 = acens.run(faceMesh, image)
            merged = np.hstack((image, res))
            merged = np.hstack((merged, res2))
            # cv2.imshow("result", merged)
            cv2.imwrite(filename.split(".")[0]+"_acnes.png", merged)
            # cv2.waitKey(0)
