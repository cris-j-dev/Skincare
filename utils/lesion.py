import numpy as np
import cv2

def run(fm, image, label):

    points = np.array(fm.points_loc[label])

    image = fm.crop(image, label)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = np.where(255 == gray, 0, gray)
    blur = cv2.GaussianBlur(gray, (17, 17), 32)
    # ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret,thresh = cv2.threshold(blur,127,255,cv2.THRESH_BINARY_INV)

    cv2.imshow("hsv", hsv)
    cv2.imshow("gray", gray)
    cv2.imshow("blur", blur)
    cv2.imshow("thresh", thresh)
    cv2.waitKey(0)
    #thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    #thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE)

    print(cv2.contourArea)
    cnt = max(contours, key=cv2.contourArea)

    #mask = np.zeros(image.shape,np.uint8)
    #cv2.drawContours(mask,cnt,-1)
    if len(cnt) > 4:
        ellipse = cv2.fitEllipse(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        area = hsv[int(y+(0.3*h)):int(y+(0.8*h)),int((0.2*w)+x):int((0.7*w)+x)]
        ellipse_cnt = cv2.ellipse2Poly( (int(ellipse[0][0]),int(ellipse[0][1]) ) ,( int(ellipse[1][0]),int(ellipse[1][1]) ),int(ellipse[2]),0,360,1)
        comp = cv2.matchShapes(cnt,ellipse_cnt,1,0.0)
        variance = cv2.meanStdDev(area)
        print(comp)
        print(variance[1])
        # cv2.ellipse(image,ellipse,(0,255,0),2)

    cv2.drawContours(image, cnt, -1, (0, 0, 255), 3)
