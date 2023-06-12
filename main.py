import sys
import cv2
import numpy as np


def thresh_bar(value):
    global thresh
    thresh = value

def mp_bar(value):
    global mask
    mask = value*2+1

def blur_bar(value):
    global blur
    blur = value*2+1


CAM_ID = 0
thresh = 180
mask = 11
blur = 3

capture = cv2.VideoCapture(CAM_ID)                        # 0번 카메라 연결
if capture.isOpened() is None: raise Exception("카메라 연결 안됨")

capture.set(cv2.CAP_PROP_FRAME_WIDTH, 400)      # 카메라 프레임 너비
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)     # 카메라 프레임 높이
capture.set(cv2.CAP_PROP_BRIGHTNESS, 100)       # 프레임 밝기 초기화

title1 = "cam"                                  # 윈도우 이름 지정
title2 = "gray"                                   # 윈도우 이름 지정
title3 = "dst"                                   # 윈도우 이름 지정
cv2.namedWindow(title2)                          # 윈도우 생성 - 반드시 생성 해야함
cv2.createTrackbar("thresh", title2, 180, 240, thresh_bar)
cv2.createTrackbar("mask", title2, 2, 50, mp_bar)
cv2.createTrackbar("blur", title2, 2, 50, blur_bar)


while True:
    ret, frame = capture.read()                 # 카메라 영상 받기
    if not ret: break
    if cv2.waitKey(30) >= 0: break

    # 흑백
    grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 블러링
    blurImg = cv2.GaussianBlur(grayImg, (blur, blur), 0)
    # 이진화
    _, binImg = cv2.threshold(blurImg, thresh, 255, cv2.THRESH_BINARY)
    # 모폴로지 연산
    kernel = np.ones((mask, mask), np.uint8)
    mpImg = cv2.morphologyEx(binImg, cv2.MORPH_CLOSE, kernel)

    # Canny
    dst = cv2.Canny(mpImg, 50, 150)

    # 윤곽선 검출
    contours, _ = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    print(len(contours))

    # for pts in contours:
    #     if cv2.contourArea(pts) < 1000:
    #         continue

        # approx = cv2.approxPolyDP(pts, cv2.arcLength(pts, True) * 0.02, True)

        # if len(approx) != 4:
        #     continue

        # w, h = 300, 400
        # srcQuad = np.array([[approx[0, 0, :]], [approx[1, 0, :]],
        #                     [approx[2, 0, :]], [approx[3, 0, :]]]).astype(np.float32)
        # dstQuad = np.array([[0, 0], [0, h], [w, h], [w, 0]]).astype(np.float32)
        #
        # pers = cv2.getPerspectiveTransform(srcQuad, dstQuad)
        # dst = cv2.warpPerspective(frame, pers, (w, h))
        #
        # cv2.polylines(frame, pts, True, (0, 0, 255))

    cv2.imshow(title1, frame)
    cv2.imshow(title2, mpImg)
    if dst is not None :
        cv2.imshow(title3, dst)

capture.release()