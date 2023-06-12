import torch
import cv2
import numpy as np

CAM_ID = 0
capture = cv2.VideoCapture(CAM_ID)                        # 0번 카메라 연결
if capture.isOpened() is None: raise Exception("카메라 연결 안됨")

title = "Cam"
cv2.namedWindow(title)
cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./cardv1.pt')
model.conf = 0.70
# model.max_det = 1

device = torch.device("cpu")
model.to(device)


while True:
    ret, frame = capture.read()                 # 카메라 영상 받기
    if not ret: break

    # Press "q" to quit
    if cv2.waitKey(30) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break

    result = model(frame)
    result.print()

    cv2.imshow(title, np.squeeze(result.render()))



capture.release()