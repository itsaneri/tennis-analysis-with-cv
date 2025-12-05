# to test the model 

from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/court_detector3/weights/best.pt")

img = cv2.imread("dataset/images/test/images/game1-Clip9-0066_jpg.rf.d34b62ae09173a12c3d9e7b6ecc169e3.jpg")
res = model(img, conf=0.05)[0]
print("num boxes:", len(res.boxes))
print("classes:", res.boxes.cls.cpu().numpy())


