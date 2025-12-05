from ultralytics import YOLO

# Train YOLOv12 ONLY for the 14 court classes
# model = YOLO("yolo12n.pt")
model = YOLO("runs/detect/court_detector3/weights/last.pt")

model.train(
    data="dataset/data.yaml",
    epochs=20,       
    imgsz=640,
    batch=4,
    device="mps",
    workers=0,
    resume=True,
    name="court_detector",
    classes=list(range(14)),  # ONLY classes 0–13 (ignore ball, player)
)
