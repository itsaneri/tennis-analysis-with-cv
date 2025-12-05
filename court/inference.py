import cv2
import numpy as np
from ultralytics import YOLO


model = YOLO("runs/detect/court_detector3/weights/best.pt")

# Edges that define a tennis court 

COURT_EDGES = [
    (6,7), (7,8), (8,9), (9,10),
    (11,12), (12,13), (13,1), (1,2),
    (3,4), (4,5),
    (11,5), (2,3),
    (6,11), (7,12), (8,13),
    (9,1), (10,2)
]

def sort_points(pts):
    pts = np.array(pts)

    # Sort by y first (top rows, bottom rows)
    idx = np.argsort(pts[:, 1])
    pts = pts[idx]

    # First 7 = top part, last 7 = bottom part
    top = pts[:7]
    bottom = pts[7:]

    # Sort each row left → right
    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]

    return np.vstack([top, bottom])


# Draw the court lines
def draw_court(img, pts):
    for a, b in COURT_EDGES:
        x1, y1 = pts[a]
        x2, y2 = pts[b]
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)


# Extract center points from YOLO detections
def get_court_keypoints(result):
    boxes = result.boxes

    pts = []
    for b in boxes:
        cls = int(b.cls[0])

        if cls >= 14:  # skip ball and player
            continue

        x1,y1,x2,y2 = b.xyxy[0]
        cx = float((x1 + x2) / 2)
        cy = float((y1 + y2) / 2)
        pts.append((cls, (cx, cy)))

    # Sort by class ID
    pts = sorted(pts, key=lambda x: x[0])

    # Keep only coord array
    return [p[1] for p in pts]


def run_video(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model(frame, verbose=False)[0]

        pts = get_court_keypoints(result)

        if len(pts) == 14:
            pts_sorted = sort_points(pts)

            # Debug: draw green points
            for (x, y) in pts_sorted:
                cv2.circle(frame, (int(x), int(y)), 4, (0,255,0), -1)

            draw_court(frame, pts_sorted)

        cv2.imshow("Court", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


run_video("dataset/videos/inference.mp4")
