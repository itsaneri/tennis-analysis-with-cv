import cv2
import numpy as np
from ultralytics import YOLO

from court_reference import CourtReference

# load model and court reference
model = YOLO("runs/best.pt")
court_ref = CourtReference()

# mapping yolo class ids → court reference point ids
REMAPPING = {
    0: 0,   # Top Left Corner (Doubles)
    6: 1,   # Top Left Singles
    7: 2,   # Top Right Singles
    8: 3,   # Top Right Corner (Doubles)
    9: 4,   # Top Mid Service Left
    10: 5,  # Top Mid Service Center
    11: 6,  # Top Mid Service Right
    12: 7,  # Bottom Mid Service Left
    13: 8,  # Bottom Mid Service Center
    1: 9,   # Bottom Mid Service Right
    2: 10,  # Bottom Left Corner (Doubles)
    3: 11,  # Bottom Left Singles
    4: 12,  # Bottom Right Singles
    5: 13   # Bottom Right Corner (Doubles)
}


def get_homography(boxes, frame_shape):
    """
    compute homography h: reference (normalized) → image (pixels)
    """
    img_h, img_w = frame_shape[:2]

    ref_pts = []
    img_pts = []

    for b in boxes:
        # guard for empty tensors
        if b.cls.numel() == 0:
            continue

        cls_id = int(b.cls[0])
        if cls_id not in REMAPPING:
            continue

        mapped_id = REMAPPING[cls_id]

        # get pixel center of bounding box
        x1, y1, x2, y2 = b.xyxy[0]
        cx = float((x1 + x2) / 2)
        cy = float((y1 + y2) / 2)

        img_pts.append([cx, cy])
        ref_pts.append(list(court_ref.court_conf[mapped_id]))

    if len(ref_pts) < 4:
        return None

    ref_pts = np.array(ref_pts, dtype=np.float32).reshape(-1, 1, 2)
    img_pts = np.array(img_pts, dtype=np.float32).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(ref_pts, img_pts, cv2.RANSAC, 3.0)
    return H


def draw_projected_court(frame, H):
    """
    draw the reference court warped into the frame
    """
    if H is None:
        return

    lines = court_ref.get_court_lines()

    pts = []
    for p1, p2 in lines:
        pts.append(p1)
        pts.append(p2)

    pts = np.array([pts], dtype=np.float32)
    warped = cv2.perspectiveTransform(pts, H)[0]

    for i in range(0, len(warped), 2):
        x1, y1 = warped[i]
        x2, y2 = warped[i + 1]
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)


def run_video(video_path):
    cap = cv2.VideoCapture(video_path)
    last_H = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.25, classes=list(range(14)), verbose=False)
        boxes = results[0].boxes

        # compute homography
        H = get_homography(boxes, frame.shape)
        if H is not None:
            last_H = H

        # draw projected court if homography is known
        if last_H is not None:
            draw_projected_court(frame, last_H)

        # draw yolo dot centers
        for b in boxes:
            x1, y1, x2, y2 = b.xyxy[0]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        cv2.imshow("court detection (homography)", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_video("dataset/videos/9.mp4")
