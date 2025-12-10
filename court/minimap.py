import cv2
import numpy as np
from ultralytics import YOLO
from court_reference import CourtReference

# configurable paths and parameters
MODEL_PATH = "runs/best.pt"
BALL_MODEL_PATH = "runs/best.pt"

PLAYER_CLASS = 15
BALL_CLASS = 14

MINIMAP_W = 350
MINIMAP_H = 700

# load models
model = YOLO(MODEL_PATH)
# ball_model = YOLO(BALL_MODEL_PATH)

court_ref = CourtReference()

# mapping yolo class ids → reference point ids
REMAPPING = {
    0: 0,
    6: 1,
    7: 2,
    8: 3,
    9: 4,
    10: 5,
    11: 6,
    12: 7,
    13: 8,
    1: 9,
    2: 10,
    3: 11,
    4: 12,
    5: 13,
}


def get_homography(boxes):
    # compute homography using detected court keypoints
    ref_pts = []
    img_pts = []

    for b in boxes:
        if b.cls.numel() == 0:
            continue

        cls_id = int(b.cls[0])
        if cls_id not in REMAPPING:
            continue

        point_id = REMAPPING[cls_id]

        # bounding box center
        x1, y1, x2, y2 = b.xyxy[0]
        cx = float((x1 + x2) / 2)
        cy = float((y1 + y2) / 2)

        img_pts.append([cx, cy])
        ref_pts.append(list(court_ref.court_conf[point_id]))

    if len(img_pts) < 4:
        return None

    img_pts = np.array(img_pts, dtype=np.float32).reshape(-1, 1, 2)
    ref_pts = np.array(ref_pts, dtype=np.float32).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(ref_pts, img_pts, cv2.RANSAC, 3.0)
    return H


def draw_projected_court(frame, H):
    # draw warped court on the original camera frame
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


def image_to_ref(points_img, H_inv):
    # transform image pixel points → normalized reference coordinates
    if len(points_img) == 0:
        return np.empty((0, 2), dtype=np.float32)

    pts = np.array([points_img], dtype=np.float32)
    return cv2.perspectiveTransform(pts, H_inv)[0]


def ref_to_minimap_xy(x, y):
    # map normalized reference coords to minimap pixel coords
    return int(x * MINIMAP_W), int(y * MINIMAP_H)


def build_minimap(H, player_pts_img, ball_pts_img):
    # create blank minimap background
    minimap = np.zeros((MINIMAP_H, MINIMAP_W, 3), dtype=np.uint8)

    # dark translucent overlay
    overlay = minimap.copy()
    overlay[:] = (30, 30, 30)
    minimap = cv2.addWeighted(minimap, 0.2, overlay, 0.8, 0)

    # draw reference court lines
    for p1, p2 in court_ref.get_court_lines():
        u1, v1 = ref_to_minimap_xy(*p1)
        u2, v2 = ref_to_minimap_xy(*p2)
        cv2.line(minimap, (u1, v1), (u2, v2), (230, 230, 230), 3)

    if H is None:
        return minimap

    H_inv = np.linalg.inv(H)

    players_ref = image_to_ref(player_pts_img, H_inv)
    balls_ref = image_to_ref(ball_pts_img, H_inv)

    # draw players
    for x, y in players_ref:
        u, v = ref_to_minimap_xy(x, y)

        if not (0 <= x <= 1 and 0 <= y <= 1):
            cv2.circle(minimap, (u % MINIMAP_W, v % MINIMAP_H), 10, (0, 255, 0), -1)
            continue

        cv2.circle(minimap, (u, v), 12, (0, 200, 0), -1)
        cv2.circle(minimap, (u, v), 12, (0, 80, 0), 2)
        cv2.putText(minimap, "P", (u - 8, v + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # draw ball
    for x, y in balls_ref:
        u, v = ref_to_minimap_xy(x, y)
        cv2.circle(minimap, (u, v), 9, (0, 255, 255), -1)
        cv2.circle(minimap, (u, v), 9, (0, 120, 120), 2)

    return minimap


def run_video(video_path):
    cap = cv2.VideoCapture(video_path)
    last_H = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = model(frame, conf=0.25, verbose=False)[0].boxes

        H = get_homography(detections)
        if H is not None:
            last_H = H

        ball_pts_img = []
        player_pts_img = []

        # process detections
        for b in detections:
            cid = int(b.cls[0])
            x1, y1, x2, y2 = b.xyxy[0]

            cx = float((x1 + x2) / 2)
            cy = float((y1 + y2) / 2)

            if cid == PLAYER_CLASS:
                player_pts_img.append((cx, cy))
            if cid == BALL_CLASS:
                ball_pts_img.append((cx, cy))

        if last_H is not None:
            draw_projected_court(frame, last_H)

        minimap = build_minimap(last_H, player_pts_img, ball_pts_img)

        mh = minimap.shape[0]
        scale = frame.shape[0] / mh
        minimap_resized = cv2.resize(minimap, (int(MINIMAP_W * scale), frame.shape[0]))

        combined = cv2.hconcat([frame, minimap_resized])
        cv2.imshow("court + minimap", combined)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


run_video("dataset/videos/8.mp4")
