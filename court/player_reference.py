
import numpy as np

class PlayerTracker:
    """
    simple 2-player tracker using y-position to assign identities.
    keeps a smoothed, dynamic bounding box for each player.
    """
    

    def __init__(self, smoothing: float = 0.7, max_missing: int = 15) -> None:
        # smoothing factor for bounding box updates (0..1)
        self.smoothing = smoothing
        # max frames allowed without detection before resetting a track
        self.max_missing = max_missing

        # current boxes as dict: {player_id: np.array([x1, y1, x2, y2])}
        self.boxes = {
            0: None,  # top player (smaller center y)
            1: None,  # bottom player (larger center y)
        }

        # how many consecutive frames each player was not detected
        self.missing_counts = {
            0: 0,
            1: 0,
        }

    @staticmethod
    def _box_center_y(box: np.ndarray) -> float:
        # return center y coordinate of a bounding box
        x1, y1, x2, y2 = box
        return float((y1 + y2) / 2.0)

    def _sort_by_vertical_position(self, boxes: list[np.ndarray]) -> list[np.ndarray]:
        # sort detected boxes by center y coordinate (top → bottom)
        return sorted(boxes, key=self._box_center_y)

    def _smooth_update(self, old_box: np.ndarray | None, new_box: np.ndarray) -> np.ndarray:
        # exponential smoothing of box coordinates
        if old_box is None:
            return new_box.copy()

        alpha = self.smoothing
        return alpha * new_box + (1.0 - alpha) * old_box

    def update(self, detected_boxes: list[list[float]]) -> None:
        """
        update tracker with a new set of detected player boxes for the current frame.

        detected_boxes: list of [x1, y1, x2, y2] in image pixels.
        """
        # convert to numpy arrays for easier math
        np_boxes = [np.array(b, dtype=np.float32) for b in detected_boxes]

        if len(np_boxes) == 0:
            # no detections, increase missing counters
            for pid in (0, 1):
                if self.boxes[pid] is not None:
                    self.missing_counts[pid] += 1
                    if self.missing_counts[pid] > self.max_missing:
                        self.boxes[pid] = None
            return

        # sort boxes by vertical position (top first, bottom second)
        sorted_boxes = self._sort_by_vertical_position(np_boxes)

        # assign first to player 0 (top), second (if exists) to player 1 (bottom)
        assignments = {
            0: sorted_boxes[0],
        }
        if len(sorted_boxes) > 1:
            assignments[1] = sorted_boxes[1]

        # update each player track with smoothing
        for pid in (0, 1):
            if pid in assignments:
                new_box = assignments[pid]
                self.boxes[pid] = self._smooth_update(self.boxes[pid], new_box)
                self.missing_counts[pid] = 0
            else:
                # player not detected in this frame
                if self.boxes[pid] is not None:
                    self.missing_counts[pid] += 1
                    if self.missing_counts[pid] > self.max_missing:
                        self.boxes[pid] = None

    def get_boxes(self) -> dict[int, np.ndarray | None]:
        """
        return current boxes as a dict {0: box_or_none, 1: box_or_none}.
        each box is a float numpy array [x1, y1, x2, y2].
        """
        return self.boxes
