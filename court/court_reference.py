
class CourtReference:
    """
    reference tennis court in normalized coordinates [0, 1] x [0, 1].
    points are stored in self.court_conf and lines in self.lines.
    """

    def __init__(self) -> None:
        # normalized court proportions
        singles_ratio = 8.23 / 10.97
        side_margin = (1.0 - singles_ratio) / 2.0

        xs_left_singles = side_margin
        xs_right_singles = 1.0 - side_margin

        y_top_base = 0.0
        y_bot_base = 1.0
        y_top_serv = 5.485 / 23.77
        y_bot_serv = 18.285 / 23.77
        x_center = 0.5

        # keypoints in normalized coordinates
        self.court_conf = {
            0: (0.0, y_top_base),
            1: (xs_left_singles, y_top_base),
            2: (xs_right_singles, y_top_base),
            3: (1.0, y_top_base),

            4: (xs_left_singles, y_top_serv),
            5: (x_center, y_top_serv),
            6: (xs_right_singles, y_top_serv),

            7: (xs_left_singles, y_bot_serv),
            8: (x_center, y_bot_serv),
            9: (xs_right_singles, y_bot_serv),

            10: (0.0, y_bot_base),
            11: (xs_left_singles, y_bot_base),
            12: (xs_right_singles, y_bot_base),
            13: (1.0, y_bot_base),
        }

        # line segments by point index
        self.lines = [
            (0, 3), (10, 13), (0, 10), (3, 13),
            (1, 11), (2, 12),
            (4, 6), (7, 9),
            (5, 8),
        ]

    def get_court_lines(self):
        # return list of ((x1, y1), (x2, y2)) line endpoints
        out = []
        for i, j in self.lines:
            out.append((self.court_conf[i], self.court_conf[j]))
        return out
