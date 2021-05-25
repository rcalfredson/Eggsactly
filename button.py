import numpy as np
from common import getTextSize
import cv2
from util import COL_W, putText


class Button:
    def __init__(
        self,
        btn_type,
        handle_press,
        text,
        coords,
        text_size,
        text_wt,
        on_color,
        off_color,
        disabled: bool = False,
        disabled_color=(44, 44, 44),
    ):
        self.type = btn_type
        self.handle_press = handle_press
        self.text = text
        self.under_cursor = False
        self.under_cursor_prev = None
        self.coords = coords
        self.text_size = text_size
        self.text_wt = text_wt
        self.on_color = on_color
        self.off_color = off_color
        self.disabled = disabled
        self.disabled_color = disabled_color

    def display(self, img):
        if self.disabled:
            color = self.disabled_color
        else:
            color = self.on_color if self.under_cursor else self.off_color
        cv2.rectangle(img, self.coords[0], self.coords[1], COL_W, 2)
        cv2.rectangle(img, self.coords[0], self.coords[1], color, cv2.FILLED)
        text_width = getTextSize(self.text, self.text_size)[1]
        putText(
            img,
            self.text,
            (
                round(np.mean([self.coords[i][0] for i in range(2)]))
                - round(0.5 * text_width),
                round(np.mean([self.coords[i][1] for i in range(2)])),
            ),
            (0, 1),
            (cv2.FONT_HERSHEY_PLAIN, self.text_size, COL_W, self.text_wt, cv2.LINE_AA),
        )

    def check_if_under_cursor(self, x, y):
        self.under_cursor_prev = self.under_cursor
        self.under_cursor = (
            x >= self.coords[0][0]
            and x <= self.coords[1][0]
            and y >= self.coords[1][1]
            and y <= self.coords[0][1]
        )
