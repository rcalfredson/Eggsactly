import cv2
import numpy as np
from typing import ByteString


def byte_to_bgr(img_in: ByteString):
    return cv2.cvtColor(
        cv2.imdecode(
            np.asarray(
                bytearray(img_in),
                dtype="uint8",
            ),
            cv2.IMREAD_COLOR,
        ),
        cv2.COLOR_RGB2BGR,
    )
