import cv2
import numpy as np

from util import *

lastX, lastY = None, None

def onMouse(event, x, y, flags, params):
    global lastX, lastY
    lastX, lastY = x, y

while True:
    img = np.zeros((300, 300, 3))
    cv2.setMouseCallback('test', onMouse)
    if lastX is not None and lastY is not None:
        print('drawing cursor at', lastX)
        cv2.circle(img, (lastX, lastY), 4, COL_Y, thickness=2)
    cv2.imshow('test', img)
    cv2.waitKey(1)


img = np.zeros((300, 300, 3)).astype(np.uint8)
cv2.circle(img, (50,50), 4, COL_Y, thickness=2)
cv2.imshow('test', img)
print('type of the img', img)
cv2.waitKey(0)