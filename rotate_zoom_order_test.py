import cv2
from circleFinder import rotate_image

core_img = cv2.imread('/media/Synology3/Robert/counting-3/imgs/obscured/2021_10_8_IMG_0002.png')
rotation_angle = 0.4

rotated_then_zoomed = cv2.resize(rotate_image(core_img, rotation_angle), (0, 0), fx=1.2, fy=1.2)
zoomed_then_rotated = rotate_image(cv2.resize(core_img, (0, 0), fx=1.2, fy=1.2), rotation_angle)

cv2.imshow('rotated then zoomed', cv2.resize(rotated_then_zoomed, (0, 0), fx=0.25, fy=0.25))
cv2.imshow('zoomed then rotated', cv2.resize(zoomed_then_rotated, (0, 0), fx=0.25, fy=0.25))
cv2.waitKey(0)
