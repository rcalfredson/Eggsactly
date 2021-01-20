import cv2
import math
from util import *
import n

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

slope = 
img = cv2.imread(r"I:\counting-condinst\fromSyno3\9_9_2020_IMG_0008.jpg")
rotatedImg = rotate_image(img, 180*math.atan(slope)/math.pi)
# rotatedImg = rotate_image(img, 25)
cv2.imshow('orig', cv2.resize(img, (0, 0), fx=0.15, fy=0.15,
      interpolation=cv2.INTER_CUBIC))
cv2.imwrite(r'P:\Egg images_9_3_2020\debug\tempRotatedImg.png', rotatedImg)
print('orig image shape:', img.shape)
print('rotationAngle:', 180*math.atan(slope)/math.pi)
print('rotated image shape:', rotatedImg.shape)
cv2.imshow('rotated', cv2.resize(rotatedImg, (0, 0), fx=0.15, fy=0.15,
      interpolation=cv2.INTER_CUBIC))
cv2.waitKey(0)
