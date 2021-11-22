import cv2
import glob, os
from util import *

imagesDir = "C:\\Users\\Tracking\\coco-annotator\\datasets\\eggs"

imagePaths = sorted(glob.glob(os.path.join(imagesDir, '*.jpg')))
with open('imagesToSort.txt', 'w') as f:
  [f.write('%s\n'%imagePath) for imagePath in imagePaths]
imageIndex = 0
while True:
  while True:
    textToDisplay = os.path.basename(imagePaths[imageIndex])
    textDims = cv2.getTextSize(textToDisplay, cv2.FONT_HERSHEY_PLAIN, 0.9, 1)[0][
      ::-1]
    img = cv2.imread(imagePaths[imageIndex])
    display = np.multiply(np.ones((img.shape[0] + textDims[0] + 10, max(textDims[1] + 10, img.shape[1]), 3)), 255)
    display[textDims[0] + 10:, :img.shape[1]] = img
    putText(display, textToDisplay, (0, 0), (0, 1), textStyle(color=COL_BK, size=0.9))
    display = display.astype(np.uint8)
    cv2.imshow('egg-laying image', display)
    k = cv2.waitKey(1)
    if k == -1: eventProcessingDone = True
    elif eventProcessingDone: break
  k &= 255
  keyCode = chr(k)
  if keyCode == '.' and imageIndex + 1 < len(imagePaths):
    imageIndex += 1
  elif keyCode == ',' and imageIndex > 0:
    imageIndex -= 1
