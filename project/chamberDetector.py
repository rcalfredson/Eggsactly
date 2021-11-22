import argparse
import cv2
import numpy as np

DEBUG = True

def options():
  """Parse options for chamber detector (debug only)."""
  p = argparse.ArgumentParser(description=
    'Count eggs in images within the given directory. If Pascal VOC XML files' +
        ' specifying bounding boxes are stored alongside images, exported' +
        ' results will also include error.')
  p.add_argument('img', help='path to an image ')
  return p.parse_args()

class CircleFinder:
  def __init__(self, imgPath):
    self.img = cv2.imread(imgPath)

  def findCircles(self):
    cntIdx = 0
    contours, hier = cv2.findContours(self.img.copy(), cv2.RETR_TREE,
      cv2.CHAIN_APPROX_SIMPLE)
    hier1 = hier[0] if len(hier) else []
    while DEBUG:
      shownImg = np.array(self.img)
      cv2.imshow('orig', self.img)
      origArea = cv2.contourArea(contours[cntIdx])
      print("%d (%.0f) " %(cntIdx, origArea),)
      childIndex, parentIndex = hier1[cntIdx][2:4]
      numChildren = 0
      # print('first child and parent:', hier1[cntIdx][2:4])
      # print('contour:', contours[cntIdx])
      # print('area before child subraction:', origArea)
      # while childIndex != -1:   # for all children
      #   childIndex = hier1[childIndex][0]
      #   print('contour area for the child', childIndex)
      #   childArea = cv2.contourArea(contours[childIndex])
      #   print(childArea)
      #   origArea -= childArea
      #   cv2.drawContours(shownImg, contours, childIndex, COL_R, 1, cv2.LINE_AA)
      #   cv2.imshow('img', shownImg)
      #   cv2.waitKey(0)
      #   numChildren += 1
      # print('numChildren:', numChildren)
      # print('area after child subraction:', origArea)
      # print('how many total pixels?', contourRegions.size)
      # #
      # print('how many non-contour pixels?', contourRegions.size - cv2.countNonZero(contourRegions))

      # print('how many contours in the converted image?', len(cv2.findContours(contourRegions.astype(np.uint8), cv2.RETR_TREE,
      # cv2.CHAIN_APPROX_SIMPLE)[0]))
      # find any black pixel.
      # cv2.imshow('contourRegions', contourRegions)
      # cv2.waitKey(0)
      print('valid?', parentIndex != -1 and hier1[parentIndex][3] == -1)
      for idx in range(0, cntIdx+1):
        cv2.drawContours(shownImg, contours, idx, COL_W, -1, cv2.LINE_AA)
        if numChildren == 0:
          cv2.drawContours(shownImg, contours, idx, COL_G, 1, cv2.LINE_AA)
      imshow('img', shownImg)
      c = chr(cv2.waitKey(0) & 255)
      if c == '+' and cntIdx < len(contours)-1:
        cntIdx += 1
      elif c == '-' and cntIdx > 0:
        cntIdx -= 1
      elif c == 'q':
        break
