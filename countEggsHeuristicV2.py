import argparse
import cv2
from sklearn.cluster import KMeans
from scipy.interpolate import splprep, splev
import numpy as np

from util import *
from common import globFiles
from image import background_color
from pascalVocHelper import PascalVOCHelper

EGG_SIZE = 175
SHAPE_FILTER = False
HULL_FILTER = False
DEBUG = True
DEBUG_TURN_DETECT = False

def options():
  """Parse options for egg counter."""
  p = argparse.ArgumentParser(description=
    'Count eggs in images within the given directory. If Pascal VOC XML files' +
        ' specifying bounding boxes are stored alongside images, exported' +
        ' results will also include error.')
  p.add_argument('-d', dest='dir', help='directory containing .png images')
  return p.parse_args()

performance_results = [['filename', 'num labelled', 'num predicted',
  'abs. error', 'pct. error']]

def getGaussianDerivs(sigma, M, gaussian, dg, d2g):
  L = int((M - 1) / 2)
  sigma_sq = sigma * sigma
  sigma_quad = sigma_sq * sigma_sq
  dg = dg[:M]
  d2g = d2g[:M]
  gaussian = gaussian[:M]
  g = cv2.getGaussianKernel(M, sigma, cv2.CV_64F)
  for i in range(-L, L+1):
    idx = i + L
    gaussian[idx] = g[idx]
    dg[idx] = (-i/sigma_sq) * g[idx]
    d2g[idx] = (-sigma_sq + i*i) / sigma_quad * g[idx]
  return gaussian, dg, d2g

def getDx(x, n, sigma, gx, dgx, d2gx, g, dg, d2g, isOpen=False):
  L = int((len(g) - 1) / 2)
  # print('N:', n)
  # print('L:', L)
  # print('len(x):', len(x))
  gx, dgx, d2gx = 0.0, 0.0, 0.0
  for k in range(-L, L+1):
    x_n_k = 0.0
    if n - k < 0:
      if isOpen:
        x_n_k = x[-(n-k)]
      else:
        x_n_k = x[len(x) + (n-k)]
    elif n-k > len(x) - 1:
      if isOpen:
        x_n_k = x[n+k]
      else:
        x_n_k = x[(n-k) - len(x)]
    else:
      x_n_k = x[n-k]
    gx += x_n_k * g[k + L]
    dgx += x_n_k * dg[k + L]
    d2gx += x_n_k * d2g[k + L]
  return gx, dgx, d2gx

def getDxCurve(x, sigma, gx, dx, d2x, g, dg, d2g, isOpen=False):
  gx = gx[:len(x)]
  dx = dx[:len(x)]
  d2x = d2x[:len(x)]
  for i in range(len(x)):
    gausX, dgx, d2gx = 0.0, 0.0, 0.0
    gausX, dgx, d2gx = getDx(x, i, sigma, gausX, dgx, d2gx, g, dg, d2g, isOpen)
    gx[i] = gausX
    dx[i] = dgx
    d2x[i] = d2gx
  return gx, dx, d2x

# counts contour
# returns label, number of eggs (0 if filtered)
def countContour(ar, le, seArs, cnt, img):
  lbl = None

  # size
  if ar < EGG_SIZE*0.5:
    return 'size', 0

  # estimate number of eggs
  nea = 1 if ar < EGG_SIZE*1.3 else round(ar/(EGG_SIZE*.75), 0)
  print('returning nea:', nea)
  if nea > 1:
    lbl = '%d'%nea

  if SHAPE_FILTER:
    l2a = le/ar
    if l2a > 0.5:
      return '%.2f'%l2a, 0

  if HULL_FILTER:
    #print('original contour?', cnt)
    #print(cnt.shape)
    #cnt = cnt[:][::2].astype(np.int32)
    #print('contour sampled every two?', cnt)
    x, y = cnt.T
    x, y = x[0], y[0]
    sigma = 8
    M = int(np.round((10.0*sigma+1) / 2.0)*2 - 1)
    if (M - 1) / 2 <= len(x):
      pass
      # assert(M % 2 == 1)
      # gaussian, dg, d2g = [[0]*M for _ in range(3)]
      # gaussian, dg, d2g = getGaussianDerivs(sigma, M, gaussian, dg, d2g)
      # gxX, dxX, d2xX = [[0]*len(x) for _ in range(3)]
      # gxY, dxY, d2xY = [[0]*len(x) for _ in range(3)]
      # gxX, dxX, d2xX = getDxCurve(x, sigma, gxX, dxX, d2xX, gaussian, dg, d2g, isOpen=False)
      # gxY, dxY, d2xY = getDxCurve(y, sigma, gxY, dxY, d2xY, gaussian, dg, d2g, isOpen=False)
      # res_array = [[[int(i[0]), int(i[1])]] for i in zip(gxX,gxY)]
      # cnt = np.asarray(res_array, dtype=np.int32)

    isClockwise = False
    anglePrev = 0
    samplingInterval = 5
    angleSignChanges = [0]*samplingInterval
    numSmallChanges = 0
    longestSmallChangeDist = 0
    currentSmallChangeDist = 0
    tot = 0
    for i in range(samplingInterval):
      if DEBUG:
        print('start of new sampling cycle')
      for ptI, cntPt in enumerate(cnt[i::samplingInterval]):
        pass
        # calculate angle between the two points
        potentialPt = i + samplingInterval*(ptI + 1)
        while potentialPt > len(cnt) - 1:
          potentialPt -= len(cnt)
        nextPt = cnt[potentialPt]
        cntPt = cntPt[0]
        nextPt = nextPt[0]
        if DEBUG:
          print('ptI + i + 1:', ptI + i + 1)
          print(cnt[ptI + i + 1])
          print('first pt:', cntPt, ' |  next pt:', nextPt)
          print('y change:', (nextPt[1] - cntPt[1]))
          print('x change:', (nextPt[0] - cntPt[0]))
        deltaAngle = np.arctan2((nextPt[1] - cntPt[1]), (nextPt[0] - cntPt[0])) * 180 / np.pi
        angleChange = angleDiff(deltaAngle, anglePrev)
        if DEBUG_TURN_DETECT:
          contourImg = np.array(img)
          cv2.drawMarker(contourImg, tuple(cntPt), (0, 255, 0), markerType=cv2.MARKER_STAR, markerSize=9)
          cv2.drawMarker(contourImg, tuple(nextPt), (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=9)
        if DEBUG:
          print('delta angle in degrees:', deltaAngle)
          print('diff with prev angle in degrees:', angleChange)
        if ptI > 0:
          if isClockwise is not (angleChange < 0) and np.abs(angleChange) > 2:
          #if np.abs(angleChange) > 2:
            if DEBUG:
              print('sign changed.')
              print('was prior change clockwise?', isClockwise)
              print('is this change clockwise?', angleChange < 0)
            #angleSignChanges.append(cntPt)
            angleSignChanges[i] += 1
          if np.abs(angleChange) < 20:
            numSmallChanges += 1
            currentSmallChangeDist += distance(cntPt, nextPt)
          else:
            if currentSmallChangeDist > longestSmallChangeDist:
              longestSmallChangeDist = currentSmallChangeDist
            currentSmallChangeDist = 0
        tot += 1
        anglePrev = deltaAngle
        if angleChange != 0:
          isClockwise = angleChange < 0
        if DEBUG_TURN_DETECT:
          cv2.imshow('cntPt', contourImg)
          cv2.waitKey(0)
        # input('enter...')
    # print(cnt.shape)
    # #smoothened.append(np.asarray(res_array, dtype=np.int32))
    # hullPerimeter = cv2.arcLength(cv2.convexHull(cnt), True)
    # #convexityDefects = cv2.convexityDefects(cnt, cv2.convexHull(cnt, returnPoints=False))
    # if convexityDefects is None:
    #   convexityDefects = []
    # numConvexityDefects = len(convexityDefects)
    # #print('hullP and peri:', hullPerimeter, le)
    # #print('ratio:', hullPerimeter / le)
    # #print('first element of convexity defects:', convexityDefects[0])
    # contourImg = np.array(img)
    #print('results from contour of the first point of the first defect:', cnt[convexityDefects[0][0][0]])
    #contourImg = np.array(img)
    contourImg = np.array(img)
    angleSignChanges = np.floor(np.mean(angleSignChanges))
    # for signChange in angleSignChanges:
    #   pass
    #   contourImg[signChange[1]-2:signChange[1]+2][:, signChange[0]-2:signChange[0]+2] = [0, 255, 0]
    # print('number of convexity defects:', numConvexityDefects)
    if DEBUG:
      pass
      print('number sign changes:', angleSignChanges)
      print('floor(0.5*nea):', np.ceil(0.5*nea))
      print('le: %.2f  |  ar: %.2f'%(le, ar))
      print('length to area:', le/ar)
      print('rejected?', nea > 4 and angleSignChanges <= np.ceil(0.5*nea))
      print('ratio of small changes to total:', numSmallChanges / tot)
      binaryContourImg = np.zeros((img.shape[0], img.shape[1]))
      cv2.drawContours(binaryContourImg, [cnt], 0, 255, 1, cv2.LINE_AA)
      #print('number of continuous regions')
      print('returning nea:', nea)
      cv2.drawContours(contourImg, [cnt], 0, COL_G, 1, cv2.LINE_AA)
      cv2.drawContours(contourImg, [cnt], 0, COL_W, -1, cv2.LINE_AA)
      cv2.imshow('contourBoundaries', contourImg)
      cv2.imshow('binary boundary', binaryContourImg)
      cv2.waitKey(0)
    #input('enter...')
    #if nea > 4 and angleSignChanges <= np.ceil(0.5*nea):
    if nea > 4 and (numSmallChanges / tot > 0.305 or longestSmallChangeDist > 350):
      # print('rejecting a contour as unnatural due to too few changes in contour sign: cd = %i AND nea = %i'%(angleSignChanges, nea))
      # cv2.drawContours(contourImg, [cnt], 0, COL_G, -1, cv2.LINE_AA)
      # cv2.imshow('contourBoundaries', contourImg)
      # print('number sign changes:', angleSignChanges)
      # print('arc length:', le)
      # print('nea:', nea)
      # cv2.waitKey(0)
      if numSmallChanges / tot > 0.305:
        print('rejecting based on number of small changes')
        print('ratio is:', numSmallChanges, '/', 'tot =', numSmallChanges / tot)
      if longestSmallChangeDist > 350:
        print('rejecting based on longest small-change distance')
      return '', 0
  if nea == 1:
    seArs.append(ar)
  return lbl, nea

def count(img, imgPath):
  """Estimate number of eggs in the inputted image."""
  dominant_color = background_color(img)
  thresholded = cv2.inRange(img, tuple([0.75 * col for col in dominant_color]), (255, 255, 255))
  # cv2.imshow('orig', img)
  # cv2.imshow('thresholded', thresholded)
  # cv2.waitKey(0)
  contours, hier = cv2.findContours(thresholded.copy(), cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE)
  hier1 = hier[0] if len(hier) else []
  cntIdx = 0
  contourRegions = np.ones((img.shape[0], img.shape[1]))

  seArs, lbls, ne = [], [], 0   # single egg areas, labels, number eggs
  for i, cnt in enumerate(contours):
    cI, pI = hier1[i][2:4]   # first child, parent
    # simply set to zero any point if it is inside the contour
    cv2.drawContours(contourRegions, [cnt], -1, 0, -1)
    if pI != -1 and hier1[pI][3] == -1:   # level 1
      ar, le = cv2.contourArea(contours[i]), cv2.arcLength(contours[i], True)
      
      #print('noncontour regions after reassign:', nonContourRegions[contours[i]])
      while cI != -1:   # for all children
        ar -= cv2.contourArea(contours[cI])
        le += cv2.arcLength(contours[cI], True)
        cI = hier1[cI][0]

      _, nea = countContour(ar, le, seArs, cnt, img)

      ne += nea
      print('running egg count:', ne)
      input('enter...')
  bbs = PascalVOCHelper('%s.xml'%imgPath.split('.')[0]).boundingBoxes()
  num_labelled = len(bbs)
  bordersize = 1
  contourRegions = cv2.copyMakeBorder(
    contourRegions,
    top=bordersize,
    bottom=bordersize,
    left=bordersize,
    right=bordersize,
    borderType=cv2.BORDER_CONSTANT,
    value=[0, 0, 0]
  )
  # cv2.imshow('contourRegions', contourRegions)
  # cv2.waitKey(0)
  remainder_contours = cv2.findContours(contourRegions.astype(np.uint8), cv2.RETR_TREE,
      cv2.CHAIN_APPROX_SIMPLE)[0]
  for contour in remainder_contours:
    # invert the contour first?
    ar, le = cv2.contourArea(contour), cv2.arcLength(contour, True)
    # _, nea = countContour(ar, le, seArs, contour, img)
    # ne += nea
  print('num labelled: %i  |  num predicted: %.2f'%(num_labelled, ne))
  abs_diff = abs(num_labelled - ne)
  rel_error = abs_diff / num_labelled if num_labelled > 0 else ''
  performance_results.append([imgPath, num_labelled, ne, abs_diff,
    rel_error*(1 if type(rel_error) is str else 100)])

  while DEBUG:
    shownImg = np.array(img)
    cv2.imshow('orig', img)
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

opts = options()
imgs = globFiles(opts.dir)

for imgPath in imgs:
  img = cv2.imread(imgPath)
  print('Checking image', imgPath)
  count(img, imgPath)
with open('egg_counts.csv', 'wt', newline='') as resultsFile:
  writer = csv.writer(resultsFile)
  writer.writerows(performance_results)
