import argparse
import csv, cv2, numpy as np
import tensorflow as tf

from common import globFiles
from object_detector import TFObjectDetector
from pascalVocHelper import PascalVOCHelper
from util import *

def options():
  """Parse options for egg counter."""
  p = argparse.ArgumentParser(description=
    'Count eggs in images within the given directory. If Pascal VOC XML files' +
        ' specifying bounding boxes are stored alongside images, exported' +
        ' results will also include error.')
  p.add_argument('-d', dest='dir', help='directory containing .png images')
  p.add_argument('--model', dest='model',
    help='path to directory containing saved TensorFlow model')
  p.add_argument('--labels', dest='labels',
    help='path to the map of indices to category names')
  p.add_argument('-I', dest='interactive', action='store_true',
    help='display results interactively (press key to advance through results)')
  return p.parse_args()
    
opts = options()
imgs = globFiles(opts.dir)
performance_results = [['filename', 'num labelled', 'num predicted',
  'abs. error', 'pct. error']]
resultsImgs = []
idxs = ((1, 0), (3, 2))
bxs_final = 0

def bBoxArea(box, shapes):
  print('two factors to calculate area:', [box[idxs[1][y]]*shapes[y] - box[idxs[0][y]]*shapes[y] for y in range(1, -1, -1)])
  return np.abs(np.product([box[idxs[1][y]]*shapes[y] - box[idxs[0][y]]*shapes[y] for y in range(1, -1, -1)]))

def drawResults(img, results):
  global bxs_final
  d_bxs, d_scores = results['detection_boxes'][0], results['detection_scores'][0]
  shps = list(reversed(img.shape[:2]))
  print('detection boxes:')
  print(d_bxs)
  for iB, bx in enumerate(d_bxs):
    print('adding these rectangles:')
    print('bx is what?', bx)
    print('shps is what?', shps)
    print('example of one arg to round:', bx[idxs[0][0]]*shps[0])
    print(*[tuple([int(round(bx[idxs[y][i]]*shps[i])) for i in range(2)])
      for y in range(2)])
    print('proportion of bBox area to total:', bBoxArea(bx, shps) / np.product(shps))
    if bBoxArea(bx, shps) / np.product(shps) < 0.2 and d_scores[iB] > 0.5:
      bxs_final += 1
      if opts.interactive:
        cv2.rectangle(img, *[tuple([int(round(bx[idxs[y][i]]*shps[i])) for i in range(2)])
          for y in range(2)], COL_Y)

with tf.Session() as sess:
  detector = TFObjectDetector(opts.model, opts.labels)
  for imgPath in imgs:
    bxs_final = 0
    print('img: %s'%imgPath)
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    results = sess.run(detector.detectObjectsInImage(img))
    drawResults(img, results)
    bbs = PascalVOCHelper('%s.xml'%imgPath.split('.')[0]).boundingBoxes()
    num_labelled = len(bbs)
    abs_diff = abs(num_labelled - bxs_final)
    rel_error = abs_diff / num_labelled if num_labelled > 0 else ''
    print('number eggs detected:', bxs_final)
    print('number eggs annotated:', num_labelled)
    performance_results.append([imgPath, num_labelled, bxs_final, abs_diff,
      rel_error*(1 if type(rel_error) is str else 100)])
    if opts.interactive:
      cv2.imshow('debug', img)
      cv2.waitKey(0)
  with open('egg_counts.csv', 'wt', newline='') as resultsFile:
        writer = csv.writer(resultsFile)
        writer.writerows(performance_results)
