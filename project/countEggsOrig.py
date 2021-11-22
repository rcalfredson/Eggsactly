import argparse, csv
import cv2
import itertools
import matplotlib.pyplot as plt
import numpy as np
import timeit

from circleFinder import CircleFinder, getSubImages
from detectors.centernet.detector_factory import detector_factory
from image import background_color
from util import *

DEBUG = True
start_t = timeit.default_timer()

def options():
  """Parse options for the automated egg counter."""
  p = argparse.ArgumentParser(description=
    'Count eggs on each agarose strip on either side of each sub-chamber ' +
    'within the inputted image.')
  p.add_argument('--counts', help='path to CSV file of human-labelled egg ' +
                                  'counts organized by egg-laying regions and' +
                                  'stacked column-wise')
  p.add_argument('--dir', help='path to directory containing images from the ' +
                               'CSV file of human-labelled egg counts')
  p.add_argument('--img', help='path to an image to analyze')
  p.add_argument('--gpus', default='0', 
                             help='-1 for CPU, use comma for multiple gpus')
  p.add_argument('--arch', default='dla_34', 
                             help='model architecture. Currently tested: '
                                  'res_18 | res_101 | resdcn_18 | resdcn_101 |'
                                  'dlav0_34 | dla_34 | hourglass')
  p.add_argument('--load_model', default='',
                             help='path to pretrained model')
  p.add_argument('--dataset', default='egg',
                             help='egg | coco | kitti | coco_hp | pascal')
  p.add_argument('--K', type=int, default=100,
                             help='max number of output objects.')
  p.add_argument('--debug', type=int, default=0,
                             help='level of visualization. '
                                  '1: only show the final detection results '
                                  '2: show the network output features '
                                  '3: use matplot to display '
                                  '4: save all visualizations to disk')
  opt = p.parse_args()

  opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
  opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >=0 else [-1]
  opt.heads = {'hm': 1, 'wh': 2, 'reg': 2}
  opt.mean = [0.40789654, 0.44719302, 0.47026115]
  opt.std = [0.28863828, 0.27408164, 0.27809835]
  opt.center_thresh = 0.1
  opt.vis_thresh = 0.3
  opt.fix_res, opt.keep_res = False, True
  opt.debugger_theme = 'white'
  opt.test_scales = '1'
  opt.test_scales = [float(i) for i in opt.test_scales.split(',')]
  opt.pad = 127 if 'hourglass' in opt.arch else 31
  opt.num_classes = 1
  opt.cat_spec_wh = False
  opt.nms = False
  opt.reg_offset = True
  opt.down_ratio = 4
  opt.head_conv = -1
  opt.flip_test = False
  return opt

opts = options()
Detector = detector_factory['ctdet']
detector = Detector(opts)
# opts.load_model = "C:\\Users\\Tracking\\centernet-test\\exp\\ctdet\\default\\model_best_egg_2020-03-09.pth"
# detector2 = Detector(opts)
if opts.img:
  img = cv2.imread(opts.img)
  # brightness = 0
  # contrast = 100
  # img = np.int16(img)
  # img = img * (contrast/127+1) - contrast + brightness
  # img = np.clip(img, 0, 255)
  # img = np.uint8(img)
  circles, avgDists, numRowsCols = CircleFinder(img).findCircles()
  subImgs = getSubImages(img, circles, avgDists, numRowsCols)
  countEstimates = []
  estimateImgs = []
  for i, img in enumerate(subImgs):
    # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    # img = cv2.filter2D(img, -1, kernel)
    run_dict = detector.run(img)
    num_predicted = len([result for result in run_dict['results'][1] if result[-1] >= 0.4])
    print('num_predicted:', num_predicted)
    countEstimates.append(num_predicted)
    newImg = np.array(img)
    putText(newImg, str(num_predicted), (5, 5), (0,1), textStyle(color=COL_R, size=3))
    estimateImgs.append(newImg)
  combinedImgs = combineImgs(estimateImgs, hdrL='predictions', nc=8)[0]
  combinedImgs = cv2.resize(combinedImgs, (0, 0), fx=0.4, fy=0.4)
  cv2.imshow('debug', combinedImgs)
  cv2.waitKey(0)
  cv2.imwrite('counts_%s.png'%(os.path.basename(opts.img).split('.')[0]), combinedImgs)
  with open('counts_%s.csv'%(os.path.basename(opts.img).split('.')[0]), 'wt', newline='') as resultsFile:
    writer = csv.writer(resultsFile)
    for i in range(numRowsCols[0]):
      row = countEstimates[slice(i*numRowsCols[1]*2, i*numRowsCols[1]*2 + numRowsCols[1]*2)]
      writer.writerow(row)
elif opts.dir and opts.counts:
  pass
  # open the CSV of egg count labels
  countsData = {}
  errors = {}
  confidNew = []
  errorNew = []
  currentImg = ''
  with open(opts.counts, 'rt') as f:
    countsDataRaw = list(csv.reader(f))
    for dataLine in countsDataRaw:
      if not dataLine[0].isdigit():
        currentImg = dataLine[0]
        countsData[currentImg] = []
        errors[currentImg] = []
      else:
        countsData[currentImg].append(dataLine)
  print('countsData:', countsData)
  for imgName in countsData.keys():
    pass
    img = cv2.imread(os.path.join(opts.dir, imgName))
    # brightness = 50
    # contrast = 50
    # img = np.int16(img)
    # img = img * (contrast/127+1) - contrast + brightness
    # img = np.clip(img, 0, 255)
    # img = np.uint8(img)
    circles, avgDists, numRowsCols = CircleFinder(img, imgName).findCircles()
    subImgs = getSubImages(img, circles, avgDists, numRowsCols)
    countEstimates = []
    confidenceLevels = []
    for i, subImg in enumerate(subImgs):
      run_dict = detector.run(subImg)
      num_predicted = len([result for result in run_dict['results'][1] if result[-1] >= 0.4])
      # run_dict2 = detector2.run(subImg)
      # num_predicted2 = len([result for result in run_dict2['results'][1] if result[-1] >= 0.4])
      # if num_predicted2 > num_predicted:
      #   num_predicted = num_predicted2
      #   run_dict = run_dict2
      # confidScores = [result[-1] for result in run_dict['results'][1] if result[-1] >= 0.4]
      # confidenceLevels.append(np.mean(confidScores) if len(confidScores) > 0 else 0)
      # print('for img', imgName)
      # print('new num predicted:', num_predicted)
      # print('old num predicted would\'ve been:', len([result for result in run_dict['results'][1] if result[-1] >= 0.3]))
      # print('confidence levels:', [result[-1] for result in run_dict['results'][1] if result[-1] >= 0.3])
      countEstimates.append(num_predicted)
    for i in range(numRowsCols[0]):
      rowSlice = slice(i*numRowsCols[1]*2, i*numRowsCols[1]*2 + numRowsCols[1]*2)
      row = countEstimates[rowSlice]
      print('row:', row)
      #confidRow = confidenceLevels[rowSlice]
      regionErr = np.abs(np.array(row) - np.array(countsData[imgName][i], dtype=np.uint8))
      errors[imgName].append(regionErr)
      #errorNew.append(regionErr)
      #confidNew.append(confidRow)
      #print('errorNew:', concat(errorNew))
      #print('confidNew:', concat(confidNew))
print('error info:', errors)
with open('error_stats_%s.csv'%(os.path.basename(opts.counts).split('.')[0]), 'wt', newline='') as resultsFile:
  writer = csv.writer(resultsFile)
  allErrs = np.asarray(concat(list(errors.values())))
  print('allErrs:', allErrs)
  try:
    writer.writerow(['overall mean err:', np.mean(allErrs), 'overall max err:', np.max(allErrs)])
  except Exception as identifier:
    print('failed to calc overall mean and max err')
  for imgName in errors:
    writer.writerow([imgName])
    errArray = np.asarray(errors[imgName])
    writer.writerow(['mean err:', np.mean(errArray), 'max err:', np.max(errArray)])
    for errRow in errors[imgName]:
      writer.writerow(errRow)
# with open('error_vs_confid_%s.csv'%(os.path.basename(opts.counts).split('.')[0]), 'wt', newline='') as resultsFile:
#   writer = csv.writer(resultsFile)
#   writer.writerow(errorNew)
#   writer.writerow(confidNew)
print('total runtime:', timeit.default_timer() - start_t)
