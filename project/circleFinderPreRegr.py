import itertools
import cv2
import timeit
from adet.config import get_cfg
from predictor import VisualizationDemo
from skimage.measure import label
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

from util import *

CONFIDENCE_THRESHOLD = 0.5
WINDOW_NAME = "COCO detections"

def setup_cfg():
  # load config from file and command-line arguments
  cfg = get_cfg()
  cfg.merge_from_file('./configs/arena_pit.yaml')
  cfg.merge_from_list(['MODEL.WEIGHTS', './models/arena_pit.pth'])
  # Set score_threshold for builtin models
  cfg.MODEL.RETINANET.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
  cfg.MODEL.FCOS.INFERENCE_TH_TEST = CONFIDENCE_THRESHOLD
  cfg.MODEL.MEInst.INFERENCE_TH_TEST = CONFIDENCE_THRESHOLD
  cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = \
      CONFIDENCE_THRESHOLD
  cfg.freeze()
  return cfg

def centroidnp(arr):
  length = arr.shape[0]
  sum_x = np.sum(arr[:, 0])
  sum_y = np.sum(arr[:, 1])
  return sum_x/length, sum_y/length

def getLargestCC(segmentation):
  labels = label(segmentation)
  if labels.max() == 0: # assume at least 1 CC
      return np.zeros(segmentation.shape).astype(np.float32)
  largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
  return np.asarray(largestCC).astype(np.float32)

# corner-finding (source: https://stackoverflow.com/a/20354078/13312013) 
def fake_image_corners(xy_sequence):
    """Get an approximation of image corners based on available data."""
    all_x, all_y = zip(*xy_sequence)
    min_x, max_x, min_y, max_y = min(all_x), max(all_x), min(all_y), max(all_y)
    d = dict()
    d['tl'] = min_x, min_y
    d['tr'] = max_x, min_y
    d['bl'] = min_x, max_y
    d['br'] = max_x, max_y
    return d

def corners(xy_sequence, image_corners):
    """Return a dict with the best point for each corner."""
    d = dict()
    d['tl'] = min(xy_sequence, key=lambda xy: distance(xy, image_corners['tl']))
    d['tr'] = min(xy_sequence, key=lambda xy: distance(xy, image_corners['tr']))
    d['bl'] = min(xy_sequence, key=lambda xy: distance(xy, image_corners['bl']))
    d['br'] = min(xy_sequence, key=lambda xy: distance(xy, image_corners['br']))
    return d
# end corner-finding

cfg = setup_cfg()
model = VisualizationDemo(cfg)

def getSubImages(img, centers, avgDists, numRowsCols):
  subImgs = []
  origins = []
  for center in centers:
    origins.append((max(center[0] - int(0.475*avgDists[0]), 0), max(center[1] - int(0.5*avgDists[1]), 0)))
    subImgs.append(img[origins[-1][1]:center[1] + int(0.5*avgDists[1]), origins[-1][0]: center[0] - int(0.18*avgDists[0])])
    origins.append((max(center[0] + int(0.18*avgDists[0]), 0), max(center[1] - int(0.5*avgDists[1]), 0)))
    subImgs.append(img[max(center[1] - int(0.5*avgDists[1]), 0):center[1] + int(0.5*avgDists[1]), max(center[0] + int(0.18*avgDists[0]), 0): center[0] + int(0.475*avgDists[0])])
  sortedSubImgs = []
  for j in range(numRowsCols[0]):
    for i in range(numRowsCols[1]):
      idx = numRowsCols[0]*2*i + 2*j
      sortedSubImgs.append(subImgs[idx])
      sortedSubImgs.append(subImgs[idx + 1])
  return sortedSubImgs, origins

class CircleFinder:
  def __init__(self, img):
      self.img = img
      self.skewed = None

  def findCircles(self):
    imageResized = cv2.resize(self.img, (0, 0), fx=0.15, fy=0.15,
      interpolation=cv2.INTER_CUBIC)
    predictions, visuals = model.run_on_image(imageResized)
    predictions = [predictions['instances'].pred_masks.cpu().numpy()[i, :, :] for i in range(predictions['instances'].pred_masks.shape[0])]
    centroids = [centroidnp(np.asarray(list(zip(*np.where(prediction == 1)))))\
      for prediction in predictions]
    centroids = [tuple(reversed(centroid)) for centroid in centroids]
    yDetections = np.asarray([centroid[1] for centroid in centroids])
    xDetections = np.asarray([centroid[0] for centroid in centroids])
    wellCoords = [[], []]
    for detI, detections in enumerate((xDetections, yDetections)):
      histResults = binned_statistic(detections, [], bins=40, statistic='count')
      binHtsOrig = histResults.statistic
      binClusters = trueRegions(binHtsOrig > 0)
      for trueRegion in binClusters:
        wellCoords[detI].append(int(round(np.mean([detections[(histResults.binnumber- 1 >= trueRegion.start) & (histResults.binnumber <= trueRegion.stop)]]))))

    for i in range(len(wellCoords)):
      wellCoords[i] = sorted(wellCoords[i])
      wellCoords[i] = reject_outliers_by_delta(np.asarray(wellCoords[i]))

    wells = list(itertools.product(wellCoords[0], wellCoords[1]))
    self.img = np.array(self.img)
    numRowsCols = [len(wellCoords[i]) for i in range(1, -1, -1)]
    diagDist = distance((0, 0), imageResized.shape[0:2])
    # cv2.imshow(WINDOW_NAME, visuals.get_image()[:, :, ::-1])
    # cv2.waitKey(0)
    for centroid in list(centroids):
      closestWell = min(wells, key=lambda xy: distance(xy, centroid))
      if distance(closestWell, centroid) > 0.02*diagDist:
        centroids.remove(centroid)
    for well in wells:
      closestDetection = min(centroids, key=lambda xy: distance(xy, well))
      if distance(closestDetection, well) > 0.02*diagDist:
        centroids.append(well)
    prelim_corners = fake_image_corners(centroids)
    true_corners = corners(centroids, prelim_corners)
    width_skew = abs((true_corners['tr'][0] - true_corners['tl'][0]) - (true_corners['br'][0] -\
      true_corners['bl'][0]))
    height_skew = abs((true_corners['br'][1] - true_corners['tr'][1]) - (true_corners['bl'][1] -\
      true_corners['tl'][1]))
    if height_skew / imageResized.shape[0] > 0.01 or width_skew / imageResized.shape[1] > 0.01:
      pass
      self.skewed = True
    else:
      self.skewed = False
    # print('height skew:', height_skew)
    # print('height skew proportion:', height_skew / imageResized.shape[0])
    # print('width skew:', width_skew)
    # print('width skew proportion:', width_skew / imageResized.shape[1])
    # print('is it skewed?', self.skewed)
    wells = [tuple(np.round(np.divide(well, 0.15)).astype(int)) for well in wells]
    for i in range(len(wellCoords)):
      wellCoords[i] = np.round(np.divide(wellCoords[i], 0.15)).astype(int)
    return (wells, [np.mean(np.diff(wellCoords[i])) for i in range(2)], numRowsCols)

def largest_within_delta_alt(X, k, delta):
  return np.where((k-delta < X) * (X < k))[0].max()

def reject_outliers(data, m=2):
  return data[abs(data - np.mean(data)) < m * np.std(data)]

def reject_outliers_by_delta(binCenters, m=1.2):
  diffs = np.diff(binCenters)
  outIdxs = list(range(len(binCenters)))
  idxs = np.squeeze(np.argwhere(~(abs(diffs - np.mean(diffs)) < m * np.std(diffs))))
  if idxs.shape == ():
    idxs = np.reshape(idxs, 1)
  for idx in idxs:
    if idx == 0:
      idxToRemove = idx
    elif idx == len(binCenters) - 2:
      idxToRemove = idx + 1
    # else:
    #   idxToRemove = idx if magnitudes[idx] < magnitudes[idx + 1] else idx + 1
    if np.mean(np.delete(diffs, idx)) * 1.5 > diffs[idx]:
      continue
    if idxToRemove in outIdxs:
      outIdxs.remove(idxToRemove)
  return binCenters[outIdxs]
