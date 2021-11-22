import argparse, csv, os

import cv2
import pycocotools.coco as coco
from sklearn.cluster import AffinityPropagation
import numpy as np

from detectors.centernet.detector_factory import detector_factory

IMG_DIR = r'C:\\Users\\Tracking\\centernet-test\\data\\cluster\\val'

def options():
  """Parse options for the automated cluster counter."""
  p = argparse.ArgumentParser(description=
    'Count the clusters in each inputting heatmap image.')
  p.add_argument('--record', help='path to COCO file containing cluster ' +
    'bounding-box data for the heatmaps to analyze')
  p.add_argument('--img', help='path to a single heatmap image to analyze')
  p.add_argument('--gpus', default='0', 
                             help='-1 for CPU, use comma for multiple gpus')
  p.add_argument('--arch', default='dla_34', 
                             help='model architecture. Currently tested: '
                                  'res_18 | res_101 | resdcn_18 | resdcn_101 |'
                                  'dlav0_34 | dla_34 | hourglass')
  p.add_argument('--load_model', default='',
                             help='path to pretrained model')
  p.add_argument('--dataset', default='cluster',
                             help='cluster | coco | kitti | coco_hp | pascal')
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

def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")

if __name__ == '__main__':
  performance_results = [['filename', 'num labelled', 'num predicted',
    'abs. error', 'pct. error']]
  opts = options()
  Detector = detector_factory['ctdet']
  detector = Detector(opts)
  if opts.record:
    parsed = coco.COCO(opts.record)
    for i, imgId in enumerate(parsed.imgs):
      file_name = parsed.imgs[imgId]['file_name']
      img = cv2.imread(os.path.join(IMG_DIR, file_name))
      print('processing image at', os.path.join(IMG_DIR, file_name))
      run_dict = detector.run(img)
      good_results = [result for result in run_dict['results'][1] if result[-1] > 0.3]
      boxes = np.array([result[0:4] for result in good_results])
      nonmax_suppressed = non_max_suppression_fast(boxes, 0.4)
      num_predicted = len(nonmax_suppressed)
      num_labelled = len([parsed.loadAnns(ids=[annID]) for annID in parsed.getAnnIds(imgIds=[imgId])])
      #print('num predicted: %i | num labelled: %i'%(num_predicted, num_labelled))
      abs_diff = abs(num_labelled - num_predicted)
      rel_error = abs_diff / num_labelled if num_labelled > 0 else ''
      performance_results.append([file_name, num_labelled, num_predicted, abs_diff,
          rel_error*(1 if type(rel_error) is str else 100)])
      print('performance results?', performance_results)
      # print(run_dict['results'])
      # # centerPoints = np.array([[0.5*(result[0] + result[2]), 0.5*(result[1] + result[3])] for result in good_results])
      # # print('centerPoints?', centerPoints)
      # # print(len(centerPoints))
      # boxes = np.array([result[0:4] for result in good_results])
      # nonmax_suppressed = non_max_suppression_fast(boxes, 0.4)
      # print('nonmax suppressed?', nonmax_suppressed)
      # # if len(centerPoints) > 1:
      # #   clustering = AffinityPropagation().fit(centerPoints)
      # #   print('clustering', clustering.labels_)
      # #   print(clustering.cluster_centers_)
      # for result in good_results:
      #   cv2.rectangle(img, (int(result[0]), result[1]), (int(result[2]), int(result[3])), (0, 0, 255), 3)
      # cv2.imshow('debug', img)
      # cv2.waitKey(0)
    with open('cluster_counts.csv', 'wt', newline='') as resultsFile:
        writer = csv.writer(resultsFile)
        writer.writerows(performance_results)
  else:
    img = cv2.imread(opts.img)
    run_dict = detector.run(img)
    good_results = [result for result in run_dict['results'][1] if result[-1] > 0.3]
    boxes = np.array([result[0:4] for result in good_results])
    good_results = non_max_suppression_fast(boxes, 0.4)
    for result in good_results:
      cv2.rectangle(img, (int(result[0]), result[1]), (int(result[2]), int(result[3])), (0, 0, 255), 3)
    cv2.imshow('debug', img)
    cv2.waitKey(0)
