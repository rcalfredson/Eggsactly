import argparse, glob, json, os
import cv2
from common import globFiles
from circleFinder import CircleFinder, getSubImages, rotate_around_point_highperf
from circleFinderPreRegr import CircleFinder as CircleFinderOld, getSubImages as getSubImagesOld
from pycocotools import coco, mask
from PIL import Image, ImageDraw
from skimage import measure
import numpy as np
from util import distance

CAT_TO_ID = dict(egg=1, blob=2)
CAT_TO_COLOR = dict(egg='#f00562', blob='#d63526')

def options():
  """Parse options for sub-image detector."""
  p = argparse.ArgumentParser(description=
    'Find chambers within an egg-laying image and write each one as a ' +
    'separate image.')
  p.add_argument('source', help='folder containing source images')
  p.add_argument('dest', help='folder to which to write images')
  return p.parse_args()

# with open('C:\\Users\\Tracking\\AdelaiDet\\datasets\\coco_backup\\coco-1598046103.4454343.json') as f:
#     blobData = json.load(f)

# with open('C:\\Users\\Tracking\\AdelaiDet\\datasets\\coco_backup\\coco-1598478182.5108423.json') as f:
#     eggData = json.load(f)

blobData = coco.COCO('C:/Users/Tracking/Downloads/blob_labels.json')
eggData = coco.COCO('C:/Users/Tracking/Downloads/egg_labels.json')
# fileNamesToIds = dict(zip([imgVal['file_name'] for imgVal in eggData.imgs.values()], eggData.imgs.keys()))
fileNamesToIds = dict()
for i, img in enumerate(eggData.imgs.values()):
  fileNamesToIds[img['file_name']] = img['id']
opts = options()
newEgg, newBlob = dict(annotations=[], categories=[], images=[]), dict(annotations=[], categories=[], images=[])
instanceIds = dict(blob=0, egg=0)
# numPerImageWithAnnotations = [0]

def addAnnotationsForImage(tp, imgIdx):
  annotationsForImage = (blobData if tp == 'blob' else eggData).imgToAnns[
    fileNamesToIds[os.path.basename(writePath)]]
  # print(tp, 'annotations for image?', annotationsForImage)
  if len(annotationsForImage) == 0: return
  # cv2.imshow('source', cv2.resize(img, (0, 0), fx=0.15, fy=0.15))
  # cv2.imshow('template', originalImg)
  # cv2.waitKey(0)
  res = cv2.matchTemplate(imgRotated, originalImg, cv2.TM_CCOEFF_NORMED)
  min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
  # threshold = 0.9
  # loc = np.where(res >= threshold)
  # pts = list(zip(*loc[::-1]))
  # print('how many pts over the threshold?', len(pts))
  # print('pts:', pts)
  # closestOrigin = min(pts, key=lambda xy: distance(xy, origins[i]))
  top_left = max_loc
  # top_left = closestOrigin
  bottom_right = (top_left[0] + originalImg.shape[1], top_left[1] + originalImg.shape[0])
  for i, annot in enumerate(annotationsForImage):
    # old location of the image: top_left
    # new location of the image: origins[i]
    # print('bbox data?', annot['bbox'])
    # print('image being analyzed:', os.path.basename(writePath))
    # print('the new origin is', origins[imgIdx])
    # print('the old origin is:', top_left)
    # input('enter...')
    # print('what image is this?')
    # cv2.imshow('original img', originalImg)
    # cv2.imshow('debug new img', newSubImages[imgIdx])
    # cv2.waitKey(0)
    # what to subtract from what?
    # example: new point (origins) is (4, 9) and old point, top_left, is (3, 10)
    # the bounding box need to move to the left, so this means to subtract origins from top_left
    # real example: original bbox coords are [64.0, 253.0, 53.0, 107.0]
    # the old origin is: (588, 966); new origin is (613, 914)
    # therefore, bbox needs to move to the left.
    # bbox X coord = X coord + (old_origin - new_origin)
    # print('new x coordinate')
    # annotationsForImage[i]['bbox'] = [annot['bbox'][0] + (top_left[0] - origins[imgIdx][0]), annot['bbox'][1] + (top_left[1] - origins[imgIdx][1])] + annot['bbox'][2:]
    translatedSegmentations = []
    for j, polygon in enumerate(annot['segmentation']):
      # if i == 0 and j == 0 and tp == 'egg':
        # numPerImageWithAnnotations[0] += 1
      # print('x values of seg:', polygon[::2][:10])
      # print('y values of seg:', polygon[1::2][:10])
      # print('original segmentation:', polygon[:20])
      # xTranslated = np.add(polygon[::2], top_left[0] - origins[imgIdx][0])
      # yTranslated = np.add(polygon[1::2], top_left[1] - origins[imgIdx][1])
      # print('original points:', polygon[:24])
      # subtract distance between center and corner of sub-image
      # if corner of sub-image is left of center, then the point will be negative
      xTranslated = np.subtract(polygon[::2], (image_center[0] - top_left[0]))
      yTranslated = np.subtract(polygon[1::2], (image_center[1] - top_left[1]))
      translatedSegmentation = [val for pair in zip(xTranslated, yTranslated) for val in pair]
      # print('polygons with respect to global origin:', translatedSegmentation[:24])
      # input('enter...')
      polygonOrderedPairs = [translatedSegmentation[j:j + 2] for j in range(0, len(translatedSegmentation), 2)]
      # rotatedOrigin = rotate_around_point_highperf((image_center[0] - origins[imgIdx][0], image_center[1] - origins[imgIdx][1]), -rotationAngle)
      for k, pt in enumerate(polygonOrderedPairs):
        # print('point:', pt)
        # polygonOrderedPairs[k] = rotate_around_point_highperf(pt, rotationAngle)
        # print('pt after rotation:', polygonOrderedPairs[k])
        polygonOrderedPairs[k][0] += image_center[0] - origins[imgIdx][0]
        polygonOrderedPairs[k][1] += image_center[1] - origins[imgIdx][1]
        # print('pt after translation to origin of the sub-image:', polygonOrderedPairs[k])
        # input('enter...')
      translatedSegmentation = [item for sublist in polygonOrderedPairs for item in sublist]
      minXVal, maxXVal = [method(translatedSegmentation[::2]) for method in (min, max)]
      minYVal, maxYVal = [method(translatedSegmentation[1::2]) for method in (min, max)]
      # print('after rotating and translating to origin of new sub-image:', translatedSegmentation[:24])
      # print('min and max for x and y:', minXVal, maxXVal, minYVal, maxYVal)
      # input('enter...')
      annotationsForImage[i]['bbox'] = [minXVal, minYVal, maxXVal - minXVal, maxYVal - minYVal]
      translatedSegmentations.append(translatedSegmentation)
    annot['segmentation'] = translatedSegmentations
    # convert the mask into sets of points
    # polygonOrderedPairs = [annot[j:j + 2] for j in range(0, len(annot), 2)]
    # annot['segmentation'][i] = rotate_around_point_highperf
    annotationsForImage[i]['image_id'] = imgObject['id']
    (newBlob if tp == 'blob' else newEgg)['annotations'].append(annot)
    # blankImg = Image.new("L", tuple(reversed(originalImg.shape[0:2])), 0)
    # ImageDraw.Draw(blankImg).polygon([int(el) for el in annot[
    #   'segmentation'][0]], outline=1, fill=1)
    # if len(annot['segmentation']) > 1:
    #   for seg in annot['segmentation'][1:]:
    #     ImageDraw.Draw(blankImg).polygon([int(el) for el in seg], outline=0, fill=0)
    # reconstructedMask = np.array(blankImg)
    # # pad or crop the left and upper first, and then crop the right and bottom?
    # if origins[i][0] > top_left[0]:
    #   # mask dimensions: first width, then height
    #   reconstructedMask = reconstructedMask[:, origins[i][0] - top_left[0]:]
    # elif origins[i][0] < top_left[0]:
    #   reconstructedMask = np.pad(reconstructedMask, ((
    #     0, 0), (top_left[0] - origins[i][0], 0)), 'constant')
    # if origins[i][1] > top_left[1]:
    #   # mask dimensions: first width, then height
    #   reconstructedMask = reconstructedMask[origins[i][1] - top_left[1]:, :]
    # elif origins[i][1] < top_left[1]:
    #   reconstructedMask = np.pad(reconstructedMask, ((top_left[1] - origins[i][1], 0), (0, 0)), 'constant')
    # # crop mask to dimensions of the new region
    # reconstructedMask = reconstructedMask[:newSubImages[i].shape[0], :newSubImages[i].shape[1]]
    # # crop the mask and convert it back to a polygon
    # fortran_ground_truth_binary_mask = np.asfortranarray(reconstructedMask)
    # encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
    # ground_truth_area = mask.area(encoded_ground_truth)
    # ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
    # contours = measure.find_contours(reconstructedMask, 0.5)
    # annotation = {
    #   "segmentation": [],
    #   "metadata": {},
    #   "area": ground_truth_area.tolist(),
    #   "iscrowd": False,
    #   "isbbox": False,
    #   "image_id": imgId + imgIdOffset,
    #   "bbox": ground_truth_bounding_box.tolist(),
    #   "category_id": CAT_TO_ID[tp],
    #   "id": instanceIds[tp],
    #   "color": annot['color']
    # }
    # instanceIds[tp] += 1
    # for contour in contours:
    #   contour = np.flip(contour, axis=1)
    #   segmentation = contour.ravel().tolist()
    #   annotation["segmentation"].append(segmentation)
    # if len(annotation['segmentation']) == 0:
    #   pass
    # (newBlob if tp == 'blob' else newEgg)['annotations'].append(annotation)

imgs = globFiles(opts.source, ext='jpg')
newBlob['categories'].append({'id': CAT_TO_ID['blob'],
  'name': 'blob', 'supercategory': "", 'color': CAT_TO_COLOR['blob'],
  'metadata': {}, 'keypoint_colors': []})
newEgg['categories'].append({'id': CAT_TO_ID['egg'],
  'name': 'egg', 'supercategory': "", 'color': CAT_TO_COLOR['egg'],
  'metadata': {}, 'keypoint_colors': []})
for imgPath in imgs:
  print('checking image', imgPath)
  # numPerImageWithAnnotations[0] = 0
  annotationsForImage = None
  img = cv2.imread(imgPath)
  image_center = tuple(np.array(img.shape[1::-1]) / 2)
  imgName = os.path.basename(imgPath).split('.')[0]
  # circlesOrig, avgDistsOrig, numRowsColsOrig = CircleFinderOld(img).findCircles()
  circles, avgDists, numRowsCols, imgRotated, rotationAngle = CircleFinder(img, imgName).findCircles()
  # origSubImages, oldOrigins = getSubImagesOld(img, circlesOrig, avgDistsOrig, numRowsColsOrig)
  newSubImages, origins = getSubImages(imgRotated, circles, avgDists, numRowsCols)
  for i in range(len(newSubImages)):
    # print('numPerImage:', numPerImageWithAnnotations[0])
    # if i > 1: continue
    # after validation, erase these early skips
    writePath = os.path.join(opts.dest, '%s_%i.jpg'%(imgName, i))
    # only update the count if there were actually annotations on the image
    print('saving sub-image to %s'%writePath)
    print('dimensions of rotated image:', newSubImages[i].shape)
    if os.path.basename(writePath) in fileNamesToIds:
      imgObject = eggData.imgs[fileNamesToIds[os.path.basename(writePath)]]
      imageData = {'id': imgObject['id'], 'path': imgObject['path'],
          'height': imgRotated.shape[0], 'width': imgRotated.shape[1],
          'file_name': imgObject['file_name'],
          'annotated': False, 'annotating': [], 'num_annotations': 0,
          'metadata': {}, 'deleted': False, 'milliseconds': 0, 'events': [],
          'regenerate_thumbnail': False}
      newBlob['images'].append(imageData)
      newEgg['images'].append(imageData)
      originalImg = cv2.imread(os.path.join('C:\\Users\\Tracking\\coco-annotator\\datasets\\eggs', os.path.basename(writePath)))
      addAnnotationsForImage('blob', i)
      addAnnotationsForImage('egg', i)
    cv2.imwrite(writePath, newSubImages[i])

with open(os.path.join(opts.dest, 'transformedBlobLabels.json'), 'w') as f:
  json.dump(newBlob, f, ensure_ascii=False, indent=4)
with open(os.path.join(opts.dest, 'transformedEggLabels.json'), 'w') as f:
  json.dump(newEgg, f, ensure_ascii=False, indent=4)