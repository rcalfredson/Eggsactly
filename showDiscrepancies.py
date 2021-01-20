import argparse
from collections import Counter
import itertools
import os
from pathlib import Path

import cv2
import numpy as np

from chamber import CT, FourCircleChamber
from circleFinder import CircleFinder
from util import *

import pickle

MAX_DIFF = 5
TEXT_SIZE_LG = 1.4
TEXT_SIZE_SM = 0.9
MARGIN_HT = 25

def options():
  """Parse options for the egg labeling visual validation tool."""
  p = argparse.ArgumentParser(description='Output visual comparisons of egg-' +\
      'laying images with large disagreements in human-generated counts.')
  p.add_argument('files', help='pickle files of egg labels to compare',
    nargs="*")
  return p.parse_args()

opts = options()

pickleFiles = {}
counts = {}
for f in opts.files:
  parent = Path(f).parent
  if parent in pickleFiles:
    pickleFiles[parent].append(f)
  else:
    pickleFiles[parent] = [f]

for directory in pickleFiles:
  counts[directory] = {}
  for f in pickleFiles[directory]:
    counts[directory][f] = {}
    with open(f, 'rb') as openedF:
      loadedData = pickle.load(openedF)
    fd = loadedData['frontierData']
    imageNamesToChamberType = {}
    for i, fileName in enumerate(loadedData['clicks']):
      extension = '.%s'%('jpg' if 'jpg' in fileName.lower() else 'png')
      imageName = '%s%s'%(fileName.split(extension)[0], extension)
      imageNamesToChamberType[imageName] = loadedData['chamberTypes'][imageName]
    for imageName in imageNamesToChamberType:
      if imageNamesToChamberType[imageName] == CT.fourCircle.name:
        possibleKeys = concat([["%s_%i_%i_%s"%(imageName, combo[0], combo[1],
          pos) for combo in itertools.product(range(FourCircleChamber().numRows
          ), range(FourCircleChamber().numCols))] for pos in ('upper', 'lower',
          'left', 'right')])
      else:
        chamberClass = getattr(CT, imageNamesToChamberType[imageName]).value()
        possibleKeys = ["%s_%i_%i"%(imageName, combo[0], combo[1]) for combo in\
          itertools.product(range(chamberClass.numRows), range(
            chamberClass.numCols*2))]
      for j, key in enumerate(possibleKeys):
        try:
          if not fd['finishedLabeling'] and imageName == fd['fileName'] and \
              j >= fd['subImgIdx']:
            continue
          clicks = loadedData['clicks'][key]
          if imageName in counts[directory][f]:
            counts[directory][f][imageName].append(clicks)
          else:
            counts[directory][f][imageName] = [clicks]
        except Exception:
          pass

for directory in counts:
  pickleFilesByImg = {}
  numImgsChecked = 0
  discrepancyIndices = []
  maxAbsDiffs = []
  for pickleFile in counts[directory]:
    for imageName in counts[directory][pickleFile]:
      if imageName in pickleFilesByImg:
        pickleFilesByImg[imageName].append(pickleFile)
      else:
        pickleFilesByImg[imageName] = [pickleFile]
  pickleFilesByImg = dict((k, v) for k, v in pickleFilesByImg.items() \
    if len(v) > 1)
  for img in pickleFilesByImg:
    print('Checking img %s'%img)
    samples = [[len(regionClicks) for regionClicks in counts[directory
      ][pickleFile][img]] for pickleFile in pickleFilesByImg[img]]
    minNumSamples = min(len(regionsLabeled) for regionsLabeled in samples)
    for i, regionsLabeled in enumerate(samples):
      samples[i] = regionsLabeled[:minNumSamples]
    samples = np.array(samples)
    maxAbsDiffs.append(np.abs(np.max(samples, axis=0) - np.min(samples, axis=0)))
    flaggedIndices = np.where(maxAbsDiffs[-1] > MAX_DIFF)[0]
    if len(flaggedIndices) == 0:
      print('\tNo difference in counts above tolerance!')
    if len(flaggedIndices) > 0:
      flaggedImg = cv2.imread(os.path.join(directory, img))
      circleFinder = CircleFinder(flaggedImg, img)
      circles, avgDists, numRowsCols, rotatedImg, _ = circleFinder.findCircles()
      subImgs = circleFinder.getSubImages(rotatedImg, circles, avgDists,
        numRowsCols)[0]
      for flaggedIndex in flaggedIndices:
        imgCopies = []
        for i, pickleFile in enumerate(pickleFilesByImg[img]):
          imgCopies.append(cv2.resize(np.array(subImgs[flaggedIndex]), (0, 0),
            fx=2, fy=2))
          imgCopies[-1] = np.vstack((np.full((MARGIN_HT, imgCopies[-1].shape[1],
            3), 255, np.uint8), imgCopies[-1], np.full((MARGIN_HT, imgCopies[-1
            ].shape[1], 3), 255, np.uint8)))
          putText(imgCopies[-1], "%s: %i"%(
            pickleFile.split('_')[-1].split('.')[0].capitalize(), samples[i][flaggedIndex]),
            (5, 5), (0, 1), textStyle(size=TEXT_SIZE_LG, color=COL_BK,
            thickness=1))
          for annotation in counts[directory][pickleFile][img][flaggedIndex]:
            cv2.circle(imgCopies[-1], tuple([2*(int(el)) + (MARGIN_HT if j\
              else 0) for j, el in enumerate(annotation)]), 2, COL_Y,
              cv2.FILLED)
        imgCopies.append(cv2.resize(np.array(subImgs[flaggedIndex]), (0, 0),
            fx=2, fy=2))
        imgCopies[-1] = np.vstack((np.full((MARGIN_HT, imgCopies[-1].shape[1],
            3), 255, np.uint8), imgCopies[-1], np.full((MARGIN_HT, imgCopies[-1
            ].shape[1], 3), 255, np.uint8)))
        putText(imgCopies[-1], 'Reference', (5, 5), (0, 1), textStyle(
          size=TEXT_SIZE_LG, color=COL_BK, thickness=1))
        combinedImgs = np.hstack(tuple(imgCopies))
        indexInTask = numImgsChecked + flaggedIndex + 1
        taskImgText = "From %s; image # %i"%(os.path.splitdrive(directory)[1
          ].replace('\\', '/'), indexInTask)
        textWidth = cv2.getTextSize(taskImgText, cv2.FONT_HERSHEY_PLAIN,
          TEXT_SIZE_SM, 1)[0][::-1][1]
        putText(combinedImgs, taskImgText, (int(0.5*(combinedImgs.shape[1] - \
          textWidth)), combinedImgs.shape[0] - 20), (0, 1), textStyle(
          size=TEXT_SIZE_SM, color=COL_BK, thickness=1))
        cv2.imwrite('P:\\Egg images_9_3_2020\\click_comparisons\\' +\
          'diff_%s_idx_%i.jpg'%(img.split('.')[0], indexInTask), combinedImgs)
        discrepancyIndices.append(numImgsChecked + flaggedIndex)
        cv2.imshow('Img: %s. Index: %i'%(img, indexInTask), combinedImgs)
        print('\tLargest difference in number of labels: %i'%
          maxAbsDiffs[-1][flaggedIndex])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    numImgsChecked += samples.shape[-1]
  maxAbsDiffs = np.array(concat(maxAbsDiffs))
  print('\nSummary for directory %s:'%directory)
  print('Total images checked: %i'%numImgsChecked)
  print('Total images with difference in counts above tol.: %i'%len(
    discrepancyIndices))
  print('\tProportion: %.3f'%(len(discrepancyIndices) / numImgsChecked))
  print('Mean difference in counts, overall: %.2f'%np.mean(maxAbsDiffs))
  print('Mean difference in counts, above tol. only: %.2f'%np.mean(maxAbsDiffs[
    discrepancyIndices]))
