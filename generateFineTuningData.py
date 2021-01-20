import os
import pickle

import cv2
import numpy as np
from objects_dmap import data_loader
from circleFinder import CircleFinder
from clickLabelManager import ClickLabelManager
from common import randID
from chamber import CT

DEST_PATH = 'P:/Robert/objects_counting_dmap/egg_source'
MODE = ['full', 'patch'][0]
POSITIONS = ('upper', 'right', 'lower', 'left')
SIDE_LENGTH = 160
NUM_IMAGES_WITH_ANNOTS = {'count': 0}

FINE_TUNE = False


def numPatches():
    if MODE == 'patch':
        return 20
    else:
        return 1


heldOutProb = 0.1

sourceImgDirs = [
    'C:/Users/Tracking/counting-3/imgs/Charlene/temp2',
    'P:/Robert/counting-3/imgs/4-circle']
picklePaths = ['%s/egg_labels_robert.pickle' % sourceImgDir for sourceImgDir in
               sourceImgDirs]

clickLabelManager = ClickLabelManager()
sourceImgs = set()

# need filenames from both
# the click file and the training/validation dataset, because the split has to be
# identical.
# the only way to find out which is which:
# open the HDF5 files for both datasets, iterate through all examples,
# and store the filenames.
# how to know which folder each image is in at that point?

# start from scratch: what am I trying to do?
# 1. Open HDF5 files and determine the original egg-laying areas/images in each dataset
# 2. Create new training and validation datasets consisting of those full-sized
#    egg-laying areas.

# the goal is different enough to justify making a new script and then refactoring whatever is duplicated.
basePath = 'P:\\Robert\\objects_counting_dmap\\egg'

datasetPaths = {'training': os.path.join(basePath, 'train.h5'),
                'validation': os.path.join(basePath, 'valid.h5')}
imagePaths = {'training': [], 'validation': []}

for p in datasetPaths:
    dataset = data_loader.H5Dataset(datasetPaths[p])
    print('images in loaded dataset?', dataset.images)
    input('enter...')

exit(0)

for i, picklePath in enumerate(picklePaths):
    with open(picklePath, 'rb') as f:
        loadedData = pickle.load(f)
    for clickKey in loadedData['clicks']:
        clickLabelManager.clicks[clickKey] = loadedData['clicks'][clickKey]
        potentialName = '%s.jpg' % clickKey.split('.jpg')[0]
        if potentialName not in sourceImgs:
            sourceImgs.add('%s/%s' % (sourceImgDirs[i], potentialName))

badImgs = [('%s.jpg' % imgName).lower() for imgName in ('Apr5_2left', 'Apr5_2right', 'Apr5_3left',
                                                        'Apr5_3right', 'Apr5_5left', 'Apr5_5right', 'Apr5_9left', 'Apr5_9right',
                                                        'Apr7_10left', 'Apr7_10right', 'Apr7_2left', 'Apr7_2right', 'Apr7_3left',
                                                        'Apr7_9right', 'Apr7_9left', 'Apr7_3right')]

for sourceImg in list(sourceImgs):
    if os.path.basename(sourceImg) in badImgs:
        sourceImgs.remove(sourceImg)

print('sourceImgs:', sourceImgs)

subImgs = dict()
chamberTypes = dict()
rowColCounts = dict()

samplePosition = {'x': 0, 'y': 0}


def saveClickPositionImage(imgPath, rowNum, colNum, x1, y1, outputPath, subImg, position=None):
    subImageKey = clickLabelManager.subImageKey(
        imgPath, rowNum, colNum, position)
    if not subImageKey in clickLabelManager.clicks:
        print('did not find key', subImageKey, 'in click keys')
        print('sample of collection of keys:', list(
            clickLabelManager.clicks.keys())[:10])
        input('enter...')
        return
    clicks = clickLabelManager.clicks[subImageKey]
    clicksInRange = tuple([tuple([int(coord) - (y1 if i else x1) for i, coord in enumerate(click)]) for click in clicks if
                           MODE == 'full' or (click[0] > x1 and click[0] < x1+SIDE_LENGTH and click[1] > y1 and click[1] < y1+SIDE_LENGTH)])
    if len(clicksInRange) == 0:
        rows, cols = (), ()
    else:
        rows, cols = zip(*clicksInRange)
    # print('bounds of the patch:', x1, x1+SIDE_LENGTH,
    #       'for x and', y1, y1+SIDE_LENGTH, 'for y.')
    print('clicks in range:', rows, cols)
    if MODE == 'patch':
        clickArray = np.zeros((SIDE_LENGTH, SIDE_LENGTH, 3))
    else:
        clickArray = np.zeros(subImg.shape)
    clickArray[cols, rows] = (0, 0, 255)
    # cv2.imshow('clicks', clickArray)
    cv2.imwrite(outputPath, clickArray)
    print('wrote image to this path:', outputPath)


def savePatches(subImg, imgPath, rowNum, colNum, position=None):
    for _ in range(numPatches()):
        if MODE == 'patch':
            x1 = np.random.randint(0, subImg.shape[1] - SIDE_LENGTH)
            y1 = np.random.randint(0, subImg.shape[0] - SIDE_LENGTH)
            patch = subImg[y1:y1+SIDE_LENGTH, x1:x1+SIDE_LENGTH]
        else:
            x1, y1 = 0, 0
            patch = np.array(subImg)

        imgId = randID()
        if MODE == 'full':
            destPath = os.path.join(DEST_PATH, 'heldout')
        else:
            destPath = DEST_PATH
        imgPath = os.path.basename(imgPath)
        imgBaseName = '%s_%i_%i_%s%s' % (
            os.path.basename(imgPath.split('.jpg')[0]), rowNum, colNum,
            '' if position == None else '%s_' % position,
            imgId)
        imgDest = os.path.join(destPath, '%s.jpg' % imgBaseName)
        dotsDest = os.path.join(destPath, '%s_dots.png' % imgBaseName)
        saveClickPositionImage(imgPath, rowNum, colNum,
                               x1, y1, dotsDest, subImg, position)
        # cv2.imshow('patch', patch)
        # cv2.waitKey(0)
        if not os.path.exists(dotsDest):
            return
        NUM_IMAGES_WITH_ANNOTS['count'] += 1
        cv2.imwrite(imgDest, patch)


for i, imgPath in enumerate(sourceImgs):
    img = cv2.imread(imgPath)
    circleFinder = CircleFinder(img, imgPath)
    circles, avgDists, numRowsCols, rotatedImg, _ = circleFinder.findCircles()
    subImgs[imgPath] = circleFinder.getSubImages(rotatedImg, circles, avgDists,
                                                 numRowsCols)[0]
    chamberTypes[imgPath] = circleFinder.ct
    rowColCounts[imgPath] = numRowsCols
    print('Finished processing', imgPath)
    print('chamber type?', chamberTypes[imgPath])
for imgPath in subImgs:
    for i, subImg in enumerate(subImgs[imgPath]):
        if chamberTypes[imgPath] == CT.fourCircle.name:
            numCirclesPerRow = rowColCounts[imgPath][0]*4
            rowNum = np.floor(i / numCirclesPerRow).astype(int)
            colNum = np.floor((i % numCirclesPerRow) / 4).astype(int)
            position = POSITIONS[i % 4]
        else:
            rowNum = int(np.floor(i / (2*rowColCounts[imgPath][1])))
            colNum = i % int(2*rowColCounts[imgPath][1])
            position = None
        print('Viewing %s, sub-image' % imgPath, i)
        print('rowNum:', rowNum)
        print('colNum:', colNum)
        print('total num rows/cols:', rowColCounts[imgPath])
        savePatches(subImg, imgPath, rowNum, colNum, position)
