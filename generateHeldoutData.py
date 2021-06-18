import json
import os
from pathlib import Path
import pickle

import cv2
import numpy as np

from circleFinder import CircleFinder
from clickLabelManager import ClickLabelManager
from common import randID
from chamber import CT

DEST_PATH = 'P:/Robert/objects_counting_dmap/egg_source'
MODE = 'full'
MODE_FOR_FULL = 'heldout'
GROUP = 'train'
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
fullSizeProb = 0.2
validationProb = 0.25
backupCounter = 0

# for group in ('train', 'valid'):
#     def mkDestDir(dirName):
#         Path(os.path.join(DEST_PATH, dirName)).mkdir(parents=True, exist_ok=True)
#     mkDestDir('fullsize_%s'%group)
#     mkDestDir('independent_fullsize_%s'%group)
#     mkDestDir(group)
#     mkDestDir('heldout')

sourceImgDirs = [
    'P:/Egg images_9_3_2020/WT_5'
]
picklePaths = ['%s/egg_count_labels_robert.pickle' % sourceImgDir for sourceImgDir in
               sourceImgDirs]

clickLabelManager = ClickLabelManager()
sourceImgs = set()
imagesAlreadyChecked = []

# with open('regionsToSkip.json', 'r') as f:
#     regionsToSkip = json.load(f)
# regionsToSkip = dict((k.lower(), v) for k,v in regionsToSkip.items())
regionsToSkip = dict()

for i, picklePath in enumerate(picklePaths):
    clickCounts = {}
    clickCounter = 0
    with open(picklePath, 'rb') as f:
        loadedData = pickle.load(f)
    for clickKey in loadedData['clicks']:
        img = clickKey.split('.jpg')[0].lower()
        if img in clickCounts:
            clickCounts[img] += 1
        else:
            clickCounts[img] = 0
        clickCounter += 1
        # skips frontier image
        if not loadedData['frontierData']['finishedLabeling'] and img in \
                loadedData['frontierData']['fileName'] and \
                loadedData['frontierData']['subImgIdx'] <= clickCounts[img]:
            continue
        if img in regionsToSkip and clickCounts[img] in regionsToSkip[img]:
            continue
        clickLabelManager.clicks[clickKey] = loadedData['clicks'][clickKey]
        print('made past checks; image name is: %s'%clickKey)
        print('clickCounts:', clickCounts[img])
        print('img:', img)
        print('regionsToSkip:', regionsToSkip)
        print('frontier information:', loadedData['frontierData']['fileName'], loadedData['frontierData']['subImgIdx'])
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
input('enter to continue...')

subImgs = dict()
chamberTypes = dict()
rowColCounts = dict()

samplePosition = {'x': 0, 'y': 0}


def saveClickPositionImage(patch, outputPath):
    subImageKey = clickLabelManager.subImageKey(
        os.path.basename(patch.imgPath), patch.rowNum, patch.colNum,
        patch.position)
    if not subImageKey in clickLabelManager.clicks:
        print('did not find key', subImageKey, 'in click keys')
        # print('sample of collection of keys:', list(
            # clickLabelManager.clicks.keys()))
        return
    x1, y1 = patch.xOffset, patch.yOffset
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


def saveSinglePatch(destPath, patch):
    imgDest = os.path.join(destPath, '%s.jpg' % patch.basename)
    dotsDest = os.path.join(destPath, '%s_dots.png' % patch.basename)
    saveClickPositionImage(patch, dotsDest)
    if not os.path.exists(dotsDest):
        return
    NUM_IMAGES_WITH_ANNOTS['count'] += 1
    cv2.imwrite(imgDest, patch.sampleArea)

class Patch:
    def __init__(self, imgPath, sourceImg, rowNum, colNum, position=None):
        self.imgPath = imgPath
        self.sourceImg = sourceImg
        if MODE == 'patch':
            self.xOffset = np.random.randint(0, subImg.shape[1] - SIDE_LENGTH)
            self.yOffset = np.random.randint(0, subImg.shape[0] - SIDE_LENGTH)
            self.sampleArea = sourceImg[self.yOffset:self.yOffset+SIDE_LENGTH,
                self.xOffset:self.xOffset+SIDE_LENGTH]
        else:
            self.xOffset, self.yOffset = 0, 0
            self.sampleArea = np.array(sourceImg)
        self.rowNum = rowNum
        self.colNum = colNum
        self.position = position
        self.basename = '%s_%i_%i_%s%s' % (
            os.path.basename(imgPath.split('.jpg')[0]), rowNum, colNum,
            '' if position == None else '%s_' % position,
            randID())


def savePatches(subImg, imgPath, rowNum, colNum, position=None):
    global MODE
    for _ in range(numPatches()):
        patch = Patch(imgPath, subImg, rowNum, colNum, position)
        if MODE == 'full':
            if MODE_FOR_FULL == 'heldout':
                destPath = os.path.join(DEST_PATH, "%s_robert_WT_5"%MODE_FOR_FULL)
            else:
                destPath = os.path.join(DEST_PATH, '%s_fullsize_%s'%(
                    MODE_FOR_FULL, GROUP))
        else:
            destPath = os.path.join(DEST_PATH, GROUP)
        saveSinglePatch(destPath, patch)
    if MODE == 'patch':
        MODE = 'full'
        patch = Patch(imgPath, subImg, rowNum, colNum, position)
        destPath = os.path.join(DEST_PATH, 'fullsize_%s'%GROUP)
        saveSinglePatch(destPath, patch)
        MODE = 'patch'

def assignToTrainingOrValidation():
    global GROUP
    if np.random.random() < validationProb:
        GROUP = 'valid'
    else:
        GROUP = 'train'    

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
        if not loadedData['frontierData']['finishedLabeling'] and \
                imgPath in loadedData['frontierData']['fileName'] and \
                i >= loadedData['frontierData']['subImgIdx']:
            continue
        if chamberTypes[imgPath] == CT.large.name:
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
