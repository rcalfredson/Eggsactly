import glob
import os
import pickle

import cv2
import numpy as np

validationProb = 0.25
matchPatchesToFullSize = False

def sortOneImg(imgPath):
    global GROUP
    if np.random.random() < validationProb:
        GROUP = 'valid'
    else:
        GROUP = 'train'
    imgPaths[GROUP].append(imgPath)

imgBasePath = os.path.join('P:\\', 'Robert', 'objects_counting_dmap', 'egg_source')
fullSizeImages = {'train': glob.glob(os.path.join(imgBasePath,
    'independent_fullsize_train', '*dots.png')),
    'valid':glob.glob(os.path.join(imgBasePath, 
    'independent_fullsize_valid', '*dots.png'))}
imgPaths = {'train': [], 'valid': []}
patchImages = glob.glob(os.path.join(imgBasePath,
    'train', '*dots.png')) +\
    glob.glob(os.path.join(imgBasePath, 
    'valid', '*dots.png'))
fullSizeToPatch = {}
for patchImg in patchImages:
    origImg = os.path.basename('_'.join(patchImg.split('_dots')[0].split('_')[:-1]))
    if origImg in fullSizeToPatch:
        fullSizeToPatch[origImg].append(patchImg)
    else:
        fullSizeToPatch[origImg] = [patchImg]
print('tried globbing this path', os.path.join(imgBasePath,
    'fullsize_train', '*dots.png'))
for group in ('train', 'valid'):
    if matchPatchesToFullSize:
        for fullSizeImage in fullSizeImages[group]:
            if np.random.random() < validationProb:
                GROUP = 'valid'
            else:
                GROUP = 'train'
            imgBaseName = os.path.basename(
                fullSizeImage).split('_dots')[0]
            pathToCheck = '_'.join(os.path.join(imgBasePath, group, imgBaseName).split('_')[:-1]) + "*dots.png"
            patchesForImage = fullSizeToPatch['_'.join(imgBaseName.split('_')[:-1])]
            print('path checked:', pathToCheck)
            print('full-size image:', fullSizeImage)
            print('patches for image?', patchesForImage)
            print('how many?', len(patchesForImage))
            imgPaths[GROUP].append(fullSizeImage)
            imgPaths[GROUP] += patchesForImage
    else:
        for img in fullSizeImages[group]:
            sortOneImg(img)
if not matchPatchesToFullSize:
    for img in patchImages:
            sortOneImg(img)

for group in ('train', 'valid'):
    with open('%sImgList.txt'%group, 'w') as f:
        for path in imgPaths[group]:
            f.write('%s\n'%path)


# for i, imgPath in enumerate(sourceImgs):
#     img = cv2.imread(imgPath)
#     circleFinder = CircleFinder(img, imgPath)
#     circles, avgDists, numRowsCols, rotatedImg, _ = circleFinder.findCircles()
#     subImgs[imgPath] = circleFinder.getSubImages(rotatedImg, circles, avgDists,
#                                                  numRowsCols)[0]
#     chamberTypes[imgPath] = circleFinder.ct
#     rowColCounts[imgPath] = numRowsCols
#     print('Finished processing', imgPath)
#     print('chamber type?', chamberTypes[imgPath])
# for imgPath in subImgs:
#     for i, subImg in enumerate(subImgs[imgPath]):
#         if np.random.random() < heldOutProb:
#             MODE = 'full'
#         else:
#             MODE = 'patch'
#             if np.random.random() < validationProb:
#                 GROUP = 'valid'
#             else:
#                 GROUP = 'train'
#         if chamberTypes[imgPath] == CT.fourCircle.name:
#             numCirclesPerRow = rowColCounts[imgPath][0]*4
#             rowNum = np.floor(i / numCirclesPerRow).astype(int)
#             colNum = np.floor((i % numCirclesPerRow) / 4).astype(int)
#             position = POSITIONS[i % 4]
#         else:
#             rowNum = int(np.floor(i / (2*rowColCounts[imgPath][1])))
#             colNum = i % int(2*rowColCounts[imgPath][1])
#             position = None
#         print('Viewing %s, sub-image' % imgPath, i)
#         print('rowNum:', rowNum)
#         print('colNum:', colNum)
#         print('total num rows/cols:', rowColCounts[imgPath])
#         savePatches(subImg, imgPath, rowNum, colNum, position)
