import argparse
import glob
import os
import pickle

import cv2
import numpy as np

from common import globFiles

# Note: this script was abandoned because its runtime is too long.

def options():
    """Parse options for image redundancy checker."""
    p = argparse.ArgumentParser(description='Check if a new prospective egg-' +
                                'counting image is already in use.')
    p.add_argument(
        'source', help='img path or a folder containing source images')
    return p.parse_args()


dirsOfImagesBeingUsed = ['P:/Egg images_9_3_2020', 'P:/Egg images_9_3_2020/askew',
                         'P:/Egg images_9_3_2020/images with increased optical zoom',
                         'P:/Egg images_9_3_2020/with_false_positives',
                         'P:/Egg images_9_3_2020/WT', 'P:/Egg images_9_3_2020/WT_1',
                         'P:/Egg images_9_3_2020/WT_2', 'P:/Egg images_9_3_2020/WT_3',
                         'C:/Users/Tracking/counting-3/imgs/Charlene/temp2',
                         'P:/Robert/counting-3/imgs/4-circle'
                         ]
pathsOfImagesBeingUsed = []
for d in dirsOfImagesBeingUsed:
    pathsOfImagesBeingUsed += glob.glob('%s/*.jpg'%d)
    pathsOfImagesBeingUsed += glob.glob('%s/*.png'%d)
print('paths of images?', pathsOfImagesBeingUsed)
print('how many?', len(pathsOfImagesBeingUsed))
opts = options()
freeImages = []
if os.path.isdir(opts.source):
    imagePaths = globFiles(opts.source, ext='jpg')
else:
    imagePaths = [opts.source]
print('image paths:', imagePaths)
existingImages = []
if not os.path.exists('existingImgs.pickle'):
    for i, existingImgPath in enumerate(pathsOfImagesBeingUsed):
        existingImages.append(cv2.imread(existingImgPath))
        print('opening image', i)
    with open('existingImgs.pickle', 'wb') as f:
        pickle.dump(existingImages, f)
else:
    pass
for path in imagePaths:
    counter = 0
    img = cv2.imread(path)
    imgName = os.path.basename(path)
    print('checking image', imgName)
    for existingImgPath in pathsOfImagesBeingUsed:
        existingImg = cv2.imread(existingImgPath)
        print('checkCount:', counter)
        counter += 1
        if not np.array_equal(existingImg, img):
            freeImages.append(img)

with open('unusedImages.txt', 'w') as f:
    [f.write('%s\n'%datum) for datum in freeImages]