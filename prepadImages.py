import glob
import os

import cv2
import numpy as np
from PIL import Image
import torch

from common import background_color

baseDir = "P:\\Robert\\objects_counting_dmap\\egg_source"
dirs = [os.path.basename(p) for p in glob.glob(os.path.join(baseDir,
    'prepadded', '*'))]
print('dirs?', dirs)
# find largest dimensions
# largestDims = {'height': 710, 'width': 445}
determineDims = False
maxWidth, maxHt = 445, 710
if determineDims:
    for d in dirs:
        if d in ('train', 'valid'): continue
        imgs = glob.glob(os.path.join(baseDir, d, '*dots.png'))
        print('num imgs in %s: %i'%(d, len(imgs)))
        for imgName in imgs:
            img = cv2.imread(os.path.join(baseDir, d, imgName))
            print('just read img')
            if img.shape[0] > largestDims['height']:
                largestDims['height'] = img.shape[0]
            if img.shape[1] > largestDims['width']:
                largestDims['width'] = img.shape[1]
    print('largest height and width:', largestDims)
for d in dirs:
    imgs = glob.glob(os.path.join(baseDir, d, '*dots.png'))
    asymmetryCorrs = {'vp': 0, 'hp': 0}
    for imgName in imgs:
        splitByDots = imgName.split('_dots')
        dotsName = imgName
        photoName = imgName.split('_dots')[0] + '.jpg'
        print('trying to open:', os.path.join(baseDir, d, photoName))
        img = cv2.imread(os.path.join(baseDir, d, photoName))
        if img.shape[1] < maxWidth:
            dividend = (maxWidth - img.shape[1]) / 2
            asymmetryCorrs['hp'] = 1 if dividend % 1 > 0 else 0
            hp = int(dividend)
        else:
            hp = 0
        if img.shape[0] < maxHt:
            dividend = (maxHt - img.shape[0]) / 2
            asymmetryCorrs['vp'] = 1 if dividend % 1 > 0 else 0
            vp = int(dividend)
        else:
            vp = 0
        padding = (hp, hp + asymmetryCorrs['hp'], vp, vp + asymmetryCorrs['vp'])
        # padded = np.multiply(img, 255).astype(np.uint8)
        bckgnd = background_color(img)
        print(type(img[0, 0, 0]))
        print('dimensions of tensor from image:', torch.Tensor(img).shape)
        print('background:', bckgnd)
        padded = torch.nn.functional.pad(torch.Tensor(img), padding, mode='constant', value=list(bckgnd))
        cv2.imwrite(os.path.join(baseDir, 'prepadded', d, photoName), padded)
        img = cv2.imread(os.path.join(baseDir, d, dotsName))
        padded = torch.nn.functional.pad(torch.Tensor(img), padding, mode='constant', value=0)
        cv2.imwrite(os.path.join(baseDir, 'prepadded', d, dotsName), padded)

