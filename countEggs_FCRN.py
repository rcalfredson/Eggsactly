import argparse
import glob
import os
import sys

import cv2
import numpy as np
from PIL import Image
import torch

from chamber import CT
from circleFinder import CircleFinder
from detectors.fcrn import model

def options():
    """Parse options for the automated egg counter."""
    p = argparse.ArgumentParser(
        description='Count eggs in the inputted images.')
    group = p.add_argument_group('inputs', 'Images, models, etc. For each of' +\
        ' these arguments, multiple paths can be separated by commas, and ' +\
        'each individual path supports wildcard patterns (via the glob ' +\
        'module)')
    group.add_argument('--img', help='Path to egg-laying image(s).')
    group.add_argument('--model', help='Path to egg-counting model(s).')
    group = p.add_argument_group('configuration', 'Script setup parameters.')
    group.add_argument('--arch', help='Network architecture to use', choices=[
        'FCRN_A', 'FCRN_B'])
    group = p.add_argument_group('flags', "ways to modify the script's" +\
        ' behavior.')
    group.add_argument('--allow_skew', action='store_true', help='try to ' +\
        'segment an image into egg-laying regions even if skew has been ' +\
        'detected in it. By default, the script exits with an error upon ' +\
        'reaching a skewed image.')
    return p.parse_args()

opts = options()

def splitAndGlob(paths):
    """Split an inputted string of comma-separated paths and glob each resulting
    element. Return all glob results as a list of depth 1.

    Arguments:
      - paths: string of comma-separated paths
    """
    paths = paths.split(',')
    globResults = []
    for path in paths:
        globResults += glob.glob(path)
    return globResults

modelPaths = splitAndGlob(opts.model)
imgPaths = splitAndGlob(opts.img)

subImgs = []
chamberTypes = []
for imgPath in imgPaths:
    continue
    img = np.array(Image.open(imgPath), dtype=np.float32)
    cf = CircleFinder(img, os.path.basename(imgPath), allowSkew=opts.allow_skew)
    chamberTypes.append(cf.ct)
    circles, avgDists, numRowsCols, rotatedImg, _ = cf.findCircles()
    subImgs.append(cf.getSubImages(rotatedImg, circles, avgDists, numRowsCols)[0])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

network = {
    'FCRN_A': model.FCRN_A,
    'FCRN_B': model.FCRN_B
}[opts.arch](input_filters=3, N=2).to(device)
network.train(False)
network = torch.nn.DataParallel(network)
predictions_by_model = []
for modelPath in modelPaths:
    network.load_state_dict(torch.load(modelPath))
    img = np.array(Image.open(imgPath), dtype=np.float32)
    img = torch.from_numpy((1/255)*np.expand_dims(np.moveaxis(img, 2, 0), 0))
    result = network(img)
    print('predicted counts:', torch.sum(result).item() / 100)
    for subImgList in subImgs:
        for i, subImg in enumerate(subImgList):
            # cv2.imshow('debug', (1/255)*np.moveaxis(subImg, 1, 0))
            subImg = torch.from_numpy((1/255)*np.expand_dims(np.moveaxis(subImg, 2, 0
                ), 0))
            result = network(subImg)
            # dMapToShow = result.cpu().detach().numpy()[0].T
            predicted_counts = torch.sum(result).item() / 100
            # cv2.imshow('density map', dMapToShow)
            print('predicted counts for image', i)
            print('on path %s:'%modelPath)
            print(predicted_counts)
            cv2.waitKey(0)

# next step: output a file
for i, ct in chamberTypes:
    CT[ct].value().writeLineFormatted(subImgs[])
