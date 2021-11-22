import argparse, csv, os
import cv2, numpy as np

from common import globFiles

def options():
  """Parse options for CSV object-count exporter."""
  p = argparse.ArgumentParser(description=
    'Export a list of all images in a given directory without square dimensions.')
  p.add_argument('-d', dest='dir', help='directory containing .png images')
  return p.parse_args()

def findNonSquareImgs(imgDir):
    nonUniforms = []
    imgPaths = globFiles(imgDir)
    for path in imgPaths:
        img = cv2.imread(path)
        if img.shape[0] != img.shape[1]:
            print('found nonuniform img:', path, 'with shape', img.shape)
        else:
            continue
        # row means height.
        if img.shape[0] < img.shape[1]: # deficient height
            padding = ((1, 0), (0, 0), (0, 0))
        else:
            padding = ((0, 1), (0, 0), (0, 0))
        paddedImg = np.pad(img, padding, 'edge')
        print('shape of the padded img', paddedImg.shape)
        cv2.imshow('post-padding', paddedImg)
        cv2.waitKey(0)
        cv2.imwrite(path, paddedImg)

opts = options()
findNonSquareImgs(opts.dir)
