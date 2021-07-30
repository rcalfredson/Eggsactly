import argparse, os
import cv2

from common import globFiles
from circleFinder import CircleFinder,  rotate_around_point_highperf, getChamberTypeByRowsAndCols

def options():
  """Parse options for sub-image detector."""
  p = argparse.ArgumentParser(description=
    'Find chambers within an egg-laying image and write each one as a ' +
    'separate image.')
  p.add_argument('source', help='img path or a folder containing source images')
  return p.parse_args()

opts = options()
unskewedImages = []
if os.path.isdir(opts.source):
  imagePaths = globFiles(opts.source, ext='jpg')
else:
  imagePaths = [opts.source]
for path in imagePaths:
    img = cv2.imread(path)
    imgName = os.path.basename(path)
    print('checking image', imgName)
    cf = CircleFinder(img, imgName)
    wells, avgDists, numRowsCols, rotatedImg, rotationAngle = cf.findCircles(debug=True)
    print('rows and col?', numRowsCols)
    if not cf.skewed:
        unskewedImages.append(path)
    getChamberTypeByRowsAndCols(numRowsCols)

with open('unskewedImgs.txt', 'w') as f:
    [f.write('%s\n'%datum) for datum in unskewedImages]
