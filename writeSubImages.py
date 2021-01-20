import argparse, os
import cv2
from common import globFiles
from circleFinder import CircleFinder, getSubImages

def options():
  """Parse options for sub-image detector."""
  p = argparse.ArgumentParser(description=
    'Find chambers within an egg-laying image and write each one as a ' +
    'separate image.')
  p.add_argument('source', help='folder containing source images')
  p.add_argument('--include', help='newline-separated list of files from the ' +\
    'source folder to include')
  p.add_argument('dest', help='folder to which to write images')
  return p.parse_args()

opts = options()
imgs = globFiles(opts.source, ext='jpg')
if opts.include:
  with open(opts.include) as f:
    includedImgs = f.read().splitlines()
else:
  includedImgs = None
print('imgs:', imgs)
print('included imgs:', includedImgs)
for imgPath in imgs:
  imgName = os.path.basename(imgPath).split('.')[0]
  print('imgName:', imgName)
  if includedImgs and imgName not in includedImgs: continue
  img = cv2.imread(imgPath)
  circles, avgDists, numRowsCols, rotatedImg, rotationAngle = CircleFinder(
    img, imgName).findCircles()
  subImgs, origins = getSubImages(rotatedImg, circles, avgDists, numRowsCols)
  for i, subImg in enumerate(subImgs):
    writePath = os.path.join(opts.dest, '%s_%i.jpg'%(imgName, i))
    print('saving sub-image to %s'%writePath)
    cv2.imwrite(os.path.join(opts.dest, '%s_%i.jpg'%(imgName, i)), subImg)
