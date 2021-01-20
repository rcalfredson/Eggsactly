import argparse
import cv2, numpy as np

from util import *
from annotator import Annotator

def options():
  """Parse options for sub-image exporter."""
  p = argparse.ArgumentParser(description=
    'Export sub-images, with bounding boxes intact, for use in ML model dev.')
  p.add_argument('-d', dest='dir', help='directory containing .png images')
  p.add_argument('-w', dest='maskWidth', type=int, default=130,
    help='sub-image width in pixels (default: %(default)s)')
  p.add_argument('-ht', dest='maskHeight', type=int, default=130,
    help='sub-image height in pixels (default: %(default)s)')
  p.add_argument('-wRng', dest='widthRange',
    help='range from which to randomly select sub-image width in pixels' +
    ' (example: "130-300")')
  p.add_argument('-htRng', dest='heightRange',
    help='range from which to randomly select sub-image height in pixels' +
    ' (example: "130-300")')
  p.add_argument('--omitPartials', dest='omitPartials', type=bool, default=False,
    help='whether to omit bounding boxes overlapping with the sub-image edge')
  return p.parse_args()
    
Annotator(options())
