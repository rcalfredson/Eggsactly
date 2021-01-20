import argparse, os
import cv2
import numpy as np
from common import globFiles
from pascalVocHelper import PascalVOCHelper
from pathlib import Path

def options():
  """Parse options for egg image labeller."""
  p = argparse.ArgumentParser(description=
    'Sort images according to label (clean or blur).')
  p.add_argument('source', help='folder containing source images')
  return p.parse_args()

class EggImageLabeller:
  """Sort sub-images according to user-supplied labels."""
  def __init__(self, opts):
    """Create image labeller interface."""
    self.bannerHeight = 30
    self.imgDir = opts.source
    self.showHelp = True
    self.setImgs()
    self.loadAnnotator()
    self.processInput()

  def setImgs(self):
    """Set self.imgs to array of paths of all PNG files in self.imgDir."""
    self.imgs = globFiles(self.imgDir)
    assert len(self.imgs) > 0
    self.idx = 0

  def loadAnnotator(self):
    """Load bounding boxes, set the source image, and update the UI view."""
    self.setSourceImg()
    self.loadBBoxesFromXML()
    self.drawBBoxes()
    self.maskImg = np.array(self.boxesImg)
    cv2.imshow('currentImg', self.maskImg)

  def processInput(self):
    """
    Handle input from keyboard and mouse.
    """
    while True:
      self.loadAnnotator()
      self.handleKeyPress()

  def loadBBoxesFromXML(self):
    """Load bounding boxes from the image's corresponding Pascal VOC file."""
    self.xmlH = PascalVOCHelper(
      self.xmlPath(), self.resizeRatio)
    self.boxes = self.xmlH.boundingBoxes(resized=True)

  def drawBBoxes(self):
    """Draw bounding box outlines on the image in the UI."""
    for box in self.boxes:
      cv2.rectangle(self.boxesImg, (
        box['xmin'], self.bannerHeight + box['ymin']),
        (box['xmax'], self.bannerHeight + box['ymax']), (0,255,255), 1)

  def imgPath(self):
    """Return path to the current image."""
    return self.imgs[self.idx]
  
  def noExtPath(self):
    """Return current image path without file extension."""
    return self.imgPath().split('.png')[0]

  def xmlPath(self):
    """Return current image path with .xml extension."""
    return "%s.xml"%self.noExtPath()

  def toggleHelp(self):
    """Negate the bool self.showHelp used for displaying help info in the UI."""
    self.showHelp = ~self.showHelp

  def addImgLabel(self):
    """Display file name of the current image in the UI."""
    cv2.putText(self.boxesImg, os.path.basename(self.imgPath()), (5, 20),
      cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 0), 1)

  def handleKeyPress(self):
    """Handle input from keyboard."""
    self.lastKeyPress = self.readKeyPress()
    command = {'.': self.stepForward,
      ',': self.stepBackward, 'b': self.sortImg, 'c': self.sortImg
    }.get(self.lastKeyPress)
    if command: command()

  def sortedImgsPath(self, subDir):
    return os.path.join(self.imgDir, subDir)

  def sortImg(self):
    """Move image and its Pascal VOC file to "blur" or "clean" folder."""
    subDir = self.sortedImgsPath('blur' if self.lastKeyPress == 'b' else "clean")
    if not os.path.exists(subDir):
      Path(subDir).mkdir(parents=True, exist_ok=True)
    os.rename(self.imgPath(), os.path.join(subDir,
      os.path.basename(self.imgPath())))
    os.rename(self.xmlPath(), os.path.join(subDir,
      os.path.basename(self.xmlPath())))
    self.stepForward()

  def readKeyPress(self):
    """Wait until a key is pressed, and then return its associated character."""
    while True:
      k = cv2.waitKey(1)
      if k == -1: eventProcessingDone = True
      elif eventProcessingDone: break
    k &= 255
    return chr(k)

  def stepForward(self):
    """Display the image succeeding the current image."""
    if self.idx + 1 >= len(self.imgs):
      return
    self.idx += 1
    self.loadAnnotator()

  def stepBackward(self):
    """Display the image preceding the current image."""
    if self.idx <= 0:
      return
    self.idx -= 1
    self.loadAnnotator()

  def setSourceImg(self):
    """Read the image to display and add labels and bounding boxes."""
    self.sourceImg = cv2.imread(self.imgPath())
    if self.sourceImg.shape[1] > 1000:
      self.resizeRatio = 1000/self.sourceImg.shape[1]
    else:
      self.resizeRatio = 1
    resizedSource = cv2.resize(self.sourceImg, (0, 0), fx=self.resizeRatio, fy=self.resizeRatio)
    self.boxesImg = np.concatenate((np.full(
      (self.bannerHeight,
      resizedSource.shape[1], 3), 255), resizedSource), axis=0
    ).astype(np.uint8)
    self.addImgLabel()
    self.mask = np.zeros(self.boxesImg.shape)

EggImageLabeller(options())