"""Annotator class"""
import glob, os, random, string
import cv2, numpy as np

from common import *
from pascalVocHelper import PascalVOCHelper

class Annotator:
  """Select and export sub-images and their corresponding bounding boxes."""
  def __init__(self, opts):
    """Create sub-image selection interface."""
    self.bannerHeight = 30
    self.randomizeDims = any(isinstance(a, str) for a in (
      opts.widthRange, opts.heightRange))
    if self.randomizeDims:
      self.setSubImgDimRanges(opts)
      self.resetSubImgDims()
    else:
      self.sbImgDims = dict(h=opts.maskHeight, w=opts.maskWidth)
    self.resizeRatio = 1
    self.imgDir = opts.dir
    self.omitPartials = opts.omitPartials
    self.showHelp = True
    self.setImgs()
    self.loadAnnotator()
    self.processInput()

  def setImgs(self):
    """Set self.imgs to array of paths of all PNG files in self.imgDir."""
    self.imgs = globFiles(self.imgDir)
    assert len(self.imgs) > 0
    self.idx = 0

  def setSubImgDimRanges(self, opts):
    """Set the ranges from which to randomly select height and/or width of
    the sub-image mask.
    """
    mDims = (opts.maskHeight, opts.maskWidth)
    for i, rngType in enumerate(('heightRange', 'widthRange')):
      setattr(self, rngType, [int(limit) for limit in getattr(opts, rngType
        ).split('-')] if getattr(opts, rngType) is not None else 2*[mDims[i]])

  def resetSubImgDims(self):
    """Randomly select new height and/or width of the sub-image mask."""
    sbImgDims = {}
    for dKey, rngKey in (('h', 'heightRange'), ('w', 'widthRange')):
      rng = getattr(self, rngKey)
      sbImgDims[dKey] = random.randrange(rng[0], rng[1])
    self.sbImgDims = sbImgDims

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

  def imgPath(self):
    """Return path to the current image."""
    return self.imgs[self.idx]

  def noExtPath(self):
    """Return current image path without file extension."""
    return self.imgPath().split('.png')[0]

  def xmlPath(self):
    """Return current image path with .xml extension."""
    return "%s.xml"%self.noExtPath()

  def addImgLabel(self):
    """Display file name of the current image in the UI."""
    cv2.putText(self.boxesImg, os.path.basename(self.imgPath()), (5, 20),
      cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 0), 1)

  def loadAnnotator(self):
    """Load bounding boxes, set the source image, and update the UI view."""
    self.setSourceImg()
    self.loadBBoxesFromXML()
    self.drawBBoxes()
    self.maskCorners = []
    self.maskImg = np.array(self.boxesImg)
    cv2.imshow('currentImg', self.maskImg)

  def loadBBoxesFromXML(self):
    """Load bounding boxes from the image's corresponding Pascal VOC file."""
    self.xmlH = PascalVOCHelper(
      self.xmlPath(), self.resizeRatio, self.omitPartials)
    self.boxes = self.xmlH.boundingBoxes(resized=True)

  def drawBBoxes(self):
    """Draw bounding box outlines on the image in the UI."""
    for box in self.boxes:
      cv2.rectangle(self.boxesImg, (
        box['xmin'], self.bannerHeight + box['ymin']),
        (box['xmax'], self.bannerHeight + box['ymax']), (0,255,255), 1)

  def maskAllowed(self):
    """Return bool True if the sub-image mask lies within the borders of the
    current image, and False otherwise.
    """
    return self.yUpper >= self.bannerHeight and\
      self.yLower <= self.maskImg.shape[0] and\
      self.xUpper >= 0 and\
      self.xLower <= self.maskImg.shape[1]

  def onMouse(self, event, x, y, flags, params):
    """Export sub-image upon mouse-click, and otherwise display sub-image
    selector based on current position of the cursor.
    """
    self.calcMaskEdges(x, y)
    if event == cv2.EVENT_LBUTTONUP:
      if not self.maskAllowed():
        return
      self.drawMaskShadows(x, y)
      self.exportSubImage()
      self.resetSubImgDims()
    else:
      self.drawMaskOutline(x, y)

  def exportSubImage(self):
    """Export sub-image using the mask last appended to self.maskCorners."""
    mCorner = self.maskCorners[-1]
    rId = randID()
    rr = self.resizeRatio
    self.hBounds = slice(round((mCorner[1] - self.bannerHeight)/rr),
      round((mCorner[1] - self.bannerHeight)/rr) + self.sbImgDims['h'])
    self.wBounds = slice(round(mCorner[0]/rr),
      round(mCorner[0]/rr) + self.sbImgDims['w'])
    self.subImg = self.sourceImg[self.hBounds, self.wBounds]
    cv2.imwrite('%s_%s.png'%(self.noExtPath(), rId), self.subImg)
    self.xmlH.exportBoundingBoxes(self.noExtPath(), rId, self.subImg.shape,
      self.wBounds, self.hBounds)

  def drawMaskShadows(self, x, y):
    """Draw shaded boxes on image in the UI to indicate existing masks."""
    self.maskCorners.append([self.xUpper, self.yUpper, self.xLower, self.yLower])
    for mCorner in self.maskCorners:
      self.mask[mCorner[1]:mCorner[3], mCorner[0]:mCorner[2]] = \
        np.full((mCorner[3] - mCorner[1], mCorner[2] - mCorner[0], 3), (0, 255, 0))
    self.maskImg = cv2.addWeighted(self.boxesImg, 1.0,
      self.mask.astype(np.uint8), 0.2, 0)
    cv2.imshow('currentImg', self.maskImg)

  def calcMaskEdges(self, x, y):
    """Calculate the upper-left and lower-right points of the mask from a given
    point at its center.
    """
    halfH, halfW = [round(self.sbImgDims[dim]* self.resizeRatio / 2) for dim in sorted(self.sbImgDims.keys())]
    self.yUpper = y - halfH
    self.yLower = y + halfH
    self.xUpper = x - halfW
    self.xLower = x + halfW
    # w: 181; h: 518 (agarose)
    # w: 881; h: 518 (arena)

  def helpText(self):
    """Display help info in the UI if self.showHelp is True."""
    if not self.showHelp:
      return
    corner, dims = (32, 15), (111, 111)
    helpMask = np.zeros(self.maskImg.shape)
    slices = [slice(corner[i], corner[i] + dims[i]) for i in range(2)]
    helpMask[slices[0], slices[1]] = np.full(dims + (3,), (255, 0, 0))
    self.maskImg = cv2.addWeighted(self.maskImg, 1.0, helpMask.astype(np.uint8), 0.2, 0)
    helpTexts = ('keyboard commands:\nh\ns\nq\nc\n.\n,',
        'toggle show help\nsave\nsave and quit\n' +
        'clear clusters for current heatmap\n' +
        'step forward 1 heatmap (autosaves clusters)\n' +
        'step back 1 heatmap (autosaves clusters)')
    for txtAndLocs in zip(helpTexts, ((23, 34), (60, 50))):
      putText(self.maskImg, txtAndLocs[0], txtAndLocs[1],
        cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 0, 0), 1)

  def drawMaskOutline(self, x, y):
    """Draw sub-image mask outline around the current position of the cursor."""
    self.drawnImage = np.array(self.maskImg)
    col = (0, 255, 0) if self.maskAllowed() else (0, 0, 255)
    cv2.rectangle(self.drawnImage, (self.xUpper, self.yUpper),
      (self.xLower, self.yLower), col)
    cv2.imshow('currentImg', self.drawnImage)

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

  def toggleHelp(self):
    """Negate the bool self.showHelp used for displaying help info in the UI."""
    self.showHelp = ~self.showHelp

  def handleKeyPress(self):
    """Handle input from keyboard."""
    command = {'.': self.stepForward,
      ',': self.stepBackward, 'h': self.toggleHelp}.get(self.readKeyPress())
    if command: command()

  def processInput(self):
    """
    Handle input from keyboard and mouse.
    """
    while True:
      self.loadAnnotator()
      #self.helpText()
      cv2.setMouseCallback('currentImg', self.onMouse)
      self.handleKeyPress()
