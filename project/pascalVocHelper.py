"""PascalVOCHelper class"""

import xml.etree.ElementTree as ET

from xmlHelper import XMLHelper

class PascalVOCHelper(XMLHelper):
  """Read and write Pascal VOC files."""
  boxCrds = ('xmin', 'ymin', 'xmax', 'ymax')

  def __init__(self, path, resizeRatio=1, omitPartials=False):
    """Initialize the instance's root XML element based on the given path and
    parse the label associated with bounding boxes.
    """
    super(PascalVOCHelper, self).__init__(path)
    self.setObjLbl()
    self.omitPartials = omitPartials
    self.resizeRatio = resizeRatio

  def setObjLbl(self):
    """Parse the label associated with the file's bounding boxes."""
    self.objLbl = self.getFirst(
      self.getFirst(self.root, 'object'), 'name').text if self.root.find('object')\
      else None

  def addBox(self, root, box):
    """Add bounding box to the given XML root element."""
    objEl = ET.SubElement(root, 'object')
    ET.SubElement(objEl, 'name').text = self.objLbl
    ET.SubElement(objEl, 'pose').text = 'Unspecified'
    ET.SubElement(objEl, 'truncated').text = '0'
    ET.SubElement(objEl, 'difficult').text = '0'
    bndBox = ET.SubElement(objEl, 'bndbox')
    for dim in self.boxCrds:
      ET.SubElement(bndBox, dim).text = str(box[dim])

  def boundingBoxes(self, resized=False):
    """Return array of bounding boxes in self.root."""
    bxs = []
    for labelled_obj in self.root.iterfind('object'):
      bBox = self.getFirst(labelled_obj, 'bndbox')
      bxs.append(dict(zip(self.boxCrds, [round(self.txt2Int(bBox, crd)*\
        (self.resizeRatio if resized else 1)) for crd in self.boxCrds])))
    return bxs

  def xmlSansBoxes(self, rId):
    """
    Return XML element tree based on self.root, excluding bounding boxes and
    appending existing file name with the given random ID.
    """
    root = ET.Element('annotation')
    for attrib in ('folder', 'source', 'segmented'):
      self.copyEl(root, attrib)
    for attrib in ('filename', 'path'):
      splitByDot = self.getFirst(self.root, attrib).text.split('.')
      ET.SubElement(root, attrib).text = '%s_%s.%s'%('.'.join(splitByDot[:-1]),
        rId, splitByDot[-1])
    return root

  def inBounds(self, box, wBounds, hBounds):
    """Return bool True if the given box falls within the given bounds with at
    least 1 pixel of padding, and False otherwise.
    """
    return (self.boxOverlaps, self.boxInBounds)[self.omitPartials](
      box, wBounds, hBounds)

  @staticmethod
  def boxOverlaps(box, wBounds, hBounds):
    """Return bool True if any portion of the given box falls within the
    given bounds, and False otherwise.
    """
    return box['xmin'] < wBounds.stop and\
      box['xmax'] > wBounds.start and\
      box['ymin'] < hBounds.stop and\
      box['ymax'] > hBounds.start

  @staticmethod
  def boxInBounds(box, wBounds, hBounds):
    """Return bool True if the given box falls within the given bounds with at
    least 1 pixel of padding, and False otherwise.
    """
    return box['xmin'] - 1 > wBounds.start and\
        box['xmax'] + 1 < wBounds.stop and\
        box['ymin'] - 1 > hBounds.start and\
        box['ymax'] + 1 < hBounds.stop

  def toMaskCoords(self, box, wBounds, hBounds):
    """Return the given bounding box offset relative to the given origin."""
    xOrg, yOrg = [getattr(bounds, 'start') for bounds in (wBounds, hBounds)]
    if self.omitPartials:
      dims = [getattr(box, dim) for dim in self.boxCrds]
    else:
      dims = (max([box['xmin'], xOrg]), max([box['ymin'], yOrg]),
        min([box['xmax'], wBounds.stop]), min([box['ymax'], hBounds.stop]))
    return dict(xmin=dims[0] - xOrg, ymin=dims[1] - yOrg,
      xmax=dims[2] - xOrg, ymax=dims[3] - yOrg)

  def exportBoundingBoxes(self, fileName, rId, imgShape, wBounds, hBounds):
    """Save Pascal VOC file containing bounding boxes within the given bounds."""
    root = self.xmlSansBoxes(rId)
    sizeEl = ET.SubElement(root, 'size')
    for i, dim in enumerate(('height', 'width')):
      ET.SubElement(sizeEl, dim).text = str(imgShape[i])
    ET.SubElement(sizeEl, 'depth').text = '3'
    for box in self.boundingBoxes():
      if self.inBounds(box, wBounds, hBounds):
        self.addBox(root, self.toMaskCoords(box, wBounds, hBounds))
    self.writeTree(root, '%s_%s.xml'%(fileName, rId))
