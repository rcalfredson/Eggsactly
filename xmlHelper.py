"""XMLHelper class"""
import copy
import xml.etree.ElementTree as ET

class XMLHelper:
  """Read and write XML files."""
  def __init__(self, path):
    """Initialize the instance's root XML element based on the given path.
    """
    self.root = ET.parse(path).getroot()

  def copyEl(self, root, path):
    """Find the first subelement in the self.root that matches the given path
    and append it to the given root.
    """
    root.append(copy.deepcopy(self.getFirst(self.root, path)))

  @staticmethod
  def writeTree(root, filename):
    """Write element tree of the given root to the given filename."""
    ET.ElementTree(root).write(filename)

  @staticmethod
  def getFirst(obj, path):
    """Return first subelement in the given root matching the given path."""
    return next(obj.iterfind(path))

  @staticmethod
  def txt2Int(obj, path):
    """Return integer parsed from text of the first subelement in the given root
    matching the given path.
    """
    return int(XMLHelper.getFirst(obj, path).text)   
