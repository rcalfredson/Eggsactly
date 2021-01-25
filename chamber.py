from abc import abstractmethod
import enum
from operator import itemgetter
from util import *

class Chamber():
  """Represent the egg-laying chamber, chiefly to facilitate reading and writing
  egg counts data to a CSV file in a way that reflects the chamber's physical
  layout.
  """
  def __init__(self):
    """Create a new egg-laying chamber."""
    self.numRows, self.numCols = 0, 0
    self.rowDist, self.colDist = 0, 0

  @abstractmethod
  def writeLineFormatted(self, eggCounts, i, writer):
    """Write egg counts to file for an entire experimental image (i.e., a
    collection of egg-laying regions)
    
    Arguments:
      - eggCounts: list of list of egg counts (one list per image)
      - i: index of the image for which to write counts
      - writer: csv.writer instance for the open file.
    """
    for j in range(self.numRows*(self.numRepeatedRowsPerCol if hasattr(
      self, 'numRepeatedRowsPerCol') else 1)):
      colCt = self.numCols*(self.numRepeatedColsPerRow if hasattr(
        self, 'numRepeatedColsPerRow') else 2)
      row = eggCounts[i][slice(j*colCt, j*colCt + colCt)]
      writer.writerow(row)

  @staticmethod
  def readCounts(reader, chamberTypes):
    """Read egg counts from a CSV file using the given reader.
    
    Arguments:
      - reader: csv.reader instance for the open file.
      - chamberTypes: dictionary whose keys are filenames of the images in the 
                      CSV file and whose values are chamber types (of type str)
    """
    counts = dict()
    for row in reader:
      if len(row) == 1:
        if len(counts.keys()) > 0 and chamberTypes[currentImg] ==\
            CT.fourCircle.name:
          counts[currentImg] = FourCircleChamber().flattenCounts(counts[
            currentImg])
        currentImg = row[0].lower()
        counts[currentImg] = []
      else:
        counts[currentImg] += row
    return counts

class NewChamber(Chamber):
  """"Represent the chamber type with 4 rows and 5 columns. Note: before being
  analyzed by this program, these images need to be rotated so their agarose
  strips run vertically, which is why they are listed below as having 5 rows
  and 4 columns."""
  def __init__(self):
    """Create a new "New"-type chamber."""
    self.numRows, self.numCols = 4, 5
    self.rowDist, self.colDist = 18, 22
    self.numRepeatedRowsPerCol = 2
    self.numRepeatedColsPerRow = 1

  def getSortedSubImgs(self, subImgs, bboxes):
    sortedSubImgs, sortedBBoxes = [], []
    proposedIndices = concat([[i + 2*self.numRows*j for j in range(
      self.numCols)] for i in range(2*self.numRows)])
    for i in proposedIndices:
      sortedSubImgs.append(subImgs[i])
      sortedBBoxes.append(bboxes[i])
    return sortedSubImgs, sortedBBoxes

class OldChamber(Chamber):
  """Represent the chamber type with 6 rows and 4 columns."""
  def __init__(self):
    """Create a new "Old"-type chamber."""
    self.numRows, self.numCols = 6, 4
    self.rowDist, self.colDist = 12, 25

class ThreeByFiveChamber(Chamber):
  """Represent the chamber type with 5 rows and 3 columns."""
  def __init__(self):
    """Create a new "3x5"-type chamber."""
    self.numRows, self.numCols = 5, 3
    self.rowDist, self.colDist = 12, 23

class FourCircleChamber(Chamber):
  """Represent the chamber type with four central points arranged in a square,
  with four agarose wells arranged in a diamond pattern around each of those
  central points."""
  def __init__(self):
    """Create a new "4-circle"-type chamber."""
    self.numRows, self.numCols = 2, 2
    self.rowDist, self.colDist = 42, 42
    self.dataIndices = tuple([item for sublist in zip(*[(el, el+18) for el in (
      1, 4, 6, 8, 9, 11, 13, 16)]) for item in sublist])
    self.csvToClockwise = (0, 4, 3, 1, 7, 5, 2, 6, 8, 12, 11, 9, 15, 13, 10, 14)

  def flattenCounts(self, counts):
    """Flatten egg counts for the 4-circle chamber. Chambers are read across
    rows, the four agarose wells around each central point are flattened
    according to North-East-South-West."""
    counts = itemgetter(*self.dataIndices)(counts)
    counts = [x for _, x in sorted(zip(self.csvToClockwise, counts))]
    return counts

  def writeLineFormatted(self, eggCounts, i, writer):
    """Write egg counts to file for an entire experimental image (i.e., a
    collection of egg-laying regions)
    
    Counts are represented in the CSV file in a true-to-life layout, i.e., in
    the positions of the wells from the original image.
    """
    uppers = ((0, 4), (8, 12))
    rights = ((1, 5), (9, 13))
    lowers = ((2, 6), (10, 14))
    lefts = ((3, 7), (11, 15))
    def noneToEmpty(myArr):
      return ['' if el is None else el for el in myArr]
    def strToArr(myS):
      return [el if el is not None else '' for el in myS.split(',')]
    for j in range(len(uppers)):
      writer.writerow(strToArr(",{},,,{},".format(*tuple(noneToEmpty([eggCounts[
        i][k] for k in uppers[j]])))))
      writer.writerow(strToArr('{},,{},{},,{}'.format(*tuple(noneToEmpty([
        eggCounts[i][k] for k in concat(zip(lefts[j], rights[j]))])))))
      writer.writerow(strToArr(',{},,,{},'.format(*tuple(noneToEmpty([eggCounts[
        i][k] for k in lowers[j]])))))

# chamber type
class CT(enum.Enum):
  """Hold constructors for all chamber types."""
  new = NewChamber
  old = OldChamber
  threeBy5 = ThreeByFiveChamber
  fourCircle = FourCircleChamber
