import pickle

class ClickLabelManager:
  def __init__(self):
    self.clicks = dict()

  @staticmethod
  def subImageKey(imageName, rowNum, colNum):
    return '%s_%i_%i'%(imageName, rowNum, colNum)

  def addClick(self, imageName, rowNum, colNum, coords):
    key = self.subImageKey(imageName, rowNum, colNum)
    if key in self.clicks:
      self.clicks[key].append(coords)
    else:
      self.clicks[key] = [coords]
    print('just added click; here is dict:', self.clicks)

  def clearClicks(self, imageName, rowNum, colNum):
    self.clicks[self.subImageKey(imageName, rowNum, colNum)] \
      = []

  def getClicks(self, imageName, rowNum, colNum):
    key = self.subImageKey(imageName, rowNum, colNum)
    return self.clicks[key] if key in self.clicks else []

with open('egg_count_labels_robert.csv','rb') as f:
    loaded = pickle.load(f)
    print('what is loaded?', loaded)
    print('and its type:', type(loaded))