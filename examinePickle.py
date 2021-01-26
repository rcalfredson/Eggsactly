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

optogeneticBasenames = (
  '9_24_2020_img_00011.jpg',
  'apr5_2left.jpg',
  'apr5_2right.jpg',
  'apr5_3left.jpg',
  'apr5_3right.jpg',
  'apr7_2left.jpg',
  'apr7_2right.jpg',
  'apr7_3left.jpg',
  'apr7_3right.jpg',
  'mar30_2left.jpg',
  'mar30_2right.jpg',
  'mar30_3left.jpg',
  'mar30_3right.jpg'
  )
with open(r"C:\Users\Tracking\counting-3\imgs\Charlene\temp2\egg_labels_robert.pickle",'rb') as f:
    loaded = pickle.load(f)
    print('loaded keys:', loaded.keys())
    print('cat')
    for keyType in loaded.keys():
      print('items for key type', keyType)
      print(loaded[keyType].keys())
    input()
    print('what is loaded?', loaded['clicks'].keys())
    for key in list(loaded['clicks'].keys()):
      for basename in optogeneticBasenames:
        if basename in key:
          del loaded['clicks'][key]
          # del loaded[]
    print('and its type:', type(loaded))