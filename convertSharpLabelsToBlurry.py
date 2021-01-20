import pickle

fileName = r"P:\Egg images_9_3_2020\WT_5\egg_count_labels_robert.pickle"
with open(fileName, 'rb') as pFile:
    loadedData = pickle.load(pFile)

if not 'isSharpLabels' in loadedData:
    exit(0)
loadedData['isBlurryLabels'] = dict((k, not v) for k, v in loadedData['isSharpLabels'].items())
del loadedData['isSharpLabels']
with open(fileName, 'wb') as pFile:
    pickle.dump(loadedData, pFile)
