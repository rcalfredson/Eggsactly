import pickle

import numpy as np

pickleFiles = (r"P:\Egg images_9_3_2020\egg_count_labels_robert.pickle",
    r"P:\Egg images_9_3_2020\images with increased optical zoom\egg_count_labels_robert.pickle",
    r"P:\Egg images_9_3_2020\WT\egg_count_labels_robert.pickle",
    r"P:\Egg images_9_3_2020\WT_1\egg_count_labels_robert.pickle",
    r"P:\Egg images_9_3_2020\WT_2\egg_count_labels_robert.pickle",
    r"P:\Egg images_9_3_2020\WT_3\egg_count_labels_robert.pickle",
    r"P:\Egg images_9_3_2020\WT_4\egg_count_labels_robert.pickle",
    r"P:\Egg images_9_3_2020\WT_5\egg_count_labels_robert.pickle")

totalNumSubImgs = 0
totalNumBlurryImgs = 0

for pFile in pickleFiles:
    with open(pFile, 'rb') as myF:
        loadedData = pickle.load(myF)
        totalNumSubImgs += len(loadedData['isBlurryLabels'])
        print('is blurry labels for file', pFile)
        print(loadedData['isBlurryLabels'].values())
        totalNumBlurryImgs += np.count_nonzero(list(loadedData['isBlurryLabels'].values()))
print('Total number sub-images:', totalNumSubImgs)
print('Total number blurry images: %i (%.2f%%)'%(totalNumBlurryImgs, 100 * totalNumBlurryImgs / totalNumSubImgs))
