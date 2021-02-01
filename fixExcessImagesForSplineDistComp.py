from glob import glob
import os
import pickle
import shutil

from chamber import CT

import numpy as np

imgs_in_train_val = glob(r"P:\Robert\splineDist\data\egg\images\*.tif")
imgs_in_heldout = glob(r"P:\Robert\splineDist\data\egg\heldout\images\*.tif")
folder_to_check = r"P:\Robert\objects_counting_dmap\egg_source"
fcrn_imgs = []
fcrn_imgs_long = []
for folder in ('train', 'valid'):
    img_list = glob('%s\\independent_fullsize_%s\\*.jpg'%(folder_to_check, folder))
    fcrn_imgs_long += img_list
    fcrn_imgs += [os.path.basename('_'.join(img.split('_')[:-1]) + '.jpg') for img in img_list]

# convert these image names to row/col style.
with open(r"C:\Users\Tracking\counting-3\imgs\Charlene\temp2\images.metadata", 'rb') as f:
    imgData = pickle.load(f)

converted_train_val = []
converted_heldout = []

for i, imgList in enumerate((imgs_in_train_val, imgs_in_heldout)):
    for img in imgList:
        partialMatch = "%s.jpg"%('_'.join(os.path.basename(img).lower().split('.tif')[0].split('_')[:-1]))
        for key in imgData.keys():
            if partialMatch in key:
                goodKey = key
                continue
        print('img name:', img)
        orig_index = int(img.lower().split('.tif')[0].split('_')[-1])
        ct = CT[imgData[goodKey]['ct']].value()
        print('ct?', ct)
        print('orig index:', orig_index)
        print('num cols?', ct.numCols)
        if imgData[goodKey]['ct'] == 'new':
            ct.numCols = ct.numRows
        rowNum = int(np.floor(orig_index / (2*ct.numCols)))
        colNum = orig_index % int(2*ct.numCols)
        convertedFilename = '%s_%i_%i.jpg'%(partialMatch.split('.jpg')[0],
            rowNum, colNum)
        if i == 0:
            converted_train_val.append(convertedFilename)
        else:
            converted_heldout.append(convertedFilename)
print('len(converted_heldout)', len(converted_heldout))
print('len(converted_train_val)', len(converted_train_val))
input()
num_not_found = 0
def to_dots(fileName):
    return fileName.split('.jpg')[0]+ '_dots.png'
for i, img in enumerate(fcrn_imgs):
    if img not in converted_train_val and img not in converted_heldout:
        print('didnot find image', img, 'in splinedist data')
        shutil.move(fcrn_imgs_long[i], '%s\\extras\\%s'%(folder_to_check, os.path.basename(img)))
        shutil.move(to_dots(fcrn_imgs_long[i]), to_dots('%s\\extras\\%s'%(folder_to_check, os.path.basename(img))))
    else:
        if img in converted_heldout:
            shutil.move(fcrn_imgs_long[i], '%s\\heldout\\%s'%(folder_to_check, os.path.basename(img)))
            shutil.move(to_dots(fcrn_imgs_long[i]), to_dots('%s\\heldout\\%s'%(folder_to_check, os.path.basename(img))))
print('total num not found:', num_not_found)
