import cv2
from glob import glob
import json
from lib.datamanagement.helpers import to_dots
import numpy as np
import os
import pickle
import shutil
import sys

from chamber import CT

with open(r"C:\Users\Tracking\counting-3\imgs\Charlene\temp2\images.metadata",
          'rb') as f:
    imgData = pickle.load(f)

output_dir = r'P:\Robert\splineDist\data\egg\to-be-labeled'

click_imgs_base_dir = r'P:\Robert\objects_counting_dmap\egg_source\archive_2021-01-14'
segment_imgs_in_train_val = glob(r"P:\Robert\splineDist\data\egg\images\*.tif")
segment_imgs_in_heldout = glob(
    r"P:\Robert\splineDist\data\egg\heldout\images\*.tif")
imgs = []
imgs_long = []
for ext_dir in ('fullsize_train', 'fullsize_valid', 'heldout',
                'independent_fullsize_train', 'independent_fullsize_valid'):
    new_imgs = glob(f'{click_imgs_base_dir}\\{ext_dir}\\*.jpg')
    imgs_long += new_imgs
    new_imgs = [
        os.path.basename(f"{'_'.join(img_name.split('_')[:-1])}.jpg")
        for img_name in new_imgs
    ]
    imgs += new_imgs

print('first ten image results:')
for name in imgs[:10]:
    print(name)
print('total number images found:', len(imgs))

splinedist_imgs = []

for imgList in (segment_imgs_in_train_val, segment_imgs_in_heldout):
    for img in imgList:
        partialMatch = "%s.jpg" % ('_'.join(
            os.path.basename(img).lower().split('.tif')[0].split('_')[:-1]))
        for key in imgData.keys():
            if partialMatch in key:
                goodKey = key
                continue
        # print('img name:', img)
        orig_index = int(img.lower().split('.tif')[0].split('_')[-1])
        ct = CT[imgData[goodKey]['ct']].value()
        # print('ct?', ct)
        if imgData[goodKey]['ct'] == 'new':
            ct.numCols = ct.numRows
        rowNum = int(np.floor(orig_index / (2 * ct.numCols)))
        colNum = orig_index % int(2 * ct.numCols)
        convertedFilename = '%s_%i_%i.jpg' % (partialMatch.split('.jpg')[0],
                                              rowNum, colNum)
        splinedist_imgs.append(convertedFilename)
        # print('converted filename:', convertedFilename)
        # print('is this in imgs?', convertedFilename in imgs)
        # input()

disjunct_union = np.setdiff1d(imgs, splinedist_imgs)
print('how many splineDist images?', len(splinedist_imgs))
print('how many in FCRN without accompanying segmentation?',
      len(disjunct_union))
with open(os.path.join(output_dir, 'manifest.jsonl'), 'w') as f:
    for i, img_name in enumerate(imgs_long):
        shutil.copy(img_name, os.path.join(output_dir, imgs[i]))
        click_img = cv2.imread(to_dots(imgs_long[i]))
        # print('tried to open', to_dots(imgs_long[i]))
        # print(click_img.shape)
        # np.set_printoptions(threshold=sys.maxsize)
        # with open('debug', 'w') as f:
        #     print(click_img, file=f)
        # cv2.imshow('debug', click_img)
        # cv2.waitKey(0)
        # need to recursively convert the contents of these tuples to ints.
        click_data = list(
            zip(*
                [[int(el) for el in arr]# at this level is an array of tuples.
                 for arr in reversed(np.where(click_img[:, :, -1] > 0))]))
        print('click data?', click_data)
        with open(
                os.path.join(output_dir,
                             imgs[i].split('.jpg')[0] + '_clicks.json'),
                'w') as click_f:
            json.dump(click_data, click_f, ensure_ascii=False)
        # print(np.count_nonzero(click_data))
        f.write('{{"source-ref":"s3://egg-laying/images/{}"}}\n'.format(
            imgs[i]))
        print(f"Copied img {i + 1}")
