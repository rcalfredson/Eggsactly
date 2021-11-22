from glob import glob
import os

import numpy as np

img_files = np.array(glob(r'C:\Users\Tracking\counting-3\imgs\Charlene\temp2\*jpg'))
with open(r"P:\Robert\objects_counting_dmap\egg_source\splineDistComparisionImgs.txt", 'r') as myF:
    files_in_use = np.array(myF.read().splitlines())
img_files = np.array([os.path.basename(p).lower() for p in img_files])
files_in_use = np.array([os.path.basename(p).lower() for p in files_in_use])
print('how many images? %i'%len(img_files))
print('how many files in use?', len(files_in_use))
free_files = np.setdiff1d(img_files, files_in_use)
print('img_files?', img_files)
print('files_in_use?', files_in_use)
print('how many free files?', len(free_files))
with open('P:/Robert/objects_counting_dmap/egg_source/splineDistLargeHeldoutSet.txt', 'w') as myF:
    for free_file in free_files:
        myF.write(free_file + '\n')
