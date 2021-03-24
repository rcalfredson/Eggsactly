# goal of script:
#   - start from single directory of paired images and dot annotations for
#     individual egg-laying regions
#   - choose training or validation as the category for the region. Copy the
#     full-sized image first, then sample 20 160x160 patches from it into a
#     separate folder.

import argparse
from common import randID
import cv2
from glob import glob
import numpy as np
import os
from pathlib import Path
from random import shuffle
import shutil

TRAIN_PROPORTION = 0.8
DEST_PATH = "P:/Robert/objects_counting_dmap/egg_source/archive_2021-03-22"
NUM_PATCHES = 20
PATCH_SQUARE_SIZE = 160

parser = argparse.ArgumentParser(
    description="Sample patches from egg-laying regions"
    " that have already been segmented."
)

parser.add_argument("dir", help="Directory containing the patches to sort")
opts = parser.parse_args()
imgs = glob(os.path.join(opts.dir, "*.jpg"))
shuffle(imgs)
print("num imgs?", imgs)
imgs = imgs[:221]
for img_path in imgs:
    is_train = np.random.random() < TRAIN_PROPORTION
    category = "train" if is_train else "valid"
    file_basename = os.path.basename(img_path)
    basename_no_ext = ".".join(file_basename.split(".")[:-1])
    dots_file = "%s_dots.png" % basename_no_ext
    dots_path = os.path.join(Path(img_path).parent, dots_file)
    img_dest = os.path.join(
        DEST_PATH, "fullsize_%s" % category, "%s.png" % basename_no_ext
    )
    dots_dest = os.path.join(DEST_PATH, "fullsize_%s" % category, dots_file)
    shutil.copyfile(img_path, img_dest)
    shutil.copyfile(dots_path, dots_dest)
    img = cv2.imread(img_dest)
    dots = cv2.imread(dots_dest)
    # now sample 20 patches from these
    for _ in range(NUM_PATCHES):
        x_offset = np.random.randint(0, img.shape[1] - PATCH_SQUARE_SIZE)
        y_offset = np.random.randint(0, img.shape[0] - PATCH_SQUARE_SIZE)
        patch_rand_id = randID()
        print("original image basename?", file_basename)
        new_basename = (
            "_".join(basename_no_ext.split("_")[:-1]) + "_%s.png" % patch_rand_id
        )
        new_dotsname = (
            "_".join(basename_no_ext.split("_")[:-1]) + "_%s_dots.png" % patch_rand_id
        )
        sampled_img = img[
            y_offset : y_offset + PATCH_SQUARE_SIZE,
            x_offset : x_offset + PATCH_SQUARE_SIZE,
        ]
        sampled_dots = dots[
            y_offset : y_offset + PATCH_SQUARE_SIZE,
            x_offset : x_offset + PATCH_SQUARE_SIZE,
        ]
        print("size of the original:", img.shape)
        print("shape of annotations:", dots.shape)
        print("x and y offsets:", x_offset, y_offset)
        img_dest = os.path.join(DEST_PATH, category, new_basename)
        dots_dest = os.path.join(DEST_PATH, category, new_dotsname)
        # cv2.imshow("image example:", sampled_img)
        # cv2.imshow("dots example:", sampled_dots)
        print("wrote image to", img_dest)
        cv2.imwrite(img_dest, sampled_img)
        cv2.imwrite(dots_dest, sampled_dots)
        # cv2.waitKey(0)
