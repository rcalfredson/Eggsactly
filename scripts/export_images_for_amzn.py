import sys

sys.path.append(r"P:\Robert\counting-3")

import argparse
from chamber import CT
from circleFinder import CircleFinder, rotate_image, subImagesFromBBoxes
import cv2
from glob import glob
import json
import numpy as np
import os
from pathlib import Path
import pickle
from PIL import Image

POSITIONS = ("upper", "right", "lower", "left")

parser = argparse.ArgumentParser(
    description="Export egg-laying regions, click data,"
    " and accompanying manifest for upload to Amazon S3."
)
parser.add_argument(
    "pickle_files",
    help="Pickle files containing egg clicks for the images to export."
    " Note: all labeled images will be exported.",
    nargs="*",
)
parser.add_argument("output_dir")

currentImgData = {"row_cols": "", "ct": ""}


def segment_sub_image_from_existing_data(img, lbn, metadata):
    rotatedImg = rotate_image(img, metadata[lbn]["rotationAngle"])
    subImgs = subImagesFromBBoxes(rotatedImg, metadata[lbn]["bboxes"])
    currentImgData["row_cols"] = metadata[lbn]["rowColCounts"]
    currentImgData["ct"] = metadata[lbn]["ct"]
    return subImgs


def segment_sub_image_via_GPU(img, img_name):
    cf = CircleFinder(img, os.path.basename(img_name), allowSkew=False)
    circles, avgDists, numRowsCols, rotatedImg, rotAngle = cf.findCircles()
    currentImgData["row_cols"] = numRowsCols
    currentImgData["ct"] = cf.ct
    return cf.getSubImages(rotatedImg, circles, avgDists, numRowsCols)


def row_num(i):
    if currentImgData["ct"] == CT.fourCircle.name:
        numCirclesPerWell = 4
        numCirclesPerRow = currentImgData["row_cols"][0] * numCirclesPerWell
        return np.floor(i / numCirclesPerRow).astype(int)
    return int(np.floor(i / (2 * currentImgData["row_cols"][1])))


def col_num(i):
    if currentImgData["ct"] == CT.fourCircle.name:
        numCirclesPerWell = 4
        numCirclesPerRow = currentImgData["row_cols"][0] * numCirclesPerWell
        return np.floor((i % numCirclesPerRow) / numCirclesPerWell).astype(int)
    return i % int(2 * currentImgData["row_cols"][1])


def generate_outputs(lbn, subImgs):
    for i, sub_img in enumerate(sub_imgs):
        print("export name is:", lbn)
        print("row", row_num(i))
        print("col", col_num(i))
        if currentImgData["ct"] == CT.fourCircle.name:
            print("4c position:", POSITIONS[i % 4])
        exported_filename = "%s_%i_%i%s" % (
            lbn,
            row_num(i),
            col_num(i),
            "_%s" % (POSITIONS[i % 4])
            if currentImgData["ct"] == CT.fourCircle.name
            else "",
        )
        cv2.imwrite(os.path.join(opts.output_dir, f"{exported_filename}.png"), sub_img[:, :, ::-1])
        manifest_list.append(exported_filename)
        with open(
            os.path.join(opts.output_dir, f"{exported_filename}_clicks.json"), "w"
        ) as f:
            json.dump(click_data[exported_filename], f)


manifest_list = []
opts = parser.parse_args()
for fname in opts.pickle_files:
    print("trying to open:", fname)
    with open(fname, "rb") as f:
        click_data = pickle.load(f)["clicks"]
        print("Pickle data:", click_data)
        # split the image up into pieces.
        # the pickle file doesn't contain
        # the image basenames, but assume it's in the same
        # folder as the images.
        parent_dir = Path(fname).parents[0]
        images = glob("%s/*jpg" % (parent_dir))
        with open(os.path.join(parent_dir, "images.metadata"), "rb") as f:
            metadata = pickle.load(f)
        print("metadata:", metadata)
        for img_name in images:
            img = np.array(Image.open(img_name), dtype=np.float32)
            lbn = os.path.basename(img_name).lower()
            if lbn in metadata:
                sub_imgs = segment_sub_image_from_existing_data(img, lbn, metadata)
            else:
                sub_imgs = segment_sub_image_via_GPU(img, img_name)
            generate_outputs(lbn, sub_imgs)
        print("image list:", images)
# save the manifest file.
with open(os.path.join(opts.output_dir, "manifest.jl"), 'w') as f:
    for line in manifest_list:
        f.write(
            '{{"source-ref":"s3://egg-laying/images/{}"}}\n'.format("%s.png" % line)
        )
