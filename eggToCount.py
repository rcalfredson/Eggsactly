import argparse
import json
import random

from pycocotools import coco

from utils.coco import getNewCategory
from utils.color import randColor
import utils.polygon as polygon


def options():
    """Parse options for the egg-to-count conversion tool."""
    p = argparse.ArgumentParser(description='Unify a COCO dataset of egg' +
                                ' annotations into blobs using the union' +
                                ' operation and assign them a category ' +
                                'determined by how many eggs are in each blob.')
    p.add_argument('datasetPath', help='Path to the COCO-formatted dataset to' +
                   ' transform.')
    return p.parse_args()


def getNewCategories(counts, categories):
    """Return a list of any categories yet unrepresented. Because these
    categories are sequential, the criterion is for at least one value in
    "counts" to exceed len(categories).

    Arguments:
      - counts: list of the number of intersections of each parent with its
                children
      - categories: list of the current categories
    """
    maxCounts, numCats = max(counts), len(categories)
    if maxCounts < numCats:
        return []
    newCategories = []
    for i in range(numCats, maxCounts):
        newCategories.append(getNewCategory(i+1))
    return newCategories




def main():
    cocoOut = {'images': [], 'categories': [], 'annotations': []}
    categoriesAdded = set()
    annotationId = 1
    opts = options()
    dataset = coco.COCO(opts.datasetPath)
    for i, imgIdx in enumerate(dataset.imgs):
        annotations = dataset.imgToAnns[imgIdx]
        cocoOut['images'].append(dataset.imgs[imgIdx])
        cocoOut['images'][-1]['category_ids'] = set()
        unitedPolygons, polygons = polygon.unitePolygons(annotations, imgIdx)
        print('the image is', cocoOut['images'][-1]['file_name'])
        overlaps = polygon.calculateOverlaps(unitedPolygons, polygons)
        flattenedSegs = polygon.flattenSegs(unitedPolygons)
        for seg in flattenedSegs:
            area, bb =\
                polygon.calculateAreaAndBBox(
                    seg, (cocoOut['images'][-1]['width'],
                          cocoOut['images'][-1]['height']))
            category_id = int(overlaps[polygon.polyFakeHash(seg)])
            cocoOut['images'][-1]['category_ids'].add(category_id)
            if category_id not in categoriesAdded:
                cocoOut['categories'].append(getNewCategory(category_id))
                categoriesAdded.add(category_id)
            cocoOut['annotations'].append({
                'id': annotationId,
                'image_id': cocoOut['images'][-1]['id'],
                'category_id': category_id,
                'segmentation': seg,
                'area': int(area),
                'bbox': bb.tolist(),
                'iscrowd': False,
                'isbbox': False,
                'color': randColor(),
                'keypoints': [],
                'metadata': {}
            })
            annotationId += 1
        cocoOut['images'][-1]['category_ids'] = list(
            cocoOut['images'][-1]['category_ids'])
    with open('blobsFromEggs.json', 'w') as f:
        json.dump(cocoOut, f, ensure_ascii=False, indent=4, sort_keys=True)


if __name__ == "__main__":
    main()
