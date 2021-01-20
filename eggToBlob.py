import argparse, json, random

from pycocotools import coco

from utils.color import randColor
import utils.polygon as polygon

BLOB_CATEGORY_ID = 2

def options():
    """Parse options for the egg-to-blob conversion tool."""
    p = argparse.ArgumentParser(description='Convert a COCO dataset of egg' +\
        ' annotations to blob format using the union operation.')
    p.add_argument('datasetPath', help='Path to the COCO-formatted dataset to' +
                   ' transform from egg to blob form.')
    return p.parse_args()

def main():
    cocoOut = {'images': [], 'categories': [], 'annotations': []}
    annotationId = 1
    opts = options()
    dataset = coco.COCO(opts.datasetPath)
    cocoOut['categories'] = dataset.dataset['categories']
    cocoOut['categories'][-1]['id'] = BLOB_CATEGORY_ID
    cocoOut['categories'][-1]['name'] = 'blob'
    cocoOut['categories'][-1]['color'] = randColor()
    for i, imgIdx in enumerate(dataset.imgs):
        if i > 5:
            continue
        annotations = dataset.imgToAnns[imgIdx]
        cocoOut['images'].append(dataset.imgs[imgIdx])
        unitedPolygons = polygon.unitePolygons(annotations, imgIdx)
        print('the image is', cocoOut['images'][-1]['file_name'])
        flattenedSegs = polygon.flattenSegs(unitedPolygons)
        for seg in flattenedSegs:
            area, bb =\
                polygon.calculateAreaAndBBox(
                    seg, (cocoOut['images'][-1]['width'],
                          cocoOut['images'][-1]['height']))
            cocoOut['annotations'].append({
                'id': annotationId,
                'image_id': cocoOut['images'][-1]['id'],
                'category_id': BLOB_CATEGORY_ID,
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
    with open('blobsFromEggs.json', 'w') as f:
        json.dump(cocoOut, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
