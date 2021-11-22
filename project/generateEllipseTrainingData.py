import itertools
import json
import os
import pickle
import random
import timeit

import cv2
from descartes import PolygonPatch
from matplotlib import pyplot
from matplotlib.collections import PolyCollection
import numpy as np
import shapely.affinity
from shapely.figures import GRAY, RED, GREEN, set_limits
from shapely.geometry import box, Polygon
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.point import Point
from shapely.ops import unary_union

from common import randID
from utils.coco import getNewCategory
from utils.color import randColor
import utils.polygon as polygon

semiAxes = (12, 5)
xBounds, yBounds = (0, 190), (0, 500)
maxNumExamples = 200
numTypes = 20
numExamplesPerType = {i+1: 0 for i in range(numTypes)}
cocoOut = {'annotations': [], 'categories': [],
           'images': []}
categoriesAdded = set()


def getRandEllipse(center=None, stayInBorder=True, polygonToTouch=None):
    borderRect = box(xBounds[0], yBounds[0], xBounds[1], yBounds[1])
    distDelta = 24
    numAttempts, maxAttempts = 0, 50

    def makeEll():
        if center:
            ptRange = {'x': [center.x - distDelta, center.x + distDelta],
                       'y': [center.y - distDelta, center.y + distDelta]}
        else:
            ptRange = {'x': xBounds, 'y': yBounds}
        circ = Point((np.random.uniform(low=ptRange['x'][0],
                                        high=ptRange['x'][1]),
                      np.random.uniform(low=ptRange['y'][0],
                                        high=ptRange['y'][1])))
        circBuffered = circ.buffer(1)
        ell = shapely.affinity.scale(circBuffered, semiAxes[0], semiAxes[1])
        return shapely.affinity.rotate(ell, np.random.uniform(low=0, high=90)), circ
    ell, newCenter = makeEll()
    def tryAgain():
        if stayInBorder and not ell.within(borderRect):
            return True
        if polygonToTouch and (not polygonToTouch.intersects(ell)\
            or polygonToTouch.intersection(ell).area/polygonToTouch.area > 0.05):
            return True
        return False
    while tryAgain():
        if numAttempts == maxAttempts:
            return None
        # print('circ:', newCenter.x, newCenter.y)
        ell, newCenter = makeEll()
        numAttempts += 1
    # print('first condition: in border?')
    # print(ell.within(borderRect))
    # if polygonToTouch != None:
        # print('does new poly touch the old one?')
        # print(polygonToTouch.intersects(ell))
    return ell, newCenter


def removeNonOverlaps(polygons, numSinglesToKeep=4):
    overlappingPolygons = dict()
    nonIntersectingPolygons = dict()
    intersectingKeys = set()
    for geom1, geom2 in itertools.combinations(polygons, 2):
        geom1Hash, geom2Hash = polygon.polyFakeHash(
            geom1), polygon.polyFakeHash(geom2)
        nonIntersectingPolygons[geom1Hash] = geom1
        nonIntersectingPolygons[geom2Hash] = geom2
        if geom1.intersects(geom2):
            overlappingPolygons[geom1Hash] = geom1
            overlappingPolygons[geom2Hash] = geom2
            del nonIntersectingPolygons[geom1Hash]
            if geom2Hash in nonIntersectingPolygons:
                del nonIntersectingPolygons[geom2Hash]
            intersectingKeys.add(geom1Hash)
            intersectingKeys.add(geom2Hash)
    for singleKey in list(nonIntersectingPolygons)[:numSinglesToKeep]:
        overlappingPolygons[singleKey] = nonIntersectingPolygons[singleKey]
    return overlappingPolygons


def getCategoriesToAdd(overlaps):
    newCats = set()
    overlapsToAdd = []
    categoriesForImage = set()
    for overlapKey in overlaps:
        numOverlaps = overlaps[overlapKey]
        # print('how many overlaps?', numOverlaps)
        if numOverlaps in numExamplesPerType and\
                numExamplesPerType[numOverlaps] < maxNumExamples:
            overlapsToAdd.append(overlapKey)
            categoriesForImage.add(numOverlaps)
            numExamplesPerType[numOverlaps] += 1
            if int(numOverlaps) not in categoriesAdded:
                newCats.add(numOverlaps)
                categoriesAdded.add(numOverlaps)

    return overlapsToAdd, categoriesForImage, newCats

def addAnnotation(blob, category_id):
    flattenedPoly = polygon.flattenSegs(blob)
    annotationId = len(cocoOut['annotations']) + 1
    # print('flattened poly?', flattenedPoly)
    area, bb =\
            polygon.calculateAreaAndBBox(
                flattenedPoly, (cocoOut['images'][-1]['width'],
                                cocoOut['images'][-1]['height']))
    cocoOut['annotations'].append({
        'area': int(area),
            'bbox': bb.tolist(),
            'category_id': category_id,
            'color': randColor(),
            'id': annotationId,
            'image_id': cocoOut['images'][-1]['id'],
            'isbbox': False,
            'iscrowd': False,
            'keypoints': [],
            'metadata': {},
            'segmentation': flattenedPoly
    })

def addAnnotations_old(keysToAdd, polygonHash, overlaps):
    if len(keysToAdd) == 0:
        return
    for polygonKey in keysToAdd:
        polyToAdd = polygonHash[polygonKey]
        flattenedPoly = polygon.flattenSeg(polyToAdd)
        area, bb =\
            polygon.calculateAreaAndBBox(
                flattenedPoly, (cocoOut['images'][-1]['width'],
                                cocoOut['images'][-1]['height']))
        annotationId = len(cocoOut['annotations']) + 1
        catId = int(overlaps[polygonKey])
        annotTemp = {
            'area': int(area),
            'bbox': bb.tolist(),
            'category_id': catId,
            'color': randColor(),
            'id': annotationId,
            'image_id': cocoOut['images'][-1]['id'],
            'isbbox': False,
            'iscrowd': False,
            'keypoints': [],
            'metadata': {},
            'segmentation': flattenedPoly
        }
        if catId == 4:
            with open('savedObject', 'wb') as f:
                pickle.dump(annotTemp, f)
        cocoOut['annotations'].append(annotTemp)


def addImage(categoryForImage, coll):
    filePath = 'generated/%s.png' % randID()
    pyplot.savefig(filePath, bbox_inches='tight', transparent=False,
                   pad_inches=0, dpi=135.5)
    coll.remove()
    imageId = len(cocoOut['images']) + 1
    img = cv2.imread(filePath)
    height, width = img.shape[0:2]
    cocoOut['images'].append({
        'id': imageId,
        'category_ids': [categoryForImage],
        'path': filePath,
        'width': width,
        'height': height,
        'file_name': os.path.basename(filePath),
        'annotated': True,
        'annotating': [],
        'num_annotations': 0,
        'deleted': False,
        'events': [],
        'metadata': {},
        'milliseconds': 0,
        'regenerate_thumbnail': False
    })


def getBlob(numEggs):
    ellInit, center = getRandEllipse()
    # print('center of first ellipse:', center)
    ellipses = [ellInit]
    centers = [center]
    # generate random points around the ellipse
    for _ in range(numEggs - 1):
        randIndex = random.randint(0, len(ellipses) - 1)
        ellResult = getRandEllipse(centers[randIndex], polygonToTouch=unary_union(ellipses))
        if ellResult == None:
            print('skipping bad blob.')
            return getBlob(numEggs)
        ellipses.append(ellResult[0])
        centers.append(ellResult[1])
    return polygon.unitePolygons(ellipses, '')


if __name__ == "__main__":
    eggsRendered = 10
    eggsPerBlob = 1
    numBlobsMade = 0

    def calcQuartiles():
        return [np.floor(eggsRendered*i) - 1 for i in np.arange(0.02, 1.02, 0.02)]
    quartiles = calcQuartiles()
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    set_limits(ax, xBounds[0], xBounds[1], yBounds[0], yBounds[1])
    axes = pyplot.gca()
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    pyplot.axis('off')
    pyplot.gca().invert_yaxis()
    while eggsPerBlob <= numTypes:
        iterStart = timeit.default_timer()
        blob, childEllipses = getBlob(eggsPerBlob)
        # print('how many child ellipses:', len(childEllipses))
        print('eggsPerBlob:', eggsPerBlob)
        print('numBlobsMade:', numBlobsMade)
        numBlobsMade += 1
        blob = blob.simplify(
            0.15, preserve_topology=True)
        coll = PolyCollection([poly.exterior.coords for poly in childEllipses])
        coll.set_alpha(0.5)
        coll.set_color('gray')
        ax.add_collection(coll)
        # for ell in childEllipses:
            # print('childEllipses:', childEllipses)
            # print('ell:', ell)
            # patch = PolygonPatch(
                # ell, fc=GRAY, ec=GRAY, alpha=0.5, zorder=2)
            # ax.add_patch(patch)
        pyplot.tick_params(axis='both', which='both', bottom=False, top=False,
                           labelbottom=False, right=False, left=False, labelleft=False)
        addImage(eggsPerBlob, coll)
        if type(blob) == Polygon:
            blob = MultiPolygon([blob])
        addAnnotation(blob, eggsPerBlob)
        if numBlobsMade == 1:
            cocoOut['categories'].append(getNewCategory(eggsPerBlob))
        if numBlobsMade == maxNumExamples:
            eggsPerBlob += 1
            numBlobsMade = 0
        print('time per iteration:', timeit.default_timer() - iterStart)
    with open('generated/ellipseDataset.json', 'w') as f:
        json.dump(cocoOut, f, ensure_ascii=False, indent=4, sort_keys=True)
    exit(0)
    while not all(np.array(list(numExamplesPerType.values())) == maxNumExamples):
        ellipses = []
        ellipseCenters = []
        for i in range(eggsRendered):
            if len(ellipses) == 0:
                ellResult = getRandEllipse()
                ellipses.append(ellResult[0])
                ellipseCenters.append(ellResult[1])
            else:
                # how to find nearest 25th or nth quartile?
                if i in quartiles:
                    center = None
                else:
                    center = random.choice(ellipseCenters)
                ellResult = getRandEllipse(center)
                ellipses.append(ellResult[0])
                ellipseCenters.append(ellResult[1])
        ellipses = removeNonOverlaps(ellipses)
        unitedPolygons, polygons = polygon.unitePolygons(list(
            ellipses.values()), '')
        overlaps, parentsToChildren = polygon.calculateOverlaps(
            unitedPolygons, polygons)
        ellipseHash, blobHash = polygon.toPolyFakeHashDict(
            polygons), polygon.toPolyFakeHashDict(unitedPolygons)
        overlapsToAdd, categoriesForImage, newCategories = getCategoriesToAdd(
            overlaps)
        print('egg count:', eggsRendered)
        if len(newCategories):
            for cat in newCategories:
                cocoOut['categories'].append(getNewCategory(cat))
        if len(categoriesForImage) == 0:
            eggsRendered += 1
            quartiles = calcQuartiles()
        print('numexamplesPerType:', numExamplesPerType)
        fig = pyplot.figure()
        ax = fig.add_subplot(111)
        for overlapShape in overlapsToAdd:
            childEllipses = parentsToChildren[overlapShape]
            blobHash[overlapShape] = blobHash[overlapShape].simplify(
                0.15, preserve_topology=True)
            for ell in childEllipses:
                # print('childEllipses:', childEllipses)
                # print('ell:', ell)
                ell = ellipseHash[ell]
                patch = PolygonPatch(
                    ell, fc=GRAY, ec=GRAY, alpha=0.5, zorder=2)
                ax.add_patch(patch)
        set_limits(ax, xBounds[0], xBounds[1], yBounds[0], yBounds[1])
        pyplot.tick_params(axis='both', which='both', bottom=False, top=False,
                           labelbottom=False, right=False, left=False, labelleft=False)
        # pyplot.show()
        addImage(categoriesForImage, fig)
        addAnnotations(overlapsToAdd, blobHash, overlaps)
    with open('generated/ellipseDataset.json', 'w') as f:
        json.dump(cocoOut, f, ensure_ascii=False, indent=4, sort_keys=True)
