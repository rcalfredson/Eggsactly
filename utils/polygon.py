import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask
from shapely.geometry import Polygon
from shapely.geometry.multipolygon import MultiPolygon
from shapely.ops import unary_union
matplotlib.use('TkAgg')

from .color import randColor

def calculateAreaAndBBox(segs, imgDims):
    blankImg = Image.new("L", imgDims, 0)
    ImageDraw.Draw(blankImg).polygon([int(el)
                                      for el in segs[0]], outline=1, fill=1)
    if len(segs) > 1:
        for seg in segs[1:]:
            ImageDraw.Draw(blankImg).polygon(
                [int(el) for el in seg], outline=0, fill=0)
    reconstructedMask = np.array(blankImg)
    fortran_ground_truth_binary_mask = np.asfortranarray(reconstructedMask)
    encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
    ground_truth_area = mask.area(encoded_ground_truth)
    ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
    return ground_truth_area, ground_truth_bounding_box

def calculateOverlaps(parents, children):
    """Return a dictionary whose keys are the unique ids of the parents and
    whose values are the number of children that intersect with that parent.

    Arguments:
      - parents:
    """
    if type(parents) == Polygon:
        parents = MultiPolygon([parents])
    parentFakeHashes = [polyFakeHash(parent) for parent in parents.geoms]

    parentsToChildren = {parentFakeHashes[i]: [] for i, parent in enumerate(parents.geoms)}
    overlaps = {parentFakeHashes[i]: 0 for i, parent in enumerate(parents.geoms)}
    for child in children:
        for i, parent in enumerate(parents.geoms):
            if child.intersects(parent):
                overlaps[parentFakeHashes[i]] += 1
                parentsToChildren[parentFakeHashes[i]].append(polyFakeHash(child))
                continue
    return overlaps, parentsToChildren

def toPolyFakeHashDict(multipoly):
    if type(multipoly) == Polygon:
        multipoly = MultiPolygon([multipoly])
    if type(multipoly) == list:
        return {polyFakeHash(polygon): polygon for polygon in multipoly}
    return {polyFakeHash(polygon): polygon for polygon in multipoly.geoms}

def polyFakeHash(polygon):
    """Return a pseudo-hash for a polygon, consisting of its first two X
    coordinates rounded to the hundredths place, connected by a dash.
    """
    if type(polygon) == list:
        coordArgs = polygon[0][:4:2]
    else:
        coordArgs = [polygon.exterior.coords[i][0] for i in range(2)]
    return '%.2f-%.2f'%(tuple(coordArgs))

def flattenSeg(seg):
    def flatten(l):
        return [item for sublist in l for item in sublist]
    segOut = [flatten(list(seg.exterior.coords))]
    holes = [list(ring.coords) for ring in list(seg.interiors)]
    for holeCoords in holes:
        segOut.append(flatten(holeCoords))
    return segOut

def flattenSegs(segs):
    segsOut = []

    if type(segs) == Polygon:
        segs = MultiPolygon([segs])
    for seg in segs:
        segsOut += flattenSeg(seg)
    return segsOut

def unflattenSegs(segs):
    newSegs = []
    for seg in segs:
        seg = [[seg[2*i], seg[2*i+1]] for i in range(int(len(seg)/2))]
        newSegs.append(seg)
    return newSegs

def unitePolygons(anns, imgIdx):
    polygons = []
    if type(anns[0]) is Polygon:
        polygons = anns
    else:
        for ann in anns:
            seg = unflattenSegs(ann['segmentation'])
            try:
                polygons.append(
                    Polygon(seg[0], None if len(seg) == 1 else seg[1:]))
            except ValueError as err:
                print('encountered error while creating polygon for image',
                    imgIdx, ':', err)
                exit(0)
    try:
        unifiedShapes = unary_union(polygons)
    except ValueError as err:
        print('encountered error while trying to form unary union for image',
              imgIdx, ':', err)
        exit(0)
    return unifiedShapes, polygons

def visualizeDebug(polygons):
    colors = [randColor() for _ in range(len(polygons))]
    gdf = gpd.GeoDataFrame({'geometry': polygons, 'colors': colors})
    randColorMap = matplotlib.colors.ListedColormap(np.random.rand(256, 3))
    gdf.plot(cmap=randColorMap)
    plt.show()
