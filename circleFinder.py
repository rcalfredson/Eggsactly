import itertools
import operator
import os
import math
import threading
import cv2
import timeit
from adet.config import get_cfg
from predictor import VisualizationDemo
from skimage.measure import label
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

from util import *
from chamber import CT

CONFIDENCE_THRESHOLD = 0.5


def setup_cfg():
    """Load config of the detection model from file and command-line arguments."""
    cfg = get_cfg()
    dirname = os.path.dirname(__file__)
    cfg.merge_from_file(os.path.join(dirname, "configs/arena_pit.yaml"))
    cfg.merge_from_list(
        ["MODEL.WEIGHTS", os.path.join(dirname, "models/arena_pit.pth")]
    )
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = CONFIDENCE_THRESHOLD
    cfg.freeze()
    return cfg


def centroidnp(arr):
    """Return the column-wise averages for a first 2 columns of a Numpy array.

    Arguments:
      - arr: Numpy array with at least two columns.
    """
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x / length, sum_y / length


# corner-finding (source: https://stackoverflow.com/a/20354078/13312013)
def fake_image_corners(xy_sequence):
    """Get an approximation of image corners based on available data."""
    all_x, all_y = [
        tuple(xy_sequence[:, :, i].flatten()) for i in range(xy_sequence.shape[-1])
    ]
    min_x, max_x, min_y, max_y = min(all_x), max(all_x), min(all_y), max(all_y)
    d = dict()
    d["tl"] = min_x, min_y
    d["tr"] = max_x, min_y
    d["bl"] = min_x, max_y
    d["br"] = max_x, max_y
    return d


def fake_image_corners_old(xy_sequence):
    """Get an approximation of image corners based on available data."""
    all_x, all_y = zip(*xy_sequence)
    min_x, max_x, min_y, max_y = min(all_x), max(all_x), min(all_y), max(all_y)
    d = dict()
    d["tl"] = min_x, min_y
    d["tr"] = max_x, min_y
    d["bl"] = min_x, max_y
    d["br"] = max_x, max_y
    return d


def corners(xy_sequence, image_corners):
    """Return a dict with the best point for each corner."""
    d = dict()
    seq_shape = xy_sequence.shape
    xy_sequence = xy_sequence.reshape(seq_shape[0] * seq_shape[1], -1)
    d["tl"] = min(xy_sequence, key=lambda xy: distance(xy, image_corners["tl"]))
    d["tr"] = min(xy_sequence, key=lambda xy: distance(xy, image_corners["tr"]))
    d["bl"] = min(xy_sequence, key=lambda xy: distance(xy, image_corners["bl"]))
    d["br"] = min(xy_sequence, key=lambda xy: distance(xy, image_corners["br"]))
    return d


def corners_old(xy_sequence, image_corners):
    """Return a dict with the best point for each corner."""
    d = dict()
    d["tl"] = min(xy_sequence, key=lambda xy: distance(xy, image_corners["tl"]))
    d["tr"] = min(xy_sequence, key=lambda xy: distance(xy, image_corners["tr"]))
    d["bl"] = min(xy_sequence, key=lambda xy: distance(xy, image_corners["bl"]))
    d["br"] = min(xy_sequence, key=lambda xy: distance(xy, image_corners["br"]))
    return d


# end corner-finding

cfg = setup_cfg()
model = VisualizationDemo(cfg)


def getChamberTypeByRowsAndCols(numRowsCols):
    """Return a chamber type name based on a match with the number of rows and
    columns.

    Arguments:
      - numRowsCols: list of the form [numRows, numCols]
    """
    for ct in CT:
        if (
            numRowsCols[0] == ct.value().numRows
            and numRowsCols[1] == ct.value().numCols
        ):
            return ct.name


def subImagesFromGridPoints(img, xs, ys):
    """Split an image into a grid determined by the inputted X and Y coordinates.
    The returned images are organized in a row-dominant order.

    Arguments:
      - img: an image to segment
      - xs: grid points along X axis
      - ys: grid points along Y axis
    """
    subImgs = []
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            subImgs.append(img[y : ys[j + 1], x : xs[i + 1]])
    return subImgs


def subImagesFromBBoxes(img, bboxes):
    """Split an image according to the inputted bounding boxes (whose order
    determines the order of the returned sub-images).

    Arguments:
      - img: an image to segment
      - bboxes: a list of bounding boxes. Each bounding box is a list of the form
                [x_min, y_min, width, height] in pixels.
    """
    subImgs = []
    for bbox in bboxes:
        subImgs.append(img[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]])
    return subImgs


class CircleFinder:
    """Detect landmarks in the center of the egg-laying chambers to use in
    segmenting the image.
    """

    def __init__(self, img, imgName, allowSkew=False):
        """Create new CircleFinder instance.

        Arguments:
          - img: the image to analyze, in Numpy array form
          - imgName: the basename of the image
          - allowSkew: boolean which, if set to False, registers an error if skew is
                       detected in the image.
        """
        self.img = img
        self.imgName = imgName
        self.skewed = None
        self.allowSkew = allowSkew

    def getPixelToMMRatio(self):
        """Calculate the image's ratio of pixels to mm, averaged between the result
        for rows and for columns."""
        self.pxToMM = 0.5 * (
            self.avgDists[1] / CT[self.ct].value().rowDist
            + self.avgDists[0] / CT[self.ct].value().colDist
        )

    @staticmethod
    def getLargeChamberBBoxesAndImages(img, centers, pxToMM):
        bboxes, subImgs = [], []
        for center in centers:
            outerEdgeDistance = round((1 + 8.6 + 8 + 1) * pxToMM)
            acrossCircleD = round(10.5 * pxToMM)
            halfRealCircleD = round((0.5 * 10) * pxToMM)
            centerAdjacentDistance = round(1 * pxToMM)

            bboxes.append(
                [
                    round(max(center[0] - halfRealCircleD, 0)),
                    round(max(center[1] - outerEdgeDistance, 0)),
                ]
            )
            subImgs.append(
                img[
                    slice(round(bboxes[-1][1]), round(center[1] - centerAdjacentDistance)),
                    slice(round(bboxes[-1][0]), round(bboxes[-1][0] + acrossCircleD)),
                ]
            )
            bboxes = CircleFinder.addWidthAndHeightToBBox(bboxes, subImgs)

            bboxes.append(
                [
                    round(center[0] + centerAdjacentDistance),
                    round(max(center[1] - halfRealCircleD, 0)),
                ]
            )
            subImgs.append(
                img[
                    slice(round(bboxes[-1][1]), round(bboxes[-1][1] + acrossCircleD)),
                    slice(round(bboxes[-1][0]), round(center[0] + outerEdgeDistance)),
                ]
            )
            bboxes = CircleFinder.addWidthAndHeightToBBox(bboxes, subImgs)

            bboxes.append(
                [
                    round(max(center[0] - halfRealCircleD, 0)),
                    round(center[1] + centerAdjacentDistance),
                ]
            )
            subImgs.append(
                img[
                    slice(round(bboxes[-1][1]), round(center[1] + outerEdgeDistance)),
                    slice(round(bboxes[-1][0]), round(bboxes[-1][0] + acrossCircleD)),
                ]
            )
            bboxes = CircleFinder.addWidthAndHeightToBBox(bboxes, subImgs)

            bboxes.append(
                [
                    round(max(center[0] - outerEdgeDistance, 0)),
                    round(max(center[1] - halfRealCircleD, 0)),
                ]
            )
            subImgs.append(
                img[
                    slice(round(bboxes[-1][1]), round(bboxes[-1][1] + acrossCircleD)),
                    slice(round(bboxes[-1][0]), round(center[0] - centerAdjacentDistance)),
                ]
            )
            bboxes = CircleFinder.addWidthAndHeightToBBox(bboxes, subImgs)
        return bboxes, subImgs

    @staticmethod
    def addWidthAndHeightToBBox(bboxes, subImgs):
        bboxes[-1] += reversed(subImgs[-1].shape[0:2])
        return bboxes

    def getSubImages(self, img, centers, avgDists, numRowsCols):
        """Determine sub-images for the image based on the chamber type and the
        locations of detected arena wells.

        Arguments:
          - img: the image to segment.
          - centers: list of detected wells for the image (each center is a tuple
                     ordered pair of X and Y coordinates)
          - avgDists: tuple list of the average distances along X and Y direction
                      between neighboring wells
          - numRowsCols: tuple list of the number of rows and columns of wells.

        Returns:
          - sortedSubImgs: list of the sub-images in Numpy array form (CV2-ready)
          - sortedBBoxes: list of the bounding boxes for each sub-image
        """
        subImgs, bboxes = [], []

        def addWidthAndHeightToBBox():
            bboxes[-1] += reversed(subImgs[-1].shape[0:2])

        self.getPixelToMMRatio()
        pxToMM = self.pxToMM
        if self.ct is CT.large.name:
            bboxes, subImgs = self.getLargeChamberBBoxesAndImages(img, centers, pxToMM)
        else:
            for center in centers:
                if self.ct is CT.opto.name:
                    bboxes.append(
                        [
                            max(center[0] - int(0.5 * avgDists[0]), 0),
                            max(center[1] - int(8.5 * pxToMM), 0),
                        ]
                    )
                    subImgs.append(
                        img[
                            bboxes[-1][1] : center[1] - int(4 * pxToMM),
                            bboxes[-1][0] : center[0] + int(0.5 * avgDists[0]),
                        ]
                    )
                    bboxes = CircleFinder.addWidthAndHeightToBBox(bboxes, subImgs)
                    bboxes.append(
                        [
                            max(center[0] - int(0.5 * avgDists[0]), 0),
                            max(center[1] + int(4 * pxToMM), 0),
                        ]
                    )
                    subImgs.append(
                        img[
                            bboxes[-1][1] : center[1] + int(8.5 * pxToMM),
                            bboxes[-1][0] : center[0] + int(0.5 * avgDists[0]),
                        ]
                    )
                    bboxes = CircleFinder.addWidthAndHeightToBBox(bboxes, subImgs)
                else:
                    bboxes.append(
                        [
                            max(center[0] - int(8.5 * pxToMM), 0),
                            max(center[1] - int(0.5 * avgDists[1]), 0),
                        ]
                    )
                    subImgs.append(
                        img[
                            bboxes[-1][1] : center[1] + int(0.5 * avgDists[1]),
                            bboxes[-1][0] : center[0] - int(4 * pxToMM),
                        ]
                    )
                    bboxes = CircleFinder.addWidthAndHeightToBBox(bboxes, subImgs)
                    bboxes.append(
                        [
                            max(center[0] + int(4 * pxToMM), 0),
                            max(center[1] - int(0.5 * avgDists[1]), 0),
                        ]
                    )
                    subImgs.append(
                        img[
                            bboxes[-1][1] : center[1] + int(0.5 * avgDists[1]),
                            bboxes[-1][0] : center[0] + int(8.5 * pxToMM),
                        ]
                    )
                    bboxes = CircleFinder.addWidthAndHeightToBBox(bboxes, subImgs)
        if self.ct is CT.opto.name:
            return CT.opto.value().getSortedSubImgs(subImgs, bboxes)
        sortedSubImgs = []
        sortedBBoxes = []
        for j in range(numRowsCols[0]):
            for i in range(numRowsCols[1]):
                offset = 4 if self.ct is CT.large.name else 2
                idx = numRowsCols[0] * offset * i + offset * j
                sortedSubImgs.append(subImgs[idx])
                sortedSubImgs.append(subImgs[idx + 1])
                sortedBBoxes.append(bboxes[idx])
                sortedBBoxes.append(bboxes[idx + 1])
                if self.ct is CT.large.name:
                    for k in range(2, 4):
                        sortedBBoxes.append(bboxes[idx + k])
                        sortedSubImgs.append(subImgs[idx + k])
        return sortedSubImgs, sortedBBoxes

    def processDetections(self):
        """
        Consolidate the arena well detections by organizing their X and Y
        coordinates into histograms and finding the bins with local maxima.

        For all chamber types excluding 4-circle, interpolate any missing detections
        using linear regressions.

        Check if the image is skewed, and flag it accordingly.
        """
        self.yDetections = np.asarray([centroid[1] for centroid in self.centroids])
        self.xDetections = np.asarray([centroid[0] for centroid in self.centroids])
        self.wellCoords = [[], []]
        for detI, detections in enumerate((self.xDetections, self.yDetections)):
            histResults = binned_statistic(detections, [], bins=40, statistic="count")
            binHtsOrig = histResults.statistic
            binClusters = trueRegions(binHtsOrig > 0)

            for i, trueRegion in enumerate(binClusters):
                self.wellCoords[detI].append(
                    int(
                        round(
                            np.mean(
                                [
                                    detections[
                                        (histResults.binnumber - 1 >= trueRegion.start)
                                        & (histResults.binnumber <= trueRegion.stop)
                                    ]
                                ]
                            )
                        )
                    )
                )
        for i in range(len(self.wellCoords)):
            self.wellCoords[i] = sorted(self.wellCoords[i])
            self.wellCoords[i] = reject_outliers_by_delta(
                np.asarray(self.wellCoords[i])
            )

        wells = list(itertools.product(self.wellCoords[0], self.wellCoords[1]))
        self.img = np.array(self.img)
        self.numRowsCols = [len(self.wellCoords[i]) for i in range(1, -1, -1)]
        self.ct = getChamberTypeByRowsAndCols(self.numRowsCols)
        diagDist = distance((0, 0), self.imageResized.shape[0:2])
        for centroid in list(self.centroids):
            closestWell = min(wells, key=lambda xy: distance(xy, centroid))
            if distance(closestWell, centroid) > 0.02 * diagDist:
                self.centroids.remove(centroid)
        self.sortedCentroids = []
        for i, well in enumerate(wells):
            closestDetection = min(self.centroids, key=lambda xy: distance(xy, well))
            if self.ct is CT.large.name:
                self.sortedCentroids.append(well)
                continue
            if distance(closestDetection, well) > 0.02 * diagDist:
                self.sortedCentroids.append((np.nan, np.nan))
            else:
                self.sortedCentroids.append(closestDetection)
        self.sortedCentroids = np.array(self.sortedCentroids).reshape(
            (
                tuple(reversed(self.numRowsCols))
                + (() if None in self.sortedCentroids else (-1,))
            )
        )
        self.rowRegressions = np.zeros(self.sortedCentroids.shape[1]).astype(object)
        self.colRegressions = np.zeros(self.sortedCentroids.shape[0]).astype(object)
        if self.ct is not CT.large.name:
            self.interpolateCentroids()
        prelim_corners = fake_image_corners(self.sortedCentroids)
        true_corners = corners(self.sortedCentroids, prelim_corners)
        width_skew = abs(
            distance(true_corners["tr"], true_corners["tl"])
            - distance(true_corners["br"], true_corners["bl"])
        )
        height_skew = abs(
            distance(true_corners["br"], true_corners["tr"])
            - distance(true_corners["bl"], true_corners["tl"])
        )
        if (
            height_skew / self.imageResized.shape[0] > 0.01
            or width_skew / self.imageResized.shape[1] > 0.01
        ):
            self.skewed = True
        else:
            self.skewed = False
        if self.skewed and not self.allowSkew:
            print(
                "Warning: skew detected in image %s. To analyze " % self.imgName
                + "this image, use flag --allowSkew."
            )
            currentThread = threading.currentThread()
            setattr(currentThread, "hadError", True)

    def interpolateCentroids(self):
        """Find any centroids with NaN coordinates and interpolate their positions
        based on neighbors in their row and column.
        """
        for i, col in enumerate(self.sortedCentroids):
            for j, centroid in enumerate(col):
                row = self.sortedCentroids[:, j]
                regResult = calculateRegressions(row, col)
                if j == 0:
                    self.colRegressions[i] = regResult["col"]
                if i == 0:
                    self.rowRegressions[j] = regResult["row"]
                if len(centroid[np.isnan(centroid)]):
                    row = self.sortedCentroids[:, j]
                    self.sortedCentroids[i, j] = linearIntersection(
                        dict(row=self.rowRegressions[j], col=self.colRegressions[i])
                    )

    def findCircles(self, debug=False):
        """Find the location of arena wells for the image in attribute `self.img`.

        Returns:
          - wells: list of the coordinates of detected wells.
          - avgDists: tuple list of the average distances along X and Y direction
                      between neighboring wells.
          - numRowsCols: tuple list of the number of rows and columns of wells.
          - rotatedImg: `self.img` after being rotated to best align rows and
                        columns with the border of the image.
          - rotationAngle: angle in radians by which the image was rotated.
        """
        self.imageResized = cv2.resize(
            self.img, (0, 0), fx=0.15, fy=0.15, interpolation=cv2.INTER_CUBIC
        )
        predictions, _ = model.run_on_image(self.imageResized)
        predictions = [
            predictions["instances"].pred_masks.cpu().numpy()[i, :, :]
            for i in range(predictions["instances"].pred_masks.shape[0])
        ]
        self.centroids = [
            centroidnp(np.asarray(list(zip(*np.where(prediction == 1)))))
            for prediction in predictions
        ]
        self.centroids = [tuple(reversed(centroid)) for centroid in self.centroids]
        if debug:
            print("what are centroids?", self.centroids)
            imgCopy = np.array(self.imageResized).astype(np.uint8)
            for centroid in self.centroids:
                cv2.drawMarker(
                    imgCopy,
                    tuple([int(el) for el in centroid]),
                    COL_G,
                    cv2.MARKER_TRIANGLE_UP,
                )
            cv2.imshow("debug", imgCopy)
            cv2.waitKey(0)
        self.processDetections()
        rotationAngle = 0
        rotatedImg = self.img
        image_origin = tuple(np.array(self.imageResized.shape[1::-1]) / 2)
        if self.ct is not CT.large.name:
            rotationAngle = 0.5 * (
                math.atan(np.mean([el["slope"] for el in self.rowRegressions]))
                - math.atan(np.mean([1 / el["slope"] for el in self.colRegressions]))
            )
            rotatedImg = rotate_image(self.img, rotationAngle)
            for i, centroid in enumerate(self.centroids):
                self.centroids[i] = rotate_around_point_highperf(
                    centroid, rotationAngle, image_origin
                )
        self.processDetections()
        wells = np.array(
            [
                np.round(np.divide(well, 0.15)).astype(int)
                for well in self.sortedCentroids
            ]
        )
        for i in range(len(self.wellCoords)):
            self.wellCoords[i] = np.round(np.divide(self.wellCoords[i], 0.15)).astype(
                int
            )
        wells = wells.reshape(self.numRowsCols[0] * self.numRowsCols[1], 2).astype(int)
        self.wells = wells
        self.avgDists = [np.mean(np.diff(self.wellCoords[i])) for i in range(2)]
        self.rotatedImg, self.rotationAngle = rotatedImg, rotationAngle
        return (
            wells,
            self.avgDists,
            self.numRowsCols,
            self.rotatedImg,
            self.rotationAngle,
        )


def rotate_image(image, angle):
    """Rotate image by an inputted angle.

    Arguments:
      - image: the image to rotate.
      - angle: the degree to which to rotate (in radians).
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, 180 * angle / math.pi, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def rotate_around_point_highperf(xy, radians, origin=(0, 0)):
    """Rotate a point around a given point.
    source: https://gist.github.com/LyleScott/e36e08bfb23b1f87af68c9051f985302

    Arguments:
      - xy: tuple containing two Numpy arrays of the X and Y coordinates of
            points to rotate
      - radians: angle in radians by which to rotate the points
      - origin: origin about which to rotate the points (default: (0, 0))
    """
    x, y = xy
    offset_x, offset_y = origin
    adjusted_x = x - offset_x
    adjusted_y = y - offset_y
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
    return [qx, qy]


def calculateRegressions(row, col):
    """Calculate linear regressions for the Y values in the list of points in a
    given column and the X values in the list of points in a given row.

    Arguments:
      - row: list of points (XY tuples) in a row of interest
      - col: list of points (XY tuples) in a column of interest
    """
    colModel = LinearRegression()
    colInd = np.array([el[1] for el in col if not np.isnan(el).any()]).reshape(-1, 1)
    colModel.fit(colInd, [el[0] for el in col if not np.isnan(el).any()])
    rowModel = LinearRegression()
    rowInd = np.array([el[0] for el in row if not np.isnan(el).any()]).reshape(-1, 1)
    rowModel.fit(rowInd, [el[1] for el in row if not np.isnan(el).any()])
    a = (1 / colModel.coef_)[0]
    c = (-colModel.intercept_ / colModel.coef_)[0]
    b = rowModel.coef_[0]
    d = rowModel.intercept_

    return dict(row=dict(slope=b, intercept=d), col=dict(slope=a, intercept=c))


def linearIntersection(regressions):
    """Calculate the intersection of two linear regressions.

    Arguments:
      - regression: dictionary of the form
      {'row': {'slope': number, 'intercept': number},
       'col': {'slope': number, 'intercept': number}}
    """
    r = regressions
    interceptDiff = r["row"]["intercept"] - r["col"]["intercept"]
    slopeDiff = r["col"]["slope"] - r["row"]["slope"]
    return (
        (interceptDiff) / (slopeDiff),
        r["col"]["slope"] * interceptDiff / slopeDiff + r["col"]["intercept"],
    )


def reject_outliers_by_delta(binCenters, m=1.3):
    """Reject outliers based on the magnitude of their difference from neighboring
    points.

    Arguments:
      - binCenters: 1D Numpy array of values to check
      - m: sensitivity of the outlier test, smaller for more sensitivity
          (default: 1.3)
    """
    diffs = np.diff(binCenters)
    outIdxs = list(range(len(binCenters)))
    delta_mags = abs(diffs - np.mean(diffs))
    if np.all(delta_mags < 3):
        return binCenters
    idxs = np.squeeze(np.argwhere(~(delta_mags < m * np.std(diffs))))
    if idxs.shape == ():
        idxs = np.reshape(idxs, 1)
    for idx in idxs:
        if idx == 0:
            idxToRemove = idx
        elif idx == len(binCenters) - 2:
            idxToRemove = idx + 1
        if np.mean(np.delete(diffs, idx)) * 1.5 > diffs[idx]:
            continue
        if "idxToRemove" in locals() and idxToRemove in outIdxs:
            outIdxs.remove(idxToRemove)
    return binCenters[outIdxs]
