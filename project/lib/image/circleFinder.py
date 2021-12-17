from csbdeep.utils import normalize
import cv2
import itertools
import math
import numpy as np
import os
from scipy.stats import binned_statistic
from sklearn.linear_model import LinearRegression
import threading
import torch

from project.detectors.splinedist.config import Config
from project.detectors.splinedist.models.model2d import SplineDist2D
from project.lib.image.chamber import CT
from project.lib.util import distance, trueRegions

dirname = os.path.dirname(__file__)
ARENA_IMG_RESIZE_FACTOR = 0.186
UNET_SETTINGS = {
    "config_path": "project/configs/unet_reduced_backbone_arena_wells.json",
    "n_channel": 3,
    "weights_path": os.path.join(dirname, "../../models/arena_pit_v2.pth"),
}
unet_config = Config(UNET_SETTINGS["config_path"], UNET_SETTINGS["n_channel"])
default_model = SplineDist2D(unet_config)
default_model.cuda()
default_model.train(False)
default_model.load_state_dict(torch.load(UNET_SETTINGS["weights_path"]))


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

    def __init__(
        self,
        img: np.ndarray,
        imgName: str,
        allowSkew: bool = False,
        model: SplineDist2D = default_model,
        predict_resize_factor: float = ARENA_IMG_RESIZE_FACTOR,
    ):
        """Create new CircleFinder instance.

        Arguments:
          - img: the image to analyze, in Numpy array form
          - imgName: the basename of the image
          - allowSkew: boolean which, if set to its default value of False, registers
                       an error if skew is detected in the image.
          - model: landmark-detection model to use. Currently, only SplineDist-based
                   models are supported, or at least models with a predict_instances
                   method whose return values mirror those from SplineDist.
                   Defaults to the currently best-performing model.
          - predict_resize_factor: factor by which the image is scaled before being
                                   inputted to the model.
        """
        self.img = img
        self.imgName = imgName
        self.skewed = None
        self.allowSkew = allowSkew
        self.model = model
        self.predict_resize_factor = predict_resize_factor

    def getPixelToMMRatio(self):
        """Calculate the image's ratio of pixels to mm, averaged between the result
        for rows and for columns."""
        self.pxToMM = 0.5 * (
            self.avgDists[1] / CT[self.ct].value().rowDist
            + self.avgDists[0] / CT[self.ct].value().colDist
        )

    def findAgaroseWells(self, img, centers, pxToMM, cannyParam1=40, cannyParam2=35):
        circles = cv2.HoughCircles(
            img,
            cv2.HOUGH_GRADIENT,
            1,
            140,
            param1=cannyParam1,
            param2=cannyParam2,
            minRadius=30,
            maxRadius=50,
        )
        self.shortest_distances = {}
        self.grouped_circles = {}
        self.well_to_well_slopes = {}
        circles = np.uint16(np.around(circles))
        dist_threshold = 0.5 * 0.25 * CT.large.value().floorSideLength * pxToMM

        for i, center in enumerate(centers):
            center = np.round(np.multiply(center, 0.25)).astype(np.int)
            for circ in circles[0, :]:
                circ = circ.astype(np.int32)
                to_well_dist = distance(circ[:2], center)
                if to_well_dist > dist_threshold:
                    continue
                else:
                    if i in self.grouped_circles:
                        self.grouped_circles[i]["raw"].append(circ.tolist())
                    else:
                        self.grouped_circles[i] = {"raw": [circ.tolist()]}
                if (
                    i not in self.shortest_distances
                    or to_well_dist < self.shortest_distances[i]
                ):
                    self.shortest_distances[i] = to_well_dist
            if i not in self.grouped_circles or len(self.grouped_circles[i]["raw"]) < 2:
                continue
            (
                leftmost,
                rightmost,
                uppermost,
                lowermost,
            ) = self.getRelativePositionsOfAgaroseWells(i, center)
            self.well_to_well_slopes[i] = []
            if None not in (lowermost, uppermost):
                self.well_to_well_slopes[i].append(
                    (uppermost[0] - lowermost[0]) / (uppermost[1] - lowermost[1])
                )
            if None not in (leftmost, rightmost):
                self.well_to_well_slopes[i].append(
                    -(rightmost[1] - leftmost[1]) / (rightmost[0] - leftmost[0])
                )
        self.skew_slopes = [
            el for sub_l in list(self.well_to_well_slopes.values()) for el in sub_l
        ]

    def getLargeChamberBBoxesAndImages(self, centers, pxToMM, img=None):
        if type(img) == type(None):
            img = self.img
        bboxes, subImgs = [], []
        img = cv2.medianBlur(img, 5)
        img_for_circles = cv2.resize(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), fx=0.25, fy=0.25
        ).astype(np.uint8)

        self.findAgaroseWells(img_for_circles, centers, pxToMM)
        if len(self.skew_slopes) == 0:
            self.findAgaroseWells(
                img_for_circles, centers, pxToMM, cannyParam1=35, cannyParam2=30
            )

        center_to_agarose_dist = np.mean(list(self.shortest_distances.values())) / 0.25
        skew_slope = np.mean(self.skew_slopes)
        rotation_angle = math.atan(skew_slope)
        for i, center in enumerate(centers):
            acrossCircleD = round(10.5 * pxToMM)
            halfRealCircleD = round((0.5 * 10) * pxToMM)
            deltas = {
                "up": {
                    "x": -halfRealCircleD,
                    "y": -(center_to_agarose_dist + halfRealCircleD),
                },
                "right": {
                    "x": center_to_agarose_dist - halfRealCircleD,
                    "y": -halfRealCircleD,
                },
                "down": {
                    "x": -halfRealCircleD,
                    "y": center_to_agarose_dist - halfRealCircleD,
                },
                "left": {
                    "x": -(center_to_agarose_dist + halfRealCircleD),
                    "y": -halfRealCircleD,
                },
            }
            for position in deltas:
                if position in self.grouped_circles[i]:
                    bboxes = CircleFinder.addUpperLeftCornerToBBox(
                        bboxes,
                        np.divide(self.grouped_circles[i][position][:2], 0.25),
                        -halfRealCircleD,
                        -halfRealCircleD,
                        0,
                    )
                    subImgs = CircleFinder.sampleSubImageBasedOnBBox(
                        subImgs, img, bboxes, acrossCircleD, 0
                    )
                    bboxes = CircleFinder.addWidthAndHeightToBBox(bboxes, subImgs)
                else:
                    bboxes = CircleFinder.addUpperLeftCornerToBBox(
                        bboxes,
                        center,
                        deltas[position]["x"],
                        deltas[position]["y"],
                        rotation_angle,
                    )
                    subImgs = CircleFinder.sampleSubImageBasedOnBBox(
                        subImgs, img, bboxes, acrossCircleD, rotation_angle
                    )
                    bboxes = CircleFinder.addWidthAndHeightToBBox(bboxes, subImgs)
        return bboxes, subImgs

    def getRelativePositionsOfAgaroseWells(self, i, center):
        leftmost, rightmost, uppermost, lowermost = None, None, None, None
        for circ in self.grouped_circles[i]["raw"]:
            if type(leftmost) == type(None) or circ[0] < leftmost[0]:
                leftmost = circ
        for circ in self.grouped_circles[i]["raw"]:
            if circ == leftmost:
                continue
            if circ[0] - leftmost[0] < 100 or abs(center[0] - leftmost[0]) < 100:
                leftmost = None
                break
        if leftmost is not None:
            self.grouped_circles[i]["left"] = leftmost
            self.grouped_circles[i]["raw"].remove(leftmost)
        for circ in self.grouped_circles[i]["raw"]:
            if type(rightmost) == type(None) or circ[0] > rightmost[0]:
                rightmost = circ
        for circ in self.grouped_circles[i]["raw"] + [leftmost]:
            if None in (rightmost, circ) or circ == rightmost:
                continue
            if rightmost[0] - circ[0] < 100:
                rightmost = None
                break
            if leftmost is not None and abs(leftmost[1] - rightmost[1]) > 20:
                rightmost = None
                break
        if rightmost is not None:
            self.grouped_circles[i]["right"] = rightmost
            self.grouped_circles[i]["raw"].remove(rightmost)
        for circ in self.grouped_circles[i]["raw"]:
            if type(uppermost) == type(None) or circ[1] < uppermost[1]:
                uppermost = circ
        for circ in self.grouped_circles[i]["raw"] + [leftmost, rightmost]:
            if None in (uppermost, circ) or circ == uppermost:
                continue
            if circ[1] - uppermost[1] < 100:
                uppermost = None
        if uppermost is not None:
            self.grouped_circles[i]["up"] = uppermost
            self.grouped_circles[i]["raw"].remove(uppermost)
        for circ in self.grouped_circles[i]["raw"]:
            if type(lowermost) == type(None) or circ[1] > lowermost[1]:
                lowermost = circ
        for circ in self.grouped_circles[i]["raw"] + [leftmost, rightmost, uppermost]:
            if None in (lowermost, circ) or circ == lowermost:
                continue
            if lowermost[1] - circ[1] < 100:
                lowermost = None
        if lowermost is not None:
            self.grouped_circles[i]["down"] = lowermost
            self.grouped_circles[i]["raw"].remove(lowermost)
        return leftmost, rightmost, uppermost, lowermost

    @staticmethod
    def addUpperLeftCornerToBBox(bboxes, center, x_del, y_del, rotation_angle):
        bboxes.append(
            [
                round(
                    max(
                        center[0]
                        + x_del * math.cos(rotation_angle)
                        + y_del * math.sin(rotation_angle),
                        0,
                    )
                ),
                round(
                    max(
                        center[1]
                        - x_del * math.sin(rotation_angle)
                        + y_del * math.cos(rotation_angle),
                        0,
                    )
                ),
            ]
        )
        return bboxes

    @staticmethod
    def sampleSubImageBasedOnBBox(subImgs, img, bboxes, delta, rotation_angle):
        subImgs.append(
            img[
                slice(
                    round(bboxes[-1][1]),
                    round(
                        bboxes[-1][1]
                        + delta * (math.cos(rotation_angle) + math.sin(rotation_angle))
                    ),
                ),
                slice(
                    round(bboxes[-1][0]),
                    round(
                        bboxes[-1][0]
                        + delta * (-math.sin(rotation_angle) + math.cos(rotation_angle))
                    ),
                ),
            ]
        )
        return subImgs

    @staticmethod
    def addWidthAndHeightToBBox(bboxes, subImgs):
        bboxes[-1] += reversed(subImgs[-1].shape[0:2])
        return bboxes

    @staticmethod
    def getSubImagesFromBBoxes(img, bboxes, ignore_indices=None):
        sub_imgs = []
        for i, bbox in enumerate(bboxes):
            if ignore_indices and ignore_indices[i]:
                sub_imgs.append(None)
                continue
            sub_imgs.append(
                img[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]
            )
        return sub_imgs

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
            bboxes, subImgs = self.getLargeChamberBBoxesAndImages(
                centers, pxToMM, img=img
            )
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
            for idx in (0, -1):
                if len(binClusters) > 0 and np.sum(binHtsOrig[binClusters[idx]]) == 1:
                    del binClusters[idx]

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

    def resize_image(self):
        self.imageResized = cv2.resize(
            self.img,
            (0, 0),
            fx=self.predict_resize_factor,
            fy=self.predict_resize_factor,
            interpolation=cv2.INTER_CUBIC,
        )
        image = normalize(self.imageResized, 1, 99.8, axis=(0, 1))
        self.imageResized = image.astype(np.float32)

    def findCircles(self, debug=False, predictions=None):
        """Find the location of arena wells for the image in attribute `self.img`.

        Arguments:
          - debug: if True, displays the inputted image with markers over detected
                   landmarks and prints their coordinates to the console
          - predictions: positions of predicted landmarks. Note: its format must match
                     that of the second element of the tuple returned from a call to
                     predict_instances method of a SplineDist model. Defaults to None,
                     in which case new predictions are made.

        Returns:
          - wells: list of the coordinates of detected wells.
          - avgDists: tuple list of the average distances along X and Y direction
                      between neighboring wells.
          - numRowsCols: tuple list of the number of rows and columns of wells.
          - rotatedImg: `self.img` after being rotated to best align rows and
                        columns with the border of the image.
          - rotationAngle: angle in radians by which the image was rotated.
        """
        self.resize_image()
        if predictions is None:
            _, predictions = self.model.predict_instances(self.imageResized)
        self.centroids = predictions["points"]
        self.centroids = [tuple(reversed(centroid)) for centroid in self.centroids]
        if debug:
            print("what are centroids?", self.centroids)

            imgCopy = cv2.resize(np.array(self.imageResized), (0, 0), fx=0.5, fy=0.5)
            for centroid in self.centroids:
                cv2.drawMarker(
                    imgCopy,
                    tuple([int(el * 0.5) for el in centroid]),
                    COL_G,
                    cv2.MARKER_TRIANGLE_UP,
                )
            cv2.imshow(f"debug/{self.imgName}", imgCopy)
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
                np.round(np.divide(well, self.predict_resize_factor)).astype(int)
                for well in self.sortedCentroids
            ]
        )
        for i in range(len(self.wellCoords)):
            self.wellCoords[i] = np.round(
                np.divide(self.wellCoords[i], self.predict_resize_factor)
            ).astype(int)
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
    if colModel.coef_[0] == 0:
        colSlope = 1e-6
    else:
        colSlope = colModel.coef_[0]
    a = 1 / colSlope
    c = -colModel.intercept_ / colSlope
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
