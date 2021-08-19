import argparse
from button import Button
import csv
from enum import Enum, auto
import os
from operator import itemgetter
from queue import LifoQueue
from threading import Thread

import cv2
import numpy as np
from numpy.core.defchararray import count, lower
from pynput import keyboard
from pynput.keyboard import Key
from screeninfo import get_monitors

from chamber import CT
from circleFinder import (
    CircleFinder,
    rotate_image,
    subImagesFromBBoxes,
    subImagesFromGridPoints,
)
from clickLabelManager import ClickLabelManager
from common import getTextSize, globFiles, X_is_running
from util import *
from users import users

import pickle

DEFAULT_EXPORT_PATH = "egg_counts.csv"
DEFAULT_LABELS_PATH = "egg_count_labels.pickle"
DELETE_COLOR = COL_R_L
POSITIONS = ("upper", "right", "lower", "left")


class EditMode(Enum):
    annot = auto()
    translate = auto()


def options():
    """Parse options for the egg-count labelling tool."""
    p = argparse.ArgumentParser(
        description="Input egg counts for each region"
        + " in a series of images of egg-laying chambers."
    )
    p.add_argument("dir", help="Path to directory containing images to annotate")
    p.add_argument(
        "--countsFile",
        default=DEFAULT_EXPORT_PATH,
        help="Path to CSV to use when loading existing counts and saving new ones",
    )
    p.add_argument(
        "--labelsFile",
        default=DEFAULT_LABELS_PATH,
        help="Path to CSV to use when loading existing locations of egg click "
        + "labels and saving new ones",
    )
    p.add_argument(
        "--fix_misalign",
        action="store_true",
        help="Run the app in a mode to fix existing annotations that are offset "
        "from their correct positions by a fixed distance.",
    )
    p.add_argument(
        "--allowSkew", action="store_true", help="allow analysis of" + " skewed images."
    )
    p.add_argument(
        "--windowSize",
        help="specify a custom size for the window"
        + ' as width and height separated by commas, e.g., "1600,900"',
        metavar="W,H",
    )
    return p.parse_args()


class EggCountLabeler:
    """Display interface for annotating positions of eggs in the given images.

    Images are automatically segmented based on the detected positions of wells
    drilled in the center of each egg-laying arena, with one sub-image per arena
    half, each corresponding to an agarose strip running along the edge.
    """

    def __init__(self):
        """Create new EggCountLabeler instance."""
        if not os.path.isdir(opts.dir):
            exit("Error: The specified images directory does not exist")
        self.imgPaths = globFiles(opts.dir) + globFiles(opts.dir, "jpg")
        self.mode = EditMode.translate if opts.fix_misalign else EditMode.annot
        if self.mode == EditMode.translate:
            self.vectors = {}
            self.zoom_factor = 4
        self.processingQueue = LifoQueue()
        self.processingCompleted = set()
        if len(self.imgPaths) == 0:
            exit("Error: The specified images directory contains no images")
        self.clickLabelManager = ClickLabelManager()
        if opts.windowSize:

            class miniMon:
                def __init__(self, width, height):
                    self.width = width
                    self.height = height

            try:
                width, ht = [int(dim) for dim in opts.windowSize.split(",")]
                self.monitorInfo = miniMon(width, ht)
            except Exception:
                print(
                    "Error setting a custom window size. Make sure to enter it"
                    + " as two comma-separated values for width and height, respectively"
                    + ', e.g., "1600,900".'
                )
        else:
            try:
                self.monitorInfo = get_monitors()[0]
            except Exception:
                print(
                    "Error detecting dimensions of display; try using option "
                    + "--windowSize instead."
                )
                exit(1)
        self.setFilePaths()
        self.buttons = dict()
        self.saveBlocker = ""
        self.checkClickConfirmationStatus = False
        self.lastX, self.lastY = 0, 0
        self.lastRender = 0
        self.quitAfterConfirm = False
        self.showHelp = False
        self.justSaved = None
        self.basicImgDrawn = False
        self.eggsDrawn = False
        self.lastClosestEgg = None
        self.inModal = False
        self.steppedWithConfirm = None
        self.ignoreFilePath = os.path.join(opts.dir, "ignore.txt")
        self.thread = Thread(target=self.loadSubImages, daemon=True)
        self.thread.finishedLoadFromFile = None
        self.thread.start()
        self.secondaryKeyHandler = keyboard.Listener(
            on_press=self.checkForShiftPress, on_release=self.checkForShiftRelease
        )
        self.shiftPressed = False
        self.secondaryKeyHandler.start()

    def checkForShiftPress(self, key):
        """
        Handle a Pynput `on_press` event by checking if shift was pressed and
        setting attribute `shiftPressed` based on the result.

        Arguments:
          - key: pynput.keyboard.Key instance
        """
        if key == Key.shift:
            self.shiftPressed = True

    def checkForShiftRelease(self, key):
        """
        Handle a Pynput `on_release` event by checking if shift was released, and if
        yes, updating attribute `shiftPressed` to False.
        """
        if key == Key.shift:
            self.shiftPressed = False

    def totalNumSubImgs(self):
        """Return number of sub-images loaded by the EggCountLabeler instance."""
        countListLengths = np.array([len(ctList) for ctList in self.eggCounts])
        subImgsFromCounts = sum(countListLengths)
        if [] not in self.eggCounts:
            return subImgsFromCounts
        averageCountPerImg = np.round(np.mean(countListLengths[countListLengths != 0]))
        numUnfilledImgs = np.count_nonzero(np.asarray(countListLengths) == 0)
        return subImgsFromCounts + numUnfilledImgs * averageCountPerImg

    def loadExistingCounts(self):
        """Associate sub-images with egg counts parsed from a saved file."""
        with open(self.countsFile) as myFile:
            reader = csv.reader(myFile)
            fileLen = len(list(reader))
            myFile.seek(0)
            reader = csv.reader(myFile)
            for i, row in enumerate(reader):
                if len(row) == 1:
                    if i > 0 and getCtsFromFile:
                        self.rowColCounts[currentImg] = [numRows] + self.rowColCounts[
                            currentImg
                        ]
                    currentImg = row[0].lower()
                    cts = self.clickLabelManager.chamberTypes
                    ctUnknown = currentImg not in cts
                    getCtsFromFile = ctUnknown or cts[currentImg] != CT.large.name
                    self.existingCounts[currentImg] = []
                    if not getCtsFromFile:
                        self.rowColCounts[currentImg] = [
                            getattr(CT.large.value(), name)
                            for name in ("numRows", "numCols")
                        ]
                else:
                    if getCtsFromFile:
                        if len(self.existingCounts[currentImg]) == 0:
                            self.rowColCounts[currentImg] = [int(len(row) / 2)]
                            numRows = 1
                        else:
                            numRows += 1
                    self.existingCounts[currentImg] += row
                    if i + 1 == fileLen and getCtsFromFile:
                        self.rowColCounts[currentImg] = [numRows] + self.rowColCounts[
                            currentImg
                        ]

    def getIndicesOfIgnoredImages(self):
        for idx in self.imgs_to_ignore:
            abs_start_index = self.absolutePosition(idx, 0)
            self.abs_idxs_ignored = self.abs_idxs_ignored.union(
                set(range(abs_start_index, abs_start_index + len(self.subImgs[idx])))
            )

    def segmentSubImageFromExistingData(self, img, imgIndex):
        """Segment one image into sub-images using data loaded from a pickle file,
        having been calculated previously via GPU.
        """
        lbn = self.lowerBasenames[imgIndex]
        self.rowColCounts[lbn] = self.imgMetadata[lbn]["rowColCounts"]
        self.clickLabelManager.setChamberType(lbn, self.imgMetadata[lbn]["ct"])
        rotatedImg = rotate_image(img, self.imgMetadata[lbn]["rotationAngle"])
        self.subImgs[imgIndex] = subImagesFromBBoxes(
            rotatedImg, self.imgMetadata[lbn]["bboxes"]
        )

    def segmentSubImageViaGPU(self, img, imgPath, imgIndex):
        """Segment one image into sub-images using the GPU-dependent CircleFinder
        class.
        """
        self.cf = CircleFinder(img, os.path.basename(imgPath), allowSkew=opts.allowSkew)
        circles, avgDists, numRowsCols, rotatedImg, rotAngle = self.cf.findCircles()
        self.rowColCounts[self.lowerBasenames[imgIndex]] = numRowsCols
        self.clickLabelManager.setChamberType(self.lowerBasenames[imgIndex], self.cf.ct)
        self.subImgs[imgIndex], bboxes = self.cf.getSubImages(
            rotatedImg, circles, avgDists, numRowsCols
        )
        self.imgMetadata[self.lowerBasenames[imgIndex]] = dict(
            rowColCounts=numRowsCols,
            rotationAngle=rotAngle,
            ct=self.cf.ct,
            bboxes=bboxes,
        )
        self.saveImgMetadata()

    def saveImgMetadata(self):
        """Export a copy of the current image metadata (sub-image bounding box
        measurements, chamber types, numbers of rows and columns) to a pickle file.
        """
        with open(self.imgMetadataPath, "wb") as myFile:
            pickle.dump(self.imgMetadata, myFile)

    def segmentSubImages(self):
        """Segment images into sub-images using CircleFinder class. Processing can
        be stopped by setting attribute `doRun` of the `thread` attribute of the
        EggCountLabeler instance to False.
        """
        thread = threading.currentThread()
        while getattr(thread, "doRun", True):
            index, imgPath = self.processingQueue.get()
            if imgPath in self.processingCompleted:
                continue
            img = cv2.imread(imgPath)
            if self.lowerBasenames[index] in self.imgMetadata:
                self.segmentSubImageFromExistingData(img, index)
            else:
                self.segmentSubImageViaGPU(img, imgPath, index)
            if len(self.eggCounts[index]) == 0:
                self.eggCounts[index] = [None for _ in range(len(self.subImgs[index]))]
            self.processingCompleted.add(imgPath)

    def setEggCounts(self):
        """Initialize the lists of egg counts for each sub-image, using any data
        parsed from file if available.
        """
        for i in range(len(self.imgPaths)):
            if self.lowerBasenames[i] in self.existingCounts.keys():
                if self.lowerBasenames[i] in self.clickLabelManager.chamberTypes:
                    self.clickLabelManager.chamberTypes[
                        self.lowerBasenames[i]
                    ] = self.clickLabelManager.chamberTypes[self.lowerBasenames[i]]
                else:
                    self.clickLabelManager.chamberTypes[self.lowerBasenames[i]] = ""
                if (
                    self.clickLabelManager.chamberTypes[self.lowerBasenames[i]]
                    == CT.large.name
                ):
                    self.existingCounts[self.lowerBasenames[i]] = itemgetter(
                        *CT.large.value().dataIndices
                    )(self.existingCounts[self.lowerBasenames[i]])
                    self.existingCounts[self.lowerBasenames[i]] = [
                        x
                        for _, x in sorted(
                            zip(
                                CT.large.value().csvToClockwise,
                                self.existingCounts[self.lowerBasenames[i]],
                            )
                        )
                    ]
                for j, el in enumerate(self.existingCounts[self.lowerBasenames[i]]):
                    if len(el) > 0:
                        self.eggCounts[i].append(int(el))
                        rowNum = self.rowNum(idx=i, subImgIdx=j)
                        colNum = self.colNum(idx=i, subImgIdx=j)
                        key = self.clickLabelManager.subImageKey(
                            self.lowerBasenames[i],
                            rowNum,
                            colNum,
                            position=self.getWellPosition(i, j),
                        )
                        if key not in self.clickLabelManager.clicks:
                            self.clickLabelManager.addKey(
                                self.lowerBasenames[i],
                                rowNum,
                                colNum,
                                position=self.getWellPosition(i, j),
                            )
                    else:
                        self.eggCounts[i].append(None)

    def allImagesIgnored(self):
        return len(self.imgs_to_ignore) == len(self.imgPaths)

    def loadSubImages(self):
        """Segment images into sub-images and set up the structures that track their
        metadata (egg counts and click locations for each sub-image).
        """
        t = threading.currentThread()
        self.subImgs = [[] for _ in range(len(self.imgPaths))]
        self.eggCounts = [[] for _ in range(len(self.imgPaths))]
        self.abs_idxs_ignored = set()
        self.imgs_to_ignore = []
        self.rowColCounts = dict()
        self.existingCounts = dict()
        self.lowerBasenames = [
            os.path.basename(imgPath).lower() for imgPath in self.imgPaths
        ]
        filesExist = [
            os.path.isfile(filename)
            for filename in (
                self.countsFile,
                self.labelsFile,
                self.imgMetadataPath,
                self.ignoreFilePath,
            )
        ]
        if hasattr(self, "deltaFile"):
            filesExist.append(os.path.isfile(self.deltaFile))
        if True in filesExist:
            setattr(t, "finishedLoadFromFile", False)
        if filesExist[1]:
            with open(self.labelsFile, "rb") as myFile:
                self.clickLabelManager = ClickLabelManager()
                tempData = pickle.load(myFile)
                self.clickLabelManager.clicks = tempData["clicks"]
                for keyName in ("chamberTypes", "frontierData") + tuple(
                    ["is%sLabels" % c for c in self.clickLabelManager.categories]
                ):
                    if keyName in tempData:
                        setattr(self.clickLabelManager, keyName, tempData[keyName])
        if not filesExist[2]:
            self.imgMetadata = {}
        else:
            with open(self.imgMetadataPath, "rb") as myFile:
                self.imgMetadata = pickle.load(myFile)
        if filesExist[3]:
            with open(self.ignoreFilePath, "r") as my_f:
                self.imgs_to_ignore = [
                    self.lowerBasenames.index(img_f.lower())
                    for img_f in my_f.read().splitlines()
                ]

        if len(filesExist) == 5:

            def notifyOnErrorAndExit():
                print("Problem opening the file", self.deltaFile)
                print("Please check if the user has annotated this folder yet.")
                print("Use Ctrl + C to exit.")
                t.doRun = False
                sys.exit()

            if filesExist[4]:
                try:
                    with open(self.deltaFile, "rb") as myFile:
                        loadedData = pickle.load(myFile)
                        self.diffClickData = loadedData["clicks"]
                        self.diffFrontierData = loadedData["frontierData"]
                        self.diffFrontierData[
                            "frontierImgIdx"
                        ] = self.lowerBasenames.index(self.diffFrontierData["fileName"])
                except Exception:
                    notifyOnErrorAndExit()
            else:
                notifyOnErrorAndExit()

        if filesExist[0]:
            self.loadExistingCounts()
        self.setEggCounts()
        setattr(t, "finishedLoadFromFile", True)
        self.segmentSubImages()

    def setBlankImg(self):
        """Set display window of the EggCountLabeler instance to a black square with
        side length 400px.
        """
        self.eggsImg = np.zeros((400, 400), np.uint8)

    def setMask(self, slices, value=130):
        """Initialize a mask based on given dimensions and brightness. Note: this
        method does not overlay the mask over an image; that needs to be performed
        in a separate, subsequent step.

        Arguments:
          - slices: List of slices giving the start and stop points, in pixels, of
                    the mask in the vertical and horizontal directions, respectively
                    (should be within the dimensions of the `img` attribute of the
                    EggCountLabeler instance.)
          - value: Transparency of the mask, with range [0, 255].
                   At 0, the mask is invisible; at 255, the mask covers the image
                   completely.
        """
        self.mask = np.full((self.eggsImg.shape[0], self.eggsImg.shape[1]), 0)
        maskDims = tuple([s.stop - s.start for s in slices])
        self.mask[slices[0], slices[1]] = np.full(maskDims, value)

    def showCenteredMessage(
        self,
        message="analyzing... please wait",
        overBlank=True,
        textSize=1.3,
        color=COL_W,
        bkgndColor=COL_W,
        htOffset=0,
        wthOffset=0,
        parentShape=None,
    ):
        """Display given message in the center of the UI on top of a mask. To be
        able to span multiple lines, explicit newlines ('\\n') must be included in
        the message.

        Arguments:
          - message: String to be displayed
          - overBlank: Bool indicating whether to clear the GUI and display the
                       message over a blank square (True) or overlay image atop the
                       current contents of GUI (False)
          - textSize: Size of the message text
          - color: BGR triplet for the color of the message text
          - bkgndColor: BGR triplet for the background of the message
          - htOffset: Number between -1 and 1 indicating how far above or below
                      center to place the message. -1 means top of image, 1 means
                      bottom, and 0 means center.
          - wthOffset: Number between -1 and 1 indicating how far left or right of
                       center to place the message. -1 means at the left, 1 means
                       at the right, and 0 means center.
          - parentShape: a (height, width) tuple of a region measured from the upper
                         left corner in which the message should be centered
                         (default is the entire GUI window).
        """
        if overBlank:
            self.setBlankImg()
        if not parentShape:
            parentShape = self.eggsImg.shape
        imgDimsHalf = [dim / 2 for dim in parentShape]
        heightBuffer = 3
        textLines = message.splitlines()
        textDims = [
            cv2.getTextSize(line, cv2.FONT_HERSHEY_PLAIN, textSize, 1)[0][::-1]
            for line in textLines
        ]
        singleTextHeight = textDims[0][0]
        largestTextWidth = max([dim[1] for dim in textDims])
        textDims = [
            singleTextHeight * len(textDims) + (len(textDims) + 2) * heightBuffer,
            largestTextWidth,
        ]
        offsets = [imgDimsHalf[i] * (wthOffset if i else htOffset) for i in range(2)]
        maskSlices = [
            slice(
                int(max(imgDimsHalf[i] + offsets[i] - textDims[i] / 2 - 5, 0)),
                int(imgDimsHalf[i] + offsets[i] + textDims[i] / 2 + 5),
            )
            for i in range(2)
        ]
        self.setMask(maskSlices)
        self.eggsImg = overlay(self.eggsImg, self.mask, bkgndColor)
        putText(
            self.eggsImg,
            message,
            tuple([imgDimsHalf[i] + offsets[i] - textDims[i] / 2 for i in range(2)])[
                ::-1
            ],
            (0, 1),
            textStyle(color=color, size=textSize),
        )

    def getFrontierIndices(self):
        """Return the image and sub-image indices corresponding to the frontier (see
        documentation for jumpToFrontier method for definition of frontier).
        """
        unfilledRegions = self.getUnfilledRegions()
        if unfilledRegions == None:
            self.frontierImgIdx = None
            self.frontierSubImgIdx = None
            return
        if len(unfilledRegions) == 0:
            frontierImgIdx = [
                i for i in range(len(self.eggCounts)) if i not in self.imgs_to_ignore
            ][-1]
            subImgIdx = (
                2
                * self.rowColCounts[self.lowerBasenames[frontierImgIdx]][0]
                * self.rowColCounts[self.lowerBasenames[frontierImgIdx]][1]
                - 1
            )
            self.frontierImgIdx, self.frontierSubImgIdx = frontierImgIdx, subImgIdx
            return
        else:
            frontierImgIdx, runningSum, subImgIdx = 0, 0, 0
            for i in range(len(self.imgPaths)):
                if len(self.rowColCounts) < i + 1:
                    self.frontierImgIdx, self.frontierSubImgIdx = i, 0
                    return
                compResults = self.compareCountWithNumCells(
                    i, runningSum, unfilledRegions[-1].start
                )
                if len(compResults) < 3:
                    runningSum = compResults[0]
                else:
                    self.frontierImgIdx, self.frontierSubImgIdx = compResults[0:2]
                    return

    def jumpToFrontier(self, addPrecedingImgs=False):
        """Update the GUI to display the first unlabelled image with no subsequent
        neighbors that are labelled. A "gap image," meaning one that is unlabelled
        but whose subsequent neighbors include labelled images, is not considered
        the frontier.

        The frontier image and its subsequent neighbors are moved to the front of
        the queue for sub-image segmentation (see method `segmentSubImages`)

        Arguments:
          - addPrecedingImgs: Bool indicating whether to queue up the images
                              preceding the frontier image as well. If True, they
                              get processed after the frontier image's subsequent
                              neighbors.
        """
        self.getFrontierIndices()

        def is_ignored():
            frontier_is_none = self.frontierImgIdx is None
            if self.allImagesIgnored():
                return frontier_is_none
            return self.frontierImgIdx in self.imgs_to_ignore or frontier_is_none

        if addPrecedingImgs:
            for i in range(len(self.imgPaths)):
                self.processingQueue.put((i, self.imgPaths[i]))
        while is_ignored():
            self.getFrontierIndices()
        for i in range(len(self.imgPaths), self.frontierImgIdx, -1):
            self.processingQueue.put((i - 1, self.imgPaths[i - 1]))
        self.imgIdx = self.frontierImgIdx
        self.subImgIdx = self.frontierSubImgIdx
        self.basicImgDrawn, self.eggsDrawn = False, False
        self.renderSubImg()

    def click_was_outside_image(self, x, y):
        self.origY = (y - self.upperBorderHt) / self.scalingFactor
        self.origX = (x - self.sidebarWidth) / self.scalingFactor
        origImgDims = self.subImgs[self.imgIdx][self.subImgIdx].shape
        return (
            self.origX < 0
            or self.origY < 0
            or self.origY > origImgDims[0]
            or self.origX > origImgDims[1]
        )

    def addEggClick(self, x, y):
        """Add a marker at the clicked location.

        Arguments:
          - x: The X coordinate of the mouse event relative to the program window
          - y: The Y coordinate of the mouse event relative to the program window
        """
        if self.click_was_outside_image(x, y):
            return
        self.clickLabelManager.addClick(
            self.lowerBasenames[self.imgIdx],
            self.rowNum(),
            self.colNum(),
            (self.origX, self.origY),
            self.wellPosition,
        )
        self.renderSubImg()
        self.clicksConfirmed = False

    def deleteEggClick(self):
        """Delete a marker at the clicked location."""
        if self.closestEggIdx is None:
            return
        self.clickLabelManager.clearClick(
            self.lowerBasenames[self.imgIdx],
            self.rowNum(),
            self.colNum(),
            self.closestEggIdx,
            self.wellPosition,
        )

    def onMouse(self, event, x, y, flags, params):
        """Process mouse events, which comprise the following: 1) labelling an egg
        or 2) hovering over or clicking on either the "Enter" or "Jump to frontier"
        button.

        Arguments:
          - event: The OpenCV code of the mouse event
          - x: The X coordinate of the mouse event relative to the program window
          - y: The Y coordinate of the mouse event relative to the program window
          - flag, params: Unused parameters that are required by OpenCV's mouse
                          callback function signature.
        """
        self.lastX, self.lastY = x, y
        for b_key in self.buttons:
            self.buttons[b_key].check_if_under_cursor(x, y)
            if (
                self.buttons[b_key].under_cursor_prev
                != self.buttons[b_key].under_cursor
            ):
                self.eggsDrawn = False
        if event == cv2.EVENT_LBUTTONUP:
            for b_key in self.buttons:
                if (
                    self.buttons[b_key].under_cursor
                    and not self.buttons[b_key].disabled
                ):
                    self.buttons[b_key].handle_press()
                    return
            if self.mode == EditMode.annot:
                self.handleAnnotClick(x, y)
            elif self.mode == EditMode.translate:
                self.handle_vector_click(x, y)

        elif len(self.saveBlocker) == 0:
            pass
            self.renderSubImg()

    def handle_vector_click(self, x, y):
        if "current" not in self.vectors:
            self.start_new_vector()
        else:
            self.finish_or_delete_vector(x, y)

    @staticmethod
    def draw_vector(img, point1, point2):
        cv2.arrowedLine(img, point1, point2, (0, 30, 240), 1, cv2.LINE_AA)

    def show_current_vector(self):
        if self.mode != EditMode.translate or "current" not in self.vectors:
            return
        self.draw_vector(
            self.zoom_image,
            tuple(
                map(
                    round,
                    [self.zoom_factor * coord for coord in self.vector_start_point],
                )
            ),
            tuple(
                map(
                    round,
                    [
                        self.zoom_factor
                        * (
                            self.scalingFactor * (el - self.vectors["current"][0][i])
                            + self.vector_start_point[i]
                        )
                        for i, el in enumerate(
                            self.appFrameToImageFrame((self.lastX, self.lastY))
                        )
                    ],
                )
            ),
        )

    def show_finished_vectors(self):
        if self.mode == EditMode.annot:
            return
        for k in self.vectors:
            if k == "current":
                continue
            self.draw_vector(
                self.eggsImg,
                self.imageFrameToAppFrame(self.vectors[k][0]),
                self.imageFrameToAppFrame(self.vectors[k][1]),
            )

    def current_vector_key(self):
        return "%.2f-%.2f" % (
            self.vectors["current"][0][0],
            self.vectors["current"][0][1],
        )

    def translate_clicks_by_mean_vector(self):
        self.calculate_mean_vector()
        click_key = self.clickLabelManager.subImageKey(*self.currentImgArgs())
        for i, click in enumerate(
            self.clickLabelManager.getClicks(*self.currentImgArgs())
        ):
            self.clickLabelManager.clicks[click_key][i] = [
                click[j] + self.mean_vector[j] for j in range(len(self.mean_vector))
            ]
        self.vectors = {}
        self.eggsDrawn = False

    def calculate_mean_vector(self):
        self.mean_vector = [
            np.mean(
                [self.vectors[k][1][i] - self.vectors[k][0][i] for k in self.vectors]
            )
            for i in range(2)
        ]

    def finish_or_delete_vector(self, x, y):
        if self.click_was_outside_image(x, y):
            return
        if (
            self.closestEggIdx is None
            or self.currentClickLabels()[self.closestEggIdx]
            != self.vectors["current"][0]
        ):
            self.vectors["current"].append(self.appFrameToImageFrame((x, y)))
            self.vectors[self.current_vector_key()] = self.vectors["current"]

        self.eggsDrawn = False
        del self.vectors["current"]
        cv2.destroyWindow("Zoom Display")
        self.buttons["translate"].disabled = len(self.vectors) == 0

    def start_new_vector(self):
        if self.closestEggIdx is None:
            return
        self.vectors["current"] = [self.currentClickLabels()[self.closestEggIdx]]
        cvk = self.current_vector_key()
        if cvk in self.vectors:
            del self.vectors[cvk]

    def draw_temporary_edit_window(self):
        vec = self.vectors["current"]

        to_right, to_bottom = 50, 50
        col_start_idx = round(vec[0][0] * self.scalingFactor - 50)
        self.vector_start_point = [50, 50]
        if col_start_idx < 0:
            to_right -= col_start_idx
            self.vector_start_point[0] += col_start_idx
            col_start_idx = 0
        col_end_idx = round(vec[0][0] * self.scalingFactor + to_right)
        if col_end_idx > self.centerImg.shape[1]:
            col_start_idx -= col_end_idx - self.centerImg.shape[1]
            self.vector_start_point[0] += col_end_idx - self.centerImg.shape[1]
            col_end_idx = self.centerImg.shape[1]
        row_start_idx = round(vec[0][1] * self.scalingFactor - 50)
        if row_start_idx < 0:
            to_bottom -= row_start_idx
            self.vector_start_point[1] += row_start_idx
            row_start_idx = 0
        row_end_idx = round(vec[0][1] * self.scalingFactor + to_bottom)
        if row_end_idx > self.centerImg.shape[0]:
            row_start_idx -= row_end_idx - self.centerImg.shape[0]
            self.vector_start_point[1] += row_end_idx - self.centerImg.shape[0]
            row_end_idx = self.centerImg.shape[0]
        self.zoom_image = cv2.resize(
            self.centerImg[
                row_start_idx : row_end_idx,
                col_start_idx : col_end_idx,
            ],
            (0, 0),
            fx=self.zoom_factor,
            fy=self.zoom_factor,
        )
        self.show_current_vector()
        cv2.imshow("Zoom Display", self.zoom_image)
        cv2.moveWindow("Zoom Display", 750, 500)

    def handleAnnotClick(self, x, y):
        if self.shiftPressed:
            self.eggsDrawn = False
            self.deleteEggClick()
        else:
            self.eggsDrawn = False
            self.addEggClick(x, y)

    def cursorDisplay(self):
        """Draw a yellow circle at the position of the cursor."""
        cv2.circle(self.liveImg, (self.lastX, self.lastY), 4, COL_Y, thickness=2)

    def annotateEggCounts(self):
        """Start GUI and set up handling of keyboard and mouse events."""
        self.imgIdx = 0
        self.subImgIdx = 0
        self.jumpToFrontier(addPrecedingImgs=True)
        while True:
            if not hasattr(self, "subImgs"):
                continue
            if len(self.subImgs[self.imgIdx]) == 0:
                self.showCenteredMessage()
                while len(self.subImgs[self.imgIdx]) == 0:
                    time.sleep(1)
            self.renderSubImg()
            cv2.setMouseCallback("Egg Count Annotator", self.onMouse)
            while True:
                if getattr(self.thread, "hadError", False):
                    sys.exit()
                if self.steppedWithConfirm is not None:
                    if time.time() - self.steppedWithConfirm > 2.5:
                        self.steppedWithConfirm = None
                        self.eggsDrawn = False
                        self.renderSubImg()
                        break
                if self.justSaved is not None:
                    if time.time() - self.justSaved > 2.5:
                        self.justSaved = None
                        self.inModal, self.eggsDrawn = False, False
                        self.renderSubImg()
                        break
                self.renderSubImg()
                k = cv2.waitKey(1)
                if k == -1:
                    eventProcessingDone = True
                elif eventProcessingDone:
                    break
            k &= 255
            keyCode = chr(k)
            if len(self.saveBlocker) > 0:
                self.handleIncompleteSaveResponse(k)
            elif self.checkClickConfirmationStatus:
                self.handleClickConfirmationResponse(keyCode)
            else:
                self.handleNormalKeypress(keyCode, k)

    def saveMessageDisplay(self):
        """Display message in GUI indicating successful save."""
        if self.justSaved:
            self.showCenteredMessage("Successfully saved to file!", overBlank=False)

    def helpText(self):
        """Display help message in GUI."""
        if self.showHelp:
            corner, dims = (44, 17), (127, 435)
            self.setMask([slice(corner[i], corner[i] + dims[i]) for i in range(2)])
            self.img = overlay(self.img, self.mask, COL_W)
            helpTexts = (
                "keyboard commands:\nh\ns\nq\n0-9\nbksp\n.\n,",
                "toggle show help\nsave CSV\nsave CSV and quit\n"
                + "input egg count for current region\n"
                + "delete one digit of egg count for current region\n"
                + "step forward 1 region (autosaves counts)\n"
                + "step back 1 region (autosaves counts)",
            )
            for txtAndLocs in zip(helpTexts, ((23, 45), (70, 61))):
                putText(
                    self.img,
                    txtAndLocs[0],
                    txtAndLocs[1],
                    (0, 1),
                    textStyle(color=COL_BK),
                )

    def getUnfilledRegions(self):
        """Return list of slices corresponding to regions of unlabelled images."""
        while not getattr(self.thread, "finishedLoadFromFile", False):
            time.sleep(0.1)
        self.imgs_to_ignore.sort()
        self.eggCountsConcat = np.asarray(concat(self.eggCounts))
        trueRegionCts = trueRegions(self.eggCountsConcat == None)
        if [] in self.eggCounts and len(trueRegionCts) == 0:
            first_non_ignored_img = 0
            for i in range(len(self.eggCounts)):
                if i not in self.imgs_to_ignore:
                    first_non_ignored_img = i
                    break
            leading_imgs_not_processed = True
            for i in range(first_non_ignored_img + 1):
                self.processingQueue.put((i, self.imgPaths[i]))
            while leading_imgs_not_processed:
                leading_imgs_not_processed = any(
                    [
                        self.imgPaths[i] not in self.processingCompleted
                        for i in range(first_non_ignored_img + 1)
                    ]
                )
                time.sleep(0.3)
            start = self.absolutePosition(first_non_ignored_img, 0)
            if start == None:
                return None
            return [slice(start, start + 1)]
        for i in self.imgs_to_ignore:
            idx_start = self.absolutePosition(i, 0)
            if idx_start == None:
                return None
            idx_end = idx_start + len(self.eggCounts[i])
            for tReg in list(trueRegionCts):
                if tReg.start <= idx_end and tReg.start >= idx_start:
                    if tReg.stop - 1 > idx_end:
                        idx = trueRegionCts.index(tReg)
                        trueRegionCts[idx] = slice(idx_end, tReg.stop)
                    else:
                        trueRegionCts.remove(tReg)
        if len(trueRegionCts) == 0:
            for i in range(len(self.eggCounts)):
                if i not in self.imgs_to_ignore:
                    self.processingQueue.put((i, self.imgPaths[i]))
                    while True:
                        if self.imgPaths[i] in self.processingCompleted:
                            break
                        time.sleep(0.3)
                    trueRs = trueRegions(np.asarray(self.eggCounts[i]) == None)
                    self.eggCountsConcat = np.asarray(concat(self.eggCounts))
                    if len(trueRs) > 0:
                        return [
                            slice(
                                self.absolutePosition(i, trueRs[0].start),
                                len(self.eggCountsConcat),
                            )
                        ]
        return trueRegionCts

    def checkForSaveBlockers(self):
        """Check if there are either 1) gaps in the egg counts (see
        `findGapsInEggCounts`) or 2) discrepancies between clicked and saved egg
        counts, and store the result as attribute `saveBlocker`. Gaps in egg counts
        take priority; i.e., the latter error is reported only after the former gets
        resolved.
        """
        if self.findGapsInEggCounts():
            self.saveBlocker = "gaps"
            return
        elif self.findMismatchBetweenSavedCountsAndClickedEggs():
            self.saveBlocker = "mismatch"

    def findGapsInEggCounts(self):
        """Return bool indicating if there are any "gap images," meaning ones that
        are unlabelled but whose subsequent neighbors include labelled images.
        """
        self.unfilledRegions = self.getUnfilledRegions()
        gapsExist = not (
            (len(self.unfilledRegions) == 0)
            or (
                len(self.unfilledRegions) == 1
                and self.unfilledRegions[-1].stop == len(self.eggCountsConcat)
            )
        )
        if gapsExist:
            self.preImgIdx, self.preSubImgIdx = self.getImgAndSubImgIdx(
                self.unfilledRegions[0].start
            )
        return gapsExist

    def findMismatchBetweenSavedCountsAndClickedEggs(self):
        """Return bool indicating if there are any regions with saved counts
        whose clicked egg counts have a different value.
        """
        unofficialEggCounts = []
        self.getIndicesOfIgnoredImages()
        for i in range(len(self.imgPaths)):
            lowerBasename = self.lowerBasenames[i]
            if lowerBasename not in self.rowColCounts:
                continue
            chamberRange = [
                range(
                    ct
                    * (
                        2
                        if i
                        and self.clickLabelManager.chamberTypes[lowerBasename]
                        != CT.large.name
                        else 1
                    )
                )
                for i, ct in enumerate(self.rowColCounts[lowerBasename])
            ]
            for i, j in itertools.product(*chamberRange):
                if self.clickLabelManager.chamberTypes[lowerBasename] != CT.large.name:
                    unofficialEggCounts.append(
                        self.clickLabelManager.getNumClicks(lowerBasename, i, j)
                    )
                else:
                    for pos in POSITIONS:
                        if i == 0:
                            pass
                        unofficialEggCounts.append(
                            self.clickLabelManager.getNumClicks(
                                lowerBasename, i, j, pos
                            )
                        )
        unofficialEggCounts = np.array(unofficialEggCounts)
        cumulativeFrontierIdx = (
            sum(
                [
                    len(subImgList)
                    for subImgList in self.subImgs[0 : self.frontierImgIdx]
                ]
            )
            + self.frontierSubImgIdx
        )
        abs_idxs = list(range(len(unofficialEggCounts)))
        unofficialEggCounts = np.array([
            i
            for j, i in enumerate(unofficialEggCounts)
            if j not in self.abs_idxs_ignored and j < cumulativeFrontierIdx
        ])
        eggCountsConcatTrimmed = np.array([
            i
            for j, i in enumerate(self.eggCountsConcat)
            if j not in self.abs_idxs_ignored and j < cumulativeFrontierIdx
        ])
        abs_idxs_trimmed = [
            i
            for i in abs_idxs
            if i not in self.abs_idxs_ignored and i < cumulativeFrontierIdx
        ]
        foundMismatches = not np.array_equal(
            unofficialEggCounts, eggCountsConcatTrimmed
        )
        if foundMismatches:
            self.preImgIdx, self.preSubImgIdx = self.getImgAndSubImgIdx(
                abs_idxs_trimmed[
                    np.argmax(unofficialEggCounts != eggCountsConcatTrimmed)
                ]
            )
        return foundMismatches

    def setFilePaths(self):
        """Set paths to files the app needs to run, chiefly image metadata and
        counts/clicks on eggs.
        """
        self.setFileSuffixes()
        self.imgMetadataPath = os.path.join(opts.dir, "images.metadata")

    def setFileSuffixes(self):
        """Add a username suffix, if needed, to the names of files containing counts
        and click locations saved during a prior session.
        """
        attrNames = ["%ssFile" % fileType for fileType in ("count", "label")]
        fileExtensions = ("csv", "pickle")
        userSuffix = "_%s" % username
        filePaths = [getattr(opts, attrName).split(".")[0] for attrName in attrNames]
        for i, f in enumerate(filePaths):
            setattr(
                self,
                attrNames[i],
                filePaths[i]
                + "%s.%s"
                % ("" if f.endswith(userSuffix) else userSuffix, fileExtensions[i]),
            )
        if not userForComparison:
            return
        else:
            countsFile = getattr(self, "labelsFile")
            setattr(
                self,
                "deltaFile",
                "%s_%s.%s"
                % (
                    "_".join(countsFile.split("_")[:-1]),
                    userForComparison,
                    fileExtensions[-1],
                ),
            )

    def exportEggCountReport(self):
        """Save egg counts and click locations in CSV and binary (pickle) formats,
        respectively.
        """
        try:
            with open(self.countsFile, "wt", newline="") as resultsFile:
                for i, imgPath in enumerate(self.imgPaths):
                    if len(self.eggCounts[i]) == 0 or np.all(
                        np.asarray(self.eggCounts[i]) == None
                    ):
                        continue
                    writer = csv.writer(resultsFile)
                    writer.writerow([os.path.basename(imgPath)])
                    CT[
                        self.clickLabelManager.chamberTypes[self.lowerBasenames[i]]
                    ].value().writeLineFormatted(self.eggCounts, i, writer)
        except PermissionError as _:
            self.saveBlocker = "csvFileInUse"
            return
        self.clickLabelManager.frontierData = {
            "fileName": self.lowerBasenames[self.frontierImgIdx],
            "subImgIdx": self.frontierSubImgIdx,
            "finishedLabeling": self.frontierImgIdx + 1 == len(self.lowerBasenames)
            and self.frontierSubImgIdx + 1 == len(self.subImgs[-1]),
        }
        for clickKey in list(self.clickLabelManager.clicks):
            for frontierImageName in self.lowerBasenames[self.frontierImgIdx + 1 :]:
                if frontierImageName in clickKey:
                    del self.clickLabelManager.clicks[clickKey]
        with open(self.labelsFile, "wb") as labelsFile:
            pickle.dump(
                {
                    dKey: self.clickLabelManager.__dict__[dKey]
                    for dKey in self.clickLabelManager.__dict__
                },
                labelsFile,
            )
        self.justSaved = time.time()
        self.eggsDrawn = False

    def handleIncompleteSaveResponse(self, k):
        """Dismiss the "incomplete save" warning only if "Enter" key is pressed.

        Arguments:
          - k: numeric ID of the keypress (13 means Enter)
        """
        if k == 13:
            self.saveBlocker = ""
            self.imgIdx = self.preImgIdx
            self.subImgIdx = self.preSubImgIdx
            self.inModal = False
            self.basicImgDrawn, self.eggsDrawn = False, False

    def currentImgArgs(self):
        """Return list containing 1) lowercase basename of current image and
        2) row and 3) column numbers of the sub-image being displayed.
        """
        return [self.lowerBasenames[self.imgIdx], self.rowNum(), self.colNum()] + (
            [] if self.wellPosition is None else [self.wellPosition]
        )

    def handleClickConfirmationResponse(self, keyCode):
        """Handle keyboard input if user tries to change sub-image without first
        confirming their input. "y" key clears egg clicks and navigates to a new
        sub-image, while "n" allows the user to continue annotating the current
        sub-image.

        Arguments:
          - keyCode: String corresponding to the key that was pressed
        """
        if keyCode in ("y", "n"):
            self.checkClickConfirmationStatus = False
            self.basicImgDrawn, self.eggsDrawn = False, False
            self.inModal = False
            if keyCode == "n":
                self.clicksConfirmed = True
                getattr(self, "step%s" % self.intendedDirection)()

    def eggCountUnofficial(self, rowNum=None, colNum=None):
        """Return the number of egg-position clicks for the given sub-image
        (default: current sub-image).
        """
        if rowNum == None:
            rowNum = self.rowNum()
        if colNum == None:
            colNum = self.colNum()
        return len(
            self.clickLabelManager.getClicks(
                self.lowerBasenames[self.imgIdx],
                self.rowNum(),
                self.colNum(),
                self.wellPosition,
            )
        )

    def handleNormalKeypress(self, keyCode, k):
        """Handle keyboard events under any circumstance when a modal dialog is not
        being displayed.

        Backspace clears the count for the current sub-image.

        Enter confirms the count for the current sub-image.

        "s" exports the counts and egg-position clicks to file (see
        `exportEggCountReport`).

        "q" quits the program after also exporting data to file.

        "." and "," step the current sub-image forward and backward, respectively.

        "c" clears the egg-position clicks for the current sub-image.

        "h" toggles display of the help message.

        Arguments:
          - keyCode: String corresponding to the key that was pressed
          - k: numeric ID of the keypress for certain keys without keycodes (e.g.,
               "Enter" and "Backspace").
        """
        if k == 8:
            if (
                self.mode == EditMode.translate
                or self.eggCounts[self.imgIdx][self.subImgIdx] is None
            ):
                return
            self.eggCounts[self.imgIdx][self.subImgIdx] = None
            self.clickLabelManager.clearClicks(*self.currentImgArgs())
            self.basicImgDrawn, self.eggsDrawn = False, False
        if k == 13:
            if self.mode == EditMode.translate:
                return
            self.basicImgDrawn, self.eggsDrawn = False, False
            if (
                self.clickLabelManager.getNumClicks(
                    self.lowerBasenames[self.imgIdx],
                    self.rowNum(),
                    self.colNum(),
                    self.wellPosition,
                )
                == None
            ):
                input("adding subimage key from keyboard command")
                self.clickLabelManager.addKey(
                    self.lowerBasenames[self.imgIdx],
                    self.rowNum(),
                    self.colNum(),
                    self.wellPosition,
                )
            self.eggCounts[self.imgIdx][self.subImgIdx] = self.eggCountUnofficial()
            self.clicksConfirmed = True
            self.steppedWithConfirm = time.time()
            self.stepForward()
        if keyCode == "s" or keyCode == "q":
            self.getFrontierIndices()
            self.checkForSaveBlockers()
            if len(self.saveBlocker) > 0:
                return
            else:
                self.exportEggCountReport()
                if keyCode == "q":
                    self.thread.doRun = False
                    sys.exit()
        if keyCode == ".":
            self.basicImgDrawn, self.eggsDrawn = False, False
            self.steppedWithConfirm = None
            self.vectors = {}
            self.stepForward()
        if keyCode == "c":
            if self.mode == EditMode.translate:
                return
            self.clickLabelManager.clearClicks(
                self.lowerBasenames[self.imgIdx],
                self.rowNum(),
                self.colNum(),
                self.wellPosition,
            )
            self.basicImgDrawn, self.eggsDrawn = False, False
            self.clicksConfirmed = False
        if keyCode == ",":
            self.steppedWithConfirm = None
            self.basicImgDrawn, self.eggsDrawn = False, False
            self.vectors = {}
            self.stepBackward()
        if keyCode == "h":
            return
            # self.showHelp = ~self.showHelp
        if keyCode == "1":
            self.eggsDrawn = False
            self.clickLabelManager.addCategoryLabel(
                self.lowerBasenames[self.imgIdx],
                self.rowNum(),
                self.colNum(),
                "Blurry",
                True,
                self.wellPosition,
            )
        if keyCode == "2":
            self.eggsDrawn = False
            self.clickLabelManager.addCategoryLabel(
                self.lowerBasenames[self.imgIdx],
                self.rowNum(),
                self.colNum(),
                "Blurry",
                False,
                self.wellPosition,
            )
        if keyCode == "3":
            self.eggsDrawn = False
            self.clickLabelManager.addCategoryLabel(
                self.lowerBasenames[self.imgIdx],
                self.rowNum(),
                self.colNum(),
                "Difficult",
                True,
                self.wellPosition,
            )
        if keyCode == "4":
            self.eggsDrawn = False
            self.clickLabelManager.addCategoryLabel(
                self.lowerBasenames[self.imgIdx],
                self.rowNum(),
                self.colNum(),
                "Difficult",
                False,
                self.wellPosition,
            )

    def setImg(self):
        """Resize the sub-image, add sidebars to it, and assign it to the `baseImg`
        attribute of the EggCountLabeler instance.
        """
        if len(self.subImgs[self.imgIdx]) == 0:
            self.showCenteredMessage()
            while len(self.subImgs[self.imgIdx]) == 0:
                time.sleep(1)
        self.centerImg = self.subImgs[self.imgIdx][self.subImgIdx]
        if self.centerImg.shape[1] / self.centerImg.shape[0] >= 1.5:
            self.scalingFactor = 1.5
        else:
            self.scalingFactor = 2
        resizedDims = [self.scalingFactor * dim for dim in self.centerImg.shape]
        bufferVert = 100
        bufferHoriz, combinedHeightUpperLowerBorders = 100, 80
        self.upperBorderHt, self.sidebarWidth = 30, 230
        vidIdWidth = getTextSize(self.vidID())[1] + 15
        if (self.monitorInfo.height - bufferVert) < resizedDims[
            0
        ] + combinedHeightUpperLowerBorders:
            self.scalingFactor = (
                self.monitorInfo.height - bufferVert - combinedHeightUpperLowerBorders
            ) / self.centerImg.shape[0]
        self.sidebarWidth = max(
            self.sidebarWidth,
            int(0.5 * (vidIdWidth - self.scalingFactor * self.centerImg.shape[1])),
        )
        widthDiff = (self.monitorInfo.width - bufferHoriz) - (
            self.centerImg.shape[1] * self.scalingFactor + 2 * self.sidebarWidth
        )
        if widthDiff < 0:
            if abs(widthDiff) < 2 * self.sidebarWidth:
                self.sidebarWidth = round(
                    (
                        self.monitorInfo.width
                        - bufferHoriz
                        - self.centerImg.shape[1] * self.scalingFactor
                    )
                    / 2
                )
            else:
                self.sidebarWidth = 0
                self.scalingFactor = (
                    self.monitorInfo.width - bufferHoriz
                ) / self.centerImg.shape[1]
        self.centerImg = cv2.resize(
            self.centerImg, (0, 0), fx=self.scalingFactor, fy=self.scalingFactor
        )
        mainImg = np.hstack(
            (
                np.zeros((self.centerImg.shape[0], self.sidebarWidth, 3), np.uint8),
                self.centerImg,
                np.zeros((self.centerImg.shape[0], self.sidebarWidth, 3), np.uint8),
            )
        )
        upperBorder, lowerBorder = [
            np.full((height, mainImg.shape[1], 3), 255, np.uint8)
            for height in (self.upperBorderHt, 30)
        ]
        self.baseImg = np.vstack((upperBorder, mainImg, lowerBorder))

    def getWellPosition(self, idx=None, subImgIdx=None):
        if idx == None:
            idx = self.imgIdx
        if subImgIdx == None:
            subImgIdx = self.subImgIdx
        if (
            self.clickLabelManager.chamberTypes[self.lowerBasenames[idx]]
            != CT.large.name
        ):
            return None
        return POSITIONS[subImgIdx % 4]

    def rowNum(self, idx=None, subImgIdx=None):
        """Return row index of the current sub-image based on its position in the
        original image.
        """
        if idx == None:
            idx = self.imgIdx
        if subImgIdx == None:
            subImgIdx = self.subImgIdx
        if (
            self.clickLabelManager.chamberTypes[self.lowerBasenames[idx]]
            == CT.large.name
        ):
            return self.rowNumFourCircle(idx, subImgIdx)
        return int(
            np.floor(subImgIdx / (2 * self.rowColCounts[self.lowerBasenames[idx]][1]))
        )

    def rowNumFourCircle(self, idx, subImgIdx):
        """Return column index of the current sub-image for a four-circle chamber
        type.
        """
        numCirclesPerWell = 4
        numCirclesPerRow = (
            self.rowColCounts[self.lowerBasenames[idx]][0] * numCirclesPerWell
        )
        return np.floor(subImgIdx / numCirclesPerRow).astype(int)

    def colNum(self, idx=None, subImgIdx=None):
        """Return column index of the current sub-image based on its position in the
        original image.

        Arguments:
          - idx: index of the original image (default: the current one)
          - subImgIdx: index of the sub-image (default: the current one)
        """
        if idx == None:
            idx = self.imgIdx
        if subImgIdx == None:
            subImgIdx = self.subImgIdx
        if (
            self.clickLabelManager.chamberTypes[self.lowerBasenames[idx]]
            == CT.large.name
        ):
            return self.colNumFourCircle(idx)
        return subImgIdx % int(2 * self.rowColCounts[self.lowerBasenames[idx]][1])

    def colNumFourCircle(self, idx):
        """Return column index of the current sub-image for a four-circle chamber
        type.

        Arguments:
          - idx: index of the original image
        """
        numCirclesPerWell = 4
        numCirclesPerRow = (
            self.rowColCounts[self.lowerBasenames[idx]][0] * numCirclesPerWell
        )
        return np.floor((self.subImgIdx % numCirclesPerRow) / numCirclesPerWell).astype(
            int
        )

    def vidIDDisplay(self):
        """Display image filename, row number, and column number in the GUI."""
        putText(self.baseImg, self.vidID(), (10, 10), (0, 1), textStyle(color=COL_BK))

    def vidID(self):
        """Return image filename, row number, and column number as a
        display-friendly string.
        """
        vidIdStr = "%s, row %i,  col %i" % (
            self.imgPaths[self.imgIdx],
            self.rowNum(),
            self.colNum(),
        )
        if (
            self.clickLabelManager.chamberTypes[self.lowerBasenames[self.imgIdx]]
            == CT.large.name
        ):
            self.wellPosition = self.getWellPosition()
            vidIdStr += ", %s" % self.wellPosition
        else:
            self.wellPosition = None
        return vidIdStr

    def eggCountDisplay(self):
        """Display the saved egg count for the current sub-image (or otherwise a
        message indicating counts have not been saved).
        """
        count = self.eggCounts[self.imgIdx][self.subImgIdx]
        putText(
            self.eggsImg,
            "number eggs: %s" % ("___" if count is None else str(count)),
            (10, self.eggsImg.shape[0] - 18),
            (0, 1),
            textStyle(color=COL_BK),
        )
        if count is None and self.eggCountUnofficial() > 0:
            self.clicksConfirmed = False
        elif count is not None and self.eggCountUnofficial() != count:
            self.clicksConfirmed = False
        elif int(count or 0) > 0:
            self.clicksConfirmed = True

    def userNameDisplay(self):
        """Display the username in the GUI."""
        usernameText = "user: %s" % username
        textWidth = getTextSize(usernameText)[1]
        putText(
            self.baseImg,
            usernameText,
            (self.baseImg.shape[1] - textWidth - 10, self.baseImg.shape[0] - 18),
            (0, 1),
            textStyle(color=COL_BK),
        )

    def priorCountDisplay(self):
        """Display the egg count confirmed from the prior image for a short time
        after advancing.
        """
        if self.steppedWithConfirm is not None:
            backImgIdx, backSubImgIdx = self.getBackwardIndices()
            message = "Egg count of %i saved." % (
                self.eggCounts[backImgIdx][backSubImgIdx]
            )
            vertOffset = self.deltaMsgHeight if userForComparison else 0
            messageWidth = getTextSize(message)[1]
            maskSlices = [
                slice(60 + vertOffset, 80 + vertOffset),
                slice(7, messageWidth + 15),
            ]
            self.setMask(maskSlices)
            self.eggsImg = overlay(self.eggsImg, self.mask, COL_G)
            putText(
                self.eggsImg,
                message,
                (12, 65 + vertOffset),
                (0, 1),
                textStyle(color=COL_W),
            )

    def currentClickLabels(self):
        return self.clickLabelManager.getClicks(*self.currentImgArgs())

    def closestEgg(self):
        """Return the index of the first egg click within an given distance in pixels
        from the cursor, or -1 if there is no nearby egg.
        The distance is 8px if in annotation mode and 2px if in translate mode.
        """
        for i, egg in enumerate(self.currentClickLabels()):
            x = int(egg[0] * self.scalingFactor + self.sidebarWidth)
            y = int(egg[1] * self.scalingFactor + self.upperBorderHt)
            if distance((x, y), (self.lastX, self.lastY)) < (
                2
                if self.mode == EditMode.translate and "current" in self.vectors
                else 8
            ):
                return i
        return -1

    def getEggColor(self, i):
        """Return the color to use when rendering an egg click. The first click
        location within 8px of the cursor is shown in red, otherwise orange.

        Arguments:
          - i: index of the egg being drawn (in order of how they were clicked)
        """
        return DELETE_COLOR if i == self.lastClosestEgg else COL_O

    def showIgnoreLabel(self):
        if self.imgIdx not in self.imgs_to_ignore:
            return
        overlay = self.eggsImg.copy()
        textDims = cv2.getTextSize(
            "Image can be skipped", cv2.FONT_HERSHEY_PLAIN, 1.8, 2
        )[0][::-1]
        putText(
            overlay,
            "Image can be skipped",
            (
                0.5 * (self.eggsImg.shape[1] - textDims[1]),
                0.5 * (self.eggsImg.shape[0] - textDims[0]),
            ),
            (0, 1),
            textStyle(1.8, COL_R_D, 2),
        )
        alpha = 0.4
        self.eggsImg = cv2.addWeighted(overlay, alpha, self.eggsImg, 1 - alpha, 0)

    def imageFrameToAppFrame(self, xy):
        return (
            round(xy[0] * self.scalingFactor + self.sidebarWidth),
            round(xy[1] * self.scalingFactor + self.upperBorderHt),
        )

    def appFrameToImageFrame(self, xy):
        return (
            round((xy[0] - self.sidebarWidth) / self.scalingFactor),
            round((xy[1] - self.upperBorderHt) / self.scalingFactor),
        )

    def showEggClicks(self):
        """Display the egg-position clicks and the corresponding running count in
        the GUI.
        """
        clickLabels = self.clickLabelManager.getClicks(
            self.lowerBasenames[self.imgIdx],
            self.rowNum(),
            self.colNum(),
            self.wellPosition,
        )
        putText(
            self.eggsImg,
            "num eggs clicked: %i" % len(clickLabels),
            (10, 42),
            (0, 1),
            textStyle(0.9, COL_O),
        )
        self.closestEggIdx = None
        for i, egg in enumerate(clickLabels):
            x, y = self.imageFrameToAppFrame(egg)
            color = self.getEggColor(i)
            deleteCandidate = color == DELETE_COLOR
            if deleteCandidate:
                self.closestEggIdx = i
            cv2.drawMarker(
                self.eggsImg,
                (x, y),
                color,
                cv2.MARKER_DIAMOND,
                markerSize=12,
                thickness=3 if deleteCandidate else 1,
            )

    def getButtonCoords(self, tp):
        """Return tuple of coordinates for the upper left and lower right vertices
        of a rectangular button in the GUI of given type: `((x1, y1), (x2, y2))`.

        Arguments:
          - tp: String corresponding to a type of button, e.g., "enter."
        """
        enterHtOffset = 0.5
        enterHalfHt = 151
        if tp in ("enter", "translate"):
            heightOffset = enterHtOffset
            halfHeight = enterHalfHt
        elif tp == "jump":
            halfHeight = 40
            heightOffset = (
                0.5 * self.baseImg.shape[0] - enterHalfHt - halfHeight - 4
            ) / self.baseImg.shape[0]
        return (
            (
                self.baseImg.shape[1] - round(self.sidebarWidth * 0.8),
                round(self.baseImg.shape[0] * heightOffset) + halfHeight,
            ),
            (
                self.baseImg.shape[1] - round(self.sidebarWidth * 0.2),
                round(self.baseImg.shape[0] * heightOffset) - halfHeight,
            ),
        )

    def buttonDisplay(self, tp, coords, text, textSize, textWt, buttonColors):
        """Display a button in the GUI.

        Arguments:
          - tp: String corresponding to a type of button, e.g., "enter."
          - coords: Tuple of coordinates of upper-left and lower-right button
                    vertices (see `getButtonCoords`).
          - text: Text to display on the button
          - textSize: Size of text to display on the button
          - textWt: Weight of text to display on the button
          - buttonColors: dict containing RGB tuple for "on" and "off" keys, to
                          toggle color depending on whether the cursor is above
                          the button.
        """
        self.buttonCoords[tp] = coords
        buttonColor = (
            buttonColors["on"]
            if getattr(self, "onButton_%s" % tp, False)
            else buttonColors["off"]
        )
        cv2.rectangle(
            self.eggsImg, self.buttonCoords[tp][0], self.buttonCoords[tp][1], COL_W, 2
        )
        cv2.rectangle(
            self.eggsImg,
            self.buttonCoords[tp][0],
            self.buttonCoords[tp][1],
            buttonColor,
            cv2.FILLED,
        )
        textWidth = getTextSize(text, textSize)[1]
        putText(
            self.eggsImg,
            text,
            (
                round(np.mean([self.buttonCoords[tp][i][0] for i in range(2)]))
                - round(0.5 * textWidth),
                round(np.mean([self.buttonCoords[tp][i][1] for i in range(2)])),
            ),
            (0, 1),
            (cv2.FONT_HERSHEY_PLAIN, textSize, COL_W, textWt, cv2.LINE_AA),
        )

    def create_buttons(self):
        key = "jump"

        def handler():
            self.jumpToFrontier()

        self.buttons[key] = Button(
            key,
            handler,
            "Jump to frontier",
            self.getButtonCoords(key),
            0.7,
            1,
            (100, 30, 0),
            (200, 100, 0),
        )

        if self.mode == EditMode.annot:
            key = "enter"
            text = key.capitalize()
            text_size = 1.5
            weight = 2

            def handler():
                self.handleNormalKeypress("", 13)
                self.renderSubImg()

        else:
            key = "translate"
            text = "Move clicks"
            text_size = 1.1
            weight = 1

            def handler():
                self.translate_clicks_by_mean_vector()

        self.buttons[key] = Button(
            key,
            handler,
            text,
            self.getButtonCoords(key),
            text_size,
            weight,
            (50, 180, 0),
            (100, 255, 0),
        )

    def display_buttons(self):
        if self.mode == EditMode.translate and len(self.vectors) == 0:
            self.buttons["translate"].disabled = True
        for b_key in self.buttons:
            self.buttons[b_key].coords = self.getButtonCoords(b_key)
            self.buttons[b_key].display(self.eggsImg)

    def absolutePosition(self, idx=None, subImgIdx=None, guess=False):
        """Return index of the current sub-image."""
        if idx == None:
            idx = self.imgIdx
        if subImgIdx == None:
            subImgIdx = self.subImgIdx
        if not guess and any(
            [len(eggCountList) == 0 for eggCountList in self.subImgs[:idx]]
        ):
            return None
        return (
            sum([len(eggCountList) for eggCountList in self.subImgs[:idx]]) + subImgIdx
        )

    def getImgAndSubImgIdx(self, absIdx):
        """Given an absolute index, return its associated image and sub-image
        indices.

        Arguments:
          - absIdx: index of the sub-image after flattening the sub-images across
                    all images in row-dominant order
        """
        runningSum = 0
        for i in range(len(self.imgPaths)):
            compResults = self.compareCountWithNumCells(i, runningSum, absIdx)
            if len(compResults) < 3:
                runningSum = compResults[0]
            else:
                return compResults[0:2]

    def compareCountWithNumCells(self, imgIdx, runningSum, stopPoint):
        """Augment runningSum by the number of cells in the given image. If the
        running sum meets or exceeds the given stopPoint, then return the image
        index, with a sub-image index given by how far the stopPoint exceeds the
        cumulative sum of sub-images before the given image index.

        Returns only the augmented running sum (no indices) if the running sum fails
        to exceed the stopPoint.

        Arguments:
          - imgIdx: index of the image
          - runningSum: the number with which to compare stopPoint
          - stopPoint: index that runningSum must exceed
        """
        rowCols = self.rowColCounts[self.lowerBasenames[imgIdx]]
        is4C = (
            self.clickLabelManager.chamberTypes[self.lowerBasenames[imgIdx]]
            == CT.large.name
        )
        numCells = (2 if not is4C else 1) * rowCols[0] * rowCols[1]
        if is4C:
            numCells *= 4
        runningSum += numCells
        if runningSum == stopPoint:
            frontierImgIdx = imgIdx + 1
            subImgIdx = 0
            return frontierImgIdx, subImgIdx, runningSum
        if runningSum > stopPoint:
            frontierImgIdx = imgIdx
            runningSum -= numCells
            subImgIdx = stopPoint - runningSum
            return frontierImgIdx, subImgIdx, runningSum
        return (runningSum,)

    def positionDisplay(self):
        """Display absolute position of the current sub-image in the GUI."""
        positionText = "image %i of %i" % (
            self.absolutePosition(guess=True) + 1,
            self.totalNumSubImgs(),
        )
        textWidth = getTextSize(positionText)[1]
        putText(
            self.baseImg,
            positionText,
            (
                int(0.5 * (self.baseImg.shape[1] - textWidth)),
                self.baseImg.shape[0] - 18,
            ),
            (0, 1),
            textStyle(),
        )

    def help_text(self):
        return {
            EditMode.annot: "S- save all counts\nQ- save and quit\n"
            + "C- clear clicks for region\nShift + L Click- delete one egg\nEnter- "
            + 'confirm count for region\nBackspace- clear count for region\n"."-'
            + ' advance forward one region\n","- step backward one region',
            EditMode.translate: "S- save all counts\nQ- save and quit\n"
            + '"."- advance forward one region\n","- step backward one region\n\n'
            + "To draw a desired translation vector,\nclick on an egg"
            + " and then click\nanywhere else on the image. Double\n"
            + "click an egg to clear its vector.\n\nAll eggs will be"
            + " translated by the\naverage of all vectors drawn. Vectors\n"
            + "are not saved after moving to a\ndifferent region.",
        }[self.mode]

    def helpDisplay(self):
        """Display an info box explaining keyboard commands in the GUI."""
        introText = "keyboard and mouse commands"
        helpText = self.help_text()
        heightOffset = 0.5
        halfHeight = 0.5 * 6 * getTextSize(introText, 0.7)[0]
        if not hasattr(self, "helpTextWidth"):
            singleLineWidths = [
                getTextSize(line, size=0.7)[1] for line in helpText.split("\n")
            ]
            self.helpTextWidth = max(singleLineWidths)
        if self.sidebarWidth < self.helpTextWidth:
            return
        putText(
            self.baseImg,
            introText,
            (
                int(0.5 * (self.sidebarWidth - getTextSize(introText, 0.7)[1])),
                round(self.baseImg.shape[0] * heightOffset) - halfHeight - 20,
            ),
            (0, 1),
            textStyle(0.7, COL_W),
        )
        putText(
            self.baseImg,
            helpText,
            (
                int(0.5 * (self.sidebarWidth - self.helpTextWidth)),
                round(self.baseImg.shape[0] * heightOffset) - halfHeight,
            ),
            (0, 1),
            textStyle(0.7, COL_W),
        )

    def renderSubImg(self):
        """Display a sub-image and all accompanying information (e.g., egg clicks,
        counts, progress bar) in the GUI.
        """
        if self.mode == EditMode.translate and "current" in self.vectors:
            self.draw_temporary_edit_window()
        self.draw_main_edit_window()
        self.lastRender = time.time()

    def draw_main_edit_window(self):
        if not self.basicImgDrawn:
            self.setImg()
            self.vidIDDisplay()
            self.positionDisplay()
            self.helpDisplay()
            self.userNameDisplay()
            self.helpText()
            self.basicImgDrawn = True
        if self.eggsDrawn:
            closestEgg = self.closestEgg()
            if closestEgg != self.lastClosestEgg:
                self.eggsDrawn = False
            self.lastClosestEgg = closestEgg
        if len(self.buttons) == 0:
            self.create_buttons()
        if not self.eggsDrawn and not self.inModal:
            self.eggsImg = np.array(self.baseImg)
            self.eggCountDisplay()
            self.saveMessageDisplay()
            self.display_buttons()
            self.showIgnoreLabel()
            self.showEggClicks()
            self.show_finished_vectors()
            self.showCountDeltas()
            self.priorCountDisplay()
            self.showCategoryLabels()
            self.eggsDrawn = True
        self.liveImg = np.array(self.eggsImg)
        self.cursorDisplay()
        self.showEggCountPrompt()
        self.saveBlockerDisplay()
        cv2.imshow("Egg Count Annotator", self.liveImg)

    def showCategoryLabels(self):
        """
        Display categories (blurry or not, difficult or not) for the current
        sub-image in the GUI.
        """
        key = self.clickLabelManager.subImageKey(*self.currentImgArgs())
        labelOffsetPx = 20
        numLabelsAdded = 0
        for labelType in ("Blurry", "Difficult"):
            if key in getattr(self.clickLabelManager, "is%sLabels" % labelType):
                isPositive = getattr(self.clickLabelManager, "is%sLabels" % labelType)[
                    key
                ]
                if not isPositive:
                    prefix = "non-" if labelType == "Blurry" else "not "
                else:
                    prefix = ""
                text = "%s%s" % (prefix, labelType.lower())
                putText(
                    self.eggsImg,
                    text,
                    (self.eggsImg.shape[1] - 98, 40 + labelOffsetPx * numLabelsAdded),
                    (0, 1),
                    textStyle(color=COL_W),
                )
                numLabelsAdded += 1

    def showCountDeltas(self):
        """Display delta of current user's counts with that of another selected
        user.
        """
        if userForComparison == None:
            return
        self.getFrontierIndices()
        baselineMessage = "your count - %s's =" % userForComparison
        if (
            (self.frontierImgIdx < self.imgIdx)
            or (
                self.frontierImgIdx == self.imgIdx
                and self.frontierSubImgIdx <= self.subImgIdx
            )
        ) or (
            (self.diffFrontierData["frontierImgIdx"] < self.imgIdx)
            or (
                self.diffFrontierData["frontierImgIdx"] == self.imgIdx
                and self.diffFrontierData["subImgIdx"] <= self.subImgIdx
            )
        ):
            count = "N/A"
        else:
            key = self.clickLabelManager.subImageKey(*self.currentImgArgs())
            if key not in self.diffClickData:
                count = "N/A"
            else:
                count = str(
                    len(self.clickLabelManager.getClicks(*self.currentImgArgs()))
                    - len(self.diffClickData[key])
                )
        message = "%s %s" % (baselineMessage, count)
        msgWidth = getTextSize(message)[1]
        if self.sidebarWidth < msgWidth:
            message = message.split("'s")
            message[1] = "\n   %s" % message[1]
            message = "'s".join(message)
            self.deltaMsgHeight = 40
        else:
            self.deltaMsgHeight = 30
        putText(self.eggsImg, message, (10, 68), (0, 1), textStyle(color=COL_O))

    def saveBlockerDisplay(self):
        """
        Display alert in the GUI if flaws are found in the data that prevent saving.
        """
        if len(self.saveBlocker) == 0:
            return
        if not self.inModal:
            self.steppedWithConfirm = None
            self.inModal = True
            if self.saveBlocker == "gaps":
                self.showCenteredMessage(
                    "Saving is enabled only if egg-laying\n"
                    + "regions are labeled consecutively (i.e.,\nwithout skipping a "
                    + "region up to the\nlabeling frontier. Please press Enter to\njump "
                    + "to the first unlabeled region.",
                    overBlank=False,
                    textSize=0.9,
                    color=COL_R_D,
                )
            elif self.saveBlocker == "mismatch":
                self.showCenteredMessage(
                    "Saving is enabled only if the click count\n"
                    + "matches the saved egg count for every\nlabeled region. Please"
                    + " press Enter to\njump to the first region where a conflict\n"
                    + "was found.",
                    overBlank=False,
                    textSize=0.9,
                    color=COL_R_D,
                )
            elif self.saveBlocker == "csvFileInUse":
                self.showCenteredMessage(
                    "Error: the CSV file of egg counts for this"
                    + "\nfolder is already in use. Close any other\nprogram using it ("
                    + "likely Excel) and try to\nsave again. Press enter to dismiss\n"
                    + "this message.",
                    textSize=0.9,
                    overBlank=False,
                    color=COL_R_D,
                )

    def showEggCountPrompt(self):
        """Display message prompting user to confirm they want to move to a
        different sub-image without first saving their egg counts.
        """
        if self.checkClickConfirmationStatus and not self.inModal:
            self.inModal = True
            self.showCenteredMessage(
                "Egg count (%i)" % self.eggCountUnofficial()
                + " has not been saved.\nStay on the current image? (y/n)",
                overBlank=False,
                textSize=0.9,
                color=COL_BK,
            )

    def getBackwardIndices(self):
        """Return the image and sub-image indices for the sub-image one step
        backward.
        """
        backImgIdx, backSubImgIdx = 0, 0
        if (
            self.imgIdx + 1 == len(self.imgPaths)
            and self.subImgIdx + 1 == len(self.subImgs[self.imgIdx])
            and self.eggCounts[self.imgIdx][self.subImgIdx] is not None
        ):
            return (self.imgIdx, self.subImgIdx)
        if self.subImgIdx > 0:
            backSubImgIdx = self.subImgIdx - 1
            backImgIdx = self.imgIdx
        elif self.subImgIdx == 0 and self.imgIdx > 0:
            backImgIdx = self.imgIdx - 1
            backSubImgIdx = len(self.subImgs[backImgIdx]) - 1
        return (backImgIdx, backSubImgIdx)

    def stepBackward(self):
        """Update indices to display the sub-image one step back."""
        if hasattr(self, "clicksConfirmed") and not self.clicksConfirmed:
            self.checkClickConfirmationStatus = True
            self.intendedDirection = "Backward"
            return
        if self.subImgIdx > 0:
            self.subImgIdx -= 1
        elif self.subImgIdx == 0 and self.imgIdx > 0:
            self.imgIdx -= 1
            self.subImgIdx = len(self.subImgs[self.imgIdx]) - 1

    def stepForward(self):
        """Update indices to display the sub-image one step forward."""
        if hasattr(self, "clicksConfirmed") and not self.clicksConfirmed:
            self.checkClickConfirmationStatus = True
            self.intendedDirection = "Forward"
            return
        if self.subImgIdx + 1 < len(self.subImgs[self.imgIdx]):
            self.subImgIdx += 1
        elif self.imgIdx + 1 < len(self.imgPaths):
            self.subImgIdx = 0
            self.imgIdx += 1


def getNumericSelection(
    options, optionListName="userlist", confirm=True, zeroToSkip=False
):
    """Prompt user to select one option from a numbered list.

    Arguments:
      - options: list of options from which to choose
      - optionListName: how the list should be identified in messages to the user
                        (default: userlist)
      - confirm: bool indicating whether user must input Y to confirm selection
                 (default: True)
      - zeroToSkip: bool indicating whether an input of 0 can be used for skipping
                    the selection dialogue (default: False)
    """
    numberedOptionList = "\t".join(
        ["%i) %s" % (i + 1, option) for i, option in enumerate(options)]
    )
    reminderMessage = "\nreminder: %s is %s" % (optionListName, numberedOptionList)
    attempts = 0
    smallestPossibleVal = 0 if zeroToSkip else -1
    enteredInt = False
    selectionMade = False
    print(numberedOptionList)
    while not selectionMade:
        attempts += 1
        try:
            selection = int(input())
            enteredInt = True
        except ValueError as _:
            print("error: non-numeric input. Please input a number.")
            enteredInt = False
        if enteredInt:
            try:
                if selection <= smallestPossibleVal - 1:
                    raise IndexError
                if selection == 0:
                    selectedOption = None
                else:
                    selectedOption = options[selection - 1]
                selectionMade = True
            except IndexError as _:
                print(
                    "error: input out of range. Please input a number between"
                    + "%i and %i." % (smallestPossibleVal, len(users))
                )
        if attempts >= 3 and attempts % 3 == 0:
            print(reminderMessage)
        if selectionMade and selectedOption != None:
            print("\nyou selected:", selectedOption)
            if confirm:
                userSelected = not (input("is this correct (y/n)?") == "n")
            else:
                userSelected = True
            if not userSelected:
                print(reminderMessage)
        else:
            print("skipped")
    return selectedOption


opts = options()
print("\nEgg Count Annotation Tool - Yang Lab")
print("\nplease input the number corresponding to your name")
users = sorted(users)
username = getNumericSelection(users)
print("\nconfirmed user:", username)
print("\n\nchoose user with whom to compare counts data (0 to skip)")
userForComparison = getNumericSelection(
    [name for name in users if name != username], confirm=False, zeroToSkip=True
)
EggCountLabeler().annotateEggCounts()
