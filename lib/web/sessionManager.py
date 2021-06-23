import csv
from datetime import datetime
import inspect
import json
from lib.web.network_loader import NetworkLoader
from lib.image.manual_segmenter import ManualSegmenter
import os
import time
import warnings
warnings.filterwarnings("error")

import cv2
import numpy as np
from PIL import Image
import torch

from chamber import CT
from circleFinder import CircleFinder
from detectors.fcrn import model
from lib.web.exceptions import CUDAMemoryException, ImageAnalysisException


class SessionManager:
    """Represent and process information while using the egg-counting web app."""

    def __init__(self, socketIO, room, network_loader: NetworkLoader):
        """Create a new SessionData instance.

        Arguments:
          - socketIO: SocketIO server
        """
        self.chamberTypes = {}
        self.predictions = {}
        self.basenames = {}
        self.annotations = {}
        self.socketIO = socketIO
        self.room = room
        self.network_loader = network_loader
        self.lastPing = time.time()
        self.errorMessages = {
            ImageAnalysisException: "Image could not be analyzed.",
            CUDAMemoryException: "Error: system ran out of resources",
        }

    def clear_data(self):
        self.chamberTypes = {}
        self.predictions = {}
        self.annotations = {}

    def emit_to_room(self, evt_name, data):
        self.socketIO.emit(evt_name, data, room=self.room)

    @staticmethod
    def is_CUDA_mem_error(exc):
        exc_str = str(exc)
        return (
            "CUDA out of memory" in exc_str
            or "Unable to find a valid cuDNN" in exc_str
            or "cuDNN error" in exc_str
        )

    def report_counting_error(self, imgPath, err_type):
        prefix = {
            ImageAnalysisException: "Error",
            CUDAMemoryException: "Ran out of system resources while",
        }[err_type]
        self.emit_to_room(
            "counting-error",
            {
                "data": "%s counting eggs for image %s"
                % (prefix, self.basenames[imgPath]),
                "filename": self.basenames[imgPath],
                "error_type": err_type.__name__,
            },
        )
        self.predictions[imgPath].append(err_type)
        time.sleep(2)

    def segment_image_via_alignment_data(self, img, img_path, alignment_data):
        segmenter = ManualSegmenter(
            img, alignment_data["nodes"], alignment_data["type"]
        )
        self.subImgs, self.bboxes = segmenter.calc_bboxes_and_subimgs()
        self.chamberTypes[img_path] = alignment_data["type"]
        self.rotation_angle = segmenter.rotation_angle

    def segment_image_via_object_detection(self, img, img_path):
        self.cf = CircleFinder(img, os.path.basename(img_path), allowSkew=True)
        if self.cf.skewed:
            self.emit_to_room(
                "counting-progress",
                {
                    "data": "Skew detected in image %s;"
                    + " stopping analysis." % self.imgBasename
                },
            )
        try:
            circles, avgDists, numRowsCols, rotatedImg, _ = self.cf.findCircles()
            self.chamberTypes[img_path] = self.cf.ct
            self.subImgs, self.bboxes = self.cf.getSubImages(
                rotatedImg, circles, avgDists, numRowsCols
            )
        except Exception as exc:
            if self.is_CUDA_mem_error(exc):
                raise CUDAMemoryException
            else:
                raise ImageAnalysisException
        except RuntimeWarning:
            raise ImageAnalysisException

    def segment_img_and_count_eggs(self, img_path, alignment_data=None, index=None):
        imgBasename = os.path.basename(img_path)
        self.imgBasename = imgBasename
        img_path = os.path.normpath(img_path)
        self.imgPath = img_path
        self.predictions[img_path] = []
        self.basenames[img_path] = imgBasename
        self.emit_to_room(
            "counting-progress", {"data": "Segmenting image %s" % imgBasename}
        )
        img = np.array(Image.open(img_path), dtype=np.float32)
        if alignment_data is None:
            self.segment_image_via_object_detection(img, img_path)
        else:
            if "ignored" in alignment_data and alignment_data["ignored"]:
                self.predictions[img_path].append(ImageAnalysisException)
                return
            self.segment_image_via_alignment_data(img, img_path, alignment_data)
        self.emit_to_room(
            "counting-progress", {"data": "Counting eggs in image %s" % imgBasename}
        )
        for subImg in self.subImgs:
            self.predictions[img_path].append(self.network_loader.predict_instances(subImg))
        self.emit_to_room("counting-progress", {"data": "Finished counting eggs"})
        self.bboxes = [[int(el) for el in bbox] for bbox in self.bboxes]
        self.sendAnnotationsToClient(index)

    def sendAnnotationsToClient(self, index):
        resultsData = []
        bboxes = self.bboxes
        for i, prediction in enumerate(self.predictions[self.imgPath]):
            if self.chamberTypes[self.imgPath] == CT.large.name:
                iMod = i % 4
                if iMod in (0, 3):
                    x = bboxes[i][0] + 0.1 * bboxes[i][2]
                    y = bboxes[i][1] + 0.15 * bboxes[i][3]
                elif iMod == 1:
                    x = bboxes[i][0] + 0.4 * bboxes[i][2]
                    y = bboxes[i][1] + 0.2 * bboxes[i][3]
                elif iMod == 2:
                    x, y = (
                        bboxes[i][0] + 0.2 * bboxes[i][2],
                        bboxes[i][1] + 0.45 * bboxes[i][3],
                    )
            elif self.chamberTypes[self.imgPath] == CT.opto.name:
                x = bboxes[i][0] + 0.5 * bboxes[i][2]
                y = bboxes[i][1] + (1.4 if i % 10 < 5 else -0.1) * bboxes[i][3]
            else:
                x = bboxes[i][0] + (1.40 if i % 2 == 0 else -0.32) * bboxes[i][2]
                y = bboxes[i][1] + 0.55 * bboxes[i][3]
            resultsData.append({"count": prediction, "x": x, "y": y, "bbox": bboxes[i]})
        counting_data = {
            "data": json.dumps(resultsData, separators=(",", ":")),
            "filename": self.imgBasename,
            'rotationAngle': self.rotation_angle if hasattr(self, 'rotation_angle') else None,
            "index": index,
        }
        self.emit_to_room(
            "counting-annotations",
            counting_data,
        )
        self.annotations[os.path.normpath(self.imgPath)] = resultsData

    def createErrorReport(self, edited_counts, user):
        for imgPath in edited_counts:
            rel_path = os.path.normpath(os.path.join("./uploads", imgPath))
            img = cv2.imread(rel_path)
            for i in edited_counts[imgPath]:
                annot = self.annotations[rel_path][int(i)]
                imgSection = img[
                    annot["bbox"][1] : annot["bbox"][1] + annot["bbox"][3],
                    annot["bbox"][0] : annot["bbox"][0] + annot["bbox"][2],
                ]
                cv2.imwrite(
                    os.path.join(
                        "error_cases",
                        ".".join(os.path.basename(imgPath).split(".")[:-1])
                        + "_%s_actualCt_%s_user_%s.png"
                        % (i, edited_counts[imgPath][i], user),
                    ),
                    imgSection,
                )
        self.emit_to_room("report-ready", {})

    def saveCSV(self, edited_counts):
        resultsPath = "temp/results_ALPHA_%s.csv" % datetime.today().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
        with open(resultsPath, "wt", newline="") as resultsFile:
            writer = csv.writer(resultsFile)
            writer.writerow(["Egg Counter, ALPHA version"])
            for i, imgPath in enumerate(self.predictions):
                writer.writerow([imgPath])
                first_pred = self.predictions[imgPath][0]
                if inspect.isclass(first_pred) and issubclass(first_pred, Exception):
                    writer.writerows([[self.errorMessages[first_pred]], []])
                    continue
                base_path = os.path.basename(imgPath)
                updated_counts = self.predictions[imgPath].copy()
                if base_path in edited_counts and len(edited_counts[base_path]) > 0:
                    writer.writerow(
                        ["Note: this image's counts have been amended by hand"]
                    )
                    for region_index in edited_counts[base_path]:
                        updated_counts[int(region_index)] = int(
                            edited_counts[base_path][region_index]
                        )
                CT[self.chamberTypes[imgPath]].value().writeLineFormatted(
                    [updated_counts], 0, writer
                )
                writer.writerow([])
        self.emit_to_room("counting-csv", {"data": os.path.basename(resultsPath)})
