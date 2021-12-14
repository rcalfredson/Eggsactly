import csv
from datetime import datetime
import inspect
import json
import os
import time
import traceback
import warnings

warnings.filterwarnings("error")

import cv2
import numpy as np
from PIL import Image, ImageDraw

from project.chamber import CT
from project.circleFinder import (
    CircleFinder,
    rotate_around_point_highperf,
    rotate_image,
)
from project.lib.event import Listener
from project.lib.image import drawing
from project.lib.image.node_based_segmenter import NodeBasedSegmenter
from project.lib.web.exceptions import (
    CUDAMemoryException,
    ImageAnalysisException,
    ImageIgnoredException,
    errorMessages,
)
from project.lib.web.gpu_manager import GPUManager
from project.lib.web.gpu_task_types import GPUTaskTypes
from project.lib.web.network_loader import NetworkLoader

with open("project/models/modelRevDates.json", "r") as f:
    model_to_update_date = json.load(f)


class SessionManager:
    """Represent and process information while using the egg-counting web app."""

    def __init__(
        self, socketIO, room, network_loader: NetworkLoader, gpu_manager: GPUManager
    ):
        """Create a new SessionData instance.

        Arguments:
          - socketIO: SocketIO server
          - room: SocketIO room where messages get sent
          - network_loader: NetworkLoader instance loaded with an egg-detection model
          - gpu_manager: GPUManager instance where GPU tasks get added
        """
        self.chamberTypes = {}
        self.cfs = {}
        self.predictions = {}
        self.basenames = {}
        self.annotations = {}
        self.alignment_data = {}
        self.socketIO = socketIO
        self.room = room
        self.network_loader = network_loader
        self.gpu_manager = gpu_manager
        self.lastPing = time.time()
        self.textLabelHeight = 96

    def clear_data(self):
        self.chamberTypes = {}
        self.cfs = {}
        self.predictions = {}
        self.annotations = {}
        self.alignment_data = {}
        self.basenames = {}

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
                "width": self.img.shape[1],
                "height": self.img.shape[0],
                "data": "%s counting eggs for image %s"
                % (prefix, self.basenames[imgPath]),
                "filename": self.basenames[imgPath],
                "error_type": err_type.__name__,
            },
        )
        self.predictions[imgPath].append(err_type)
        time.sleep(2)

    def segment_image_via_bboxes(self, img, alignment_data):
        img = cv2.resize(
            img,
            (0, 0),
            fx=alignment_data.get("scaling", 1),
            fy=alignment_data.get("scaling", 1),
        )
        img = rotate_image(img, alignment_data["rotationAngle"])
        self.bboxes = alignment_data["bboxes"]
        bbox_translation = [
            -el for el in alignment_data.get("imageTranslation", [0, 0])
        ]
        alignment_data["regionsToIgnore"] = []
        translated_bboxes = []
        for bbox in self.bboxes:
            new_bbox = [
                bbox[0] + bbox_translation[0],
                bbox[1] + bbox_translation[1],
                bbox[2],
                bbox[3],
            ]
            if new_bbox[0] < 0:
                new_bbox[2] += new_bbox[0]
                new_bbox[0] = 0
            if new_bbox[1] < 0:
                new_bbox[2] += new_bbox[1]
                new_bbox[1] = 0
            translated_bboxes.append(list(map(round, new_bbox)))

        self.bboxes = translated_bboxes
        self.subImgs = CircleFinder.getSubImagesFromBBoxes(
            img, translated_bboxes, alignment_data["regionsToIgnore"]
        )
        # for sub_img in self.subImgs:
        #     cv2.imshow("debug sub-img", sub_img.astype(np.uint8))
        #     print("scaling factor:", alignment_data.get("scaling", 1))
        #     print('sub-image?', sub_img.astype(np.uint8))
        #     cv2.waitKey(0)

    def segment_image_via_alignment_data(self, img, img_path, alignment_data):
        segmenter = NodeBasedSegmenter(
            img, alignment_data["nodes"], alignment_data["type"]
        )
        self.subImgs, self.bboxes = segmenter.calc_bboxes_and_subimgs()
        self.chamberTypes[img_path] = alignment_data["type"]
        if img_path in self.alignment_data:
            self.alignment_data[img_path]["rotationAngle"] = segmenter.rotation_angle
        else:
            self.alignment_data[img_path] = {"rotationAngle": segmenter.rotation_angle}

    def enqueue_arena_detection_task(self, img, img_path):
        self.cfs[img_path] = CircleFinder(
            img, os.path.basename(img_path), allowSkew=True
        )
        taskgroup = self.gpu_manager.add_task_group(
            self.room, n_tasks=1, task_type=GPUTaskTypes.arena
        )
        self.gpu_manager.add_task(
            taskgroup,
            img_path,
        )
        taskgroup.add_completion_listener(
            Listener(self.segment_image_via_object_detection, (img_path,))
        )

    def enqueue_egg_counting_task(self, index, n_files, img_path, alignment_data):
        taskgroup = self.gpu_manager.add_task_group(
            self.room, n_tasks=1, task_type=GPUTaskTypes.egg
        )
        self.gpu_manager.add_task(taskgroup, img_path, alignment_data)
        taskgroup.add_completion_listener(
            Listener(
                self.sendAnnotationsToClient, (index, n_files, img_path, alignment_data)
            )
        )

    # now reached the point where the SessionManager should be refactored
    # in the form it would need after the GPU worker assumes the details of egg
    # counting. The next step is to ensure the necessary data reaches the GPU worker
    # after which we need to transfer the code related to calculating sub-images
    # from SessionManager to the new SubImageHelper class.
    # one question: doesn't the SessionManager rely on those subimages
    # later when processing error reports?
    # no, it uses freestanding code to generate that. I'm not sure if
    # that's a good or bad decision, because it appears not to account
    # for any transformations of the image other than rotation,
    # but that's a concern separate from this round of refactoring.

    def segment_image_via_object_detection(self, img_path, predictions):
        imgBasename = os.path.basename(img_path)
        # why are there two levels of nesting?
        # we have the results from each task in the group,
        # and within each task, there can be multiple sub-images.
        print('initial predictions:', predictions)
        if type(predictions) is list and len(predictions) == 1:
            predictions = predictions[0]
            input('trimmed using the conditional')
        print('predictions:', predictions)
        try:
            print("running circle finder for this image path:", img_path)
            (circles, avgDists, numRowsCols, rotatedImg, rotationAngle) = self.cfs[
                img_path
            ].findCircles(debug=False, predictions=predictions)
            if self.cfs[img_path].skewed:
                self.emit_to_room(
                    "counting-progress",
                    {
                        "data": "Skew detected in image %s;"
                        + " stopping analysis." % imgBasename
                    },
                )
                raise ImageAnalysisException
            self.chamberTypes[img_path] = self.cfs[img_path].ct
            self.subImgs, self.bboxes = self.cfs[img_path].getSubImages(
                rotatedImg, circles, avgDists, numRowsCols
            )
            self.bboxes = [[round(el) for el in bbox] for bbox in self.bboxes]
            if img_path not in self.alignment_data:
                self.alignment_data[img_path] = {"rotationAngle": rotationAngle}
            else:
                self.alignment_data[img_path]["rotationAngle"] = rotationAngle
            self.emit_to_room(
                "chamber-analysis",
                {
                    "rotationAngle": self.alignment_data[img_path]["rotationAngle"],
                    "filename": imgBasename,
                    "chamberType": self.cfs[img_path].ct,
                    "bboxes": self.bboxes,
                    "width": self.img.shape[1],
                    "height": self.img.shape[0],
                },
            )
        except Exception as exc:
            print("exception while finding circles:", type(exc), exc)
            if self.is_CUDA_mem_error(exc):
                raise CUDAMemoryException
            else:
                self.report_counting_error(self.imgPath, ImageAnalysisException)
        except RuntimeWarning:
            self.report_counting_error(self.imgPath, ImageAnalysisException)
        del self.cfs[img_path]

    def check_chamber_type_and_find_bounding_boxes(self, img_path):
        imgBasename = os.path.basename(img_path)
        img_path = os.path.normpath(img_path)
        self.imgPath = img_path
        self.predictions[img_path] = []
        self.basenames[img_path] = imgBasename
        self.emit_to_room(
            "counting-progress", {"data": "Segmenting image %s" % imgBasename}
        )
        self.img = np.array(Image.open(img_path), dtype=np.float32)
        img = self.img
        self.enqueue_arena_detection_task(img, img_path)

    def segment_img_and_count_eggs(self, img_path, alignment_data, index, n_files):
        imgBasename = os.path.basename(img_path)
        img_path = os.path.normpath(img_path)
        self.predictions[img_path] = []
        self.basenames[img_path] = imgBasename
        self.emit_to_room(
            "counting-progress", {"data": "Segmenting image %s" % imgBasename}
        )
        self.alignment_data[img_path] = alignment_data
        if "ignored" in alignment_data and alignment_data["ignored"]:
            self.predictions[img_path].append(ImageIgnoredException)
        else:
            if "nodes" in alignment_data:
                self.chamberTypes[img_path] = alignment_data["type"]
            self.enqueue_egg_counting_task(index, n_files, img_path, alignment_data)
        #     if "nodes" in alignment_data:
        #         self.segment_image_via_alignment_data(img, img_path, alignment_data)
        #     elif "bboxes" in alignment_data:
        #         self.segment_image_via_bboxes(img, alignment_data)
        #     self.emit_to_room(
        #         "counting-progress",
        #         {"data": "Counting eggs in image %s" % imgBasename},
        #     )
        #     num_sub_imgs = len(self.subImgs)
        #     for i, subImg in enumerate(self.subImgs):
        #         self.emit_to_room(
        #             "counting-progress",
        #             {
        #                 "overwrite": True,
        #                 "data": f"Counting eggs in region {i+1} of {num_sub_imgs}",
        #             },
        #         )
        #         self.predictions[img_path].append(
        #             self.network_loader.predict_instances(subImg)
        #         )
        # self.sendAnnotationsToClient(index, imgBasename, alignment_data)

    def sendAnnotationsToClient(
        self, index, n_files, imgPath, alignment_data, predictions, metadata
    ):
        imgBasename = os.path.basename(imgPath)
        print('predictions at start:', predictions)
        if type(predictions) is list and len(predictions) == 1:
            predictions = predictions[0]
        self.alignment_data[imgPath] = metadata['rotationAngle']
        self.predictions[imgPath] = predictions
        print('predictions for image path', imgPath, self.predictions[imgPath])
        if self.is_exception(self.predictions[imgPath][0]):
            self.emit_to_room(
                "counting-annotations",
                {
                    "index": index,
                    "filename": imgBasename,
                    "rotationAngle": 0,
                    "data": json.dumps(
                        {"error": self.predictions[imgPath][0].__name__}
                    ),
                },
            )
            return
        resultsData = []
        bboxes = self.bboxes
        do_rotate = (
            type(alignment_data) is dict
            and alignment_data["type"] == "custom"
            and alignment_data["rotationAngle"] != 0
        )
        ct = self.chamberTypes[imgPath]
        if alignment_data["type"] and alignment_data["type"] != ct:
            ct = alignment_data["type"]
        for i, prediction in enumerate(self.predictions[imgPath]):
            if ct == CT.large.name:
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
            elif ct == CT.opto.name:
                x = bboxes[i][0] + 0.5 * bboxes[i][2]
                y = bboxes[i][1] + (1.4 if i % 10 < 5 else -0.1) * bboxes[i][3]
            else:  # position the label inside the upper left corner
                x = bboxes[i][0]
                y = bboxes[i][1]
                if do_rotate:
                    x, y = self.rotate_pt(x, y, -alignment_data["rotationAngle"])
                x += 50 + (0 if prediction["count"] < 10 else 40)
                y += self.textLabelHeight

            resultsData.append({**prediction, **{"x": x, "y": y, "bbox": bboxes[i]}})
        counting_data = {
            "data": json.dumps(resultsData, separators=(",", ":")),
            "filename": imgBasename,
            "rotationAngle": self.alignment_data[imgPath]["rotationAngle"]
            if hasattr(self, "rotation_angle")
            else None,
            "index": index,
        }
        self.emit_to_room(
            "counting-annotations",
            counting_data,
        )
        self.annotations[os.path.normpath(imgPath)] = resultsData
        print("index is", index, type(index), "and n_files is", n_files, type(n_files))
        self.emit_to_room(
            "counting-progress",
            {"data": "Finished processing image %i of %i" % (index, n_files)},
        )
        if index + 1 == n_files:
            self.emit_to_room("counting-progress", {"data": "Finished counting eggs"})
            self.emit_to_room("counting-done", {"is_retry": True})

    def rotate_pt(self, x, y, radians):
        img_center = list(reversed([el / 2 for el in self.img.shape[:2]]))
        return rotate_around_point_highperf((x, y), radians, img_center)

    def createErrorReport(self, edited_counts, user):
        font = drawing.loadFont(14)
        for imgPath in edited_counts:
            rel_path = os.path.normpath(os.path.join("./uploads", self.room, imgPath))
            img = rotate_image(
                cv2.imread(rel_path), self.alignment_data[imgPath]["rotationAngle"]
            )
            for i in edited_counts[imgPath]:
                error_report_image_basename = ".".join(
                    os.path.basename(imgPath).split(".")[:-1]
                ) + "_region_%s_actualCt_%s_user_%s" % (
                    i,
                    edited_counts[imgPath][i],
                    user,
                )
                annot = self.annotations[rel_path][int(i)]
                imgSection = img[
                    annot["bbox"][1] : annot["bbox"][1] + annot["bbox"][3],
                    annot["bbox"][0] : annot["bbox"][0] + annot["bbox"][2],
                ]
                cv2.imwrite(
                    os.path.join("error_cases", f"{error_report_image_basename}.png"),
                    imgSection,
                )
                for outline in self.predictions[rel_path][int(i)]["outlines"]:
                    reversed_outline = [list(reversed(el)) for el in outline]
                    drawing.draw_line(imgSection, [reversed_outline])
                imgSection = Image.fromarray(imgSection)
                draw = ImageDraw.Draw(imgSection)
                draw.text(
                    (15, 15),
                    f"Orig: {self.predictions[rel_path][int(i)]['count']}",
                    font=font,
                    fill=drawing.orangeRed,
                )
                draw.text(
                    (15, 35),
                    f"Edited: {edited_counts[imgPath][i]}",
                    font=font,
                    fill=drawing.orangeRed,
                )
                cv2.imwrite(
                    os.path.join(
                        "error_cases",
                        f"{error_report_image_basename}_outlines.png",
                    ),
                    np.array(imgSection),
                )
        self.emit_to_room("report-ready", {})

    @staticmethod
    def is_exception(my_var):
        return inspect.isclass(my_var) and issubclass(my_var, Exception)

    def saveCSV(self, edited_counts, row_col_layout, ordered_counts):
        resultsPath = "temp/egg_counts_ALPHA_%s.csv" % datetime.today().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
        with open(resultsPath, "wt", newline="") as resultsFile:
            writer = csv.writer(resultsFile)
            writer.writerow(["Egg Counter, ALPHA version"])
            writer.writerow(
                [
                    "Egg-detection model updated: "
                    + model_to_update_date.get(
                        os.path.basename(self.network_loader.model_path), "Unknown"
                    )
                ]
            )
            for i, imgPath in enumerate(self.predictions):
                writer.writerow([os.path.basename(imgPath)])
                first_pred = self.predictions[imgPath][0]
                if self.is_exception(first_pred):
                    writer.writerows([[errorMessages[first_pred]], []])
                    continue
                base_path = os.path.basename(imgPath)
                updated_counts = [el["count"] for el in self.predictions[imgPath]]
                if base_path in edited_counts and len(edited_counts[base_path]) > 0:
                    writer.writerow(
                        ["Note: this image's counts have been amended by hand"]
                    )
                    for region_index in edited_counts[base_path]:
                        updated_counts[int(region_index)] = int(
                            edited_counts[base_path][region_index]
                        )

                if row_col_layout[i]:
                    for j, row in enumerate(row_col_layout[i]):
                        num_entries_added = 0
                        row_entries = []
                        for col in range(row[-1] + 1):
                            if col in row:
                                row_entries.append(
                                    ordered_counts[i][j][num_entries_added]
                                )
                                num_entries_added += 1
                            else:
                                row_entries.append("")
                        writer.writerow(row_entries)
                else:
                    CT[self.chamberTypes[imgPath]].value().writeLineFormatted(
                        [updated_counts], 0, writer
                    )
                writer.writerow([])
        self.emit_to_room("counting-csv", {"data": os.path.basename(resultsPath)})
