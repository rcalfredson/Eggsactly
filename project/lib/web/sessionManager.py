import csv
import cv2
from datetime import datetime
import inspect
from io import BytesIO, StringIO
import json
import numpy as np
from PIL import Image, ImageDraw
import os
import time
import traceback
import warnings

from project import backend_type, db
from project.lib.datamanagement.models import EggLayingImage, ErrorReport
from project.lib.event import Listener
from project.lib.image import drawing
from project.lib.image.chamber import CT
from project.lib.image.circleFinder import (
    CircleFinder,
    rotate_around_point_highperf,
    rotate_image,
)
from project.lib.web.backend_types import BackendTypes
from project.lib.web.exceptions import (
    CUDAMemoryException,
    errorMessages,
    ImageAnalysisException,
)
from project.lib.web.gpu_manager import GPUManager
from project.lib.web.gpu_task_types import GPUTaskTypes

warnings.filterwarnings("error")

with open("project/models/modelRevDates.json", "r") as f:
    model_to_update_date = json.load(f)


class SessionManager:
    """Represent and process information while using the egg-counting web app."""

    def __init__(self, socketIO, room, gpu_manager: GPUManager):
        """Create a new SessionData instance.

        Arguments:
          - socketIO: SocketIO server
          - room: SocketIO room where messages get sent
          - gpu_manager: GPUManager instance where GPU tasks get added
        """
        self.chamberTypes = {}
        self.cfs = {}
        self.predictions = {}
        self.basenames = {}
        self.img_shapes = {}
        self.paths_to_indices = {}
        self.annotations = {}
        self.bboxes = {}
        self.alignment_data = {}
        self.socketIO = socketIO
        self.room = room
        self.gpu_manager = gpu_manager
        self.counting_task_group = None
        self.lastPing = time.time()
        self.textLabelHeight = 96

    def clear_data(self):
        self.chamberTypes = {}
        self.cfs = {}
        self.predictions = {}
        self.annotations = {}
        self.bboxes = {}
        self.alignment_data = {}
        self.basenames = {}
        self.img_shapes = {}
        self.paths_to_indices = {}

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
                "width": self.img_shapes[imgPath][1],
                "height": self.img_shapes[imgPath][0],
                "data": "%s counting eggs for image %s"
                % (prefix, self.basenames[imgPath]),
                "filename": self.basenames[imgPath],
                "error_type": err_type.__name__,
            },
        )
        self.predictions[imgPath] = [err_type]
        time.sleep(2)

    def enqueue_arena_detection_task(self, img_path):
        self.cfs[img_path] = CircleFinder(
            os.path.basename(img_path),
            self.img_shapes[img_path],
            self.room,
            allowSkew=True,
        )
        taskgroup = self.gpu_manager.add_task_group(
            self.room, n_tasks=1, task_type=GPUTaskTypes.arena
        )
        self.gpu_manager.add_task(taskgroup, img_path)
        taskgroup.add_completion_listener(
            Listener(self.segment_image_via_object_detection, (img_path,))
        )

    def enqueue_egg_counting_task(self, img_path, alignment_data):
        self.gpu_manager.add_task(self.counting_task_group, img_path, alignment_data)

    def segment_image_via_object_detection(self, img_path, predictions):
        imgBasename = os.path.basename(img_path)
        try:
            (circles, avgDists, numRowsCols, rotationAngle, _) = self.cfs[
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
            self.bboxes[img_path] = self.cfs[img_path].getSubImageBBoxes(
                circles, avgDists, numRowsCols
            )
            self.bboxes[img_path] = [
                [round(el) for el in bbox] for bbox in self.bboxes[img_path]
            ]
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
                    "bboxes": self.bboxes[img_path],
                    "width": self.img_shapes[img_path][1],
                    "height": self.img_shapes[img_path][0],
                },
            )
        except Exception as exc:
            print("exception while finding circles:")
            traceback.print_exc()
            if self.is_CUDA_mem_error(exc):
                raise CUDAMemoryException
            else:
                self.report_counting_error(img_path, ImageAnalysisException)
        except RuntimeWarning:
            self.report_counting_error(img_path, ImageAnalysisException)
        del self.cfs[img_path]

    @staticmethod
    def open_image(img_path, dtype=np.float32):
        if backend_type == BackendTypes.gcp:
            path_split = img_path.split(os.path.sep)
            img = BytesIO(
                EggLayingImage.query.filter_by(
                    session_id=path_split[-2], basename=path_split[-1]
                )
                .first()
                .image
            )
        elif backend_type == BackendTypes.local:
            img = img_path
        return np.array(Image.open(img), dtype=dtype)

    def check_chamber_type_and_find_bounding_boxes(self, img_path, i, n_files):
        img_path = os.path.normpath(img_path)
        imgBasename = os.path.basename(img_path)
        self.basenames[img_path] = imgBasename
        self.emit_to_room(
            "counting-progress",
            {"data": "Segmenting image %i of %i" % (i + 1, n_files)},
        )
        img = self.open_image(img_path)
        self.img_shapes[img_path] = img.shape
        self.enqueue_arena_detection_task(img_path)

    def segment_img_and_count_eggs(self, img_path, alignment_data, index, n_files):
        if int(index) == 0:
            self.counting_task_group = self.gpu_manager.add_task_group(
                self.room, n_tasks=n_files, task_type=GPUTaskTypes.egg
            )
            self.counting_task_group.add_completion_listener(
                Listener(
                    self.send_annotations_for_task_group,
                )
            )
        imgBasename = os.path.basename(img_path)
        img_path = os.path.normpath(img_path)
        self.basenames[img_path] = imgBasename
        self.paths_to_indices[img_path] = index
        self.alignment_data[img_path] = alignment_data
        if not img_path in self.chamberTypes or (
            "type" in alignment_data
            and img_path in self.chamberTypes
            and self.chamberTypes[img_path] != alignment_data["type"]
        ):
            self.chamberTypes[img_path] = alignment_data["type"]
        self.enqueue_egg_counting_task(img_path, alignment_data)

    def send_annotations_for_task(self, prediction_set, metadata, img_index):
        imgPath = self.img_paths[img_index]
        imgBasename = os.path.basename(imgPath)
        self.alignment_data[imgPath]["rotationAngle"] = metadata.get("rotationAngle", 0)
        if "bboxes" in metadata:
            self.bboxes[imgPath] = metadata["bboxes"]
        self.predictions[imgPath] = prediction_set
        alignment_data = self.alignment_data[imgPath]
        if self.is_exception(self.predictions[imgPath][0]):
            self.emit_to_room(
                "counting-annotations",
                {
                    "index": str(img_index),
                    "filename": imgBasename,
                    "rotationAngle": 0,
                    "data": json.dumps(
                        {"error": self.predictions[imgPath][0].__name__}
                    ),
                },
            )
        else:
            resultsData = []
            bboxes = self.bboxes[imgPath]
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
                        x, y = self.rotate_pt(
                            x, y, -alignment_data["rotationAngle"], imgPath
                        )
                    x += 50 + (0 if prediction["count"] < 10 else 40)
                    y += self.textLabelHeight

                resultsData.append(
                    {**prediction, **{"x": x, "y": y, "bbox": bboxes[i]}}
                )
            counting_data = {
                "data": json.dumps(resultsData, separators=(",", ":")),
                "filename": imgBasename,
                "rotationAngle": self.alignment_data[imgPath]["rotationAngle"]
                if "rotationAngle" in self.alignment_data[imgPath]
                else None,
                "index": str(img_index),
            }
            self.emit_to_room(
                "counting-annotations",
                counting_data,
            )
            self.annotations[os.path.normpath(imgPath)] = resultsData

    def send_annotations_for_task_group(self, predictions, metadata):
        self.img_paths = list(self.basenames.keys())
        for i, prediction_set in enumerate(predictions):
            self.send_annotations_for_task(prediction_set, metadata[i], i)
        self.emit_to_room("counting-done", {"is_retry": True})

    def rotate_pt(self, x, y, radians, img_path):
        img_center = list(reversed([el / 2 for el in self.img_shapes[img_path][:2]]))
        return rotate_around_point_highperf((x, y), radians, img_center)

    def createErrorReport(self, edited_counts, user):
        font = drawing.loadFont(14)
        for imgPath in edited_counts:
            rel_path = os.path.normpath(os.path.join("./uploads", self.room, imgPath))
            img = SessionManager.open_image(rel_path, dtype=np.uint8)
            img = rotate_image(img, self.alignment_data[rel_path]["rotationAngle"])
            for i in edited_counts[imgPath]:
                error_report_image_basename = ".".join(
                    os.path.basename(imgPath).split(".")[:-1]
                ) + "_region_%s_actualCt_%s_user_%s" % (
                    i,
                    edited_counts[imgPath][i],
                    user.id,
                )
                annot = self.annotations[rel_path][int(i)]
                img_section = img[
                    annot["bbox"][1] : annot["bbox"][1] + annot["bbox"][3],
                    annot["bbox"][0] : annot["bbox"][0] + annot["bbox"][2],
                ]
                if backend_type == BackendTypes.local:
                    cv2.imwrite(
                        os.path.join(
                            "error_cases", f"{error_report_image_basename}.png"
                        ),
                        img_section,
                    )
                outline_img_section = np.array(img_section)
                for outline in self.predictions[rel_path][int(i)]["outlines"]:
                    reversed_outline = [list(reversed(el)) for el in outline]
                    drawing.draw_line(outline_img_section, [reversed_outline])
                outline_img_section = Image.fromarray(outline_img_section)
                draw = ImageDraw.Draw(outline_img_section)
                original_ct = self.predictions[rel_path][int(i)]["count"]
                draw.text(
                    (15, 15),
                    f"Orig: {original_ct}",
                    font=font,
                    fill=drawing.orangeRed,
                )
                draw.text(
                    (15, 35),
                    f"Edited: {edited_counts[imgPath][i]}",
                    font=font,
                    fill=drawing.orangeRed,
                )
                outline_img_section = np.array(outline_img_section)
                if backend_type == BackendTypes.local:
                    cv2.imwrite(
                        os.path.join(
                            "error_cases",
                            f"{error_report_image_basename}_outlines.png",
                        ),
                        outline_img_section,
                    )
                if backend_type == BackendTypes.gcp:
                    img_section, outline_img_section = [
                        cv2.imencode(".png", im)[1]
                        for im in (img_section, outline_img_section)
                    ]
                    ErrorReport(
                        image=img_section,
                        outline_image=outline_img_section,
                        img_path=imgPath,
                        region_index=i,
                        original_ct=original_ct,
                        edited_ct=edited_counts[imgPath][i],
                        user=user,
                    )
                    db.session.commit()

        self.emit_to_room("report-ready", {})

    @staticmethod
    def is_exception(my_var):
        return inspect.isclass(my_var) and issubclass(my_var, Exception)

    def sendCSV(self, edited_counts, row_col_layout, ordered_counts):
        resultsBasename = "egg_counts_ALPHA_%s.csv" % datetime.today().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
        csvfile = StringIO()
        writer = csv.writer(csvfile)
        writer.writerow(["Egg Counter, ALPHA version"])
        writer.writerow(
            [
                "Egg-detection model updated: "
                + model_to_update_date["models"].get(
                    model_to_update_date["latest"], "Unknown"
                )
            ]
        )
        for i, imgPath in enumerate(self.predictions):
            img_basename = os.path.basename(imgPath)
            writer.writerow([img_basename])
            first_pred = self.predictions[imgPath][0]
            if self.is_exception(first_pred):
                writer.writerows([[errorMessages[first_pred]], []])
                continue
            base_path = os.path.basename(imgPath)
            updated_counts = [el["count"] for el in self.predictions[imgPath]]
            if base_path in edited_counts and len(edited_counts[base_path]) > 0:
                writer.writerow(["Note: this image's counts have been amended by hand"])
                for region_index in edited_counts[base_path]:
                    updated_counts[int(region_index)] = int(
                        edited_counts[base_path][region_index]
                    )

            if row_col_layout[img_basename]:
                for j, row in enumerate(row_col_layout[img_basename]):
                    num_entries_added = 0
                    row_entries = []
                    for col in range(row[-1] + 1):
                        if col in row:
                            row_entries.append(
                                ordered_counts[img_basename][j][num_entries_added]
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
        self.emit_to_room(
            "counting-csv", {"data": csvfile.getvalue(), "basename": resultsBasename}
        )
