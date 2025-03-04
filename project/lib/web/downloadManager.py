import cv2
import inspect
import numpy as np
import os
from pathlib import Path
from PIL import Image, ImageDraw

from project import backend_type, db
from project.lib.datamanagement.models import EggLayingImage
from project.lib.image import drawing
from project.lib.image.circleFinder import rotate_around_point_highperf
from project.lib.web.backend_types import BackendTypes
from project.lib.web.exceptions import errorMessages


class DownloadManager:
    """Render and save downloadable copies of annotated egg-counting images."""

    def __init__(self):
        self.sessions = {}

    def addNewSession(self, session_manager, ts, id, edited_counts=None):
        self.sessions[id] = {
            "session_manager": session_manager,
            "folder": "annotated_images_ALPHA_%s" % ts,
            "edited_counts": edited_counts,
            "ts": ts,
        }
        if backend_type == BackendTypes.filesystem:
            Path(self.sessions[ts]["folder"]).mkdir(parents=True, exist_ok=True)

    def calculateEggPositionsForImage(self, sm, path):
        img = sm.open_image(path, dtype=np.uint8)
        zoom = sm.alignment_data[path].get("scaling", 1)
        ht, wd = img.shape[:2]
        center = (wd / 2, ht / 2)
        rot = sm.alignment_data[path]["rotationAngle"]
        sm.egg_positions_full_scale[path] = []
        for i, annotation in enumerate(sm.annotations[path]):
            for pt in sm.predictions[path][i]["points"]:
                pt = [
                    (1 / zoom) * (pt[1] + annotation["bbox"][0]),
                    (1 / zoom) * (pt[0] + annotation["bbox"][1]),
                ]
                pt = rotate_around_point_highperf(pt, -rot, center)
                sm.egg_positions_full_scale[path].append(pt)
    
    def prepareAnnotatedImage(self, sm, id, path, path_base):
        self.font = drawing.loadFont(80)
        img = sm.open_image(path, dtype=np.uint8)
        check_counts = path_base in self.sessions[id]["edited_counts"]
        if check_counts:
            been_edited = [
                i in self.sessions[id]["edited_counts"][path_base]
                for i in map(str, range(len(sm.annotations[path])))
            ]
        else:
            been_edited = [False] * len(sm.annotations[path])
        zoom = sm.alignment_data[path].get("scaling", 1)
        ht, wd = img.shape[:2]
        center = (wd / 2, ht / 2)
        rot = sm.alignment_data[path]["rotationAngle"]
        for i, annotation in enumerate(sm.annotations[path]):
            color = (3, 44, 252) if been_edited[i] else (25, 25, 25)
            box_coords = [  # UL, UR, BL, BR
                [annotation["bbox"][0], annotation["bbox"][1]],
                [
                    annotation["bbox"][0] + annotation["bbox"][2],
                    annotation["bbox"][1],
                ],
                [
                    annotation["bbox"][0],
                    annotation["bbox"][1] + annotation["bbox"][3],
                ],
                [
                    annotation["bbox"][0] + annotation["bbox"][2],
                    annotation["bbox"][1] + annotation["bbox"][3],
                ],
            ]

            for j, pt in enumerate(box_coords):
                for k, el in enumerate(pt):
                    box_coords[j][k] = (1 / zoom) * el
                box_coords[j] = rotate_around_point_highperf(
                    box_coords[j], -rot, center
                )
                box_coords[j] = [round(el) for el in box_coords[j]]
            xs = [el[0] for el in box_coords]
            ys = [el[1] for el in box_coords]

            # draw rotated rectangle
            rect = cv2.minAreaRect(np.float32(box_coords))
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (255, 191, 0), 2)

            for outline in sm.predictions[path][i]["outlines"]:
                outline = [list(reversed(el)) for el in outline]
                outline = [
                    [
                        (1 / zoom) * (el[0] + annotation["bbox"][0]),
                        (1 / zoom) * (el[1] + annotation["bbox"][1]),
                    ]
                    for el in outline
                ]
                outline = [
                    rotate_around_point_highperf(el, -rot, center) for el in outline
                ]
                drawing.draw_line(img, [outline], color=(255, 255, 255))
        img = Image.fromarray(img)
        for i, annotation in enumerate(sm.annotations[path]):
            draw = ImageDraw.Draw(img)

            def draw_single_text(msg, edit_status):
                y_offset = {"orig_w_edit": -40, "edited": 40, "orig": 0}[edit_status]
                color = (3, 44, 252, 0) if edit_status == "edited" else (0, 0, 0, 0)
                w, h = draw.textsize(msg)
                text_position = [
                    (1 / zoom) * (annotation["x"] - 3 * w),
                    (1 / zoom) * (annotation["y"] - 6 * h - y_offset),
                ]
                text_position = tuple(map(round, text_position))

                draw.text(
                    text_position,
                    msg,
                    font=self.font,
                    fill=color,
                )

            if been_edited[i]:
                draw_single_text(
                    self.sessions[id]["edited_counts"][path_base][str(i)],
                    edit_status="edited",
                )
            draw_single_text(
                str(annotation["count"]),
                edit_status="orig_w_edit" if been_edited[i] else "orig",
            )
        return np.array(img)

    def prepareImageWithError(self, path, sm, error_type):
        self.font = drawing.loadFont(140)
        img = sm.open_image(path, dtype=np.uint8)
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        w, h = img.size
        msg = errorMessages[error_type]
        msgW, msgH = draw.textsize(msg, font=self.font)
        draw.text(
            (int(0.5 * (w - msgW)), int(0.5 * (h - msgH))),
            msg,
            font=self.font,
            fill=(3, 44, 252, 0),
        )
        return np.array(img)

    def calculateFullScaleEggPositions(self, id):
        sm = self.sessions[id]["session_manager"]
        for path in sm.predictions:
            if inspect.isclass(sm.predictions[path][0]) and issubclass(
                sm.predictions[path][0], Exception
            ):
                continue
            self.calculateEggPositionsForImage(sm, path)

    def createImagesForDownload(self, id):
        sm = self.sessions[id]["session_manager"]
        for path in sm.predictions:
            path_base = os.path.basename(path)
            if inspect.isclass(sm.predictions[path][0]) and issubclass(
                sm.predictions[path][0], Exception
            ):
                img = self.prepareImageWithError(path, sm, sm.predictions[path][0])
            else:
                img = self.prepareAnnotatedImage(sm, id, path, path_base)

            if backend_type == BackendTypes.sql:

                img_entity = EggLayingImage.query.filter_by(
                    session_id=sm.room, basename=path_base
                ).first()
                img_entity.annotated_img = cv2.imencode(
                    f".{os.path.splitext(path_base)[1]}",
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                )[1].tobytes()
                db.session.commit()
            elif backend_type == BackendTypes.filesystem:
                cv2.imwrite(os.path.join(self.sessions[id]["folder"], path_base), img)
