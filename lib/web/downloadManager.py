import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw

from lib.image import drawing
from util import putText, textStyle


class DownloadManager():
    """Render and save downloadable copies of annotated egg-counting images.
    """

    def __init__(self):
        self.sessions = {}

    def addNewSession(self, session_manager, ts, edited_counts):
        ts = str(ts)
        self.sessions[ts] = {'session_manager': session_manager,
                             'folder': 'temp/results_ALPHA_%s' % ts,
                             'edited_counts': edited_counts}
        Path(self.sessions[ts]['folder']).mkdir(parents=True,
                                                exist_ok=True)

    def createImagesForDownload(self, ts):
        sm = self.sessions[ts]['session_manager']
        for path in sm.predictions:
            path_base = os.path.basename(path)
            img = cv2.imread(path)
            check_counts = path_base in self.sessions[ts]['edited_counts']
            if check_counts:
                been_edited = [i in self.sessions[ts]['edited_counts'][path_base] for i in
                               map(str, range(len(sm.annotations[path])))]
            else:
                been_edited = [False]*len(sm.annotations[path])
            for i, annotation in enumerate(sm.annotations[path]):
                color = (3, 44, 252) if been_edited[i] else (25, 25, 25)
                img = drawing.rounded_rectangle(img,
                                                (annotation['bbox'][0],
                                                 annotation['bbox'][1]),
                                                (annotation['bbox'][1] + annotation['bbox'][3],
                                                    annotation['bbox'][0] + annotation['bbox'][2]),
                                                color=color)
            img = Image.fromarray(img)
            for i, annotation in enumerate(sm.annotations[path]):
                font = ImageFont.truetype(os.path.join(
                    Path(__file__).parent.absolute(), '../../static/fonts/arial.ttf'), 80)
                draw = ImageDraw.Draw(img)

                def draw_single_text(msg, edit_status):
                    y_offset = {'orig_w_edit': -40, 'edited': 40, 'orig': 0}[edit_status]
                    color = (3, 44, 252, 0) if edit_status == 'edited' else (0, 0, 0, 0)
                    w, h = draw.textsize(msg)
                    draw.text((int(annotation['x'] - 3*w), int(annotation['y'] - 6*h - y_offset)),
                              msg, font=font, fill=color)
                if been_edited[i]:
                    draw_single_text(
                        self.sessions[ts]['edited_counts'][path_base][str(i)],
                        edit_status='edited')
                draw_single_text(str(annotation['count']),
                    edit_status='orig_w_edit' if been_edited[i] else 'orig')
            img = np.array(img)
            cv2.imwrite(os.path.join(
                self.sessions[ts]['folder'], path_base), img)
