import csv
from datetime import datetime
import json
import os
import time

import cv2
import numpy as np
from PIL import Image
import torch

from chamber import CT
from circleFinder import CircleFinder
from detectors.fcrn import model

MODEL_PATH = 'models/egg_FCRN_A_150epochs_Yang-Lab-Dell2_2021-01-06' + \
             ' 17-04-53.765866.pth'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
network = model.FCRN_A(input_filters=3, N=2).to(device)
network.train(False)
network = torch.nn.DataParallel(network)
network.load_state_dict(torch.load(MODEL_PATH))


class SessionManager():
    """Represent and process information while using the egg-counting web app.
    """

    def __init__(self, socketIO, room):
        """Create a new SessionData instance.

        Arguments:
          - socketIO: SocketIO server
        """
        self.chamberTypes = {}
        self.predictions = {}
        self.annotations = {}
        self.socketIO = socketIO
        self.room = room
        self.lastPing = time.time()

    def clear_data(self):
        self.chamberTypes = {}
        self.predictions = {}
        self.annotations = {}

    def emit_to_room(self, evt_name, data):
        self.socketIO.emit(evt_name,
            data, room=self.room)

    def register_image(self, imgPath):
        imgBasename = os.path.basename(imgPath)
        imgPath = os.path.normpath(imgPath)
        self.imgPath = imgPath
        self.predictions[imgPath] = []
        self.emit_to_room('counting-progress',
                           {'data': 'Segmenting image %s' % imgBasename})
        img = np.array(Image.open(imgPath), dtype=np.float32)
        self.cf = CircleFinder(img, os.path.basename(imgPath), allowSkew=True)
        if self.cf.skewed:
            self.emit_to_room('counting-progress',
                               {'data': 'Skew detected in image %s;' +
                                ' stopping analysis.' % imgBasename})
        circles, avgDists, numRowsCols, rotatedImg, _ = self.cf.findCircles()
        self.chamberTypes[imgPath] = self.cf.ct
        subImgs, bboxes = self.cf.getSubImages(
            rotatedImg, circles, avgDists, numRowsCols)
        self.emit_to_room('counting-progress',
                           {'data': 'Counting eggs in image %s' % imgBasename})
        for subImg in subImgs:
            subImg = torch.from_numpy((1/255)*np.expand_dims(
                np.moveaxis(subImg, 2, 0), 0))
            result = network(subImg)
            self.predictions[imgPath].append(
                int(torch.sum(result).item() / 100))
        self.emit_to_room('counting-progress',
                           {'data': 'Finished counting eggs'})
        self.bboxes = [[int(el) for el in bbox] for bbox in bboxes]
        self.imgBasename = imgBasename
        self.sendAnnotationsToClient()

    def sendAnnotationsToClient(self):
        resultsData = []
        bboxes = self.bboxes
        for i, prediction in enumerate(self.predictions[self.imgPath]):
            if self.cf.ct == CT.fourCircle.name:
                iMod = i % 4
                if iMod in (0, 3):
                    x = bboxes[i][0] + 0.1*bboxes[i][2]
                    y = bboxes[i][1] + 0.15 * bboxes[i][3]
                elif iMod == 1:
                    x = bboxes[i][0] + 0.4*bboxes[i][2]
                    y = bboxes[i][1] + 0.2*bboxes[i][3]
                elif iMod == 2:
                    x, y = bboxes[i][0] + 0.2 * \
                        bboxes[i][2], bboxes[i][1] + 0.45*bboxes[i][3]
            elif self.cf.ct == CT.new.name:
                x = bboxes[i][0] + 0.5*bboxes[i][2]
                y = bboxes[i][1] + (1.4 if i % 10 < 5 else -0.1)*bboxes[i][3]
            else:
                x = bboxes[i][0] + (1.40 if i % 2 == 0 else -0.32)*bboxes[i][2]
                y = bboxes[i][1] + 0.55*bboxes[i][3]
            resultsData.append({'count': prediction, 'x': x, 'y': y,
                                'bbox': bboxes[i]})
        self.emit_to_room('counting-annotations',
                           {'data': json.dumps(resultsData,
                                               separators=(',', ':')),
                            'filename': self.imgBasename})
        self.annotations[os.path.normpath(self.imgPath)] = resultsData

    def createErrorReport(self, edited_counts, user):
        for imgPath in edited_counts:
            rel_path = os.path.normpath(
                os.path.join('./uploads', imgPath))
            img = cv2.imread(rel_path)
            for i in edited_counts[imgPath]:
                annot = self.annotations[rel_path][int(i)]
                imgSection = img[annot['bbox'][1]:annot['bbox'][1] + annot['bbox'][3],
                                 annot['bbox'][0]:annot['bbox'][0] + annot['bbox'][2]]
                cv2.imwrite(os.path.join('error_cases', '.'.join(
                    os.path.basename(imgPath).split('.')[:-1]) +
                    '_%s_actualCt_%s_user_%s.png' % (i, edited_counts[
                        imgPath][i], user)), imgSection)
        self.emit_to_room('report-ready', {})

    def saveCSV(self, edited_counts):
        resultsPath = 'temp/results_ALPHA_%s.csv' % datetime.today().strftime(
            '%Y-%m-%d_%H-%M-%S')
        with open(resultsPath, 'wt', newline='') as resultsFile:
            writer = csv.writer(resultsFile)
            writer.writerow(['Egg Counter, ALPHA version'])
            for i, imgPath in enumerate(self.predictions):
                writer.writerow([imgPath])
                base_path = os.path.basename(imgPath)
                if base_path in edited_counts and len(edited_counts[base_path]) > 0:
                    writer.writerow(
                        ["Note: this image's counts have been amended by hand"])
                CT[self.chamberTypes[imgPath]].value().writeLineFormatted(
                    [self.predictions[imgPath]], 0, writer)
                writer.writerow([])
        self.emit_to_room('counting-csv',
                           {'data': os.path.basename(resultsPath)})
