import csv
from datetime import datetime
import json
from pathlib import Path
import os
import time

from flask import (
    Flask,
    abort,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    url_for
)
from flask_socketio import SocketIO, emit
import numpy as np
from PIL import Image
import torch
from werkzeug.utils import secure_filename

from chamber import CT
from circleFinder import CircleFinder
from detectors.fcrn import model

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'tif'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.urandom(24)
socketIO = SocketIO(app)

MODEL_PATH = 'models/egg_FCRN_A_150epochs_Yang-Lab-Dell2_2021-01-06' + \
             ' 17-04-53.765866.pth'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
network = model.FCRN_A(input_filters=3, N=2).to(device)
network.train(False)
network = torch.nn.DataParallel(network)
network.load_state_dict(torch.load(MODEL_PATH))

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def remove_old_files(folder):
    now = time.time()
    for f in os.listdir(folder):
        f = os.path.join(folder, f)
        if now - os.stat(f).st_mtime > 60 * 60:
            if os.path.isfile(f):
                os.remove(f)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/csvResults/<filename>')
def send_csv(filename):
    fileToDownload = os.path.join('temp', filename)
    if not os.path.isfile(fileToDownload):
        abort(404)
    return send_file(fileToDownload, as_attachment=True)

@app.route('/upload', methods=['POST'])
def handle_upload():
    for f in os.listdir('uploads'):
        os.unlink(os.path.join('uploads', f))
    if 'img-upload' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['img-upload']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filePath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filePath)
        socketIO.emit('clear-all')
        socketIO.emit('counting-progress', {'data': 'File uploaded!'})
        remove_old_files('temp')
        register_image(filePath)
        return 'OK'

@app.route('/', methods=['GET', 'POST'])
def show_mainpage():
    return render_template('registration.html')

def register_image(imgPath):
    predictions = []
    imgBasename = os.path.basename(imgPath)
    socketIO.emit('counting-progress',
        {'data': 'Segmenting image %s'%imgBasename})
    img = np.array(Image.open(imgPath), dtype=np.float32)
    cf = CircleFinder(img, os.path.basename(imgPath), allowSkew=True)
    if cf.skewed:
        socketIO.emit('counting-progress',
            {'data': 'Skew detected in image %s; stopping analysis.'%imgBasename})
    circles, avgDists, numRowsCols, rotatedImg, _ = cf.findCircles()
    subImgs, bboxes = cf.getSubImages(rotatedImg, circles, avgDists, numRowsCols)
    socketIO.emit('counting-progress',
        {'data': 'Counting eggs in image %s'%imgBasename})
    for subImg in subImgs:
        subImg = torch.from_numpy((1/255)*np.expand_dims(np.moveaxis(subImg, 2, 0
            ), 0))
        result = network(subImg)
        predictions.append(int(torch.sum(result).item() / 100))
    socketIO.emit('counting-progress',
        {'data': 'Finished counting eggs'})
    resultsData = []
    bboxes = [[int(el) for el in bbox] for bbox in bboxes]
    for i, prediction in enumerate(predictions):
        if cf.ct == CT.fourCircle.name:
            iMod = i%4
            if iMod in (0, 3):
                x, y = bboxes[i][0] + 0.1*bboxes[i][2], bboxes[i][1] + 0.15 * bboxes[i][3]
            elif iMod == 1:
                x, y = bboxes[i][0] + 0.4*bboxes[i][2], bboxes[i][1] + 0.2*bboxes[i][3]
            elif iMod == 2:
                x, y = bboxes[i][0] + 0.2*bboxes[i][2], bboxes[i][1] + 0.45*bboxes[i][3]
        else:
            x = bboxes[i][0] + (1.40 if i%2 == 0 else -0.32)*bboxes[i][2]
            y = bboxes[i][1] + 0.55*bboxes[i][3]
        resultsData.append({'count': prediction, 'x': x, 'y': y,
            'bbox': bboxes[i]})
    socketIO.emit('counting-annotations',
        {'data': json.dumps(resultsData, separators=(',', ':'))})
    resultsPath = 'temp/results_%s.csv'%datetime.today().strftime('%Y-%m-%d_%H-%M')
    with open(resultsPath,
        'wt', newline='') as resultsFile:
        writer = csv.writer(resultsFile)
        writer.writerow([imgPath])
        CT[cf.ct].value().writeLineFormatted([predictions], 0, writer)
    socketIO.emit('counting-csv',
        {'data': os.path.basename(resultsPath)})

if __name__ == '__main__':
    print('about to start app.')
    socketIO.run(app)
