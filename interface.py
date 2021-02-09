import csv
from datetime import datetime
import json
from pathlib import Path
import os
import shutil
import time
import zipfile

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
from PIL import Image
from werkzeug.utils import secure_filename

from lib.web.downloadManager import DownloadManager
from lib.web.sessionManager import SessionManager
from users import users

sessions = {}

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'tif'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.urandom(24)
socketIO = SocketIO(app)
downloadManager = DownloadManager()

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

def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.path.join(path, '..')))

@socketIO.on('connect')
def connected():
    sessions[request.sid] = SessionManager(socketIO,
        request.sid)
    socketIO.emit('sid-from-server', {'sid': request.sid}, room=request.sid)

@socketIO.on('disconnect')
def disconnected():
    del sessions[request.sid]

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

@socketIO.on('prepare-csv')
def prepare_csv(data):
    sessions[data['sid']].saveCSV(data['editedCounts'])

@socketIO.on('submit-error-report')
def submit_error_report(data):
    sessions[data['sid']].createErrorReport(data['editedCounts'], data['user'])

@socketIO.on('prepare-annot-imgs-zip')
def setup_imgs_download(data):
    ts = str(data['time'])
    downloadManager.addNewSession(sessions[data['sid']],
        ts, data['editedCounts'])
    downloadManager.createImagesForDownload(str(data['time']))
    zipfName = '%s.zip'%(downloadManager.sessions[ts]['folder'])
    zipf = zipfile.ZipFile(zipfName, 'w', zipfile.ZIP_DEFLATED)
    zipdir(downloadManager.sessions[ts]['folder'], zipf)
    zipf.close()
    downloadManager.sessions[ts]['zipfile'] = zipfName
    socketIO.emit('zip-annots-ready', {'time': ts},
        room=data['sid'])

@app.route('/annot-img/<ts>', methods=['GET'])
def return_zipfile(ts):
    shutil.rmtree(downloadManager.sessions[ts]['folder'])
    zipfile_name = downloadManager.sessions[ts]['zipfile']
    del downloadManager.sessions[ts]
    return send_file(zipfile_name, as_attachment=True)

@app.route('/annot-img', methods=['POST'])
def handle_annot_img_upload():
    ts = request.form['time']
    sid = request.form['sid']
    for file in request.files.getlist('img'):
        filename = secure_filename(request.form['imgName'])
        filePath = os.path.join(
            downloadManager.sessions[ts]['folder'], filename)
        file.save(filePath)
        downloadManager.sessions[ts]['imgs_saved'] += 1
    if downloadManager.sessions[ts]['imgs_saved'] == \
        downloadManager.sessions[ts]['total_imgs']:
        zipfName = '%s.zip'%(downloadManager.sessions[ts]['folder'])
        zipf = zipfile.ZipFile(zipfName, 'w', zipfile.ZIP_DEFLATED)
        zipdir(downloadManager.sessions[ts]['folder'], zipf)
        zipf.close()
        downloadManager.sessions[ts]['zipfile'] = zipfName
        socketIO.emit('zip-annots-ready', {'time': ts},
            room=sid)
    return 'OK'

@app.route('/upload', methods=['POST'])
def handle_upload():
    sid = request.form['sid']
    uniq_names = json.loads(request.form['uniqueNames'])
    sessionManager = sessions[sid]
    sessionManager.clear_data()
    for dirName in ('uploads', 'temp'):
        remove_old_files(dirName)
    socketIO.emit('clear-all',
        room=sid)
    if 'img-upload-1' not in request.files:
        flash('No file part')
        return redirect(request.url)
    files = []
    numFiles = None
    counter = 1
    while numFiles != 0:
        newFiles = request.files.getlist('img-upload-%i'%counter)
        numFiles = len(newFiles)
        for file in newFiles:
            if file.content_type != 'application/octet-stream':
                files.append(file)
        counter += 1
    if len(files) == 0:
        flash('No selected file')
        return redirect(request.url)
    for i, file in enumerate(files):
        if file and allowed_file(uniq_names[i]):
            socketIO.emit('clear-display', room=sid)
            socketIO.emit('counting-progress',
                {'data': 'Uploading image %i of %i'%(i+1, len(files))},
                room=sid)
            filename = secure_filename(uniq_names[i])
            filePath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filePath)
            img = Image.open(filePath)
            img = img.rotate(360)
            img.save(filePath)
            socketIO.emit('counting-progress',
                {'data': 'Processing image %i of %i'%(i+1, len(files))},
                room=sid)
            sessionManager.register_image(filePath)
    socketIO.emit('counting-done', room=sid)
    return 'OK'

@app.route('/', methods=['GET', 'POST'])
def show_mainpage():
    return render_template('registration.html', users=map(str.capitalize, sorted(users)))

@app.route('/isolatedRedrawTest', methods=["GET"])
def show_testpage():
    return render_template('canvasLoadTest.html')

if __name__ == '__main__':
    socketIO.run(app)
