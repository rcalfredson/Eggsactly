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
from werkzeug.utils import secure_filename

from lib.web.sessionManager import SessionManager

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'tif'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.urandom(24)
socketIO = SocketIO(app)

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
    sessionManager = SessionManager(socketIO)
    for dirName in ('uploads', 'temp'):
        remove_old_files(dirName)
    socketIO.emit('clear-all')
    if 'img-upload' not in request.files:
        flash('No file part')
        return redirect(request.url)
    files = request.files.getlist('img-upload')
    if len(files) == 0:
        flash('No selected file')
        return redirect(request.url)
    for i, file in enumerate(files):
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filePath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filePath)
            socketIO.emit('clear-display')
            socketIO.emit('counting-progress',
                {'data': 'Processing image %i of %i'%(i+1, len(files))})
            sessionManager.register_image(filePath)
    sessionManager.saveCSV()
    return 'OK'

@app.route('/', methods=['GET', 'POST'])
def show_mainpage():
    return render_template('registration.html')

if __name__ == '__main__':
    socketIO.run(app)
