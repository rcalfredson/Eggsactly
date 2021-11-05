from enum import Enum
from glob import glob
import json
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
)
from flask_socketio import SocketIO, emit
from PIL import Image
from werkzeug.utils import secure_filename

from lib.image.exif import correct_via_exif
from lib.os.pauser import PythonPauser
from lib.web.downloadManager import DownloadManager
from lib.web.exceptions import CUDAMemoryException, ImageAnalysisException
from lib.web.network_loader import NetworkLoader
from lib.web.scheduler import Scheduler
from lib.web.sessionManager import SessionManager
from users import users

sessions = {}

UPLOAD_FOLDER = "./uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "tif"}
NET_ARCH = ("fcrn", "splinedist")[1]
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = os.urandom(24)
socketIO = SocketIO(app)
downloadManager = DownloadManager()
network_loader = NetworkLoader(NET_ARCH)
pauser = PythonPauser()
scheduler = Scheduler(1)


def prune_old_sessions():
    current_time = time.time()
    for sid in list(sessions.keys()):
        if current_time - sessions[sid].lastPing > 60 * 10:
            del sessions[sid]


scheduler.schedule.every(5).minutes.do(prune_old_sessions)
scheduler.run_continuously()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


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
            ziph.write(
                os.path.join(root, file),
                os.path.relpath(os.path.join(root, file), os.path.join(path, "..")),
            )


@socketIO.on("ping")
def field_ping(data):
    if data["sid"] not in sessions:
        return
    sessions[data["sid"]].lastPing = time.time()
    if request.sid != data["sid"] and request.sid in sessions:
        sessions[data["sid"]].room = request.sid
        del sessions[request.sid]
    sessions[data["sid"]].emit_to_room("pong", {})


@socketIO.on("connect")
def connected():
    sessions[request.sid] = SessionManager(socketIO, request.sid, network_loader)
    socketIO.emit("sid-from-server", {"sid": request.sid}, room=request.sid)


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/csvResults/<filename>")
def send_csv(filename):
    fileToDownload = os.path.join("temp", filename)
    if not os.path.isfile(fileToDownload):
        abort(404)
    return send_file(fileToDownload, as_attachment=True)


@socketIO.on("save-custom-mask")
def save_custom_mask(data):
    with open(os.path.join("configs", "masks", f"{data['maskName']}.json"), "w") as f:
        json.dump(data, f)
    socketIO.emit(
        "mask-list-update",
        {"names": get_mask_list(data, emit=False), "currentMask": data["maskName"]},
    )


@socketIO.on("load-custom-mask")
def load_custom_mask(data):
    with open(os.path.join("configs", "masks", f"{data['maskName']}.json")) as f:
        socketIO.emit("loaded-custom-mask", json.load(f), room=data["sid"])


@socketIO.on("prepare-csv")
def prepare_csv(data):
    sessions[data["sid"]].saveCSV(
        data["editedCounts"], data["rowColLayout"], data["orderedCounts"]
    )


@socketIO.on("submit-error-report")
def submit_error_report(data):
    sessions[data["sid"]].createErrorReport(data["editedCounts"], data["user"])


@socketIO.on("prepare-annot-imgs-zip")
def setup_imgs_download(data):
    ts = str(data["time"])
    downloadManager.addNewSession(sessions[data["sid"]], ts, data["editedCounts"])
    downloadManager.createImagesForDownload(str(data["time"]))
    zipfName = "%s.zip" % (downloadManager.sessions[ts]["folder"])
    zipf = zipfile.ZipFile(zipfName, "w", zipfile.ZIP_DEFLATED)
    zipdir(downloadManager.sessions[ts]["folder"], zipf)
    zipf.close()
    downloadManager.sessions[ts]["zipfile"] = zipfName
    socketIO.emit("zip-annots-ready", {"time": ts}, room=data["sid"])


@socketIO.on("mask-list")
def get_mask_list(data, emit=True):
    names = [
        os.path.basename(mask.split(".json")[0])
        for mask in glob("configs/masks/*.json")
    ]
    if emit:
        socketIO.emit("mask-list", {"names": names})
    else:
        return names


@app.route("/annot-img/<ts>", methods=["GET"])
def return_zipfile(ts):
    shutil.rmtree(downloadManager.sessions[ts]["folder"])
    zipfile_name = downloadManager.sessions[ts]["zipfile"]
    del downloadManager.sessions[ts]
    return send_file(zipfile_name, as_attachment=True)


@app.route("/annot-img", methods=["POST"])
def handle_annot_img_upload():
    ts = request.form["time"]
    sid = request.form["sid"]
    for file in request.files.getlist("img"):
        filename = secure_filename(request.form["imgName"])
        filePath = os.path.join(downloadManager.sessions[ts]["folder"], filename)
        file.save(filePath)
        downloadManager.sessions[ts]["imgs_saved"] += 1
    if (
        downloadManager.sessions[ts]["imgs_saved"]
        == downloadManager.sessions[ts]["total_imgs"]
    ):
        zipfName = "%s.zip" % (downloadManager.sessions[ts]["folder"])
        zipf = zipfile.ZipFile(zipfName, "w", zipfile.ZIP_DEFLATED)
        zipdir(downloadManager.sessions[ts]["folder"], zipf)
        zipf.close()
        downloadManager.sessions[ts]["zipfile"] = zipfName
        socketIO.emit("zip-annots-ready", {"time": ts}, room=sid)
    return "OK"


@app.route("/manual-recount", methods=["POST"])
def manual_recount():
    pauser.set_resume_timer()
    process_imgs(request.json["sid"], data_type=AllowedDataTypes.json)
    pauser.set_resume_timer()
    socketIO.emit("counting-done", {"is_retry": True}, room=request.json["sid"])
    return "OK"


@app.route("/reset-session", methods=["POST"])
def reset_session():
    sessions[request.form["sid"]].clear_data()


@app.route("/upload", methods=["POST"])
def handle_upload():
    pauser.set_resume_timer()
    sid = request.form["sid"]
    for dir_name in ("uploads", "temp"):
        remove_old_files(dir_name)
    check_chamber_type_of_imgs(sid)
    return "OK"


class AllowedDataTypes(Enum):
    file = 1
    json = 2


def check_chamber_type_of_imgs(sid):
    MAX_ATTEMPTS_PER_IMG = 1
    file_list = request.files
    n_files = len(request.files)
    for i, file in enumerate(file_list):
        attempts = 0
        succeeded = False
        while True:
            try:
                check_chamber_type_of_img(i, file, sid, n_files, attempts)
                succeeded = True
            except CUDAMemoryException:
                attempts += 1
                socketIO.emit(
                    "counting-progress",
                    {
                        "data": "Ran out of system resources. Trying"
                        " to reallocate and try again..."
                    },
                    room=sid,
                )
                pauser.end_high_impact_py_prog()
            except ImageAnalysisException:
                sessions[sid].report_counting_error(
                    sessions[sid].imgPath, ImageAnalysisException
                )
                break
            if succeeded:
                break
            elif attempts > MAX_ATTEMPTS_PER_IMG:
                sessions[sid].report_counting_error(
                    sessions[sid].imgPath, CUDAMemoryException
                )


def check_chamber_type_of_img(i, file, sid, n_files, attempts):
    if file and allowed_file(file):
        socketIO.emit("clear-display", room=sid)
        socketIO.emit(
            "counting-progress",
            {"data": "Uploading image %i of %i" % (i + 1, n_files)},
            room=sid,
        )
        filename = secure_filename(file)
        filePath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        if attempts == 0:
            request.files[file].save(filePath)
        socketIO.emit(
            "counting-progress",
            {"data": "Checking chamber type of image %i of %i" % (i + 1, n_files)},
            room=sid,
        )
        sessions[sid].check_chamber_type_and_find_bounding_boxes(filePath)


def process_imgs(sid, data_type):
    MAX_ATTEMPTS_PER_IMG = 1

    if data_type == AllowedDataTypes.file:
        file_list = request.files
        n_files = len(request.files)
    elif data_type == AllowedDataTypes.json:
        file_list = [
            {
                "index": entry,
                "file_name": request.json["chamberData"][entry]["file_name"],
            }
            for entry in request.json["chamberData"]
        ]
        n_files = len(request.json["chamberData"])

    for i, file in enumerate(file_list):
        attempts = 0
        succeeded = False
        while True:
            try:
                process_img(
                    i,
                    file,
                    sid,
                    n_files,
                    attempts,
                    manual_recount=data_type == AllowedDataTypes.json,
                )
                succeeded = True
            except CUDAMemoryException:
                attempts += 1
                socketIO.emit(
                    "counting-progress",
                    {
                        "data": "Ran out of system resources. Trying"
                        " to reallocate and try again..."
                    },
                    room=sid,
                )
                pauser.end_high_impact_py_prog()
            except ImageAnalysisException:
                sessions[sid].report_counting_error(
                    sessions[sid].imgPath, ImageAnalysisException
                )
                break
            if succeeded:
                break
            elif attempts > MAX_ATTEMPTS_PER_IMG:
                sessions[sid].report_counting_error(
                    sessions[sid].imgPath, CUDAMemoryException
                )


def process_img(i, file, sid, n_files, attempts, manual_recount=False):
    if manual_recount:
        file, index = file["file_name"], file["index"]
    if file and allowed_file(file):
        socketIO.emit("clear-display", room=sid)
        socketIO.emit(
            "counting-progress",
            {"data": "Uploading image %i of %i" % (i + 1, n_files)},
            room=sid,
        )
        filename = secure_filename(file)
        filePath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        if not manual_recount and attempts == 0:
            request.files[file].save(filePath)
            correct_via_exif(filePath)
        socketIO.emit(
            "counting-progress",
            {"data": "Processing image %i of %i" % (i + 1, n_files)},
            room=sid,
        )
        if manual_recount:
            kwargs = {
                "alignment_data": request.json["chamberData"][index],
                "index": index,
            }
        else:
            kwargs = {}
        sessions[sid].segment_img_and_count_eggs(filePath, **kwargs)


@app.route("/", methods=["GET", "POST"])
def show_mainpage():
    return render_template(
        "registration.html", users=map(str.capitalize, sorted(users))
    )


@app.route("/isolatedRedrawTest", methods=["GET"])
def show_testpage():
    return render_template("canvasLoadTest.html")


if __name__ == "__main__":
    socketIO.run(app)
