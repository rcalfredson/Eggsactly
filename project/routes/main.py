from enum import Enum
from flask import (
    abort,
    Blueprint,
    render_template,
    request,
    send_from_directory,
    send_file,
)

from flask_login import login_required, current_user
import os
import shutil
import time
from werkzeug.utils import secure_filename

from project.lib.os.pauser import PythonPauser
from project.lib.web.exceptions import CUDAMemoryException, ImageAnalysisException
from project.users import users
from .. import app, socketIO

main = Blueprint("main", __name__)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "tif"}
pauser = PythonPauser()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def remove_old_files(folder):
    now = time.time()
    for f in os.listdir(folder):
        f = os.path.join(folder, f)
        if now - os.stat(f).st_mtime > 60 * 60:
            if os.path.isfile(f):
                os.remove(f)


@main.route("/", methods=["GET", "POST"])
def index():
    return render_template(
        "registration.html", users=map(str.capitalize, sorted(users))
    )


@main.route("/profile")
@login_required
def profile():
    return render_template('profile.html', name=current_user.name)


@main.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(
        os.path.join("../", app.config["UPLOAD_FOLDER"]), filename
    )


@main.route("/csvResults/<filename>")
def send_csv(filename):
    fileToDownload = os.path.join("temp", filename)
    if not os.path.isfile(fileToDownload):
        abort(404)
    return send_file(os.path.join("../", fileToDownload), as_attachment=True)


@main.route("/annot-img/<ts>", methods=["GET"])
def return_zipfile(ts):
    shutil.rmtree(app.downloadManager.sessions[ts]["folder"])
    zipfile_name = app.downloadManager.sessions[ts]["zipfile"]
    del app.downloadManager.sessions[ts]
    return send_file(os.path.join("../", zipfile_name), as_attachment=True)


@main.route("/annot-img", methods=["POST"])
def handle_annot_img_upload():
    ts = request.form["time"]
    sid = request.form["sid"]
    for file in request.files.getlist("img"):
        filename = secure_filename(request.form["imgName"])
        filePath = os.path.join(app.downloadManager.sessions[ts]["folder"], filename)
        file.save(filePath)
        app.downloadManager.sessions[ts]["imgs_saved"] += 1
    if (
        app.downloadManager.sessions[ts]["imgs_saved"]
        == app.downloadManager.sessions[ts]["total_imgs"]
    ):
        zipfName = "%s.zip" % (app.downloadManager.sessions[ts]["folder"])
        zipf = zipfile.ZipFile(zipfName, "w", zipfile.ZIP_DEFLATED)
        zipdir(app.downloadManager.sessions[ts]["folder"], zipf)
        zipf.close()
        app.downloadManager.sessions[ts]["zipfile"] = zipfName
        socketIO.emit("zip-annots-ready", {"time": ts}, room=sid)
    return "OK"


@main.route("/manual-recount", methods=["POST"])
def manual_recount():
    pauser.set_resume_timer()
    process_imgs(request.json["sid"], data_type=AllowedDataTypes.json)
    pauser.set_resume_timer()
    socketIO.emit("counting-done", {"is_retry": True}, room=request.json["sid"])
    return "OK"


@main.route("/upload", methods=["POST"])
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
                app.sessions[sid].report_counting_error(
                    app.sessions[sid].imgPath, ImageAnalysisException
                )
                break
            if succeeded:
                break
            elif attempts > MAX_ATTEMPTS_PER_IMG:
                app.sessions[sid].report_counting_error(
                    app.sessions[sid].imgPath, CUDAMemoryException
                )


def check_chamber_type_of_img(i, file, sid, n_files, attempts):
    if file and allowed_file(file):
        # socketIO.emit("clear-display", room=sid)
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
        app.sessions[sid].check_chamber_type_and_find_bounding_boxes(filePath)


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
                app.sessions[sid].report_counting_error(
                    app.sessions[sid].imgPath, ImageAnalysisException
                )
                break
            if succeeded:
                break
            elif attempts > MAX_ATTEMPTS_PER_IMG:
                app.sessions[sid].report_counting_error(
                    app.sessions[sid].imgPath, CUDAMemoryException
                )


def process_img(i, file, sid, n_files, attempts, manual_recount=False):
    if manual_recount:
        file, index = file["file_name"], file["index"]
    if file and allowed_file(file):
        # socketIO.emit("clear-display", room=sid)
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
        app.sessions[sid].segment_img_and_count_eggs(filePath, **kwargs)
