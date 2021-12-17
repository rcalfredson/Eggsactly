from enum import Enum
from flask import (
    abort,
    Blueprint,
    render_template,
    request,
    send_from_directory,
    send_file,
)
from flask_dance.contrib.google import google
from flask_login import login_required, current_user
import os
from pathlib import Path
import shutil
import time
from werkzeug.utils import secure_filename
import zipfile

from project.lib.common import zipdir
from project.lib.datamanagement.models import login_google_user
from project.lib.image.exif import correct_via_exif
from project.lib.os.pauser import PythonPauser
from project.lib.web.exceptions import CUDAMemoryException, ImageAnalysisException
from project.lib.users import users
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
            else:
                shutil.rmtree(f)


@main.route("/", methods=["GET", "POST"])
def index():
    origin_flag = request.args.get("origin")
    if origin_flag in ("external", "local"):
        is_local = origin_flag == "local"
    else:
        is_local = current_user.is_local if hasattr(current_user, "is_local") else False
    name = current_user.name if hasattr(current_user, "name") else None
    user_info_endpoint = "/oauth2/v2/userinfo"
    result = login_google_user()
    return render_template(
        "counting.html",
        name=result["name"] if result["name"] is not None else name,
        google_data=result["data"],
        is_local=is_local,
    )


@main.route("/uploads/<sid>/<filename>")
def uploaded_file(sid, filename):
    return send_from_directory(
        os.path.join("../", app.config["UPLOAD_FOLDER"], sid), filename
    )


@main.route("/csvResults/<filename>")
def send_csv(filename):
    fileToDownload = os.path.join("downloads", filename)
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


@main.route("/upload", methods=["POST"])
def handle_upload():
    pauser.set_resume_timer()
    sid = request.form["sid"]
    for dir_name in ("uploads", "downloads"):
        remove_old_files(dir_name)
    check_chamber_type_of_imgs(sid)
    return "OK"


def check_chamber_type_of_imgs(sid):
    file_list = request.files
    n_files = len(request.files)
    for i, file in enumerate(file_list):
        check_chamber_type_of_img(i, file, sid, n_files)


def check_chamber_type_of_img(i, file, sid, n_files):
    if file and allowed_file(file):
        filename = secure_filename(file)
        socketIO.emit(
            "counting-progress",
            {"data": "Uploading image %i of %i (%s)" % (i + 1, n_files, filename)},
            room=sid,
        )
        folder_path = os.path.join(app.config["UPLOAD_FOLDER"], sid)
        if not os.path.exists(folder_path):
            Path(folder_path).mkdir(exist_ok=True, parents=True)
        filePath = os.path.join(folder_path, filename)
        request.files[file].save(filePath)
        correct_via_exif(filePath)
        socketIO.emit(
            "counting-progress",
            {"data": "Checking chamber type of image %i of %i" % (i + 1, n_files)},
            room=sid,
        )
        app.sessions[sid].check_chamber_type_and_find_bounding_boxes(
            filePath, i, n_files
        )


@main.route("/count-eggs", methods=["POST"])
def count_eggs_in_batch():
    sid = request.json["sid"]
    pauser.set_resume_timer()
    MAX_ATTEMPTS_PER_IMG = 1
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
                count_eggs_in_img(i, file, sid, n_files)
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
    pauser.set_resume_timer()
    return "OK"


def count_eggs_in_img(i, file, sid, n_files):
    file, index = file["file_name"], file["index"]
    if file and allowed_file(file):
        filename = secure_filename(file)
        folder_path = os.path.join(app.config["UPLOAD_FOLDER"], sid)
        if not os.path.exists(folder_path):
            Path(folder_path).mkdir(exist_ok=True, parents=True)
        filePath = os.path.join(folder_path, filename)
        kwargs = {
            "alignment_data": request.json["chamberData"][index],
            "index": index,
            "n_files": n_files,
        }
        app.sessions[sid].segment_img_and_count_eggs(filePath, **kwargs)
