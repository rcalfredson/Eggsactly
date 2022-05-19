from io import BytesIO
from flask import (
    abort,
    Blueprint,
    render_template,
    request,
    Response,
    send_file,
    send_from_directory,
)
from flask_login import current_user
import os
from pathlib import Path
import shutil
import sys
import time
from werkzeug.utils import secure_filename
import zipstream

from project import app, backend_type, db, socketIO
from project.lib.datamanagement.models import (
    delete_expired_rows,
    EggLayingImage,
    login_google_user,
    SocketIOUser,
)
from project.lib.image.exif import correct_via_exif
from project.lib.os.pauser import PythonPauser
from project.lib.web.backend_types import BackendTypes
from project.lib.web.exceptions import CUDAMemoryException, ImageAnalysisException


main = Blueprint("main", __name__)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "tif"}
pauser = PythonPauser()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def remove_old_files(folder):
    now = time.time()
    for fname in os.listdir(folder):
        if fname == ".gitkeep":
            continue
        f = os.path.join(folder, fname)
        if now - os.stat(f).st_mtime > 60 * 60:
            if os.path.isfile(f):
                os.remove(f)
            else:
                shutil.rmtree(f)


def remove_old_sql_rows():
    for cls in (SocketIOUser, EggLayingImage):
        delete_expired_rows(cls)
    # with db.engine.begin() as conn:
    #     conn.execute("VACUUM FULL")


@main.route("/", methods=["GET", "POST"])
def index():
    origin_flag = request.args.get("origin")
    if origin_flag in ("external", "local"):
        is_local = origin_flag == "local"
    else:
        is_local = current_user.is_local if hasattr(current_user, "is_local") else False
    name = current_user.name if hasattr(current_user, "name") else None
    result = login_google_user()
    return render_template(
        "counting.html",
        name=result["name"] if result["name"] is not None else name,
        google_data=result["data"],
        is_local=is_local,
    )


@main.route("/uploads/<sid>/<filename>")
def uploaded_file(sid, filename):
    if backend_type == BackendTypes.sql:
        query = EggLayingImage.query.filter_by(
            session_id=sid, basename=filename
        ).first_or_404()
        return send_file(
            BytesIO(query.image),
            mimetype=f"image/{os.path.splitext(filename)[0]}",
            as_attachment=False,
        )
    elif backend_type == BackendTypes.filesystem:
        return send_from_directory(
            os.path.join("../", app.config["UPLOAD_FOLDER"], sid), filename
        )


def zip_generator(ts):
    z = zipstream.ZipFile(mode="w", compression=zipstream.ZIP_DEFLATED)
    sm = app.downloadManager.sessions[ts]["session_manager"]
    for path in sm.basenames.values():
        img = img_as_bytes(sm.room, path)
        if img.getbuffer().nbytes > 0:
            z.write_iter(path, img)
    for chunk in z:
        yield chunk


def img_as_bytes(session_id, basename):
    with app.app_context():
        return BytesIO(
            EggLayingImage.query.filter_by(session_id=session_id, basename=basename)
            .first()
            .annotated_img
        )


@main.route("/annot-img/<ts>", methods=["GET"])
def return_zipfile(ts):
    zipfile_name = app.downloadManager.sessions[ts]["zipfile"]
    if backend_type == BackendTypes.sql:
        response = Response(zip_generator(ts), mimetype="application/zip")
        response.headers["Content-Disposition"] = "attachment; filename={}".format(
            zipfile_name
        )
        return response
    elif backend_type == BackendTypes.filesystem:
        shutil.rmtree(app.downloadManager.sessions[ts]["folder"])
        del app.downloadManager.sessions[ts]
        return send_file(os.path.join("../", zipfile_name), as_attachment=True)


@main.route("/upload", methods=["POST"])
def handle_upload():
    pauser.set_resume_timer()
    sid = request.form["sid"]
    if backend_type == BackendTypes.sql:
        remove_old_sql_rows()
    elif backend_type == BackendTypes.filesystem:
        for dir_name in ("uploads", "downloads"):
            remove_old_files(dir_name)
    check_chamber_type_of_imgs(sid)
    return "OK"


def check_chamber_type_of_imgs(sid):
    file_list = request.files
    n_files = len(request.files)
    for i, file in enumerate(file_list):
        check_chamber_type_of_img(i, file, sid, n_files)


def save_img_as_file(file, file_path):
    folder_path = Path(file_path).parent
    if not os.path.exists(folder_path):
        Path(folder_path).mkdir(exist_ok=True, parents=True)
    request.files[file].save(file_path)
    correct_via_exif(path=file_path)


def save_img_as_sql_blob(sid, file, file_path):
    user = SocketIOUser.query.filter_by(id=sid).first()
    if not user:
        user = SocketIOUser(id=sid)
        db.session.add(user)
    data = correct_via_exif(
        data=request.files[file].read(),
        format=os.path.splitext(file_path)[1][1:],
    )
    EggLayingImage(image=data, basename=os.path.basename(file_path), user=user)
    db.session.commit()


def save_uploaded_img(sid, file, file_path):
    if backend_type == BackendTypes.sql:
        save_img_as_sql_blob(sid, file, file_path)
    elif backend_type == BackendTypes.filesystem:
        save_img_as_file(file, file_path)


def check_chamber_type_of_img(i, file, sid, n_files):
    if file and allowed_file(file):
        filename = secure_filename(file)
        socketIO.emit(
            "counting-progress",
            {"data": "Uploading image %i of %i (%s)" % (i + 1, n_files, filename)},
            room=sid,
        )
        folder_path = os.path.join(app.config["UPLOAD_FOLDER"], sid)
        file_path = os.path.join(folder_path, filename)
        save_uploaded_img(sid, file, file_path)
        app.sessions[sid].check_chamber_type_and_find_bounding_boxes(
            file_path, i, n_files
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
    if len(request.json["chamberData"]) == 1:
        lowest_idx = int(list(request.json["chamberData"])[0])
    else:
        lowest_idx = min(*[int(i) for i in request.json["chamberData"].keys()])
    for file in file_list:
        attempts = 0
        succeeded = False
        while True:
            try:
                count_eggs_in_img(
                    file,
                    sid,
                    n_files,
                    is_lowest_idx=int(file["index"]) == lowest_idx,
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
    pauser.set_resume_timer()
    return "OK"


def count_eggs_in_img(file, sid, n_files, is_lowest_idx):
    file, index = file["file_name"], file["index"]
    if file and allowed_file(file):
        filename = secure_filename(file)
        folder_path = os.path.join(app.config["UPLOAD_FOLDER"], sid)
        if backend_type == BackendTypes.local and not os.path.exists(folder_path):
            Path(folder_path).mkdir(exist_ok=True, parents=True)
        filePath = os.path.join(folder_path, filename)
        kwargs = {
            "alignment_data": request.json["chamberData"][index],
            "index": index,
            "n_files": n_files,
            "is_lowest_idx": is_lowest_idx,
        }
        app.sessions[sid].segment_img_and_count_eggs(filePath, **kwargs)
