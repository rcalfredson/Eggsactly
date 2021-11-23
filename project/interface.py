from glob import glob
import json
import os
import time
import zipfile

from flask import request
from flask_socketio import SocketIO, emit
from PIL import Image

from project.lib.image.exif import correct_via_exif
from project.lib.web.downloadManager import DownloadManager
from project.lib.web.network_loader import NetworkLoader
from project.lib.web.scheduler import Scheduler
from project.lib.web.sessionManager import SessionManager

from . import create_app


UPLOAD_FOLDER = "./uploads"
NET_ARCH = ("fcrn", "splinedist")[1]
app = create_app()
app.sessions = {}
app.downloadManager = DownloadManager()
network_loader = NetworkLoader(NET_ARCH)


def prune_old_sessions():
    current_time = time.time()
    for sid in list(app.sessions.keys()):
        if current_time - app.sessions[sid].lastPing > 60 * 10:
            del app.sessions[sid]


scheduler = Scheduler(1)
scheduler.schedule.every(5).minutes.do(prune_old_sessions)
scheduler.run_continuously()


def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(
                os.path.join(root, file),
                os.path.relpath(os.path.join(root, file), os.path.join(path, "..")),
            )


@app.socketIO.on("ping")
def field_ping(data):
    if data["sid"] not in app.sessions:
        return
    app.sessions[data["sid"]].lastPing = time.time()
    if request.sid != data["sid"] and request.sid in app.sessions:
        app.sessions[data["sid"]].room = request.sid
        del app.sessions[request.sid]
    app.sessions[data["sid"]].emit_to_room("pong", {})


@app.socketIO.on("connect")
def connected():
    app.sessions[request.sid] = SessionManager(
        app.socketIO, request.sid, network_loader
    )
    app.socketIO.emit("sid-from-server", {"sid": request.sid}, room=request.sid)


@app.socketIO.on("save-custom-mask")
def save_custom_mask(data):
    with open(
        os.path.join("project", "configs", "masks", f"{data['maskName']}.json"), "w"
    ) as f:
        json.dump(data, f)
    app.socketIO.emit(
        "mask-list-update",
        {"names": get_mask_list(data, emit=False), "currentMask": data["maskName"]},
    )


@app.socketIO.on("remove-img")
def remove_img(data):
    del app.sessions[data["sid"]].predictions[
        os.path.normpath(os.path.join(app.config["UPLOAD_FOLDER"], data["imgName"]))
    ]


@app.socketIO.on("load-custom-mask")
def load_custom_mask(data):
    with open(
        os.path.join("project", "configs", "masks", f"{data['maskName']}.json")
    ) as f:
        app.socketIO.emit("loaded-custom-mask", json.load(f), room=data["sid"])


@app.socketIO.on("prepare-csv")
def prepare_csv(data):
    app.sessions[data["sid"]].saveCSV(
        data["editedCounts"], data["rowColLayout"], data["orderedCounts"]
    )


@app.socketIO.on("submit-error-report")
def submit_error_report(data):
    app.sessions[data["sid"]].createErrorReport(data["editedCounts"], data["user"])


@app.socketIO.on("prepare-annot-imgs-zip")
def setup_imgs_download(data):
    ts = str(data["time"])
    app.downloadManager.addNewSession(app.sessions[data["sid"]], ts, data["editedCounts"])
    app.downloadManager.createImagesForDownload(str(data["time"]))
    zipfName = "%s.zip" % (app.downloadManager.sessions[ts]["folder"])
    zipf = zipfile.ZipFile(zipfName, "w", zipfile.ZIP_DEFLATED)
    zipdir(app.downloadManager.sessions[ts]["folder"], zipf)
    zipf.close()
    app.downloadManager.sessions[ts]["zipfile"] = zipfName
    app.socketIO.emit("zip-annots-ready", {"time": ts}, room=data["sid"])


@app.socketIO.on("mask-list")
def get_mask_list(data, emit=True):
    names = [
        os.path.basename(mask.split(".json")[0])
        for mask in glob("./project/configs/masks/*.json")
    ]
    if emit:
        app.socketIO.emit("mask-list", {"names": names})
    else:
        return names


@app.socketIO.on("reset-session")
def reset_session(data):
    app.sessions[data["sid"]].clear_data()


if __name__ == "__main__":
    app.socketIO.run(app)
