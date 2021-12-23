from flask import request
from flask_login import current_user
from glob import glob
import json
import os
from pathlib import Path
import time
import zipfile

from project import app, backend_type, db
from project.lib.common import zipdir
from project.lib.datamanagement.models import EggRegionTemplate
from project.lib.datamanagement.socket_io_auth import authenticated_only
from project.lib.web.backend_types import BackendTypes
from project.lib.web.sessionManager import SessionManager


def setup_event_handlers():
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
            app.socketIO, request.sid, app.gpu_manager
        )
        app.socketIO.emit("sid-from-server", {"sid": request.sid}, room=request.sid)

    @app.socketIO.on("save-custom-mask")
    @authenticated_only
    def save_custom_mask(data):
        if data["no_auth"]:
            app.socketIO.emit("mask-list-update", {"fail": "no_auth"})
        if backend_type == BackendTypes.gcp:
            mask = EggRegionTemplate.query.filter_by(
                name=data["maskName"], user=current_user
            ).first()
            if not mask:
                EggRegionTemplate(
                    data=data["maskData"], name=data["maskName"], user=current_user
                )
            else:
                mask.data = data["maskData"]
            db.session.commit()
        elif backend_type == BackendTypes.local:
            mask_dir = os.path.join("project", "configs", "masks", str(current_user.id))
            if not os.path.exists(mask_dir):
                Path(mask_dir).mkdir(parents=True)
            with open(os.path.join(mask_dir, f"{data['maskName']}.json"), "w") as f:
                data["user_id"] = current_user.id
                json.dump(data, f)
        app.socketIO.emit(
            "mask-list-update",
            {"names": get_mask_list(emit=False), "currentMask": data["maskName"]},
        )

    @app.socketIO.on("remove-img")
    def remove_img(data):
        del app.sessions[data["sid"]].predictions[
            os.path.normpath(os.path.join(app.config["UPLOAD_FOLDER"], data["imgName"]))
        ]

    @app.socketIO.on("load-custom-mask")
    def load_custom_mask(data):
        if backend_type == BackendTypes.gcp:
            queried_mask = EggRegionTemplate.query.filter_by(
                user=current_user, name=data["maskName"]
            ).first()
            mask_data = {"maskData": queried_mask.data, "maskName": data["maskName"]}
        elif backend_type == BackendTypes.local:
            with open(
                os.path.join(
                    "project",
                    "configs",
                    "masks",
                    str(current_user.id),
                    f"{data['maskName']}.json",
                )
            ) as f:
                mask_data = json.load(f)

        app.socketIO.emit("loaded-custom-mask", mask_data, room=data["sid"])

    @app.socketIO.on("prepare-csv")
    def prepare_csv(data):
        app.sessions[data["sid"]].sendCSV(
            data["editedCounts"], data["rowColLayout"], data["orderedCounts"]
        )

    @app.socketIO.on("submit-error-report")
    @authenticated_only
    def submit_error_report(data):
        if data["no_auth"]:
            app.socketIO.emit("report-ready", {"fail": "no_auth"}, room=data["sid"])
            return
        app.sessions[data["sid"]].createErrorReport(data["editedCounts"], current_user)

    @app.socketIO.on("prepare-annot-imgs-zip")
    def setup_imgs_download(data):
        ts = str(data["time"])
        app.downloadManager.addNewSession(
            app.sessions[data["sid"]], ts, data["editedCounts"]
        )
        app.downloadManager.createImagesForDownload(str(data["time"]))
        zipfName = "%s.zip" % (app.downloadManager.sessions[ts]["folder"])
        if backend_type == BackendTypes.local:
            zipf = zipfile.ZipFile(zipfName, "w", zipfile.ZIP_DEFLATED)
            zipdir(app.downloadManager.sessions[ts]["folder"], zipf)
            zipf.close()
        app.downloadManager.sessions[ts]["zipfile"] = zipfName
        app.socketIO.emit("zip-annots-ready", {"time": ts}, room=data["sid"])

    @app.socketIO.on("mask-list")
    def get_mask_list(emit=True):
        names = []
        if current_user.is_authenticated:
            if backend_type == BackendTypes.gcp:
                names = [
                    el[0]
                    for el in (
                        db.session.query(EggRegionTemplate.name)
                        .filter(EggRegionTemplate.user == current_user)
                        .all()
                    )
                ]
            elif backend_type == BackendTypes.local:
                names = [
                    os.path.basename(mask.split(".json")[0])
                    for mask in glob(
                        f"./project/configs/masks/{current_user.id}/*.json"
                    )
                ]
        if emit:
            app.socketIO.emit("mask-list", {"names": names})
        else:
            return names

    @app.socketIO.on("reset-session")
    def reset_session(data):
        app.sessions[data["sid"]].clear_data()
