import argparse
import os
import time


from project.lib.web.downloadManager import DownloadManager
from project.lib.web.gpu_manager import GPUManager
from project.lib.web.network_loader import NetworkLoader
from project.lib.web.scheduler import Scheduler
from project.routes import socket_events

from . import create_app


UPLOAD_FOLDER = "./uploads"
NET_ARCH = ("fcrn", "splinedist")[1]


def prune_old_sessions():
    current_time = time.time()
    for sid in list(app.sessions.keys()):
        if current_time - app.sessions[sid].lastPing > 60 * 10:
            del app.sessions[sid]


is_main = __name__ == "__main__"
is_dev = os.environ["FLASK_ENV"] == "development"
if is_dev or is_main:
    app = create_app()
    app.sessions = {}
    app.downloadManager = DownloadManager()
    app.network_loader = NetworkLoader(NET_ARCH)
    app.gpu_manager = GPUManager()
    socket_events.setup_event_handlers()
    if is_main:
        app.socketIO.run(app)
    scheduler = Scheduler(1)
    scheduler.schedule.every(5).minutes.do(prune_old_sessions)
    scheduler.run_continuously()
