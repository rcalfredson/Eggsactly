from dotenv import load_dotenv
from flask import abort, Blueprint, request
from flask.json import jsonify
from jwt.exceptions import InvalidTokenError
import numpy as np
import os
from threading import Event, Thread

from project import app
from project.lib.web.auth_helper import AuthDecoder
from project.lib.web.exceptions import CUDAMemoryException, ImageIgnoredException
from project.lib.web.gpu_task import GPUTask
from project.lib.web.gpu_task_types import GPUTaskTypes


load_dotenv()
SAFE_TIMEOUT = int(os.environ["GPU_WORKER_TIMEOUT"]) - 1
tasks = Blueprint("tasks", __name__)
num_workers = int(os.environ["NUM_GPU_WORKERS"])
auth_decoder = AuthDecoder(
    [f"project/auth/gpu_worker_{i+1}_id_rsa.pub" for i in range(num_workers)]
)


class TaskFinalizer(Thread):
    def __init__(self, group_id, results):
        Thread.__init__(self)
        self.group_id = group_id
        self.room = app.gpu_manager.task_groups[group_id].room
        self.task_type = app.gpu_manager.task_groups[group_id].task_type
        self.results = results

    def handle_error(self):
        if self.results["error"] == repr(CUDAMemoryException()):
            if self.results["will_retry"]:
                app.socketIO.emit(
                    "counting-progress",
                    {
                        "data": "Ran out of system resources. "
                        "Trying to reallocate and try again..."
                    },
                    room=self.room,
                )
            else:
                app.sessions[self.room].report_counting_error(
                    self.results["img_path"], CUDAMemoryException
                )

    def run(self):
        if "error" in self.results:
            self.handle_error()
        if self.task_type == GPUTaskTypes.arena:
            results = {
                "predictions": [
                    {k: np.array(prediction_set[k]) for k in prediction_set}
                    for prediction_set in self.results["predictions"]
                ],
                "metadata": self.results["metadata"],
            }
        else:
            results = self.results
            if "ignored" in results["metadata"] and results["metadata"]["ignored"]:
                results["predictions"] = [ImageIgnoredException]
        app.gpu_manager.register_completed_task(results, self.group_id)


def check_auth(request):
    if not request.headers.get("Authorization"):
        abort(400)
    try:
        auth_decoder.decode_token(
            request.headers.get("Authorization").split("access_token ")[1]
        )
    except InvalidTokenError:
        abort(401)


@tasks.route("/tasks/gpu")
def get_task():
    check_auth(request)

    def task_as_json(task: GPUTask):
        return jsonify(
            img_path=task.img_path,
            type=task.task_type.name,
            room=task.task_group.room,
            group_id=task.task_group.id,
            data=task.data,
        )

    task: GPUTask
    task = app.gpu_manager.get_task()
    if type(task) is not GPUTask:
        task_request = Event()
        app.gpu_manager.add_task_request(task_request)
        task_available = task_request.wait(timeout=SAFE_TIMEOUT)
        if task_request in app.gpu_manager.task_requests:
            app.gpu_manager.task_requests.remove(task_request)
        if task_available:
            return task_as_json(app.gpu_manager.get_task())
        else:
            return jsonify({})
    else:
        return task_as_json(task)


@tasks.route("/tasks/gpu/report", methods=["POST"])
def report_task_progress():
    info = request.get_json()
    task_type = info["task_type"]
    room = app.gpu_manager.task_groups[info["group_id"]].room
    s = app.sessions[room]
    path_map = s.paths_to_indices
    img_index = int(path_map[info["img_path"]])
    if task_type == "egg":
        message = (
            f"Counting eggs in image {img_index + 1} of {s.n_files}"
            + f" (region {info['region_index'] + 1} of {info['tot_regions']})"
        )
    elif task_type == "arena":
        message = f"Segmenting image {img_index + 1} of {s.n_files}"
    s.emit_to_room(
        "counting-progress",
        {
            "overwrite": True if info["region_index"] else False,
            "ellipses": False,
            "data": message,
        },
    )
    return ("", 204)


@tasks.route("/tasks/gpu/<group_id>", methods=["POST"])
def receive_task_results(group_id):
    check_auth(request)
    results = request.get_json()
    task_finalizer = TaskFinalizer(group_id, results)
    task_finalizer.start()
    return group_id
