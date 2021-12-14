from flask import abort, Blueprint, request
from flask.json import jsonify
from jwt.exceptions import InvalidTokenError
import numpy as np
from threading import Thread

from project.lib.web.auth_helper import AuthDecoder
from project.lib.web.exceptions import CUDAMemoryException
from project.lib.web.gpu_task import GPUTask
from project.lib.web.gpu_task_types import GPUTaskTypes
from .. import app


tasks = Blueprint("tasks", __name__)
num_workers = 1
auth_decoder = AuthDecoder(
    [f"project/configs/gpu_worker_{i+1}_id_rsa.pub" for i in range(num_workers)]
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
    task: GPUTask
    task = app.gpu_manager.get_task()
    if type(task) is not GPUTask:
        return jsonify({})
    return jsonify(
        img_path=task.img_path,
        type=task.task_type.name,
        room=task.task_group.room,
        group_id=task.task_group.id,
        data=task.data,
    )


@tasks.route("/tasks/gpu/<group_id>", methods=["POST"])
def receive_task_results(group_id):
    check_auth(request)
    results = request.get_json()
    task_finalizer = TaskFinalizer(group_id, results)
    task_finalizer.start()
    return group_id
