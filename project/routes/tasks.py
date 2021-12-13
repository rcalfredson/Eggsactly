from flask import abort, Blueprint, request
from flask.json import jsonify
from jwt.exceptions import InvalidTokenError
import numpy as np
from threading import Thread

from project.lib.web.auth_helper import AuthDecoder
from project.lib.web.gpu_task import GPUTask
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
        self.results = results

    def run(self):
        results = {
            k: np.array(self.results["predictions"][k])
            for k in self.results["predictions"]
        }
        app.gpu_manager.task_groups[self.group_id].register_completed_task(results)


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
    task = app.gpu_manager.get_task()
    if type(task) is not GPUTask:
        return jsonify({})
    return jsonify(
        img_path=task.img_path,
        type=task.task_type.name,
        group_id=task.group_id,
    )


@tasks.route("/tasks/gpu/<group_id>", methods=["POST"])
def receive_task_results(group_id):
    check_auth(request)
    results = request.get_json()
    task_finalizer = TaskFinalizer(group_id, results)
    task_finalizer.start()
    return group_id
