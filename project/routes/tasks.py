from flask import Blueprint, request
from flask.json import jsonify
import numpy as np
from threading import Thread

from project.lib.web.gpu_task import GPUTask
from .. import app


tasks = Blueprint("tasks", __name__)


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


@tasks.route("/tasks/gpu")
def get_task():
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
    results = request.get_json()
    task_finalizer = TaskFinalizer(group_id, results)
    task_finalizer.start()
    return group_id
