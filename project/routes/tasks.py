import base64
from flask import Blueprint, request
import io
import numpy as np
from PIL import Image
from flask.json import jsonify

from project.lib.web.gpu_task import GPUTask
from .. import app


tasks = Blueprint("tasks", __name__)


@tasks.route("/tasks/gpu")
def get_task():
    task = app.gpu_manager.get_task()
    if type(task) is not GPUTask:
        return jsonify({})
    in_mem_file = io.BytesIO()
    img = Image.open(task.img_path)
    img.save(in_mem_file, format="PNG")
    in_mem_file.seek(0)
    img_bytes = in_mem_file.read()
    base64_image = base64.b64encode(img_bytes)
    return jsonify(
        image=base64_image.decode("ascii"),
        type=task.task_type.name,
        group_id=task.group_id,
    )


@tasks.route("/tasks/gpu/<group_id>", methods=["POST"])
def receive_task_results(group_id):
    results = request.get_json()
    results = {k: np.array(results['predictions'][k]) for k in results['predictions']}
    app.gpu_manager.task_groups[group_id].register_completed_task(
        results
    )
    return group_id
