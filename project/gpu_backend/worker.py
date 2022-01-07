from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from csbdeep.utils import normalize
import cv2
import datetime
from dotenv import load_dotenv
import json
import jwt
import numpy as np
import os
import requests
import time
import timeit
import torch

from project import app, create_app
from project.detectors.splinedist.config import Config
from project.detectors.splinedist.models.model2d import SplineDist2D
from project.lib.datamanagement.models import EggLayingImage
from project.lib.image.circleFinder import ARENA_IMG_RESIZE_FACTOR
from project.lib.image.converter import byte_to_bgr
from project.lib.image.drawing import get_interpolated_points
from project.lib.image.sub_image_helper import SubImageHelper
from project.lib.os.pauser import PythonPauser
from project.lib.web.exceptions import CUDAMemoryException
from project.lib.web.gpu_task_types import GPUTaskTypes
from project.lib.web.sessionManager import SessionManager


NETWORK_CONSTS = {
    GPUTaskTypes.arena: {
        "wts": "project/models/arena_pit_v2.pth",
        "config": "project/configs/unet_reduced_backbone_arena_wells.json",
    },
    GPUTaskTypes.egg: {
        "wts": "project/models/"
        + "splinedist_unet_full_400epochs_NZXT-U_2021-08-12 08-39-05.733572.pth",
        "config": "project/configs/unet_backbone_rand_zoom.json",
    },
}
MAX_ATTEMPTS_PER_IMG = 2
MAX_SQL_QUERIES_PER_IMG = 10


class AuthHelper:
    def __init__(self, keypath):
        with open(keypath, "rb") as f:
            self.private_key = serialization.load_pem_private_key(
                f.read(), password=None, backend=default_backend()
            )
        self.start_time = "{date:%Y-%m-%d_%H:%M:%S}".format(
            date=datetime.datetime.now()
        )

    def get_jwt(self):
        jwt_token = jwt.encode(
            {"name": f"gpu-worker-{self.start_time}"},
            key=self.private_key,
            algorithm="RS256",
        )
        return jwt_token


load_dotenv()
create_app()
server_uri = os.environ["MAIN_SERVER_URI"]
key_holder = AuthHelper(os.environ["PRIVATE_KEY_PATH"])
reconnect_attempt_delay = int(os.environ["GPU_WORKER_RECONNECT_ATTEMPT_DELAY"])
request_headers = {"Authorization": f"access_token {key_holder.get_jwt()}"}
active_tasks = {}
networks = {}
pauser = PythonPauser()
with open("project/models/modelRevDates.json", "r") as f:
    model_to_update_date = json.load(f)
    latest_model = model_to_update_date["models"].get(
        model_to_update_date["latest"], "unknown"
    )


def request_work():
    print("\ngetting a task")
    if len(active_tasks) > 0:
        print("returning early")
        return
    try:
        r = requests.get(
            f"{server_uri}/tasks/gpu",
            headers=request_headers,
        )
        if r.status_code >= 400 and r.status_code < 500:
            print("server rejected request. status:", r.status_code)
            return
        try:
            task = r.json()
        except json.decoder.JSONDecodeError:
            print("Failed to decode the egg-counting server response")
            print("Raw response:", r.text)
            return
        if len(task.keys()) == 0:
            print("no work found")
            return
        print("starting a task")
        active_tasks[task["group_id"]] = task
        attempts = 0
        succeeded = False
        while True:
            try:
                perform_task(attempts)
                succeeded = True
            except CUDAMemoryException as exc:
                attempts += 1
                post_results_to_server(
                    task["group_id"],
                    {
                        "error": repr(exc),
                        "will_retry": attempts < MAX_ATTEMPTS_PER_IMG,
                        "img_path": task["img_path"],
                    },
                )
                if attempts < MAX_ATTEMPTS_PER_IMG:
                    pauser.end_high_impact_py_prog()
                else:
                    break
            if succeeded:
                break
        if succeeded:
            print("task complete")
        else:
            print("unable to complete task due to error")
            del active_tasks[task["group_id"]]
        request_work()
    except requests.exceptions.ConnectionError:
        print("failed to connect to the egg-counting server")
        time.sleep(reconnect_attempt_delay)


def report_progress_to_server(group_id, region_index, tot_regions, img_path):
    requests.request(
        "POST",
        f"{server_uri}/tasks/gpu/report",
        json={
            "group_id": group_id,
            "region_index": region_index,
            "tot_regions": tot_regions,
            "img_path": img_path,
        },
        headers=request_headers,
    )


def post_results_to_server(task_key, result):
    requests.request(
        "POST",
        f"{server_uri}/tasks/gpu/{task_key}",
        json=result,
        headers=request_headers,
    )


def perform_task(attempt_ct=0):
    pauser.set_resume_timer()
    for task_key in list(active_tasks.keys()):
        start_t = timeit.default_timer()
        task = active_tasks[task_key]
        if task["type"] not in GPUTaskTypes.__members__:
            del active_tasks[task_key]
            return
        task_type = GPUTaskTypes[task["type"]]
        if attempt_ct == 0:
            print("task type:", task_type.name)
        print("num attempts:", attempt_ct + 1)
        decode_start_t = timeit.default_timer()
        num_tries, img_entity = 0, None

        while not img_entity and num_tries < MAX_SQL_QUERIES_PER_IMG:
            with app.app_context():
                img_entity = EggLayingImage.query.filter_by(
                    session_id=task["room"], basename=os.path.basename(task["img_path"])
                ).first()
            num_tries += 1
            if not img_entity and num_tries < MAX_SQL_QUERIES_PER_IMG:
                print("Couldn't find image; retrying...")
                print("amount for sleep:", num_tries * 2)
                time.sleep(num_tries * 2)

        if not img_entity:
            print("Couldn't find image specified in task")
            print(
                "Queried room",
                task["room"],
                "and basename",
                os.path.basename(task["img_path"]),
            )
            return
        img = byte_to_bgr(img_entity.image)
        print("time spent decoding:", timeit.default_timer() - decode_start_t)
        resize_norm_start_t = timeit.default_timer()
        if task_type == GPUTaskTypes.arena:
            img = cv2.resize(
                img,
                (0, 0),
                fx=ARENA_IMG_RESIZE_FACTOR,
                fy=ARENA_IMG_RESIZE_FACTOR,
                interpolation=cv2.INTER_CUBIC,
            )
        img = normalize(img, 1, 99.8, axis=(0, 1))
        metadata = {}
        predictions = []
        if task_type == GPUTaskTypes.arena:
            imgs = (img,)
        elif task_type == GPUTaskTypes.egg:
            metadata["model"] = latest_model
            metadata["filename"] = os.path.basename(task["img_path"])
            metadata["index"] = task["data"]["index"]
            if "ignored" in task["data"] and task["data"]["ignored"]:
                metadata["ignored"] = True
                print("image marked as ignored; skipping")
                post_req_start_t = timeit.default_timer()
                post_results_to_server(
                    task_key, {"predictions": [], "metadata": metadata}
                )
                clean_up_task(task_key, start_t, post_req_start_t)
                return
            helper = SubImageHelper()
            helper.get_sub_images(img, task["img_path"], task["data"], task["room"])
            metadata["rotationAngle"] = helper.rotation_angle
            metadata["bboxes"] = helper.bboxes
            imgs = helper.subImgs
        predict_start_t = timeit.default_timer()
        print(
            "time spent resizing and normalizing:",
            predict_start_t - resize_norm_start_t,
        )
        for i, img in enumerate(imgs):
            try:
                if task_type == GPUTaskTypes.egg:
                    report_progress_to_server(task_key, i, len(imgs), task["img_path"])
                results = networks[task_type].predict_instances(img)[1]
                results["count"] = len(results["points"])
                results["outlines"] = get_interpolated_points(results["coord"])
                predictions.append(results)
            except Exception as exc:
                print("encountered an exception.", exc)
                print(type(exc))
                if SessionManager.is_CUDA_mem_error(exc):
                    raise CUDAMemoryException

        converted_predictions = []
        for prediction_set in predictions:
            converted_set = {}
            for k in prediction_set:
                if type(prediction_set[k]) is np.ndarray:
                    converted_set[k] = prediction_set[k].tolist()
                else:
                    converted_set[k] = prediction_set[k]
            converted_predictions.append(converted_set)
        post_req_start_t = timeit.default_timer()
        print("time spent predicting:", post_req_start_t - predict_start_t)
        post_results_to_server(
            task_key, {"predictions": converted_predictions, "metadata": metadata}
        )
        clean_up_task(task_key, start_t, post_req_start_t)


def clean_up_task(task_key, start_t, post_req_start_t):
    del active_tasks[task_key]
    end_t = timeit.default_timer()
    print("time spent making post request:", end_t - post_req_start_t)
    print("total time for task:", end_t - start_t)


def init_networks():
    for type in GPUTaskTypes:
        init_splinedist_network(type)


def init_splinedist_network(type):
    networks[type] = SplineDist2D(
        Config(NETWORK_CONSTS[type]["config"], n_channel_in=3)
    )
    networks[type].cuda()
    networks[type].train(False)
    networks[type].load_state_dict(torch.load(NETWORK_CONSTS[type]["wts"]))


init_networks()
while True:
    request_work()
