from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from csbdeep.utils import normalize
import cv2
import datetime
from dotenv import load_dotenv
import jwt
import numpy as np
import os
import requests
import time
import timeit
import torch
from project.circleFinder import ARENA_IMG_RESIZE_FACTOR
from project.detectors.splinedist.config import Config
from project.detectors.splinedist.models.model2d import SplineDist2D
from project.lib.os.pauser import PythonPauser
from project.lib.web.exceptions import CUDAMemoryException
from project.lib.web.gpu_task_types import GPUTaskTypes
from project.lib.web.sessionManager import SessionManager

import timeit

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
server_uri = os.environ["MAIN_SERVER_URI"]
key_holder = AuthHelper(os.environ["PRIVATE_KEY_PATH"])
request_headers = {"Authorization": f"access_token {key_holder.get_jwt()}"}
active_tasks = {}
networks = {}
pauser = PythonPauser()


def request_work():
    print("\ngetting a task")
    if len(active_tasks) > 0:
        print("returning early")
        return
    try:
        r = requests.get(f"{server_uri}/tasks/gpu", headers=request_headers)
        if r.status_code >= 400 and r.status_code < 500:
            print("server rejected request. status:", r.status_code)
            return
        task = r.json()
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
            print("path:", task["img_path"])
        print("num attempts:", attempt_ct + 1)
        decode_start_t = timeit.default_timer()

        img = cv2.cvtColor(
            cv2.imdecode(
                np.asarray(
                    bytearray(
                        requests.get(
                            f"{server_uri}/{task['img_path']}", stream=True
                        ).raw.read()
                    ),
                    dtype="uint8",
                ),
                cv2.IMREAD_COLOR,
            ),
            cv2.COLOR_RGB2BGR,
        )
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
        predict_start_t = timeit.default_timer()
        print(
            "time spent resizing and normalizing:",
            predict_start_t - resize_norm_start_t,
        )
        try:
            predictions = networks[task_type].predict_instances(img)[1]
        except Exception as exc:
            if SessionManager.is_CUDA_mem_error(exc):
                raise CUDAMemoryException
        predictions = {k: predictions[k].tolist() for k in predictions}
        post_req_start_t = timeit.default_timer()
        print("time spent predicting:", post_req_start_t - predict_start_t)
        post_results_to_server(task_key, {"predictions": predictions})

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
    start_t = timeit.default_timer()
    request_work()
    end_t = timeit.default_timer()
    time_balance = 0.75 - (end_t - start_t)
    if time_balance > 0:
        time.sleep(time_balance)
