from csbdeep.utils import normalize
import cv2
from dotenv import load_dotenv
import numpy as np
import os
import requests
import torch
from project.circleFinder import ARENA_IMG_RESIZE_FACTOR
from project.detectors.splinedist.config import Config
from project.detectors.splinedist.models.model2d import SplineDist2D
from project.lib.web.gpu_task_types import GPUTaskTypes
from project.lib.web.scheduler import Scheduler

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


load_dotenv()
server_uri = os.environ["MAIN_SERVER_URI"]
active_tasks = {}
networks = {}


def request_work():
    if len(active_tasks) > 0:
        return
    print("\ngetting a task")
    try:
        r = requests.get(f"{server_uri}/tasks/gpu")
        task = r.json()
        if len(task.keys()) == 0:
            print("no work found")
            return
        print("starting a task")
        active_tasks[task["group_id"]] = task
        perform_task()
        print("task complete")
        request_work()
    except requests.exceptions.ConnectionError:
        print("Failed to connect to the egg-counting server")


scheduler = Scheduler(1)
scheduler.schedule.every(1).second.do(request_work)


def perform_task():
    for task_key in list(active_tasks.keys()):
        start_t = timeit.default_timer()
        task = active_tasks[task_key]
        if task["type"] not in GPUTaskTypes.__members__:
            del active_tasks[task_key]
            return
        task_type = GPUTaskTypes[task["type"]]
        print("task type:", task_type.name)
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
        predictions = networks[task_type].predict_instances(img)[1]
        predictions = {k: predictions[k].tolist() for k in predictions}
        post_req_start_t = timeit.default_timer()
        print("time spent predicting:", post_req_start_t - predict_start_t)
        requests.request(
            "POST",
            f"{server_uri}/tasks/gpu/{task_key}",
            json={"predictions": predictions},
        )
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
scheduler.run_continuously()
