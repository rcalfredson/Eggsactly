from project.lib.web.gpu_task_group import GPUTaskGroup


class GPUTask:
    def __init__(self, task_group: GPUTaskGroup, img_path, task_type):
        self.task_group = task_group
        self.img_path = img_path
        self.task_type = task_type
