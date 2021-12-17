from project.lib.web.gpu_task_group import GPUTaskGroup


class GPUTask:
    def __init__(self, task_group: GPUTaskGroup, img_path, data):
        self.task_group = task_group
        self.img_path = img_path
        self.data = data

    @property
    def task_type(self):
        return self.task_group.task_type
