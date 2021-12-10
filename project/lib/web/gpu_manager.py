import queue

from project.lib.web.gpu_task import GPUTask
from project.lib.web.gpu_task_group import GPUTaskGroup


class GPUManager:
    def __init__(self):
        self.queue = queue.Queue()
        self.task_groups = {}

    def add_task_group(self, sid, n_tasks) -> GPUTaskGroup:
        self.task_groups[sid] = GPUTaskGroup(n_tasks)
        return self.task_groups[sid]

    def add_task(self, group_id, img_path, task_type):
        task = GPUTask(group_id, img_path, task_type)
        self.queue.put(task)

    def get_task(self):
        try:
            return self.queue.get(timeout=0.5)
        except queue.Empty:
            return {}
