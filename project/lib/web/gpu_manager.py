import queue

from project.lib.web.gpu_task import GPUTask
from project.lib.web.gpu_task_group import GPUTaskGroup


class GPUManager:
    def __init__(self):
        self.queue = queue.Queue()
        self.task_groups = {}

    def add_task_group(self, room, n_tasks, task_type) -> GPUTaskGroup:
        new_taskgroup = GPUTaskGroup(n_tasks, room, task_type)
        self.task_groups[new_taskgroup.id] = new_taskgroup
        return self.task_groups[new_taskgroup.id]

    def register_completed_task(self, results, group_id):
        self.task_groups[group_id].register_completed_task(results)
        if self.task_groups[group_id].complete:
            del self.task_groups[group_id]

    def add_task(self, task_group, img_path, data={}):
        task = GPUTask(task_group, img_path, data)
        self.queue.put(task)

    def get_task(self):
        try:
            return self.queue.get(timeout=0.5)
        except queue.Empty:
            return {}
