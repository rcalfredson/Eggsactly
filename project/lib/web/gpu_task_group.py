import uuid
from project.lib.event import Event, Listener


class GPUTaskGroup:
    """sends a notification when a group of tasks has been completed."""

    def __init__(self, n_tasks, room, task_type):
        self.id = str(uuid.uuid1())
        self.room = room
        self.n_tasks = n_tasks
        self.task_type = task_type
        self.results = []
        self.on_completion = Event()

    @property
    def complete(self):
        return len(self.results) == self.n_tasks

    def add_completion_listener(self, listener: Listener):
        self.on_completion += listener

    def register_completed_task(self, results):
        self.results.append(results)
        if self.complete:
            self.notify_complete()

    def notify_complete(self):
        notify_args = {k: [] for k in self.results[0].keys() if len(self.results[0][k])}
        for result in self.results:
            for k in notify_args:
                if (
                    len(self.results) == 1
                    and type(result[k]) is list
                    and len(result[k]) == 1
                ):
                    notify_args[k] = result[k][0]
                else:
                    notify_args[k].append(result[k])

        self.on_completion.notify(notify_args)
