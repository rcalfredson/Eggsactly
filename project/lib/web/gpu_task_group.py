import uuid
from project.lib.event import Event, Listener


class GPUTaskGroup:
    """sends a notification when a group of tasks has been completed."""

    def __init__(self, n_tasks, room):
        self.id = str(uuid.uuid1())
        self.room = room
        self.n_tasks = n_tasks
        self.results = []
        self.on_completion = Event()

    @property
    def complete(self):
        return len(self.results) == self.n_tasks

    def add_completion_listener(self, listener: Listener):
        self.on_completion += listener

    def register_completed_task(self, predictions):
        self.results.append(predictions)
        if self.complete:
            self.notify_complete()

    def notify_complete(self):
        self.on_completion.notify({"predictions": self.results})
