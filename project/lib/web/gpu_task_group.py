from project.lib.event import Event, Listener


class GPUTaskGroup:
    """sends a notification when a group of tasks has been completed."""

    def __init__(self, n_tasks):
        self.n_tasks = n_tasks
        self.results = []
        self.on_completion = Event()

    def add_completion_listener(self, listener: Listener):
        self.on_completion += listener

    def register_completed_task(self, predictions):
        self.results.append(predictions)
        if len(self.results) == self.n_tasks:
            self.notify_complete()

    def notify_complete(self):
        self.on_completion.notify({"predictions": self.results})
