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
        # we are no longer receiving "predictions,"
        # but instead a dict that has two keys: "predictions"
        # and "metadata"
        self.results.append(results)
        if self.complete:
            self.notify_complete()

    def notify_complete(self):
        print("self results before filtering keys:", self.results)
        # for result in self.results:
        #     for k in notify_args:
        #         print('what is k while building notify args:', k)
        #         if k != 'predictions':
        #             print('and associated value:', result[k])
        #         input()

        #         notify_args[k].(result[k])
        # print("notify args?", notify_args)
        # at this point in the code, self.results
        # contains one dictionary of results for every task in the group.
        # however, each task can also contain the results of
        # multiple predictions, and this is where I'm getting confused.
        # why not just send the raw data and let the downstream functions
        # sort it out?
        # why am I having trouble converting to the desired forms?
        # for segment_image_via_object_detection,
        # we want a list with one element, a dictionary whose keys are
        # 'coord', etc., and we don't care about metadata.
        # for sendAnnotationsToClient, we want
        # a list with as many elements as there are regions in the image.

        notify_args = {k: [] for k in self.results[0].keys() if len(self.results[0][k])}
        # make an array.
        for result in self.results:
            for k in notify_args:
                print("what is k while building notify args:", k)
                if k != "predictions":
                    print("and associated value:", result[k])
                # what is being appended to the list?
                # a list is being appended to the list, but that's not what we want.
                # it's a list because
                if (
                    len(self.results) == 1
                    and type(result[k]) is list
                    and len(result[k]) == 1
                ):
                    notify_args[k] = result[k][0]
                elif len(self.results) == 1 and type(result[k]) is dict:
                    notify_args[k] = result[k]
                else:
                    notify_args[k].append(result[k])
        # why can't I make this more direct? we don't need this.
        # the concept of task group is useless now.
        print("predictions in notify args:", notify_args["predictions"])
        if "metadata" in notify_args:
            print("metadata:", notify_args["metadata"])

        self.on_completion.notify(notify_args)
