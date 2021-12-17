class Listener:
    def __init__(self, func, args=[], kwargs={}):
        self.func = func
        self.args = args
        self.kwargs = kwargs


class Event:
    # adapted from https://stackoverflow.com/a/57069782
    def __init__(self):
        self.listeners = []

    def __iadd__(self, listener: Listener):
        """Shortcut for using += to add a listener."""
        self.listeners.append(listener)
        return self

    def notify(self, kwargs={}):
        listener: Listener
        for listener in self.listeners:
            listener.func(*listener.args, **{**kwargs, **listener.kwargs})
