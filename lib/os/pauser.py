import os
import psutil

from lib.web.scheduler import Scheduler


class PythonPauser():
    def __init__(self):
        self.paused_parents = set()
        self.my_pid = os.getpid()
        self.scheduler = Scheduler(1)
        self.resume_is_scheduled = False

    def is_flaggable_process(self, proc: psutil.Process):
        if proc.parent() == None:
            return False
        grandparent_proc = proc.parent().parent()
        if grandparent_proc == None:
            return False
        return proc.pid != self.my_pid and \
            proc.cpu_percent(interval=1) > 20 and \
            'python' in grandparent_proc.name()

    def get_high_cpu_py_processes_with_py_parents(self):
        """Collect a list of Python processes matching these criteria:
            - PID differs from that of the current process
            - % CPU utilization exceeds 20
            - Is a subprocess of another Python process
        """
        py_processes = [proc.info['pid'] for proc in psutil.process_iter(
            ['pid', 'name']) if 'python' in proc.info['name'].lower()]
        self.flagged_processes = []
        for proc in py_processes:
            proc = psutil.Process(proc)
            if self.is_flaggable_process(proc):
                self.flagged_processes.append(proc)

    def set_resume_timer(self):
        if hasattr(self, 'scheduled_job') and self.scheduled_job is not None:
            self.scheduler.schedule.cancel_job(self.scheduled_job)
            self.stop_run_continuously.set()
        self.scheduled_job = self.scheduler.schedule.every(25).minutes.do(
            self.resume_high_cpu_py_processes)
        self.stop_run_continuously = self.scheduler.run_continuously()
        self.resume_is_scheduled = True

    def resume_high_cpu_py_processes(self):
        pid: int
        for pid in self.paused_parents:
            try:
                psutil.Process(pid).resume()
            except psutil.NoSuchProcess:
                print('Error: tried to resume process', pid,
                      "but it doesn't exist anymore")
        self.paused_parents = set()
        return self.scheduler.schedule.CancelJob

    def end_high_impact_py_prog(self):
        self.get_high_cpu_py_processes_with_py_parents()
        proc: psutil.Process
        for proc in self.flagged_processes:
            grandparent = proc.parent().parent()
            self.paused_parents.add(grandparent.pid)
            grandparent.suspend()
            proc.kill()
