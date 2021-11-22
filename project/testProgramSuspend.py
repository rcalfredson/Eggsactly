import os
import psutil
import subprocess
import time

pid = '18048'

# subprocess.call([
#     'bin/NtSuspendProcess64.exe',
#     '-pid', pid
# ])

class PythonPauser():
    def __init__(self):
        self.paused_programs = []

    def get_high_cpu_py_processes(self):
        my_pid = os.getpid()
        py_processes = [proc.info['pid'] for proc in psutil.process_iter(
            ['pid', 'name']) if 'python' in proc.info['name'].lower()]
        self.py_high_cpu_processes = []
        for proc in py_processes:
            proc = psutil.Process(proc)
            if proc.pid != my_pid and proc.cpu_percent(interval=1) > 20:
                self.py_high_cpu_processes.append(proc)

    def pause_high_cpu_py_processes(self):
        self.get_high_cpu_py_processes()
        for proc in self.py_high_cpu_processes:
            proc.suspend()
            self.paused_programs.append(proc.pid)

    def resume_high_cpu_py_processes(self):
        for pid in self.paused_programs:
            psutil.Process(pid).resume()

        self.paused_programs = []

pauser = PythonPauser()
pauser.pause_high_cpu_py_processes()
print('paused high cpu Python programs')
time.sleep(5)
print('now trying to resume them')
pauser.resume_high_cpu_py_processes()