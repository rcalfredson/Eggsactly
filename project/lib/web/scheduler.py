import schedule
import threading
import time


class Scheduler:
    def __init__(self, interval):
        self.interval = interval
        self.schedule = schedule

    def run_continuously(self):
        """Continuously run, while executing pending jobs at each
        elapsed time interval.
        @return cease_continuous_run: threading. Event which can
        be set to cease continuous run. Please note that it is
        *intended behavior that run_continuously() does not run
        missed jobs*. For example, if you've registered a job that
        should run every minute and you set a continuous run
        interval of one hour then your job won't be run 60 times
        at each interval but only once.
        """
        cease_continuous_run = threading.Event()

        class ScheduleThread(threading.Thread):
            @classmethod
            def run(cls):
                while not cease_continuous_run.is_set():
                    print('top of the while loop')
                    schedule.run_pending()
                    time.sleep(self.interval)
                print('the thread can be killed now.')

        self.continuous_thread = ScheduleThread()
        self.continuous_thread.start()
        return cease_continuous_run
