from pathlib import Path

class DownloadManager():
    def __init__(self):
        self.sessions = {}

    def addNewSession(self, num_imgs, ts):
        ts = str(ts)
        self.sessions[ts] = {'total_imgs': num_imgs,
                             'imgs_saved': 0,
                             'folder': 'temp/results_ALPHA_%s'%ts}
        Path(self.sessions[ts]['folder']).mkdir(parents=True,
            exist_ok=True)
