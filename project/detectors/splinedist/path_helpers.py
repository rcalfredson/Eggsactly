from datetime import datetime
import os
import platform

onload_ts = datetime.now()
dirname = os.path.dirname(__file__)


def data_dir(must_exist=True, as_abs_path=False):
    prospective_dir = os.path.join(
        "data_by_host", f"{platform.node()}_{onload_ts}".replace(":", "-")
    )
    if (must_exist and os.path.isdir(prospective_dir)) or not must_exist:
        return prospective_dir
    elif must_exist and not os.path.isdir(prospective_dir):
        if as_abs_path:
            return os.path.join(dirname, "constants")
        return "./project/detectors/splinedist/constants/"
