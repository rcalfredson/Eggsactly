"""Shared methods"""

import glob, platform, random, string

import cv2
from PIL import Image

IS_WINDOWS = platform.system() == "Windows"


def randID(N=5):
    """Generate uppercase string of alphanumeric characters of length N."""
    return "".join(
        random.SystemRandom().choice(string.ascii_uppercase + string.digits)
        for _ in range(N)
    )


def insensitive_glob(pattern):
    def either(c):
        return "[%s%s]" % (c.lower(), c.upper()) if c.isalpha() else c

    return glob.glob("".join(map(either, pattern)))


def globFiles(dirName, ext="png"):
    """Return files matching the given extension in the given directory."""
    query = dirName + "/*.%s" % ext
    return glob.glob(query) if IS_WINDOWS else insensitive_glob(query)


def X_is_running():
    if IS_WINDOWS:
        return False
    from subprocess import Popen, PIPE

    p = Popen(["xset", "-q"], stdout=PIPE, stderr=PIPE)
    p.communicate()
    return p.returncode == 0


def background_color(img):
    print("image type:", type(img))
    if type(img) != Image.Image:
        pil_img = Image.fromarray(img)
    else:
        pil_img = img
    return max(pil_img.getcolors(pil_img.size[0] * pil_img.size[1]))[1]


def getTextSize(text, size=0.9):
    """Return the height and width of inputted text using Hershey Plain font.

    Arguments:
      - size: text size in OpenCV units (default: 0.9)
    """
    return cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, size, 1)[0][::-1]
