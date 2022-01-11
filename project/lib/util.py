#
# utilities
#
# 4 Aug 2013 by Ulrich Stern
#

import itertools
import numpy as np

# concatenates lists in the given list of lists, returning, e.g., [1, 2, 3] for
#  [[1, 2], [3]]
def concat(l, asIt=False):
    it = itertools.chain.from_iterable(l)
    return it if asIt else list(it)


# returns the distance of two points
def distance(pnt1, pnt2):
    return np.linalg.norm(np.array(pnt1) - pnt2)


def dashed_datetime(dt_in):
    return dt_in.replace("/", "-").replace(" ", "").replace(":", "-").replace(",", "-")


# returns slice objects for all contiguous true regions in the given array
def trueRegions(a):
    r = np.ma.flatnotmasked_contiguous(np.ma.array(a, mask=~a))
    return [] if r is None else r


COL_G = (0, 255, 0)
