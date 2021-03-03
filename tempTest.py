# import numpy as np

from util import *

# arr1 = np.array([3, 1, 11])
# arr2 = np.array([7, 20, 3])
# arr3 = np.array([14, 3, 5])
# unitedArr = np.vstack((arr1, arr2, arr3))

# maxDiffs = np.abs(np.max(unitedArr, axis=0) - np.min(unitedArr, axis=0))
# print('what is argmax?', np.max(unitedArr, axis=0))
# print('max diffs:', maxDiffs)

print(meanConfInt([
0.097,
0.096,
0.12,
0.099,
0.118,
0.107,
0.101,
0.109,
0.108,

], asDelta=True))
