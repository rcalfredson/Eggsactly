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
1.337,
1.422,
1.325,
1.525,
1.79,
2.005,
1.522,
1.453,
1.401,
2.43,
1.432,
2.935,
2.185,
1.692,
1.903,
1.533,
2.011,
2.029,
1.92,
1.865,
1.724,
2.304,
1.567,
1.454,
1.664,
1.794,
2.011,
1.498,
1.671,
1.477,
1.831,

], asDelta=True))
