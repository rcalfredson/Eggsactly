import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import matplotlib

npol, nvrts = 30, 5
cnts = 100 * (np.random.random((npol,2)) - 0.5)
print('what are cnts?', cnts)
offs = 10 * (np.random.random((nvrts,npol,2)) - 0.5)
print('what are offs?', offs)
vrts = cnts + offs
vrts = np.swapaxes(vrts, 0, 1)

z = np.random.random(npol) * 500

# fig, ax = plt.subplots()
# coll = PolyCollection(vrts, array=z, cmap=matplotlib.cm.jet)
# ax.add_collection(coll)
# ax.autoscale()
# plt.show()