import cv2, numpy as np
from sklearn.cluster import KMeans

def background_color(img):
  reshape = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape((img.shape[0] * img.shape[1], 3))
  cluster = KMeans(n_clusters=1).fit(reshape)
  return tuple([int(v) for v in cv2.cvtColor(np.array([[cluster.cluster_centers_[0]]]).astype(np.float32), cv2.COLOR_RGB2BGR)[0][0]])