# imgCopy = cv2.resize(np.array(img), (0, 0), fx=0.15, fy=0.15,
#   interpolation=cv2.INTER_CUBIC)
# for every sub-image:
# cv2.drawMarker(imgCopy, (int(getSlice(forHt=False).stop*0.15), int(getSlice().stop*0.15)), COL_G, cv2.MARKER_TILTED_CROSS, 15)
# cv2.drawMarker(imgCopy, (int(getSlice(forHt=False).start*0.15), int(getSlice().start*0.15)), COL_G, cv2.MARKER_TILTED_CROSS, 15)
# cv2.imshow('testinginnerPoint', imgCopy)
# cv2.waitKey(0)

def largest_within_delta_alt(X, k, delta):
  return np.where((k-delta < X) * (X < k))[0].max()

def reject_outliers(data, m=2):
  return data[abs(data - np.mean(data)) < m * np.std(data)]