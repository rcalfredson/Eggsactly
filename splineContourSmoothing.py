# smooth the contour
x,y = cnt.T
# Convert from numpy arrays to normal arrays
# https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
x, y = x[0], y[0]
okay = np.where(np.abs(np.diff(x)) + np.abs(np.diff(y)) > 0)
x = np.r_[x[okay], x[-1], x[0]]
y = np.r_[y[okay], y[-1], y[0]]
x = x.tolist()
y = y.tolist()
#print('x and y passed to splprep:', x, y)
tck, u = splprep([x,y], s=1.0, per=1)
# https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
u_new = np.linspace(u.min(), u.max(), 25)
# https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
x_new, y_new = splev(u_new, tck, der=0)
# Convert it back to numpy format for opencv to be able to display it
res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new,y_new)]
#cnt = np.asarray(res_array, dtype=np.int32)