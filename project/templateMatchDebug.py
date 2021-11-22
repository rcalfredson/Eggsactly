import cv2
sourceImg = cv2.imread('C:\\Users\\Tracking\\counting-3\\imgs\\Charlene\\9_11_2020_IMG_0008.jpg')
templateImg = cv2.imread('C:\\Users\\Tracking\\coco-annotator\\datasets\\eggs\\9_11_2020_IMG_0008_8.jpg')
w, h = templateImg.shape[2:0:-1]

res = cv2.matchTemplate(sourceImg, templateImg, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(sourceImg, top_left, bottom_right, 255, 2)
cv2.imshow('source', cv2.resize(sourceImg, (0, 0), fx=0.15, fy=0.15))
cv2.imshow('template', templateImg)
cv2.waitKey(0)
print('origin of match:', top_left)