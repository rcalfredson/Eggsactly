cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for i in circles[0, :]:
    # draw the outer circle
    cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
cv2.imshow("detected circles", cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()