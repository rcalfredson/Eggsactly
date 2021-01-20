# for dI, defect in enumerate(convexityDefects):
#   #print('for contour %i, coloring this point green:'%dI, cnt[defect[0][0]][0])
#   #print('shape of contourImg:', contourImg.shape)
#   #print('what is the content of contourImg at this point?', contourImg[cnt[defect[0][0]][0]])
#   #print('trying to get one point from contourImg:', contourImg[0][0])
#   #print('range of data accessed:', cnt[defect[0][0]][0][1]-2, cnt[defect[0][0]][0][1]+2)
#   #print('and for second axis:', cnt[defect[0][0]][0][0]-2, cnt[defect[0][0]][0][0]+2)
#   #print('img data from first axis only:', contourImg[cnt[defect[0][0]][0][1]-2:cnt[defect[0][0]][0][1]+2])
#   #print('shape of data from first axis:', contourImg[cnt[defect[0][0]][0][1]-2:cnt[defect[0][0]][0][1]+2].shape)
#   #print('img data before overwriting:', contourImg[cnt[defect[0][0]][0][1]-2:cnt[defect[0][0]][0][1]+2][:, cnt[defect[0][0]][0][0]-2:cnt[defect[0][0]][0][0]+2])
#   contourImg[cnt[defect[0][0]][0][1]-2:cnt[defect[0][0]][0][1]+2][:, cnt[defect[0][0]][0][0]-2:cnt[defect[0][0]][0][0]+2] = [0, 255, 0]
#   #print('and after overwriting:', contourImg[cnt[defect[0][0]][0][1]-2:cnt[defect[0][0]][0][1]+2][:, cnt[defect[0][0]][0][0]-2:cnt[defect[0][0]][0][0]+2])
#   #print('second (end) point on the contour?', cnt[defect[0][1]][0])
#   #print('start:', cnt[defect[0][0]][0][1], cnt[defect[0][0]][0][0])
#   #print('end:', cnt[defect[0][1]][0][1], cnt[defect[0][1]][0][0])
#   cv2.drawMarker(contourImg, (cnt[defect[0][1]][0][0], cnt[defect[0][1]][0][1]), (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=9)
#   #contourImg[cnt[defect[0][1]][0][1]-2:cnt[defect[0][1]][0][1]+2][:, cnt[defect[0][1]][0][0]-2:cnt[defect[0][1]][0][0]+2] = [0, 0, 255]
