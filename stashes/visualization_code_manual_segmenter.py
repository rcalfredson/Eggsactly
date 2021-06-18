debug_resize_factor = 0.25
im_copy = cv2.resize(
    np.array(self.image).astype(np.uint8),
    (0, 0),
    fx=debug_resize_factor,
    fy=debug_resize_factor,
)
for bbox in bboxes:
    cv2.rectangle(
        im_copy,
        (
            int(bbox[0] * debug_resize_factor),
            int(bbox[1] * debug_resize_factor),
        ),
        (
            int((bbox[0] + bbox[2]) * debug_resize_factor),
            int((bbox[1] + bbox[3]) * debug_resize_factor),
        ),
        (255, 0, 0),
        2,
    )
for div in latitude_divisions:
    cv2.drawMarker(
        im_copy,
        tuple(
            # reversed(
            [int(el * debug_resize_factor) for el in div]
            # )
        ),
        COL_G,
        cv2.MARKER_TILTED_CROSS,
        markerSize=15,
    )
cv2.imshow("debug", im_copy)
cv2.waitKey(0)