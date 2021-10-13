from shapely.geometry import LineString, Polygon
import matplotlib.pyplot as plt
import numpy as np


def rotate_point(pt, center, angle):
    print("angle by which to rotate:", 180 * angle / np.pi)
    temp_x = pt[1] - center[1]
    temp_y = pt[0] - center[0]
    print("x and y after reorienting:", temp_x, temp_y)
    rotated_x = temp_x * np.cos(angle) - temp_y * np.sin(angle)
    rotated_y = temp_x * np.sin(angle) + temp_y * np.cos(angle)
    print("after rotating:", rotated_x, rotated_y)
    return (rotated_y + center[0], rotated_x + center[1])


polygon1 = Polygon([(5, 0), (1, 3), (4, 10), (6, 3)])

plt.plot(*polygon1.exterior.xy)
plt.plot([polygon1.centroid.x], [polygon1.centroid.y], "bo")
bnds = polygon1.bounds
bounds_box = Polygon(
    [(bnds[0], bnds[1]), (bnds[0], bnds[3]), (bnds[2], bnds[3]), (bnds[2], bnds[1])]
)
plt.plot(
    *bounds_box.exterior.xy,
    "r",
)
plt.plot([bounds_box.centroid.x], [bounds_box.centroid.y], "ro")
print("centroid of point:", polygon1.centroid)
print("bounds of shape:", polygon1.bounds)
# Y axis marks 0 degrees
print("horiz distance:", (polygon1.centroid.x - bnds[0]))
print("vert distance:", (bnds[3] - polygon1.centroid.y))
angle_to_upper_left = np.arctan2(
    (polygon1.centroid.x - bnds[0]),
    (bnds[3] - polygon1.centroid.y),
)
plt.gca().axis("equal")
print("angle from centroid to upper left corner:", 180 * angle_to_upper_left / np.pi)
for rot_angle in range(360):
    print("rotation angle:", rot_angle)
    rotated_pt_outside_shape = list(
        reversed(
            rotate_point(
                (bnds[3], bnds[0]),
                (
                    polygon1.centroid.y,
                    polygon1.centroid.x,
                ),
                -(rot_angle * np.pi / 180 + angle_to_upper_left),
            )
        )
    )
    intersect_finder_line = LineString(
        [
            (polygon1.centroid.x, polygon1.centroid.y),
            (rotated_pt_outside_shape[0], rotated_pt_outside_shape[1]),
        ]
    )
    # amount by which to rotate would be the actual rotation angle + angle to upper left.
    plt.plot([rotated_pt_outside_shape[1]], [rotated_pt_outside_shape[0]], "go")
plt.show()
