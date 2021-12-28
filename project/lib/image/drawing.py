import cv2
import numpy as np
import os
from pathlib import Path
from PIL import ImageFont
import random

import project.detectors.splinedist.spline_generator as sg
from project.detectors.splinedist.path_helpers import data_dir

phi = np.load(os.path.join(data_dir(), "phi_" + str(8) + ".npy"))

orangeRed = (3, 44, 252, 0)


def loadFont(size):
    return ImageFont.truetype(
        os.path.join(Path(__file__).parent.absolute(), "../../static/fonts/arial.ttf"),
        size,
    )


def get_rand_color(pastel_factor=0.8):
    return [
        (x + pastel_factor) / (1.0 + pastel_factor)
        for x in [random.uniform(0, 1.0) for i in [1, 2, 3]]
    ]


def draw_line(img, point_lists, color=None):
    line_width = 1
    if color == None:
        color = [int(256 * i) for i in reversed(get_rand_color())]
    for line in point_lists:
        pts = np.array(line, dtype=np.int32)
        cv2.polylines(
            img, [pts], True, color, thickness=line_width, lineType=cv2.LINE_AA
        )


def get_interpolated_points(data, n_points=30):
    M = np.shape(data)[2]

    SplineContour = sg.SplineCurveVectorized(
        M, sg.B3(), True, np.transpose(data, [0, 2, 1])
    )
    more_coords = SplineContour.sampleSequential(phi)
    # choose 30 points evenly divided by the range.
    sampling_interval = more_coords.shape[1] // n_points
    sampled_points = more_coords[:, slice(0, more_coords.shape[1], sampling_interval)]
    sampled_points = np.add(sampled_points, 1)
    sampled_points = sampled_points.astype(float).tolist()
    return sampled_points


def rounded_rectangle(
    src,
    top_left,
    bottom_right,
    radius=0.1,
    color=(25, 25, 25),
    thickness=1,
    line_type=cv2.LINE_AA,
):
    # source: https://stackoverflow.com/a/60210706/13312013
    #  corners:
    #  p1 - p2
    #  |     |
    #  p4 - p3

    p1 = top_left
    p2 = (bottom_right[1], top_left[1])
    p3 = (bottom_right[1], bottom_right[0])
    p4 = (top_left[0], bottom_right[0])

    height = abs(bottom_right[0] - top_left[1])

    if radius > 1:
        radius = 1

    corner_radius = int(radius * (height / 2))

    if thickness < 0:

        # big rect
        top_left_main_rect = (int(p1[0] + corner_radius), int(p1[1]))
        bottom_right_main_rect = (int(p3[0] - corner_radius), int(p3[1]))

        top_left_rect_left = (p1[0], p1[1] + corner_radius)
        bottom_right_rect_left = (p4[0] + corner_radius, p4[1] - corner_radius)

        top_left_rect_right = (p2[0] - corner_radius, p2[1] + corner_radius)
        bottom_right_rect_right = (p3[0], p3[1] - corner_radius)

        all_rects = [
            [top_left_main_rect, bottom_right_main_rect],
            [top_left_rect_left, bottom_right_rect_left],
            [top_left_rect_right, bottom_right_rect_right],
        ]

        [cv2.rectangle(src, rect[0], rect[1], color, thickness) for rect in all_rects]

    # draw straight lines
    cv2.line(
        src,
        (p1[0] + corner_radius, p1[1]),
        (p2[0] - corner_radius, p2[1]),
        color,
        abs(thickness),
        line_type,
    )
    cv2.line(
        src,
        (p2[0], p2[1] + corner_radius),
        (p3[0], p3[1] - corner_radius),
        color,
        abs(thickness),
        line_type,
    )
    cv2.line(
        src,
        (p3[0] - corner_radius, p4[1]),
        (p4[0] + corner_radius, p3[1]),
        color,
        abs(thickness),
        line_type,
    )
    cv2.line(
        src,
        (p4[0], p4[1] - corner_radius),
        (p1[0], p1[1] + corner_radius),
        color,
        abs(thickness),
        line_type,
    )

    # draw arcs
    cv2.ellipse(
        src,
        (p1[0] + corner_radius, p1[1] + corner_radius),
        (corner_radius, corner_radius),
        180.0,
        0,
        90,
        color,
        thickness,
        line_type,
    )
    cv2.ellipse(
        src,
        (p2[0] - corner_radius, p2[1] + corner_radius),
        (corner_radius, corner_radius),
        270.0,
        0,
        90,
        color,
        thickness,
        line_type,
    )
    cv2.ellipse(
        src,
        (p3[0] - corner_radius, p3[1] - corner_radius),
        (corner_radius, corner_radius),
        0.0,
        0,
        90,
        color,
        thickness,
        line_type,
    )
    cv2.ellipse(
        src,
        (p4[0] + corner_radius, p4[1] - corner_radius),
        (corner_radius, corner_radius),
        90.0,
        0,
        90,
        color,
        thickness,
        line_type,
    )

    return src
