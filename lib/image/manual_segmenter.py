from util import COL_G, distance, toInt
from circleFinder import rotate_around_point_highperf, rotate_image, subImagesFromBBoxes
import cv2
import math
from chamber import CT
import numpy as np


class ManualSegmenter:
    def __init__(self, image, alignment_data, chamber_type: str):
        self.image = image
        self.alignment_data = alignment_data
        self.chamber_type = CT[chamber_type]
        self.line_types = [f"{tp}_line" for tp in ("horiz", "vert")]

    def calc_line_slopes(self, from_points=False):
        def calc_slope(line):
            return (line["end"][1] - line["start"][1]) / (
                line["end"][0] - line["start"][0]
            )

        if from_points:
            for line_type in self.line_types:
                line = getattr(self, line_type)
                try:
                    getattr(self, line_type)["slope"] = calc_slope(line["points"])
                except ZeroDivisionError:
                    getattr(self, line_type)["slope"] = np.infty
        else:
            slopes = []
            for line in self.alignment_data:
                try:
                    slopes.append(calc_slope(line))
                except ZeroDivisionError:
                    slopes.append(np.infty)
            return slopes

    def determine_vert_and_horiz_lines(self):
        slopes = []
        for line in self.alignment_data:
            try:
                slopes.append(
                    (line["end"][1] - line["start"][1])
                    / (line["end"][0] - line["start"][0])
                )
            except ZeroDivisionError:
                slopes.append(np.infty)
        abs_slopes = [abs(slope) for slope in slopes]
        min_slope_index = abs_slopes.index(min(abs_slopes))
        max_slope_index = abs_slopes.index(max(abs_slopes))
        self.horiz_line = {
            "points": self.alignment_data[min_slope_index],
            "slope": slopes[min_slope_index],
        }
        self.vert_line = {
            "points": self.alignment_data[max_slope_index],
            "slope": slopes[max_slope_index],
        }

    def calc_rotation_angle(self):
        self.rotation_angle = 0.5 * (
            math.atan(self.horiz_line["slope"]) - math.atan(1 / self.vert_line["slope"])
        )

    def rotate_img_and_lines(self):
        image_origin = tuple(np.array(self.image.shape[1::-1]) / 2)
        self.image = rotate_image(self.image, self.rotation_angle)
        for line_type in self.line_types:
            orig_data = getattr(self, line_type)
            setattr(
                self,
                line_type,
                {
                    "points": {
                        k: rotate_around_point_highperf(
                            orig_data["points"][k], self.rotation_angle, image_origin
                        )
                        for k in orig_data["points"]
                    }
                },
            )
        self.calc_line_slopes(from_points=True)

    @staticmethod
    def interpolate(start_pt, end_pt, proportion):
        x_dist = end_pt[0] - start_pt[0]
        y_dist = end_pt[1] - start_pt[1]
        return [start_pt[0] + x_dist * proportion, start_pt[1] + y_dist * proportion]

    def calc_px_to_mm_ratio(self):
        if self.chamber_type == CT.opto:
            line_to_measure = self.horiz_line
        elif self.chamber_type in (CT.sixByFour, CT.fiveByThree):
            line_to_measure = self.vert_line
        elif self.chamber_type == CT.large:
            raise NotImplementedError
        self.px_to_mm = (
            distance(
                line_to_measure["points"]["start"], line_to_measure["points"]["end"]
            )
            / self.chamber_type.value().dist_along_agarose
        )

    def calculate_distance_from_origin_to_point(self, dist_type):
        return self.interpolate(
            self.latitudinal_line["points"]["start"],
            self.latitudinal_line["points"]["end"],
            getattr(self.chamber_type.value(), dist_type)
            / self.total_ideal_latitudinal_dist,
        )

    def calculate_chamber_distance_along_line(self, dist_type):
        point_rel_to_origin = self.calculate_distance_from_origin_to_point(dist_type)
        return [
            point_rel_to_origin[0] - self.latitudinal_line["points"]["start"][0],
            point_rel_to_origin[1] - self.latitudinal_line["points"]["start"][1],
        ]

    @staticmethod
    def scale_pt(pt, multiplier):
        return [el * multiplier for el in pt]

    @staticmethod
    def add_pts(pt1, pt2):
        return [pt1[0] + pt2[0], pt1[1] + pt2[1]]

    @staticmethod
    def subtract_pts(pt1, pt2):
        return [pt1[0] - pt2[0], pt1[1] - pt2[1]]

    @staticmethod
    def list_to_int(list_in):
        return [int(el) for el in list_in]

    def divide_img(self):
        if self.chamber_type == CT.large:
            self.divide_img_large_CT()
            return
        latitude_divisions = []
        if self.chamber_type == CT.opto:
            self.latitudinal_line = self.vert_line
            self.longitudinal_line = self.horiz_line
        elif self.chamber_type in (CT.sixByFour, CT.fiveByThree):
            self.latitudinal_line = self.horiz_line
            self.longitudinal_line = self.vert_line

        if self.chamber_type == CT.opto:
            num_latitude_divs = self.chamber_type.value().numRows
            num_longitude_divs = self.chamber_type.value().numCols

        else:
            num_latitude_divs = self.chamber_type.value().numCols
            num_longitude_divs = self.chamber_type.value().numRows

        self.total_ideal_latitudinal_dist = (
            num_latitude_divs * self.chamber_type.value().dist_across_arena
            + (num_latitude_divs - 1) * self.chamber_type.value().dist_between_arenas
        )
        across_arena_delta = self.calculate_chamber_distance_along_line(
            "dist_across_arena"
        )
        between_arenas_delta = self.calculate_chamber_distance_along_line(
            "dist_between_arenas"
        )

        latitude_divisions.append(self.latitudinal_line["points"]["start"])
        if self.chamber_type == CT.opto:
            line_start = self.longitudinal_line["points"]["start"]
            line_end = self.longitudinal_line["points"]["end"]
        else:
            line_start = self.interpolate(
                self.longitudinal_line["points"]["start"],
                self.longitudinal_line["points"]["end"],
                self.chamber_type.value().dist_trough_to_first_arena
                / self.chamber_type.value().dist_along_agarose,
            )
            line_end = self.interpolate(
                self.longitudinal_line["points"]["start"],
                self.longitudinal_line["points"]["end"],
                1
                - (
                    self.chamber_type.value().dist_trough_to_first_arena
                    / self.chamber_type.value().dist_along_agarose
                ),
            )
            latitude_divisions[-1] = self.add_pts(
                latitude_divisions[-1],
                self.subtract_pts(
                    line_start, self.longitudinal_line["points"]["start"]
                ),
            )

        for i in range(num_latitude_divs):
            latitude_divisions.append(
                [
                    latitude_divisions[-1][0] + across_arena_delta[0],
                    latitude_divisions[-1][1] + across_arena_delta[1],
                ]
            )
            if i < num_latitude_divs - 1:
                latitude_divisions.append(
                    [
                        latitude_divisions[-1][0] + between_arenas_delta[0],
                        latitude_divisions[-1][1] + between_arenas_delta[1],
                    ]
                )
        deltas_from_horiz_start_point = []

        for i in range(num_longitude_divs):

            start = line_start
            end = self.interpolate(
                start,
                line_end,
                (i + 1) / num_longitude_divs,
            )
            deltas_from_horiz_start_point.append([end[0] - start[0], end[1] - start[1]])
        orig_lat_divisions = list(latitude_divisions)
        for d in deltas_from_horiz_start_point:
            for v in orig_lat_divisions:
                latitude_divisions.append([v[0] + d[0], v[1] + d[1]])

        row_col_shape = (2 * num_latitude_divs, num_longitude_divs + 1)
        if self.chamber_type != CT.opto:
            row_col_shape = tuple(reversed(row_col_shape))
        grid_vertices = np.reshape(
            np.array(latitude_divisions),
            row_col_shape + (2,),
            order="F" if self.chamber_type == CT.opto else "C",
        )
        bboxes = []
        if self.chamber_type == CT.opto:
            outer_range = grid_vertices.shape[1] - 1
            inner_range = num_latitude_divs
        else:
            outer_range = num_latitude_divs
            inner_range = grid_vertices.shape[0] - 1
        for i in range(outer_range):
            for j in range(inner_range):
                agarose_width = self.chamber_type.value().agarose_width * self.px_to_mm
                outward_buffer = 0.6 * self.px_to_mm
                if self.chamber_type == CT.opto:
                    x_delta = grid_vertices[j, i + 1][0] - grid_vertices[j, i][0]
                    inward_buffer = 0.6 * self.px_to_mm
                    bbox_1_xmin = grid_vertices[j * 2, i][0] - outward_buffer
                    bbox_1_ymin = grid_vertices[j * 2, i][1]
                    bbox_1_width = x_delta
                    bbox_1_height = agarose_width + inward_buffer
                    bbox_2_xmin = grid_vertices[j, i][0]
                    bbox_2_ymin = (
                        grid_vertices[j * 2 + 1, i][1]
                        - agarose_width
                        - 0.5 * inward_buffer
                    )
                    bbox_2_width = x_delta
                    bbox_2_height = agarose_width + outward_buffer
                else:
                    inward_buffer = 2.5 * self.px_to_mm
                    y_delta = grid_vertices[j + 1, i][1] - grid_vertices[j, i][1]
                    bbox_1_xmin = grid_vertices[j, i * 2][0] + 3.5 * self.px_to_mm
                    bbox_1_ymin = grid_vertices[j, i * 2][1]
                    bbox_1_width = agarose_width - 2.7 * self.px_to_mm
                    bbox_1_height = y_delta
                    bbox_2_xmin = grid_vertices[j, i * 2 + 1][0] - agarose_width
                    bbox_2_ymin = grid_vertices[j, i * 2][1]
                    bbox_2_width = agarose_width - 2.7* self.px_to_mm
                    bbox_2_height = y_delta

                bboxes.append(
                    self.list_to_int(
                        [
                            bbox_1_xmin,
                            bbox_1_ymin,
                            bbox_1_width,
                            bbox_1_height,
                        ]
                    )
                )
                bboxes.append(
                    self.list_to_int(
                        [
                            bbox_2_xmin,
                            bbox_2_ymin,
                            bbox_2_width,
                            bbox_2_height,
                        ]
                    )
                )
        nr = self.chamber_type.value().numRows
        nc = self.chamber_type.value().numCols

        if self.chamber_type.value() == CT.opto:
            bboxes = np.reshape(np.array(bboxes), (nc, nr * 2, -1))
            bboxes = np.transpose(bboxes, (1, 0, 2))
            bboxes = np.reshape(bboxes, (2 * nr * nc, -1))
        else:
            bboxes_old = list(bboxes)
            bboxes = []
            offset = 2
            for i in range(nr):
                for j in range(nc):
                    index = i * offset + 2 * nr * j
                    bboxes.append(bboxes_old[index])
                    bboxes.append(bboxes_old[index + 1])
        self.bboxes = bboxes
        self.sub_imgs = subImagesFromBBoxes(self.image, self.bboxes)

    def ensure_start_pt_closer_to_origin(self):
        for line_type in self.line_types:
            dists_from_origin = [
                distance([0, 0], getattr(self, line_type)["points"][pos])
                for pos in ("start", "end")
            ]
            if dists_from_origin.index(min(dists_from_origin)) == 1:
                old_start = getattr(self, line_type)["points"]["start"]
                getattr(self, line_type)["points"]["start"] = getattr(self, line_type)[
                    "points"
                ]["end"]
                getattr(self, line_type)["points"]["end"] = old_start

    def divide_img_large_CT(self):
        pass

    def calc_bboxes_and_subimgs(self):
        self.determine_vert_and_horiz_lines()
        self.calc_rotation_angle()
        self.ensure_start_pt_closer_to_origin()
        self.calc_px_to_mm_ratio()
        self.rotate_img_and_lines()
        self.divide_img()
        return self.sub_imgs, self.bboxes
