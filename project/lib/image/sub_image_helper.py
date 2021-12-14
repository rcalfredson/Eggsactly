import cv2
from project.circleFinder import CircleFinder, rotate_image

from project.lib.image.node_based_segmenter import NodeBasedSegmenter


class SubImageHelper:
    def __init__(self) -> None:
        pass

    def get_sub_images(self, img, alignment_data):
        if "nodes" in alignment_data:
            self.segment_image_via_alignment_data(img, alignment_data)
        elif 'bboxes' in alignment_data:
            # do we have access to the rotation angle, scaling, etc? I think...
            self.segment_image_via_bboxes(img, alignment_data)
        # once we get to this point, we should have the sub-images at least,
        # even if the same updates aren't yet being sent to the client,
        # but the messaging will be added later.
        print('have sub-images been calculated?',
            self.subImgs)

    def segment_image_via_alignment_data(self, img, alignment_data):
        segmenter = NodeBasedSegmenter(
            img, alignment_data["nodes"], alignment_data["type"]
        )
        self.subImgs, self.bboxes = segmenter.calc_bboxes_and_subimgs()
        self.rotation_angle = segmenter.rotation_angle

    def segment_image_via_bboxes(self, img, alignment_data):
        img = cv2.resize(
            img,
            (0, 0),
            fx=alignment_data.get("scaling", 1),
            fy=alignment_data.get("scaling", 1),
        )
        img = rotate_image(img, alignment_data["rotationAngle"])
        self.bboxes = alignment_data["bboxes"]
        bbox_translation = [
            -el for el in alignment_data.get("imageTranslation", [0, 0])
        ]
        alignment_data["regionsToIgnore"] = []
        translated_bboxes = []
        for bbox in self.bboxes:
            new_bbox = [
                bbox[0] + bbox_translation[0],
                bbox[1] + bbox_translation[1],
                bbox[2],
                bbox[3],
            ]
            if new_bbox[0] < 0:
                new_bbox[2] += new_bbox[0]
                new_bbox[0] = 0
            if new_bbox[1] < 0:
                new_bbox[2] += new_bbox[1]
                new_bbox[1] = 0
            translated_bboxes.append(list(map(round, new_bbox)))

        self.rotation_angle = alignment_data["rotationAngle"]
        self.bboxes = translated_bboxes
        self.subImgs = CircleFinder.getSubImagesFromBBoxes(
            img, translated_bboxes, alignment_data["regionsToIgnore"]
        )
        # for sub_img in self.subImgs:
        #     cv2.imshow("debug sub-img", sub_img.astype(np.uint8))
        #     print("scaling factor:", alignment_data.get("scaling", 1))
        #     print('sub-image?', sub_img.astype(np.uint8))
        #     cv2.waitKey(0)
