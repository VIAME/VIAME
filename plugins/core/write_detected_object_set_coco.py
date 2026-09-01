# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Write detections to COCO/kwcoco-format JSON files.

Emits the shared VIAME COCO profile described in utilities_coco: 1-based ids,
a ``file_name`` and unique ``name`` on every image, ``area``/``iscrowd`` on
every annotation, image-coordinate polygon segmentations, and keypoints in the
kwcoco dict-list format with an auto-generated keypoint_categories table.
Arbitrary per-annotation attributes round-trip from DetectedObject notes.
"""

from kwiver.vital.algo import DetectedObjectSetOutput

from viame.core.utilities_coco import (
    VIDEO_ID,
    global_categories,
    default_video_name,
    detection_to_annotation,
    write_coco_json,
)


class WriteDetectedObjectSetCoco(DetectedObjectSetOutput):
    """
    COCO/kwcoco-formatted output for DetectedObjectSets.

    Writes detections to a JSON file with support for:
    - bbox in [x, y, width, height] format
    - Segmentation masks (as RLE) or single polygons
    - Keypoints (kwcoco dict-list format, with keypoint_categories)
    - Arbitrary annotation attributes (from DetectedObject notes)
    """

    # Kept for backwards compatibility — now delegates to the shared
    # global_categories dict in coco_writer_utils.
    categories = global_categories

    def __init__(self):
        DetectedObjectSetOutput.__init__(self)
        self.detections = []
        self.images = []
        self.category_start_id = 1
        self.top_n_classes = 0
        self.global_categories = True
        self.aux_image_labels = ""
        self.aux_image_extensions = ""
        self.video_name = ""
        self.version_identifier = ""
        self.frame_rate = ""
        self.contributor = ""
        self.file = None
        self._local_categories = {}
        self._output_path = ""
        # Video input names no frame; an image list names every one.
        self._saw_file_name = False

    def get_configuration(self):
        cfg = super(DetectedObjectSetOutput, self).get_configuration()
        cfg.set_value("category_start_id", str(self.category_start_id))
        cfg.set_value("top_n_classes", str(self.top_n_classes))
        cfg.set_value("global_categories", str(self.global_categories))
        cfg.set_value("aux_image_labels", ','.join(self.aux_image_labels))
        cfg.set_value("aux_image_extensions", ','.join(self.aux_image_extensions))
        cfg.set_value("video_name", self.video_name)
        cfg.set_value("version_identifier", self.version_identifier)
        cfg.set_value("frame_rate", self.frame_rate)
        cfg.set_value("contributor", self.contributor)
        return cfg

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)
        self.category_start_id = int(cfg.get_value("category_start_id"))
        self.top_n_classes = int(cfg.get_value("top_n_classes"))
        self.global_categories = self._strtobool(cfg.get_value("global_categories"))
        self.aux_image_labels = str(cfg.get_value("aux_image_labels"))
        self.aux_image_extensions = str(cfg.get_value("aux_image_extensions"))
        self.video_name = str(cfg.get_value("video_name"))
        self.version_identifier = str(cfg.get_value("version_identifier"))
        self.frame_rate = str(cfg.get_value("frame_rate"))
        self.contributor = str(cfg.get_value("contributor"))

        self.aux_image_labels = self.aux_image_labels.rstrip().split(",")
        self.aux_image_extensions = self.aux_image_extensions.rstrip().split(",")

        if len(self.aux_image_labels) != len(self.aux_image_extensions):
            print("Auxiliary image labels and extensions must be same size")
            return False
        if not self.global_categories:
            self._local_categories = {}

    def _strtobool(self, val):
        """Convert a string representation of truth to True or False."""
        val = val.lower()
        if val in ("y", "yes", "t", "true", "on", "1"):
            return True
        elif val in ("n", "no", "f", "false", "off", "0"):
            return False
        else:
            raise ValueError("Invalid truth value %r" % (val,))

    def check_configuration(self, cfg):
        return True

    def open(self, file_name):
        self._output_path = file_name
        self.file = open(file_name, 'w')

    def close(self):
        if self.file:
            self.file.close()

    def write_set(self, detected_object_set, file_name):
        cats = self._local_categories
        # No timestamp reaches this algorithm, so a frame is the Nth
        # write_set call, the same way viame_csv numbers its frame column.
        # Blank frames keep their slot: video names no frame, so dropping
        # them would slide later detections towards the start of the clip.
        frame_index = len(self.images)
        image_id = frame_index + 1
        if file_name:
            self._saw_file_name = True
        for det in detected_object_set:
            d = detection_to_annotation(
                det, image_id, cats,
                self.category_start_id, self.global_categories,
                self.top_n_classes)
            self.detections.append(d)
        self.images.append(dict(file_name=file_name, frame_index=frame_index))

    def _trim_trailing_blank_frames(self):
        """Drop trailing frames carrying neither imagery nor detections.

        Conversion pipelines run without imagery (gt_only) pad the end of the
        stream with nameless empty frames. Interior blanks stay, so frames
        keep the numbering the source file gave them.
        """
        last_used = max(
            (d["image_id"] for d in self.detections), default=0)
        while len(self.images) > last_used and not self.images[-1]["file_name"]:
            self.images.pop()

    def complete(self):
        cats = self._local_categories
        self._trim_trailing_blank_frames()

        videos = None
        if self.video_name or not self._saw_file_name:
            videos = [dict(
                id=VIDEO_ID,
                name=default_video_name(self._output_path, self.video_name))]
            for entry in self.images:
                entry["video_id"] = VIDEO_ID

        write_coco_json(
            self.file,
            self.detections,
            self.images,
            cats,
            self.global_categories,
            self.aux_image_labels, self.aux_image_extensions,
            description="Created by WriteDetectedObjectSetCoco",
            version=self.version_identifier,
            fps=self.frame_rate,
            contributor=self.contributor,
            videos=videos)


def __vital_algorithm_register__():
    from viame.core.vital_registration import register_vital_algorithm

    register_vital_algorithm(
        WriteDetectedObjectSetCoco, "coco", "Write detections to COCO-style JSON format"
    )
