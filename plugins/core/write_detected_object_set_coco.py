# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Write detections to COCO-format JSON files.

The COCO format is a widely used annotation format that contains:
- info: metadata about the dataset
- categories: list of category definitions with 'id' and 'name'
- images: list of image definitions with 'id' and 'file_name'
- annotations: list of annotations with 'id', 'image_id', 'category_id', 'bbox', 'score', and optionally 'segmentation'
"""

from kwiver.vital.algo import DetectedObjectSetOutput

from viame.core.utilities_coco import (
    global_categories,
    detection_to_annotation,
    write_coco_json,
)


class WriteDetectedObjectSetCoco(DetectedObjectSetOutput):
    """
    COCO-formatted output for DetectedObjectSets.

    Writes detections to a JSON file in COCO format with:
    - info: creation metadata
    - categories: list of {id, name} category definitions
    - images: list of {id, file_name} image definitions
    - annotations: list of {id, image_id, category_id, bbox, score, segmentation?}

    The bbox is written in [x, y, width, height] format.
    """

    # Kept for backwards compatibility — now delegates to the shared
    # global_categories dict in coco_writer_utils.
    categories = global_categories

    def __init__(self):
        DetectedObjectSetOutput.__init__(self)
        self.detections = []
        self.images = []
        self.category_start_id = 1
        self.global_categories = True
        self.aux_image_labels = ""
        self.aux_image_extensions = ""
        self.file = None
        self._local_categories = {}

    def get_configuration(self):
        cfg = super(DetectedObjectSetOutput, self).get_configuration()
        cfg.set_value("category_start_id", str(self.category_start_id))
        cfg.set_value("global_categories", str(self.global_categories))
        cfg.set_value("aux_image_labels", ','.join(self.aux_image_labels))
        cfg.set_value("aux_image_extensions", ','.join(self.aux_image_extensions))
        return cfg

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)
        self.category_start_id = int(cfg.get_value("category_start_id"))
        self.global_categories = self._strtobool(cfg.get_value("global_categories"))
        self.aux_image_labels = str(cfg.get_value("aux_image_labels"))
        self.aux_image_extensions = str(cfg.get_value("aux_image_extensions"))

        self.aux_image_labels = self.aux_image_labels.rstrip().split(',')
        self.aux_image_extensions = self.aux_image_extensions.rstrip().split(',')

        if len(self.aux_image_labels) != len(self.aux_image_extensions):
            print("Auxiliary image labels and extensions must be same size")
            return False
        if not self.global_categories:
            self._local_categories = {}

    def _strtobool(self, val):
        """Convert a string representation of truth to True or False."""
        val = val.lower()
        if val in ('y', 'yes', 't', 'true', 'on', '1'):
            return True
        elif val in ('n', 'no', 'f', 'false', 'off', '0'):
            return False
        else:
            raise ValueError("Invalid truth value %r" % (val,))

    def check_configuration(self, cfg):
        return True

    def open(self, file_name):
        self.file = open(file_name, 'w')

    def close(self):
        if self.file:
            self.file.close()

    def write_set(self, detected_object_set, file_name):
        cats = self._local_categories
        for det in detected_object_set:
            d = detection_to_annotation(
                det, len(self.images), cats,
                self.category_start_id, self.global_categories)
            self.detections.append(d)
        self.images.append(file_name)

    def complete(self):
        cats = self._local_categories
        write_coco_json(
            self.file, self.detections, self.images, cats,
            self.global_categories,
            self.aux_image_labels, self.aux_image_extensions,
            description="Created by WriteDetectedObjectSetCoco")


def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    implementation_name = "coco"

    if algorithm_factory.has_algorithm_impl_name(
            WriteDetectedObjectSetCoco.static_type_name(),
            implementation_name,
    ):
        return

    algorithm_factory.add_algorithm(
        implementation_name,
        "Write detections to COCO-style JSON format",
        WriteDetectedObjectSetCoco,
    )

    algorithm_factory.mark_algorithm_as_loaded(implementation_name)
