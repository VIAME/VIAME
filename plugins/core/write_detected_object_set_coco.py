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

import datetime
import json
import os

from kwiver.vital.algo import DetectedObjectSetOutput


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

    # ID mappings of categories such that they're shared across all
    # WriteDetectedObjectSetCoco class instantiations in the running instance
    categories = {}

    def __init__(self):
        DetectedObjectSetOutput.__init__(self)
        # List of dicts corresponding to elements of the output
        # "annotations" attribute, minus the "id" attribute
        self.detections = []
        # List of image paths
        self.images = []
        # The first ID to be assigned to a category (and then counting
        # up from there)
        self.category_start_id = 1
        # Have consistent category ids across multiple coco writers
        # within the same program
        self.global_categories = True
        # Optional auxiliary image information to write out to json file
        self.aux_image_labels = ""
        self.aux_image_extensions = ""
        # The current file object or None
        self.file = None

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
            WriteDetectedObjectSetCoco.categories = {}

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
        for det in detected_object_set:
            bbox = det.bounding_box
            d = dict(
                image_id=len(self.images),
                bbox=[
                    bbox.min_x(),
                    bbox.min_y(),
                    bbox.width(),
                    bbox.height(),
                ],
                score=det.confidence,
            )
            polygon = det.get_flattened_polygon()
            if polygon:
                # Downstream applications expect ints, not floats
                d['segmentation'] = [int(round(p)) for p in polygon]
            if det.type is not None:
                d['category_id'] = self.get_cat_id(det.type)
            self.detections.append(d)
        self.images.append(file_name)

    def get_cat_id(self, dot):
        if self.global_categories:
            return type(self).categories.setdefault(
                dot.get_most_likely_class(),
                len(type(self).categories) + self.category_start_id)
        else:
            return self.categories.setdefault(
                dot.get_most_likely_class(),
                len(type(self).categories) + self.category_start_id)

    def fill_aux(self, file_name):
        output = []
        for label, aug_ext in zip(self.aux_image_labels, self.aux_image_extensions):
            file_name_base, file_ext = os.path.splitext(file_name)
            adj_file_name = file_name_base + aug_ext + file_ext
            output.append(dict(file_name=adj_file_name, channels=label))
        return output

    def complete(self):
        now = datetime.datetime.now(datetime.timezone.utc).astimezone()
        if len(self.aux_image_extensions) > 0 and self.aux_image_extensions[0]:
            image_dict = [dict(id=i, file_name=im, auxillary=self.fill_aux(im))
                          for i, im in enumerate(self.images)]
        else:
            image_dict = [dict(id=i, file_name=im)
                          for i, im in enumerate(self.images)]
        if self.global_categories:
            category_dict = [dict(id=i, name=c)
                             for c, i in type(self).categories.items()]
        else:
            category_dict = [dict(id=i, name=c)
                             for c, i in self.categories.items()]
        json.dump(dict(
            info=dict(
                year=now.year,
                description="Created by WriteDetectedObjectSetCoco",
                date_created=now.replace(microsecond=0).isoformat(' '),
            ),
            annotations=[dict(d, id=i)
                         for i, d in enumerate(self.detections)],
            categories=category_dict,
            images=image_dict,
        ), self.file, indent=2)
        self.file.flush()


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
