# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Shared utilities for writing COCO-format JSON files.

Used by both the detection writer (write_detected_object_set_coco) and
the track writer (write_object_track_set_coco).
"""

import datetime
import json
import os


# Global category mapping shared across all COCO writer instances so
# that detection and track writers in the same pipeline produce
# consistent category IDs.
global_categories = {}


def get_cat_id(dot, categories, category_start_id, use_global):
    """Return the integer category ID for a detected_object_type, creating one if needed."""
    mapping = global_categories if use_global else categories
    return mapping.setdefault(
        dot.get_most_likely_class(),
        len(mapping) + category_start_id,
    )


def detection_to_annotation(det, image_id, categories, category_start_id,
                            use_global):
    """Convert a single detected_object into a COCO annotation dict.

    Returns a dict WITHOUT the 'id' key — callers assign that.
    """
    bbox = det.bounding_box
    d = dict(
        image_id=image_id,
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
        d['segmentation'] = [int(round(p)) for p in polygon]
    if det.type is not None:
        d['category_id'] = get_cat_id(
            det.type, categories, category_start_id, use_global)
    return d


def build_image_list(images, aux_image_labels, aux_image_extensions):
    """Build the 'images' section of the COCO output.

    Each element of *images* is either:
      - a plain filename string  (detection writer), or
      - a dict with at least 'file_name' and optionally 'frame_index',
        'timestamp', 'video_id'.
    """
    has_aux = len(aux_image_extensions) > 0 and aux_image_extensions[0]
    result = []
    for i, im in enumerate(images):
        if isinstance(im, dict):
            entry = dict(id=i, **im)
        else:
            entry = dict(id=i, file_name=im, frame_index=i)
        if has_aux:
            fn = entry.get("file_name", "")
            aux = []
            for label, ext in zip(aux_image_labels, aux_image_extensions):
                base, fext = os.path.splitext(fn)
                aux.append(dict(file_name=base + ext + fext, channels=label))
            entry['auxiliary'] = aux
        result.append(entry)
    return result


def write_coco_json(file_obj, annotations, images, categories,
                    use_global, aux_image_labels, aux_image_extensions,
                    description="Created by VIAME COCO writer",
                    videos=None, tracks=None):
    """Serialize accumulated data to a file in COCO JSON format."""
    now = datetime.datetime.now(datetime.timezone.utc).astimezone()

    mapping = global_categories if use_global else categories
    category_dict = [dict(id=i, name=c) for c, i in mapping.items()]
    image_dict = build_image_list(images, aux_image_labels, aux_image_extensions)

    output = dict(
        info=dict(
            year=now.year,
            description=description,
            date_created=now.replace(microsecond=0).isoformat(' '),
        ),
        annotations=[dict(d, id=i) for i, d in enumerate(annotations)],
        categories=category_dict,
        images=image_dict,
    )
    if videos is not None:
        output['videos'] = videos
    if tracks is not None:
        output['tracks'] = tracks
    json.dump(output, file_obj, indent=2)
    file_obj.flush()
