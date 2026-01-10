"""
Utilities for kwimage / vital conversions
"""
import numpy as np
try:
    from kwiver.vital.types import BoundingBoxD
except ImportError:
    from kwiver.vital.types import BoundingBox as BoundingBoxD

from kwiver.vital.types import DetectedObjectSet
from kwiver.vital.types import DetectedObject
from kwiver.vital.types import DetectedObjectType


def kwimage_to_kwiver_detections(detections):
    """
    Convert kwimage detections to kwiver deteted object sets

    TODO: move to a utility module

    Args:
        detected_objects (kwimage.Detections)

    Returns:
        kwiver.vital.types.DetectedObjectSet
    """
    from kwiver.vital.types.types import ImageContainer, Image

    segmentations = None
    # convert segmentation masks
    if 'segmentations' in detections.data:
        segmentations = detections.data['segmentations']

    try:
        boxes = detections.boxes.to_ltrb()
    except Exception:
        boxes = detections.boxes.to_tlbr()

    scores = detections.scores
    class_idxs = detections.class_idxs

    if not segmentations:
        # Placeholders
        segmentations = (None,) * len(boxes)

    # convert to kwiver format, apply threshold
    detected_objects = DetectedObjectSet()

    for tlbr, score, cidx, seg in zip(boxes.data, scores, class_idxs, segmentations):
        class_name = detections.classes[cidx]

        bbox_int = np.round(tlbr).astype(np.int32)
        bounding_box = BoundingBoxD(
            bbox_int[0], bbox_int[1], bbox_int[2], bbox_int[3])

        detected_object_type = DetectedObjectType(class_name, score)
        detected_object = DetectedObject(
            bounding_box, score, detected_object_type)
        if seg:
            mask = seg.to_relative_mask().numpy().data
            detected_object.mask = ImageContainer(Image(mask))

        detected_objects.add(detected_object)
    return detected_objects


def kwiver_to_kwimage_detections(detected_objects):
    """
    Convert vital detected object sets to kwimage.Detections

    TODO: move to a utility module

    Args:
        detected_objects (kwiver.vital.types.DetectedObjectSet)

    Returns:
        kwimage.Detections
    """
    import ubelt as ub
    import kwimage
    boxes = []
    scores = []
    class_idxs = []

    classes = []
    if len(detected_objects) > 0:
        obj = ub.peek(detected_objects)
        try:
            classes = obj.type.all_class_names()
        except AttributeError:
            classes = obj.type().all_class_names()

    for obj in detected_objects:
        try:
            box = obj.bounding_box()
        except TypeError:
            box = obj.bounding_box
        ltrb = [box.min_x(), box.min_y(), box.max_x(), box.max_y()]
        try:
            score = obj.confidence()
            cname = obj.type().get_most_likely_class()
        except TypeError:
            score = obj.confidence
            cname = obj.type.get_most_likely_class()
        cidx = classes.index(cname)
        boxes.append(ltrb)
        scores.append(score)
        class_idxs.append(cidx)

    dets = kwimage.Detections(
        boxes=kwimage.Boxes(np.array(boxes), 'ltrb'),
        scores=np.array(scores),
        class_idxs=np.array(class_idxs),
        classes=classes,
    )
    return dets


def vital_to_kwimage_box(vital_bbox):
    """
    Args:
        vital_bbox (kwiver.vital.types.BoundingBox)

    Returns:
        kwimage.Box
    """
    import kwimage
    bbox = vital_bbox
    xyxy = [bbox.min_x(), bbox.min_y(), bbox.max_x(), bbox.max_y()]
    kw_bbox = kwimage.Box.coerce(xyxy, format='ltrb')
    return kw_bbox
