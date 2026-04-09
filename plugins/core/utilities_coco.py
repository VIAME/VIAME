# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Shared utilities for reading and writing COCO/kwcoco-format JSON files.

Used by both the detection and track readers/writers.  Supports:

- Segmentation masks (RLE), single polygons, multi-polygons, and
  kwcoco exterior/interiors polygons with holes
- Keypoints in COCO flat, kwcoco dict-list, and kwcoco column formats
- Arbitrary per-annotation and per-track attributes (round-tripped
  through DetectedObject notes as JSON)
"""

import datetime
import json
import os

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


# Global category mapping shared across all COCO writer instances so
# that detection and track writers in the same pipeline produce
# consistent category IDs.
global_categories = {}

# Standard COCO annotation keys — not treated as custom attributes.
_STANDARD_ANNOTATION_KEYS = frozenset({
    'id', 'image_id', 'category_id', 'bbox', 'score', 'segmentation',
    'keypoints', 'track_id', 'area', 'iscrowd',
})


# ------------------------------------------------------------------
# Category helpers
# ------------------------------------------------------------------

def get_cat_id(dot, categories, category_start_id, use_global):
    """Return the integer category ID for a detected_object_type, creating one if needed."""
    mapping = global_categories if use_global else categories
    return mapping.setdefault(
        dot.get_most_likely_class(),
        len(mapping) + category_start_id,
    )


# ------------------------------------------------------------------
# Mask / RLE helpers
# ------------------------------------------------------------------

def mask_to_rle(mask_array):
    """Convert a 2D binary mask (numpy uint8) to COCO RLE format.

    Uses column-major (Fortran) order per the COCO convention.
    Returns a dict with ``counts`` (list of ints) and ``size`` [h, w].
    """
    flat = mask_array.flatten(order='F').astype(np.uint8)
    padded = np.concatenate([[0], flat, [0]])
    changes = np.where(np.diff(padded) != 0)[0]
    lengths = np.diff(changes).tolist()
    # RLE starts with a run of 0s; if the mask starts with 1 prepend a
    # zero-length run.
    if flat[0] == 1:
        lengths = [0] + lengths
    return {'counts': lengths, 'size': list(mask_array.shape)}


def rle_to_mask(rle):
    """Convert a COCO RLE dict to a 2D binary numpy mask.

    Supports both integer-array counts and pycocotools-style LEB128
    string counts.
    """
    h, w = rle['size']
    counts = rle['counts']
    if isinstance(counts, str):
        counts = _decode_rle_string(counts)
    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    for i, c in enumerate(counts):
        if i % 2 == 1:  # odd runs are foreground
            flat[pos:pos + c] = 1
        pos += c
    return flat.reshape((h, w), order='F')


def _decode_rle_string(s):
    """Decode a pycocotools-style LEB128 RLE string to integer counts."""
    counts = []
    p = 0
    while p < len(s):
        x = 0
        shift = 0
        more = True
        while more:
            c = ord(s[p]) - 48
            p += 1
            x |= (c & 0x1f) << shift
            shift += 5
            more = (c & 0x20) != 0
        if x & 1:
            x = -(x >> 1)
        else:
            x = x >> 1
        counts.append(x)
    # Undo differential encoding
    for i in range(1, len(counts)):
        counts[i] += counts[i - 1]
    return counts


def polygons_to_mask(segmentation, height, width):
    """Rasterize a COCO/kwcoco polygon segmentation to a binary mask.

    Handles plain polygon coordinate lists and kwcoco ``{exterior,
    interiors}`` dicts.  Requires OpenCV; returns *None* if cv2 is not
    available.
    """
    if not _HAS_CV2:
        return None
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in segmentation:
        if isinstance(poly, dict):
            ext = np.array(poly['exterior'], dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [ext], 1)
            for hole in poly.get('interiors', []):
                hole_pts = np.array(hole, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [hole_pts], 0)
        elif isinstance(poly, list):
            pts = np.array(poly, dtype=np.float64).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
    return mask


# ------------------------------------------------------------------
# Writer: DetectedObject -> COCO annotation dict
# ------------------------------------------------------------------

def detection_to_annotation(det, image_id, categories, category_start_id,
                            use_global):
    """Convert a single detected_object into a COCO annotation dict.

    Returns a dict WITHOUT the ``id`` key — callers assign that.

    Handles:

    - Segmentation masks (written as RLE) and single polygons
    - Keypoints (written in kwcoco dict-list format)
    - Custom attributes from notes (JSON-encoded dicts are unpacked as
      top-level annotation keys; plain strings go into a ``notes`` list)
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

    # Segmentation: prefer mask -> RLE, fall back to polygon
    mask = det.mask
    if mask is not None:
        mask_arr = mask.asarray()
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[:, :, 0]
        d['segmentation'] = mask_to_rle((mask_arr > 0).astype(np.uint8))
    else:
        polygon = det.get_flattened_polygon()
        if polygon:
            d['segmentation'] = [[int(round(p)) for p in polygon]]

    # Keypoints (kwcoco dict-list format)
    keypoints = det.keypoints
    if keypoints:
        kp_list = []
        for name, pt in keypoints.items():
            v = pt.value
            kp_list.append(dict(
                xy=[float(v[0]), float(v[1])],
                keypoint_category=str(name),
                visible=2,
            ))
        d['keypoints'] = kp_list

    # Category
    if det.type is not None:
        d['category_id'] = get_cat_id(
            det.type, categories, category_start_id, use_global)

    # Custom attributes from notes
    notes = det.notes
    if notes:
        plain_notes = []
        for note in notes:
            try:
                attrs = json.loads(note)
                if isinstance(attrs, dict):
                    for k, v in attrs.items():
                        if k not in _STANDARD_ANNOTATION_KEYS and k not in d:
                            d[k] = v
                    continue
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
            plain_notes.append(note)
        if plain_notes:
            d['notes'] = plain_notes

    return d


# ------------------------------------------------------------------
# Reader: COCO annotation dict -> DetectedObject
# ------------------------------------------------------------------

def annotation_to_detection(ann, categories, image_dims=None,
                            kp_cat_names=None):
    """Convert a COCO annotation dict to a DetectedObject.

    Parameters
    ----------
    ann : dict
        A single COCO annotation.
    categories : dict
        Mapping of ``category_id`` -> ``category_name``.
    image_dims : tuple of (width, height), optional
        Image dimensions used when rasterizing multi-polygon
        segmentations to masks.
    kp_cat_names : list of str, optional
        Keypoint category names ordered by ID, for decoding COCO
        flat-format keypoints.
    """
    import kwiver.vital.types as vt

    x, y, w, h = ann['bbox']
    score = ann.get('score', 1.0)

    det = vt.DetectedObject(
        bbox=vt.BoundingBoxD(x, y, x + w, y + h),
        confidence=score,
    )

    # Category
    cat_id = ann.get('category_id')
    if cat_id is not None and cat_id in categories:
        det.type = vt.DetectedObjectType(categories[cat_id], score)

    # Segmentation
    if 'segmentation' in ann:
        _apply_segmentation(det, ann['segmentation'], image_dims)

    # Keypoints
    if 'keypoints' in ann:
        _apply_keypoints(det, ann['keypoints'], kp_cat_names)

    # Custom attributes -> notes
    custom = {}
    for k, v in ann.items():
        if k in _STANDARD_ANNOTATION_KEYS:
            continue
        if k == 'notes':
            if isinstance(v, list):
                for note in v:
                    det.add_note(str(note))
            else:
                det.add_note(str(v))
        else:
            custom[k] = v
    if custom:
        det.add_note(json.dumps(custom))

    return det


def _apply_segmentation(det, seg, image_dims=None):
    """Apply segmentation data to a DetectedObject."""
    import kwiver.vital.types as vt

    if isinstance(seg, dict):
        # RLE mask
        mask_arr = rle_to_mask(seg)
        det.mask = vt.ImageContainer(mask_arr)
        return

    if not isinstance(seg, list) or not seg:
        return

    # Flat list of numbers -> single polygon (non-standard but common)
    if isinstance(seg[0], (int, float)):
        if len(seg) % 2 == 0:
            det.set_flattened_polygon([float(v) for v in seg])
        return

    # List of items: polygon lists or exterior/interior dicts
    has_dict = any(isinstance(p, dict) for p in seg)
    is_multi = len(seg) > 1 or has_dict

    if is_multi:
        # Multi-polygon or polygon-with-holes: rasterize to mask
        if image_dims and _HAS_CV2:
            iw, ih = image_dims
            mask_arr = polygons_to_mask(seg, ih, iw)
            if mask_arr is not None:
                det.mask = vt.ImageContainer(mask_arr)
        # Also set first plain polygon as flattened polygon
        for p in seg:
            if isinstance(p, list) and len(p) % 2 == 0:
                det.set_flattened_polygon([float(v) for v in p])
                break
    elif len(seg) == 1 and isinstance(seg[0], list):
        # Single polygon: [[x1,y1,...]]
        poly = seg[0]
        if len(poly) % 2 == 0:
            det.set_flattened_polygon([float(v) for v in poly])


def _apply_keypoints(det, kps, kp_cat_names=None):
    """Apply keypoint data to a DetectedObject."""
    import kwiver.vital.types as vt

    if isinstance(kps, list):
        if not kps:
            return
        if isinstance(kps[0], dict):
            # kwcoco dict-list format
            for kp in kps:
                xy = kp.get('xy', [0, 0])
                name = kp.get('keypoint_category',
                              str(kp.get('keypoint_category_id', '')))
                det.add_keypoint(str(name),
                                 vt.Point2d(float(xy[0]), float(xy[1])))
        elif isinstance(kps[0], (int, float)):
            # COCO flat format: [x1,y1,v1, x2,y2,v2, ...]
            for i in range(0, len(kps) - 2, 3):
                idx = i // 3
                if kp_cat_names and idx < len(kp_cat_names):
                    name = kp_cat_names[idx]
                else:
                    name = str(idx)
                det.add_keypoint(
                    name,
                    vt.Point2d(float(kps[i]), float(kps[i + 1])))
    elif isinstance(kps, dict):
        # kwcoco column format: {x: [...], y: [...], ...}
        xs = kps.get('x', [])
        ys = kps.get('y', [])
        names = kps.get('keypoint_category',
                        [str(i) for i in range(len(xs))])
        for i, (kx, ky) in enumerate(zip(xs, ys)):
            name = names[i] if i < len(names) else str(i)
            det.add_keypoint(str(name), vt.Point2d(float(kx), float(ky)))


# ------------------------------------------------------------------
# Image list builder
# ------------------------------------------------------------------

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


def _collect_keypoint_categories(annotations):
    """Extract unique keypoint category names from annotation dicts.

    Returns a list of ``{id, name}`` dicts, or *None* if no keypoints
    are present.
    """
    names = {}
    for ann in annotations:
        for kp in ann.get('keypoints', []):
            if isinstance(kp, dict):
                name = kp.get('keypoint_category', '')
                if name and name not in names:
                    names[name] = len(names) + 1
    if not names:
        return None
    return [dict(id=i, name=n) for n, i in names.items()]


# ------------------------------------------------------------------
# JSON serialisation
# ------------------------------------------------------------------

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

    kp_cats = _collect_keypoint_categories(annotations)
    if kp_cats is not None:
        output['keypoint_categories'] = kp_cats

    json.dump(output, file_obj, indent=2)
    file_obj.flush()
