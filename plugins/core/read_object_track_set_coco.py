# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Read object tracks from COCO/kwcoco-format JSON files.

Supports standard COCO annotations with an optional ``track_id`` field
on each annotation and optional top-level ``videos`` and ``tracks``
tables.  Annotations without a ``track_id`` are each placed into their
own single-state track.

Additional kwcoco features supported:
- Segmentation: single polygon, multi-polygon, exterior/interiors,
  and RLE masks
- Keypoints: COCO flat, kwcoco dict-list, kwcoco column formats
- Arbitrary per-annotation attributes (round-tripped as JSON notes)

Two read modes are supported (configured via ``batch_load``):
- **batch** (default): the first ``read_set`` call returns every track
  in a single ObjectTrackSet; subsequent calls signal EOF.
- **streaming**: each ``read_set`` call returns the tracks active on
  the next frame, ordered by ``frame_index`` (or image ``id``).
"""

import json

from kwiver.vital.algo import ReadObjectTrackSet
import kwiver.vital.types as vt

from viame.core.utilities_coco import annotation_to_detection


class ReadObjectTrackSetCoco(ReadObjectTrackSet):
    """
    Object track set reader for COCO-format JSON files.

    Reads JSON with:
    - categories: list of {id, name}
    - images: list of {id, file_name, frame_index?, timestamp?, video_id?}
    - annotations: list of {image_id, category_id, bbox, score?, segmentation?,
      track_id?}
    - tracks (optional): list of {id, name}

    The bbox is in [x, y, width, height] format.
    """

    def __init__(self):
        ReadObjectTrackSet.__init__(self)
        self.batch_load = True
        self.video_name = ""
        self.file = None
        self.loaded = False

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def get_configuration(self):
        cfg = super(ReadObjectTrackSet, self).get_configuration()
        cfg.set_value("batch_load", str(self.batch_load))
        cfg.set_value("video_name", self.video_name)
        return cfg

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)
        self.batch_load = _strtobool(cfg.get_value("batch_load"))
        self.video_name = str(cfg.get_value("video_name"))

    def check_configuration(self, cfg):
        return True

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def open(self, file_name):
        self.file = open(file_name)
        self.loaded = False

    def close(self):
        if self.file:
            self.file.close()

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def read_set(self):
        self._ensure_loaded()

        if self.batch_load:
            if self._batch_returned:
                return None
            self._batch_returned = True
            tracks = list(self._all_tracks.values())
            return vt.ObjectTrackSet(tracks)

        if self._stream_idx >= len(self._frame_order):
            return None

        frame_id = self._frame_order[self._stream_idx]
        self._stream_idx += 1
        track_ids = self._tracks_by_frame.get(frame_id, [])

        tracks = []
        seen = set()
        for tid in track_ids:
            if tid not in seen:
                seen.add(tid)
                tracks.append(self._all_tracks[tid])

        return vt.ObjectTrackSet(tracks)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_loaded(self):
        if self.loaded:
            return

        data = json.load(self.file)

        categories = {cat['id']: cat['name'] for cat in data.get('categories', [])}

        # Resolve video_id filter from video_name config
        filter_video_id = None
        if self.video_name:
            for vid in data.get('videos', []):
                if vid.get('name') == self.video_name:
                    filter_video_id = vid['id']
                    break

        # Build image info lookup: image_id -> (frame_index, timestamp)
        # Use frame_index if present, otherwise fall back to image id
        image_info = {}
        image_dims = {}
        for im in data.get('images', []):
            if filter_video_id is not None:
                if im.get('video_id') != filter_video_id:
                    continue
            img_id = im['id']
            frame_index = im.get('frame_index', img_id)
            timestamp = im.get('timestamp', None)
            image_info[img_id] = (frame_index, timestamp)
            w, h = im.get('width'), im.get('height')
            if w and h:
                image_dims[img_id] = (w, h)

        # Keypoint category names for decoding COCO flat-format keypoints
        kp_cat_names = None
        kp_cats = data.get('keypoint_categories', [])
        if kp_cats:
            kp_cats_sorted = sorted(kp_cats, key=lambda c: c['id'])
            kp_cat_names = [c['name'] for c in kp_cats_sorted]

        # Group annotations by track_id; annotations without track_id
        # get a unique synthetic track id each.
        next_synth_id = -1

        # track_id -> list of (frame_index, timestamp, annotation)
        track_states = {}

        for ann in data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in image_info:
                continue
            frame_index, timestamp = image_info[img_id]

            tid = ann.get('track_id', None)
            if tid is None:
                tid = next_synth_id
                next_synth_id -= 1

            track_states.setdefault(tid, []).append(
                (frame_index, timestamp, ann))

        # Build Track objects
        all_tracks = {}
        tracks_by_frame = {}

        for tid, states in track_states.items():
            states.sort(key=lambda s: s[0])
            trk = vt.Track(id=int(tid))

            for frame_index, timestamp, ann in states:
                dims = image_dims.get(ann.get('image_id'))
                det = annotation_to_detection(
                    ann, categories, image_dims=dims,
                    kp_cat_names=kp_cat_names)

                if timestamp is not None:
                    time_usec = int(timestamp * 1e6)
                else:
                    time_usec = frame_index

                ots = vt.ObjectTrackState(frame_index, time_usec, det)
                trk.append(ots)

                tracks_by_frame.setdefault(frame_index, []).append(tid)

            all_tracks[tid] = trk

        # Frame ordering for streaming mode
        frame_order = sorted(tracks_by_frame.keys())

        self._all_tracks = all_tracks
        self._tracks_by_frame = tracks_by_frame
        self._frame_order = frame_order
        self._stream_idx = 0
        self._batch_returned = False
        self.loaded = True


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _strtobool(val):
    """Convert a string representation of truth to True or False."""
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("Invalid truth value %r" % (val,))


# ------------------------------------------------------------------
# Plugin registration
# ------------------------------------------------------------------

def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    implementation_name = "coco"

    if algorithm_factory.has_algorithm_impl_name(
            ReadObjectTrackSetCoco.static_type_name(),
            implementation_name,
    ):
        return

    algorithm_factory.add_algorithm(
        implementation_name,
        "Read object tracks from COCO-style JSON format",
        ReadObjectTrackSetCoco,
    )

    algorithm_factory.mark_algorithm_as_loaded(implementation_name)
