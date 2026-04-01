# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Write object tracks to COCO-format JSON files.

Produces the standard COCO annotation format with an additional
``track_id`` field on each annotation so that per-frame detections
belonging to the same object can be linked across time.
"""

from kwiver.vital.algo import WriteObjectTrackSet

from viame.core.utilities_coco import (
    global_categories,
    detection_to_annotation,
    write_coco_json,
)


class WriteObjectTrackSetCoco(WriteObjectTrackSet):
    """
    COCO-formatted output for ObjectTrackSets.

    Each annotation carries:
    - id, image_id, category_id, bbox, score, segmentation (standard COCO)
    - track_id: the VIAME track identifier that links detections over time

    Images are registered in frame order.  The writer accumulates all
    track states across ``write_set`` calls and serialises them on
    ``close()``.
    """

    categories = global_categories

    def __init__(self):
        WriteObjectTrackSet.__init__(self)
        self.annotations = []
        self.images = []
        self.category_start_id = 1
        self.global_categories = True
        self.aux_image_labels = ""
        self.aux_image_extensions = ""
        self.file = None
        self._local_categories = {}
        # Map frame_id -> index in self.images
        self._frame_to_image_id = {}
        # Accumulate full tracks; keyed by track id
        self._tracks = {}
        # Map frame_id -> frame_identifier string
        self._frame_ids = {}
        # Map frame_id -> time in seconds
        self._frame_times = {}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def get_configuration(self):
        cfg = super(WriteObjectTrackSet, self).get_configuration()
        cfg.set_value("category_start_id", str(self.category_start_id))
        cfg.set_value("global_categories", str(self.global_categories))
        cfg.set_value("aux_image_labels", ','.join(self.aux_image_labels))
        cfg.set_value("aux_image_extensions", ','.join(self.aux_image_extensions))
        return cfg

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)
        self.category_start_id = int(cfg.get_value("category_start_id"))
        self.global_categories = _strtobool(cfg.get_value("global_categories"))
        self.aux_image_labels = str(cfg.get_value("aux_image_labels"))
        self.aux_image_extensions = str(cfg.get_value("aux_image_extensions"))

        self.aux_image_labels = self.aux_image_labels.rstrip().split(',')
        self.aux_image_extensions = self.aux_image_extensions.rstrip().split(',')

        if len(self.aux_image_labels) != len(self.aux_image_extensions):
            print("Auxiliary image labels and extensions must be same size")
            return False
        if not self.global_categories:
            self._local_categories = {}

    def check_configuration(self, cfg):
        return True

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def open(self, file_name):
        self.file = open(file_name, 'w')

    def close(self):
        self._flush_tracks()
        if self.file:
            self.file.close()

    # ------------------------------------------------------------------
    # Per-frame callback
    # ------------------------------------------------------------------

    def write_set(self, track_set, timestamp, frame_identifier):
        if not track_set:
            return

        frame_id = timestamp.get_frame() if timestamp.has_valid_frame() else None

        if frame_id is not None:
            if frame_identifier:
                self._frame_ids[frame_id] = frame_identifier
            if timestamp.has_valid_time():
                self._frame_times[frame_id] = timestamp.get_time_seconds()

        for trk in track_set.tracks():
            self._tracks[trk.id] = trk

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def _get_image_id(self, frame_id):
        """Return a stable image index for *frame_id*, creating an entry if needed."""
        if frame_id not in self._frame_to_image_id:
            idx = len(self.images)
            self._frame_to_image_id[frame_id] = idx
            entry = dict(
                file_name=self._frame_ids.get(frame_id, ""),
            )
            if frame_id in self._frame_times:
                entry["time"] = self._frame_times[frame_id]
            self.images.append(entry)
        return self._frame_to_image_id[frame_id]

    def _flush_tracks(self):
        """Convert accumulated tracks to COCO annotations and write JSON."""
        cats = self._local_categories

        for trk in self._tracks.values():
            track_id = trk.id

            for state in trk:
                det = state.detection()
                if det is None:
                    continue

                fid = state.frame_id
                if fid not in self._frame_times and state.time_usec > 0:
                    self._frame_times[fid] = state.time_usec / 1e6

                image_id = self._get_image_id(fid)

                d = detection_to_annotation(
                    det, image_id, cats,
                    self.category_start_id, self.global_categories)
                d['track_id'] = track_id
                self.annotations.append(d)

        write_coco_json(
            self.file, self.annotations, self.images, cats,
            self.global_categories,
            self.aux_image_labels, self.aux_image_extensions,
            description="Created by WriteObjectTrackSetCoco")


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
            WriteObjectTrackSetCoco.static_type_name(),
            implementation_name,
    ):
        return

    algorithm_factory.add_algorithm(
        implementation_name,
        "Write object tracks to COCO-style JSON format with track_id field",
        WriteObjectTrackSetCoco,
    )

    algorithm_factory.mark_algorithm_as_loaded(implementation_name)
