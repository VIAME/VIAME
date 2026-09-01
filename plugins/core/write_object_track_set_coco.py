# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Write object tracks to COCO/kwcoco-format JSON files.

Emits the shared VIAME COCO profile described in utilities_coco, plus the
kwcoco ``videos`` and ``tracks`` tables and a per-annotation ``track_id`` so
per-frame detections belonging to the same object can be linked across time.

Images are ordered by frame, and ``frame_index`` carries the pipeline's own
frame number so it round-trips through the reader and lines up with the frame
column the VIAME CSV writer emits. Unlike the other tables, ``track_id`` keeps
the pipeline's native track numbering rather than being renumbered from 1.
"""

from kwiver.vital.algo import WriteObjectTrackSet

from viame.core.utilities_coco import (
    VIDEO_ID,
    global_categories,
    default_video_name,
    detection_to_annotation,
    seconds_to_iso8601,
    write_coco_json,
)


class WriteObjectTrackSetCoco(WriteObjectTrackSet):
    """
    COCO-formatted output for ObjectTrackSets.

    Produces JSON with a top-level ``tracks`` table alongside standard
    COCO ``images``, ``annotations`` and ``categories``, plus a ``videos``
    table when the input is a video rather than an image list.

    Each annotation carries:
    - id, image_id, category_id, bbox, score, segmentation (standard COCO)
    - track_id: links the annotation to an entry in the ``tracks`` table

    Each image carries:
    - file_name, name, frame_index, and timestamp when the frame has a time
    - video_id, for video input only

    The writer accumulates all track states across ``write_set`` calls
    and serialises them on ``close()``.
    """

    categories = global_categories

    def __init__(self):
        WriteObjectTrackSet.__init__(self)
        self.annotations = []
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
        # Map frame_id -> index in self.images
        self._frame_to_image_id = {}
        # Accumulate full tracks; keyed by track id
        self._tracks = {}
        # Map frame_id -> frame_identifier string
        self._frame_ids = {}
        # Map frame_id -> time in seconds
        self._frame_times = {}
        # Frames seen, for pipelines that supply no timestamp
        self._frame_count = 0

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def get_configuration(self):
        cfg = super(WriteObjectTrackSet, self).get_configuration()
        cfg.set_value("category_start_id", str(self.category_start_id))
        cfg.set_value("top_n_classes", str(self.top_n_classes))
        cfg.set_value("global_categories", str(self.global_categories))
        cfg.set_value("aux_image_labels", ",".join(self.aux_image_labels))
        cfg.set_value("aux_image_extensions", ",".join(self.aux_image_extensions))
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
        self.global_categories = _strtobool(cfg.get_value("global_categories"))
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

    def check_configuration(self, cfg):
        return True

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def open(self, file_name):
        self._output_path = file_name
        self.file = open(file_name, 'w')

    def close(self):
        self._flush_tracks()
        if self.file:
            self.file.close()

    # ------------------------------------------------------------------
    # Per-frame callback
    # ------------------------------------------------------------------

    def write_set(self, track_set, timestamp, frame_identifier):
        # Conversion pipelines have no frame clock of their own, so fall back
        # to counting calls the way the detection writer numbers its frames.
        frame_id = timestamp.get_frame() if timestamp.has_valid_frame() else None
        if frame_id is None:
            frame_id = self._frame_count
        self._frame_count += 1

        # Recorded even for empty frames so the images table covers the whole
        # sequence rather than only the frames that happen to carry a track.
        if frame_identifier:
            self._frame_ids[frame_id] = frame_identifier
        if timestamp.has_valid_time():
            self._frame_times[frame_id] = timestamp.get_time_seconds()

        if not track_set:
            return

        for trk in track_set.tracks():
            self._tracks[trk.id] = trk

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def _get_image_id(self, frame_id):
        """Return the 1-based image id for *frame_id*, creating an entry if needed.

        Frames must be registered in ascending order (see _register_frames) for
        image ids to follow time.
        """
        if frame_id not in self._frame_to_image_id:
            self._frame_to_image_id[frame_id] = len(self.images) + 1
            entry = dict(frame_index=frame_id)
            file_name = self._frame_ids.get(frame_id, "")
            if file_name:
                entry["file_name"] = file_name
            if frame_id in self._frame_times:
                entry["timestamp"] = seconds_to_iso8601(self._frame_times[frame_id])
            self.images.append(entry)
        return self._frame_to_image_id[frame_id]

    def _register_frames(self):
        """Create image entries for every known frame, in ascending frame order.

        Track states are visited track by track, so registering frames lazily
        while converting annotations would number images and frame indices by
        track iteration order instead of by time.
        """
        frame_ids = set(self._frame_ids)
        frame_ids.update(self._frame_times)

        for trk in self._tracks.values():
            for state in trk:
                if state.detection() is None:
                    continue
                frame_ids.add(state.frame_id)
                if state.frame_id not in self._frame_times and state.time_usec > 0:
                    self._frame_times[state.frame_id] = state.time_usec / 1e6

        for frame_id in sorted(frame_ids):
            self._get_image_id(frame_id)

    def _flush_tracks(self):
        """Convert accumulated tracks to COCO annotations and write JSON."""
        cats = self._local_categories

        self._register_frames()

        # Build the tracks table
        tracks = []
        for trk in self._tracks.values():
            track_id = trk.id
            tracks.append(dict(id=track_id, name=str(track_id)))

            for state in trk:
                det = state.detection()
                if det is None:
                    continue

                image_id = self._get_image_id(state.frame_id)

                d = detection_to_annotation(
                    det, image_id, cats,
                    self.category_start_id, self.global_categories,
                    self.top_n_classes)
                d['track_id'] = track_id
                self.annotations.append(d)

        # An image list names every frame; video input names none. Both are
        # temporally ordered, so kwcoco would accept a videos table either
        # way -- it is gated because DIVE reads `videos` as "this is a video"
        # and skips its frame_index-against-filename ordering check when one
        # is present. Image sequences keep that safety net.
        videos = None
        if self.video_name or not self._frame_ids:
            videos = [dict(
                id=VIDEO_ID,
                name=default_video_name(self._output_path, self.video_name))]
            for entry in self.images:
                entry["video_id"] = VIDEO_ID

        write_coco_json(
            self.file,
            self.annotations,
            self.images,
            cats,
            self.global_categories,
            self.aux_image_labels,
            self.aux_image_extensions,
            description="Created by WriteObjectTrackSetCoco",
            version=self.version_identifier,
            fps=self.frame_rate,
            contributor=self.contributor,
            videos=videos, tracks=tracks)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _strtobool(val):
    """Convert a string representation of truth to True or False."""
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    elif val in ("n", "no", "f", "false", "off", "0"):
        return False
    else:
        raise ValueError("Invalid truth value %r" % (val,))


# ------------------------------------------------------------------
# Plugin registration
# ------------------------------------------------------------------


def __vital_algorithm_register__():
    from viame.core.vital_registration import register_vital_algorithm

    register_vital_algorithm(
        WriteObjectTrackSetCoco,
        "coco",
        "Write object tracks to COCO-style JSON format with track_id field",
    )
