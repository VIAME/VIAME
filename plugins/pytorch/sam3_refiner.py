# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
SAM3 (Segment Anything Model 3) Track Refiner

This refiner uses SAM 2.1 for refining object tracks with text prompts.
It uses SAM3's native video predictor for temporal tracking with memory
attention, providing high-quality mask propagation across frames.

Features:
- Re-segments existing track bounding boxes with SAM for better masks
- Detects new objects using Grounding DINO text queries
- Propagates seed boxes across frames using SAM3 video predictor
- Adds non-overlapping new detections as new tracks
- Filters out tracks with low-quality masks
- Adjusts bounding boxes to fit refined masks
- Generates polygon and/or point outputs from masks
"""

import os
import shutil
import tempfile

import scriptconfig as scfg
import numpy as np

from kwiver.vital.algo import RefineTracks, RefineDetections
from kwiver.vital.types import (
    BoundingBoxD, DetectedObject, DetectedObjectSet, DetectedObjectType,
    ObjectTrackState, Track, ObjectTrackSet, ImageContainer
)
from kwiver.vital.util import VitalPIL
from PIL import Image as PILImage

from viame.pytorch.sam3_utilities import (
    SAM3BaseConfig, SAM3ModelManager,
    mask_to_polygon, mask_to_points, box_from_mask, compute_iou,
    image_to_rgb_numpy, get_autocast_context, parse_bool
)
from viame.pytorch.utilities import vital_config_update, report_cuda_errors


# Bounds and cost model for auto-sizing SAM3's grounding batch (see
# SAM3RefinerConfig.grounding_batch_size).  SAM3 works at a fixed 1008px
# internally, so the per-frame activation cost does not depend on source
# resolution.  Measured on a 16 GB Turing card at 1080p: 4.25 GiB resident
# for the weights and a 12.72 GiB peak at batch 7, i.e. 8.47 GiB of
# activations, or ~1.21 GiB per batched frame.  The headroom covers transient
# allocations and allocator fragmentation.
_GROUNDING_VRAM_PER_FRAME_GB = 1.2
_GROUNDING_HEADROOM_GB = 2.0
_GROUNDING_BATCH_MIN = 1
_GROUNDING_BATCH_MAX = 16


def _safe_class_name(class_name, fallback=None):
    """A class name kwiver will accept.

    DetectedObjectType rejects an empty name outright, and the grounding model
    returns one for its weakest matches, so lowering detection_threshold far
    enough turns a detection that should simply be kept unlabelled into a
    ValueError that fails the whole frame. Name it after the query that found
    it, or 'unknown'.
    """
    if class_name is not None and str(class_name).strip():
        return str(class_name)

    if fallback is not None and str(fallback).strip():
        return str(fallback)

    return 'unknown'


class _FrameBuffer:
    """
    Append-only frame buffer that keeps frames in memory while they fit and
    spills to disk once they do not.

    The video-propagation path collects every frame before running SAM3 in
    ``finalize()``.  Held as decoded PIL images that costs width*height*3
    bytes per frame, so a 10k-frame 1080p sequence needs ~65 GB and exhausts
    host RAM.  Past a budget derived from currently-available memory the
    frames are written out as JPEGs and re-decoded per chunk, keeping
    resident memory proportional to ``video_chunk_size`` rather than to
    video length.

    Short sequences therefore behave exactly as before -- no encode, decode
    or disk traffic -- and only sequences that would not have fit pay for
    spilling.

    Supports only the operations the propagation path uses: ``append``,
    ``len``, truthiness, and slicing (which returns PIL images).
    """

    JPEG_QUALITY = 95
    # Fraction of available RAM the in-memory buffer may occupy before
    # spilling.  Deliberately well under half: SAM3's weights, the CUDA host
    # allocations and the per-chunk decode all draw on the same pool.
    MEMORY_FRACTION = 0.25
    FALLBACK_BUDGET_BYTES = 4 * 1024 ** 3

    def __init__(self):
        self._frames = []      # in-memory PIL images (before spilling)
        self._paths = []       # on-disk JPEG paths (after spilling)
        self._dir = None
        self._bytes = 0
        self._budget = None

    @classmethod
    def _memory_budget(cls):
        """Bytes the in-memory buffer may use before spilling to disk."""
        try:
            with open('/proc/meminfo') as fh:
                for line in fh:
                    if line.startswith('MemAvailable:'):
                        avail_kb = int(line.split()[1])
                        return int(avail_kb * 1024 * cls.MEMORY_FRACTION)
        except Exception:
            pass
        return cls.FALLBACK_BUDGET_BYTES

    def _spill(self):
        """Move everything held in memory out to disk and switch modes."""
        self._dir = tempfile.mkdtemp(prefix="sam3_frames_")
        for image in self._frames:
            self._write(image)
        self._frames = []

    def _write(self, pil_image):
        path = os.path.join(self._dir, "%08d.jpg" % len(self._paths))
        pil_image.save(path, "JPEG", quality=self.JPEG_QUALITY)
        self._paths.append(path)

    def append(self, pil_image):
        if self._dir is not None:
            self._write(pil_image)
            return

        if self._budget is None:
            self._budget = self._memory_budget()

        width, height = pil_image.size
        self._bytes += width * height * 3
        if self._bytes > self._budget:
            self._spill()
            self._write(pil_image)
            return

        self._frames.append(pil_image)

    def __len__(self):
        return len(self._frames) + len(self._paths)

    def __bool__(self):
        return bool(self._frames) or bool(self._paths)

    def __getitem__(self, key):
        if self._dir is None:
            return self._frames[key]
        if isinstance(key, slice):
            return [PILImage.open(p).convert("RGB") for p in self._paths[key]]
        return PILImage.open(self._paths[key]).convert("RGB")

    def cleanup(self):
        self._frames = []
        if self._dir is not None:
            shutil.rmtree(self._dir, ignore_errors=True)
            self._dir = None
        self._paths = []
        self._bytes = 0


class SAM3RefinerConfig(SAM3BaseConfig):
    """
    Configuration for SAM3Refiner.

    Extends SAM3BaseConfig with track refinement specific options.
    """
    # Track refinement parameters
    iou_threshold = scfg.Value(
        0.5,
        help='IoU threshold for matching new detections to existing tracks'
    )
    min_mask_area = scfg.Value(
        10,
        help='Minimum mask area in pixels; tracks with smaller masks are removed'
    )
    resegment_existing = scfg.Value(
        True,
        help='Whether to re-segment existing track bounding boxes with SAM'
    )
    add_new_objects = scfg.Value(
        True,
        help='Whether to add new objects detected by text query that do not overlap'
    )
    replace_existing = scfg.Value(
        False,
        help='If True, discard all pre-existing input annotations and output '
             'only newly detected objects. Default False keeps existing '
             'annotations and adds new detections alongside them.'
    )
    filter_by_quality = scfg.Value(
        True,
        help='If True, remove tracks with poor mask quality'
    )
    adjust_boxes = scfg.Value(
        True,
        help='Whether to adjust bounding boxes based on refined masks'
    )
    max_new_objects = scfg.Value(
        50,
        help='Maximum number of new objects to add per frame'
    )
    # Whether to propagate tracked objects across frames using SAM3 video
    # predictor.  Enable for track-user-selections (seed boxes forwarded
    # across frames).  Disable for text-query pipelines where grounding
    # DINO re-detects on every frame independently.
    # Externally supplied detections, rather than a handful of seed boxes,
    # are the input in a detector-comparison setting: every tracker replays
    # the same detection set so the comparison is of association alone.
    #
    # SAM 3.1 resets its inference state on every add_prompt call, so the
    # seed path hands it all boxes in one call WITHOUT per-box obj_ids and
    # lets SAM 3 number the objects itself. Those numbers cannot match the
    # ids of the tracks that supplied the boxes, so with
    # add_new_objects=False every propagated object is filtered out as
    # "not one we seeded" and the refiner emits nothing at all. Enabling
    # this records the seed order and maps SAM 3's ids back onto the input
    # track ids, which is what makes add_new_objects=False usable.
    #
    # Off by default: it changes the object ids in the output, and the
    # text-query pipelines depend on the existing numbering.
    # Replay an external detection set, by frame index, as the seeds.
    #
    # The pipeline's read_object_track matches rows to frames by image
    # file name and hands the refiner nothing per frame when those names
    # do not line up, which makes the whole run silently empty. Every
    # other tracker in this benchmark sidesteps that by leaving
    # image_file_name unconnected and replaying detections positionally.
    # This does the same: give it a VIAME CSV and the detections on the
    # Nth frame of the CSV seed the Nth frame of the video, whatever the
    # file names say.
    #
    # Empty by default, so the ordinary seed-track and text-query
    # pipelines are unaffected.
    # How much a detection must overlap a live object to count as that
    # object continuing, rather than as a new one. Only used when
    # detections_file is set.
    link_iou = scfg.Value(
        0.3,
        help='IoU at which a detection is treated as a continuation of an '
             'object already being propagated instead of seeding a new one.'
    )

    # How long an object stays linkable after its last supporting detection.
    # Detectors miss animals for a few frames at a time; expiring too eagerly
    # splits one animal into several tracks.
    link_max_gap = scfg.Value(
        5,
        help='Frames an object remains available for linking after its last '
             'matching detection.'
    )

    # SAM 3 caps concurrent tracked objects and its memory cost scales with
    # them, so this bounds how many are alive at once. Without a bound a
    # crowded sequence exhausts system memory and the run is killed.
    max_seed_objects = scfg.Value(
        16,
        help='Maximum objects seeded concurrently when replaying an external '
             'detection set.'
    )

    # SAM 3.1 refuses to instantiate a new object once this many are already
    # being tracked, silently dropping the lowest-scoring new detections. The
    # model builder's default is 16, which is well under what a crowded frame
    # holds -- a quarter of the sequences here exceed it -- so it has to be
    # raised for a detection set to be replayed faithfully rather than
    # truncated. Costs memory and time roughly in proportion to the objects
    # actually present, so scenes below the cap are unaffected.
    max_tracked_objects = scfg.Value(
        0,
        help='Override SAM 3.1\'s concurrent tracked-object limit. 0 leaves '
             'the model default (16) alone.'
    )

    # Every tracker in a detector-shared comparison gates the detection set
    # before it will start a track -- ByteTrack and OC-SORT here keep only the
    # 15% above 0.804. Replaying an ungated set instead makes SAM 3 open a
    # track on detections the others silently discard, so it is charged with
    # false positives they never had the chance to emit and its precision is
    # not comparable with theirs. Off by default: the seed-track and
    # text-query pipelines supply their own detections and do not want them
    # second-guessed.
    detections_threshold = scfg.Value(
        0.0,
        help='Drop detections below this confidence before seeding. Set it to '
             'the new-track threshold of the trackers being compared against, '
             'so every tracker starts tracks on the same detections.'
    )

    detections_file = scfg.Value(
        '',
        help='VIAME CSV of detections to replay as seeds, matched to '
             'frames by index rather than by image file name. Intended '
             'for replaying a shared detection set so SAM 3 can be '
             'compared against other trackers on equal input.'
    )

    preserve_seed_ids = scfg.Value(
        False,
        help='Map propagated object ids back onto the ids of the input '
             'tracks that seeded them, so seeded objects survive an '
             'add_new_objects=False filter. Required when replaying an '
             'external detection set through SAM 3.'
    )

    propagate_tracked = scfg.Value(
        True,
        help='Propagate seed boxes across frames using SAM3 video predictor'
    )
    # How often (in frames) to re-run text detection to find new objects
    # entering the scene.  Set to 0 to only detect on the first frame.
    reinit_interval = scfg.Value(
        10,
        help='Frames between text re-detection for new objects (0=first only)'
    )
    # SAM3 video predictor internal detection confidence threshold.
    # Lowering this lets SAM3 detect less prominent objects.
    video_detection_threshold = scfg.Value(
        0.3,
        help='SAM3 video predictor detection score threshold (default 0.5 in SAM3)'
    )
    # Threshold for a detection to be promoted to a new tracked object.
    # SAM3 default is 0.7 which is very aggressive — lower for recall.
    video_new_det_threshold = scfg.Value(
        0.1,
        help='Min score for a new detection to become a tracked object (default 0.7 in SAM3)'
    )
    # Hotstart delay: SAM3 holds outputs this many frames for filtering.
    # During hotstart, unmatched/duplicate tracks are pruned.  Set to 0
    # to disable hotstart filtering entirely.
    video_hotstart_delay = scfg.Value(
        0,
        help='Frames to hold outputs for hotstart filtering (default 15 in SAM3, 0=disable)'
    )
    # Frames batched together through SAM3's grounding detector and
    # post-processing.  SAM3's model_builder hardcodes 16, which assumes a
    # datacenter GPU and OOMs on smaller cards regardless of
    # ``video_chunk_size`` (the batch is internal to SAM3's per-frame
    # inference, not to our chunk loop).  'auto' sizes it to the GPU memory
    # actually free once the weights are resident.
    grounding_batch_size = scfg.Value(
        'auto',
        help="Frames per SAM3 grounding batch ('auto'=fit to free VRAM)"
    )
    # Maximum number of frames to process per video chunk.  SAM3 video
    # predictor keeps per-frame features in GPU memory; processing very
    # long or high-resolution videos in one shot can cause OOM.  The
    # video is split into overlapping chunks of this size and the chunks
    # are processed sequentially.  Set to 0 for no chunking.
    video_chunk_size = scfg.Value(
        100,
        help='Max frames per video propagation chunk (0=no chunking)'
    )
    # Starting value for IDs of tracks the refiner creates (for detections
    # that don't match any input seed track when add_new_objects=True).
    # The refiner skips any IDs already used by input tracks, and remaps
    # later input tracks whose IDs would collide with already-assigned
    # refiner IDs, so the final output is always collision-free regardless
    # of this starting value.
    new_track_id_start = scfg.Value(
        1,
        help='Starting ID for refiner-created tracks (default 1)'
    )
    # When enabled, the per-frame path (``propagate_tracked=False``)
    # maintains a simple tracker: seed detections are re-segmented on each
    # subsequent frame by feeding their last-known box back to the image
    # predictor. This is the SAM2 strategy ported to SAM3 and is the right
    # choice for "track user selections" style pipelines where there is no
    # text query to drive the multiplex tracker.
    track_new_objects = scfg.Value(
        False,
        help='Propagate seed boxes forward by re-segmenting them on each '
             'subsequent frame (per-frame mode only)'
    )
    lost_track_frames = scfg.Value(
        10,
        help='Number of frames to keep a tracked object alive after it stops '
             'being seen in the input (per-frame tracker only)'
    )


def _ensure_binary_mask(mask):
    """Ensure mask is a numpy uint8 binary array suitable for contour finding."""
    if isinstance(mask, _PackedMask):
        return mask.unpack()
    if not isinstance(mask, np.ndarray):
        import torch
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        else:
            mask = np.array(mask)
    return (mask > 0.5).astype(np.uint8)


class _PackedMask(object):
    """A propagated mask stored as the bit-packed crop of its bounding box.

    Video propagation keeps one mask per object per frame for the whole
    video. At full-frame resolution that grows by gigabytes per chunk and
    long sequences are OOM-killed before they can finish; the objects
    themselves cover a tiny fraction of the frame, so the crop packs into
    a few kilobytes.
    """

    __slots__ = ('_bits', '_crop', 'shape', 'bbox')

    def __init__(self, mask):
        if isinstance(mask, np.ndarray) and mask.dtype == np.uint8:
            m = mask
        else:
            m = _ensure_binary_mask(mask)
        self.shape = m.shape
        rows = np.any(m, axis=1)
        if not rows.any():
            self._bits = None
            self._crop = (0, 0, 0, 0)
            self.bbox = [0.0, 0.0, 0.0, 0.0]
            return
        cols = np.any(m, axis=0)
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        y2, x2 = int(y2) + 1, int(x2) + 1
        y1, x1 = int(y1), int(x1)
        self._crop = (y1, y2, x1, x2)
        self.bbox = [float(x1), float(y1), float(x2), float(y2)]
        self._bits = np.packbits(m[y1:y2, x1:x2] > 0)

    def unpack(self):
        """The full-frame uint8 mask this was built from."""
        full = np.zeros(self.shape, dtype=np.uint8)
        if self._bits is not None:
            y1, y2, x1, x2 = self._crop
            n = (y2 - y1) * (x2 - x1)
            crop = np.unpackbits(self._bits, count=n)
            full[y1:y2, x1:x2] = crop.reshape(y2 - y1, x2 - x1)
        return full

    def area(self):
        """Number of set pixels (packbits pads with zeros, so a plain
        popcount over the packed bytes is exact)."""
        if self._bits is None:
            return 0
        return int(np.unpackbits(self._bits).sum())


def _mask_bbox(mask):
    """Get [x1, y1, x2, y2] bounding box from a binary mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows):
        return [0, 0, 0, 0]
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [float(x1), float(y1), float(x2 + 1), float(y2 + 1)]


def _set_mask_on_detection(det, mask, bbox):
    """Set a binary mask (cropped to the detection bbox) on a detection."""
    binary_mask = _ensure_binary_mask(mask)
    x1 = max(0, int(bbox.min_x()))
    y1 = max(0, int(bbox.min_y()))
    x2 = min(binary_mask.shape[1], int(bbox.max_x()))
    y2 = min(binary_mask.shape[0], int(bbox.max_y()))
    if x2 <= x1 or y2 <= y1:
        return
    cropped = binary_mask[y1:y2, x1:x2]
    if cropped.size == 0:
        return
    pil_img = PILImage.fromarray(cropped)
    vital_img = ImageContainer(VitalPIL.from_pil(pil_img))
    det.mask = vital_img


def _set_polygon_on_detection(det, mask, simplification):
    """Set a flattened polygon on a detection from a binary mask."""
    binary_mask = _ensure_binary_mask(mask)
    poly_pts = mask_to_polygon(binary_mask, simplification)
    if poly_pts is not None:
        det.set_flattened_polygon(poly_pts)


def _box_iou(a, b):
    """Intersection over union of two (x1, y1, x2, y2) boxes."""
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])

    if x1 <= x0 or y1 <= y0:
        return 0.0

    inter = (x1 - x0) * (y1 - y0)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])

    return inter / (aa + bb - inter)


def _load_viame_csv_by_frame(path):
    """Read a VIAME CSV into {frame_index: [(id, (x1,y1,x2,y2), conf, cls)]}.

    Frame index is column 3 as written in the file. Rows are keyed by that
    value directly, so the Nth annotated frame seeds the Nth frame of the
    video; this is the same positional replay the other trackers use, and
    it does not depend on image file names agreeing.
    """
    import csv as _csv

    by_frame = {}
    try:
        with open(path, errors='replace') as handle:
            for row in _csv.reader(handle):
                if not row or row[0].lstrip().startswith('#'):
                    continue
                if len(row) < 8:
                    continue
                try:
                    det_id = int(row[0])
                    frame = int(row[2])
                    box = tuple(float(v) for v in row[3:7])
                    conf = float(row[7])
                except ValueError:
                    continue
                cls = row[9] if len(row) > 9 else ''
                by_frame.setdefault(frame, []).append((det_id, box, conf, cls))
    except OSError as error:
        print('[SAM3] could not read detections file %s: %s' % (path, error))
        return {}
    return by_frame


class SAM3Refiner(RefineTracks):
    """
    SAM3-based track refiner with native video tracking.

    Uses SAM3's video predictor for temporal mask propagation when
    ``propagate_tracked`` is enabled.  Seed boxes (single-state input
    tracks) are added as prompts and propagated across all subsequent
    frames with SAM3's memory-attention mechanism.

    When ``propagate_tracked`` is disabled (text-query pipelines),
    each frame is processed independently with the image predictor.

    Example:
        >>> from viame.pytorch.sam3_refiner import SAM3Refiner
        >>> refiner = SAM3Refiner()
        >>> refiner.set_configuration({'text_query': 'fish, crab'})
        >>> refined_tracks = refiner.refine(timestamp, image, tracks)
    """

    def __init__(self):
        RefineTracks.__init__(self)
        self._config = SAM3RefinerConfig()
        self._model_manager = SAM3ModelManager()
        # Track-ID allocation state. ``_next_track_id`` is the next
        # candidate for a refiner-created track; ``_allocate_next_id``
        # skips over IDs that have already been used by either another
        # refiner-created track or an input seed track, so the output is
        # always collision-free. Input IDs that collide with IDs already
        # handed out to refiner-created tracks are remapped once via
        # ``_input_id_remap`` and reused on subsequent frames.
        self._next_track_id = 1
        self._assigned_ids = set()
        self._known_input_ids = set()
        self._input_id_remap = {}
        # Per-frame tracker state (used when propagate_tracked=False and
        # track_new_objects=True). Maps track_id → {
        #   'last_box': [x1,y1,x2,y2], 'class_name': str, 'lost': int,
        #   'history': [ObjectTrackState, ...] }
        self._tracked_objects = {}

        self._preserve_seed_ids = False
        self._detections_file = ''
        self._external_dets = {}
        self._clamped_out = 0
        self._link_iou = 0.3
        self._link_max_gap = 5
        self._max_seed_objects = 16
        self._max_tracked_objects = 0
        self._detections_threshold = 0.0
        self._below_threshold = 0
        self._active_seeds = {}   # obj_id -> (last_frame, box_xyxy)
        self._seeded = 0
        self._linked = 0
        self._seed_capacity_drops = 0
        # Ids handed to objects SAM 3 discovered rather than ones that were
        # seeded. Negative and strictly decreasing, so they stay distinct from
        # every real track id and from each other across chunks.
        self._discovered_id_seq = 0
        self._seeds_submitted = 0
        self._seeds_mapped = 0

        # Video predictor state (used when propagate_tracked=True)
        self._video_predictor = None
        self._pil_frames = _FrameBuffer()   # frames for init_state (spills if large)
        self._frame_prompts = {}       # frame_idx -> [(obj_id, box_rel_xywh)]
        self._text_prompt_frames = {}  # frame_idx -> text_query_string
        self._obj_id_to_class = {}     # obj_id -> class_name
        self._propagated_tracks = {}   # obj_id -> [ObjectTrackState, ...]
        self._timestamps = {}          # frame_idx -> timestamp
        self._img_width = 0
        self._img_height = 0

    def get_configuration(self):
        """Get the algorithm configuration."""
        cfg = super(RefineTracks, self).get_configuration()
        for key, value in self._config.items():
            cfg.set_value(key, str(value))
        return cfg

    @report_cuda_errors("SAM3 refiner initialization")
    def set_configuration(self, cfg_in):
        """Set the algorithm configuration and initialize models."""
        cfg = self.get_configuration()
        vital_config_update(cfg, cfg_in)

        for key in self._config.keys():
            self._config[key] = str(cfg.get_value(key))

        self._config.__post_init__()

        # Convert types
        self._iou_threshold = float(self._config.iou_threshold)
        self._min_mask_area = int(self._config.min_mask_area)
        self._resegment_existing = parse_bool(self._config.resegment_existing)
        self._add_new_objects = parse_bool(self._config.add_new_objects)
        self._replace_existing = parse_bool(self._config.replace_existing)
        self._filter_by_quality = parse_bool(self._config.filter_by_quality)
        self._adjust_boxes = parse_bool(self._config.adjust_boxes)
        self._max_new_objects = int(self._config.max_new_objects)
        self._detection_threshold = float(self._config.detection_threshold)
        self._text_threshold = float(self._config.text_threshold)
        self._polygon_simplification = float(self._config.polygon_simplification)
        self._num_points = int(self._config.num_points)
        self._output_type = self._config.output_type
        self._text_query_list = self._config.text_query_list
        self._propagate_tracked = parse_bool(self._config.propagate_tracked)
        self._preserve_seed_ids = parse_bool(
            self._config.preserve_seed_ids)
        self._link_iou = float(self._config.link_iou)
        self._link_max_gap = int(self._config.link_max_gap)
        self._max_seed_objects = int(self._config.max_seed_objects)
        self._max_tracked_objects = int(self._config.max_tracked_objects)
        self._detections_threshold = float(
            self._config.detections_threshold)
        self._detections_file = str(self._config.detections_file or '')
        self._external_dets = {}
        if self._detections_file:
            self._external_dets = _load_viame_csv_by_frame(
                self._detections_file)
            print('[SAM3] replaying %d frames of external detections from %s'
                  % (len(self._external_dets), self._detections_file))
        self._reinit_interval = int(self._config.reinit_interval)
        self._next_track_id = int(self._config.new_track_id_start)
        self._assigned_ids.clear()
        self._known_input_ids.clear()
        self._input_id_remap.clear()
        self._track_new_objects = parse_bool(self._config.track_new_objects)
        self._lost_track_frames = int(self._config.lost_track_frames)
        self._tracked_objects.clear()

        # When propagation is disabled (per-frame mode), load the image
        # predictor and grounding DINO now.
        # When propagation is enabled, defer to the video predictor which
        # handles text detection natively — no grounding DINO needed.
        self._video_predictor_initialized = False
        if not self._propagate_tracked:
            self._model_manager.init_models(self._config, use_video_predictor=False)

        return True

    def check_configuration(self, cfg):
        """Check if the configuration is valid."""
        return True

    # ------------------------------------------------------------------
    # Video predictor helpers
    # ------------------------------------------------------------------

    def _ensure_video_predictor(self):
        """Lazily initialize the video predictor on first use."""
        if not self._video_predictor_initialized and self._propagate_tracked:
            self._model_manager.init_models(
                self._config, use_video_predictor=True,
            )
            self._video_predictor = self._model_manager._video_predictor
            self._video_predictor_initialized = True

            # Override SAM3 video predictor thresholds.
            # The built-in defaults (score_threshold_detection=0.5,
            # new_det_thresh=0.7, hotstart_delay=15) are tuned for
            # general video and are too aggressive for many domains.
            vp = self._video_predictor
            det_thresh = float(self._config.video_detection_threshold)
            new_det_thresh = float(self._config.video_new_det_threshold)
            hotstart_delay = int(self._config.video_hotstart_delay)

            self._set_predictor_attr(vp, 'score_threshold_detection',
                                     det_thresh)
            self._set_predictor_attr(vp, 'new_det_thresh', new_det_thresh)
            self._set_predictor_attr(vp, 'hotstart_delay', hotstart_delay)
            # SAM3 requires the unmatch/dup suppression windows to fit inside
            # the hotstart window (it asserts this at construction, but not
            # for values assigned afterwards).  Lowering only the delay leaves
            # suppression thresholds larger than the window they index into,
            # which drops tracklets that should have been kept.
            for dependent in ('hotstart_unmatch_thresh', 'hotstart_dup_thresh'):
                current = self._get_predictor_attr(vp, dependent)
                if current is not None and current > hotstart_delay:
                    self._set_predictor_attr(vp, dependent, hotstart_delay)
            # Enable detection NMS to suppress overlapping detections.
            # The base class default is 0.0 (disabled) but SAM3's own
            # model_builder sets 0.1 when constructing the video model.
            if self._get_predictor_attr(vp, 'det_nms_thresh', 1.0) <= 0:
                self._set_predictor_attr(vp, 'det_nms_thresh', 0.1)

            # Shrink SAM3's frame batches to what this GPU can hold. Done
            # after init_models so the weights are already resident and
            # mem_get_info reports the memory genuinely left for activations.
            # Only the grounding batch is resized: that is what the backbone
            # and segmentation heads allocate against, and what the OOM
            # traceback points at.  ``postprocess_batch_size`` is left alone
            # deliberately — it gates output batching, and any trailing
            # partial batch is emitted upstream as DUMMY_OUTPUT, so lowering
            # it silently discards detections on real frames.
            batch = self._resolve_grounding_batch_size()
            if batch <= 1:
                # Batched grounding is what allocates the large multi-frame
                # activations.  At a batch of one there is nothing to gain
                # from it, and SAM3's unbatched path has a much smaller
                # peak, so take that route instead of a degenerate batch.
                self._set_predictor_attr(vp, 'use_batched_grounding', False)
            else:
                self._set_predictor_attr(vp, 'batched_grounding_batch_size',
                                         batch)

            if self._max_tracked_objects > 0:
                self._set_predictor_attr(vp, 'max_num_objects',
                                         self._max_tracked_objects)
                # SAM 3 post-processes masks for a whole batch of frames at
                # once, concatenating one full-resolution mask per object per
                # frame. That allocation scales with the object limit and is
                # what runs a 16 GB card out of memory as soon as the limit is
                # raised, so take the per-frame path instead: same masks, peak
                # divided by the batch.
                self._set_predictor_attr(vp, 'postprocess_batch_size', 1)
                print("[SAM3] concurrent tracked-object limit raised to %d "
                      "(mask post-processing unbatched to fit)"
                      % self._max_tracked_objects)

            # Echo what the detector will actually run with.  These are
            # assembled from several config layers and silently defaulted
            # when a pipe key does not reach the refiner, so an empty or
            # unexpected query here explains an empty result set.
            print("[SAM3] effective text query=%r | det_thresh=%s "
                  "new_det_thresh=%s hotstart_delay=%s nms=%s | "
                  "add_new_objects=%s propagate_tracked=%s"
                  % (self._text_query_list, det_thresh, new_det_thresh,
                     hotstart_delay,
                     self._get_predictor_attr(vp, 'det_nms_thresh'),
                     self._add_new_objects, self._propagate_tracked))

    @staticmethod
    def _predictor_targets(vp):
        """
        Objects that may carry SAM3's tunables, outermost first.

        On the SAM 3.1 path ``vp`` is ``_Sam3p1VideoPredictorAdapter``, a thin
        shim holding only ``_p`` / ``_model`` / ``device``; the thresholds and
        batch sizes live on the wrapped multiplex model.  Setting them on the
        adapter silently does nothing, so resolve the real owner instead.
        """
        inner = getattr(vp, '_p', None)
        candidates = [vp,
                      getattr(vp, '_model', None),
                      inner,
                      getattr(inner, 'model', None)]
        seen, targets = set(), []
        for obj in candidates:
            if obj is not None and id(obj) not in seen:
                seen.add(id(obj))
                targets.append(obj)
        return targets

    @classmethod
    def _get_predictor_attr(cls, vp, name, default=None):
        for obj in cls._predictor_targets(vp):
            if hasattr(obj, name):
                return getattr(obj, name)
        return default

    @classmethod
    def _set_predictor_attr(cls, vp, name, value):
        """
        Set ``name`` on whichever wrapped object declares it.

        Returns True if it landed somewhere.  A miss is reported rather than
        ignored — a silently unapplied memory or threshold override looks
        exactly like the setting having no effect.
        """
        applied = False
        for obj in cls._predictor_targets(vp):
            if hasattr(obj, name):
                setattr(obj, name, value)
                applied = True
        if not applied:
            print("[SAM3] WARNING: could not apply %s=%s — no wrapped "
                  "predictor object declares it" % (name, value))
        return applied

    def _resolve_grounding_batch_size(self):
        """
        Frames to batch through SAM3's grounding detector.

        Honors an explicit ``grounding_batch_size``; otherwise divides the
        free VRAM (less a headroom allowance for transient allocations and
        fragmentation) by the measured per-frame cost.
        """
        configured = str(self._config.grounding_batch_size).strip().lower()
        if configured and configured != 'auto':
            try:
                return max(1, int(float(configured)))
            except ValueError:
                pass

        try:
            import torch
            free_bytes, _total = torch.cuda.mem_get_info()
        except Exception:
            # No CUDA, or the driver would not report — leave SAM3's default.
            return _GROUNDING_BATCH_MAX

        usable_gb = (free_bytes / (1024.0 ** 3)) - _GROUNDING_HEADROOM_GB
        batch = int(usable_gb / _GROUNDING_VRAM_PER_FRAME_GB)

        # Deliberately all-or-nothing.  Measured on a 16 GB Turing card:
        # batched grounding OOMs at the stock batch of 16, and every reduced
        # batch (7, 4, 2) completes "successfully" while emitting no
        # detections at all.  A partial batch is therefore not a usable
        # middle ground -- it trades a loud failure for a silent one.  If the
        # full batch does not fit, fall back to the unbatched path, which is
        # slower but produces correct output.
        if batch < _GROUNDING_BATCH_MAX:
            print("[SAM3] %.1f GiB VRAM free -> too little for a full "
                  "%d-frame grounding batch; using the unbatched path"
                  % (free_bytes / (1024.0 ** 3), _GROUNDING_BATCH_MAX))
            return 1

        print("[SAM3] %.1f GiB VRAM free -> grounding batch size %d"
              % (free_bytes / (1024.0 ** 3), _GROUNDING_BATCH_MAX))
        return _GROUNDING_BATCH_MAX

    def _run_video_propagation(self):
        """
        Run SAM3 video predictor on accumulated frames and prompts.
        Splits long videos into chunks to avoid GPU OOM.
        Returns a dict of (obj_id, frame_idx) -> (binary_mask, box_xyxy).
        """
        import torch
        import sys

        if not self._pil_frames:
            return {}
        if not self._frame_prompts and not self._text_prompt_frames:
            return {}

        self._ensure_video_predictor()
        if self._video_predictor is None:
            return {}

        chunk_size = int(self._config.video_chunk_size) if hasattr(self._config, 'video_chunk_size') else 100
        total_frames = len(self._pil_frames)

        if chunk_size <= 0 or total_frames <= chunk_size:
            # Process all at once
            return self._run_video_propagation_chunk(
                self._pil_frames[:], 0, self._frame_prompts,
                self._text_prompt_frames)

        # Process in overlapping chunks with ID reconciliation.
        # Each chunk assigns its own object IDs independently.  We use
        # mask IoU in the overlap region to match new-chunk IDs to the
        # IDs already established in all_results, so that the same
        # physical object keeps the same track ID across chunks.
        print(f"[SAM3] Processing {total_frames} frames in chunks of {chunk_size}")
        all_results = {}
        next_global_id = 0  # monotonically increasing global ID counter
        overlap = max(10, chunk_size // 10)
        start = 0

        while start < total_frames:
            end = min(start + chunk_size, total_frames)
            chunk_frames = self._pil_frames[start:end]

            # Remap prompts to chunk-local indices
            chunk_box_prompts = {}
            for fidx, prompts in self._frame_prompts.items():
                if start <= fidx < end:
                    chunk_box_prompts[fidx - start] = prompts

            chunk_text_prompts = {}
            for fidx, text in self._text_prompt_frames.items():
                if start <= fidx < end:
                    chunk_text_prompts[fidx - start] = text

            # If no text prompt falls in this chunk, add one on the
            # first frame of the chunk so detection stays active.
            if not chunk_text_prompts and self._text_prompt_frames:
                first_text = next(iter(self._text_prompt_frames.values()))
                chunk_text_prompts[0] = first_text

            # A span of video with no seeds in it at all: nothing to
            # propagate, and SAM3 raises rather than returning empty if
            # asked to run without a single prompt.
            if not chunk_box_prompts and not chunk_text_prompts:
                start += chunk_size - overlap
                continue

            chunk_results = self._run_video_propagation_chunk(
                chunk_frames, start, chunk_box_prompts, chunk_text_prompts)

            if self._preserve_seed_ids:
                # The chunk has already mapped its objects back onto the ids
                # of the detections that seeded them, and those ids are the
                # input track ids -- global across the whole video by
                # construction. Renumbering them here would undo exactly the
                # identity this mode exists to preserve, and the renumbered
                # ids then collide with the seed-id space, so the
                # add_new_objects=False filter keeps arbitrary objects
                # instead of dropping SAM 3's own discoveries.
                for (cid, local_fidx), val in chunk_results.items():
                    key = (cid, local_fidx + start)
                    if key in all_results:
                        continue
                    all_results[key] = val
                start += chunk_size - overlap
                continue

            if not all_results:
                # First chunk — adopt IDs directly, offset to global range
                chunk_oids = set(oid for oid, _ in chunk_results.keys())
                id_map = {}
                for cid in sorted(chunk_oids):
                    id_map[cid] = next_global_id
                    next_global_id += 1
                for (cid, local_fidx), val in chunk_results.items():
                    gid = id_map[cid]
                    all_results[(gid, local_fidx + start)] = val
            else:
                # Build ID mapping by matching masks in the overlap region.
                id_map = self._match_chunk_ids(
                    all_results, chunk_results, start, overlap,
                    self._iou_threshold)

                # Assign new global IDs for unmatched chunk objects
                for cid in set(oid for oid, _ in chunk_results.keys()):
                    if cid not in id_map:
                        id_map[cid] = next_global_id
                        next_global_id += 1

                # Merge, skipping overlap frames that already have data
                overlap_end = start + overlap
                for (cid, local_fidx), val in chunk_results.items():
                    global_fidx = local_fidx + start
                    gid = id_map[cid]
                    key = (gid, global_fidx)
                    # In overlap region, keep prior chunk's results
                    if global_fidx < overlap_end and key in all_results:
                        continue
                    all_results[key] = val

            start += chunk_size - overlap

        return all_results

    @staticmethod
    def _match_chunk_ids(all_results, chunk_results, chunk_start,
                         overlap, iou_thresh):
        """
        Match object IDs from a new chunk to existing global IDs using
        mask IoU in the overlap region.

        Returns a dict mapping chunk_obj_id -> global_obj_id for matched
        objects.  Unmatched chunk objects are not included.
        """
        # Collect masks per object in the overlap frames from both sides
        overlap_end = chunk_start + overlap

        # Existing global results in the overlap region: gid -> list of masks
        global_masks = {}
        for (gid, gfidx), rtup in all_results.items():
            if chunk_start <= gfidx < overlap_end:
                global_masks.setdefault(gid, {})[gfidx] = rtup[0]

        # New chunk results in the overlap region: cid -> list of masks
        chunk_masks = {}
        for (cid, local_fidx), rtup in chunk_results.items():
            gfidx = local_fidx + chunk_start
            if chunk_start <= gfidx < overlap_end:
                chunk_masks.setdefault(cid, {})[gfidx] = rtup[0]

        if not global_masks or not chunk_masks:
            return {}

        # Compute average IoU between each (chunk_id, global_id) pair
        # across shared overlap frames
        id_map = {}
        used_gids = set()

        # Score all pairs
        pairs = []
        for cid, c_frames in chunk_masks.items():
            for gid, g_frames in global_masks.items():
                shared = set(c_frames.keys()) & set(g_frames.keys())
                if not shared:
                    continue
                total_iou = 0.0
                for fidx in shared:
                    total_iou += compute_iou(
                        c_frames[fidx].bbox,
                        g_frames[fidx].bbox)
                avg_iou = total_iou / len(shared)
                if avg_iou > iou_thresh:
                    pairs.append((avg_iou, cid, gid))

        # Greedy matching: best IoU first, each ID used at most once
        pairs.sort(reverse=True)
        used_cids = set()
        for iou_val, cid, gid in pairs:
            if cid in used_cids or gid in used_gids:
                continue
            id_map[cid] = gid
            used_cids.add(cid)
            used_gids.add(gid)

        return id_map

    def _run_video_propagation_chunk(self, pil_frames, global_offset,
                                     frame_prompts, text_prompt_frames):
        """
        Run SAM3 video propagation on a single chunk of frames.
        frame_prompts and text_prompt_frames use chunk-local indices.
        Returns results keyed by (obj_id, local_frame_idx).
        """
        import torch
        import contextlib

        torch.cuda.empty_cache()

        # Build inference context: inference_mode + fp16 autocast + SDPA
        # fallback for pre-Ampere GPUs (e.g. Turing RTX 5000).  The SAM 3.1
        # adapter already wraps calls this way, but the raw SAM 3.0 video
        # predictor does not.  Running in fp32 without this causes OOM on
        # 16 GB GPUs even with small frame counts.
        cm = contextlib.ExitStack()
        cm.enter_context(torch.inference_mode())
        try:
            cm.enter_context(get_autocast_context(
                str(self._model_manager.device)))
        except Exception:
            pass
        try:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            cm.enter_context(sdpa_kernel(
                [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
            ))
        except Exception:
            pass

        with cm:
            state = self._video_predictor.init_state(
                pil_frames, offload_video_to_cpu=True,
            )

            from viame.pytorch.sam3_utilities import _Sam3p1VideoPredictorAdapter
            is_sam31 = isinstance(self._video_predictor,
                                  _Sam3p1VideoPredictorAdapter)

            # Extract the (single) text query for this chunk, if any.
            text_query = None
            text_seed_frame = 0
            for fidx, tq in text_prompt_frames.items():
                text_query = tq
                text_seed_frame = fidx
                break

            # A comma-separated text query is an open-vocabulary list of
            # class names; SAM3's add_prompt treats it as one opaque string,
            # so detections come back with no way to tell which term they
            # matched. When the caller supplied >1 class and no seed boxes
            # (pure text-driven detection), run a separate propagation pass
            # per class, assign each pass its own obj_id namespace, and tag
            # each obj_id with its class term so downstream output carries
            # the correct label.
            class_terms = []
            if text_query:
                class_terms = [t.strip() for t in text_query.split(',')
                               if t.strip()]
            multi_class = len(class_terms) > 1 and not frame_prompts

            # Suppress tqdm progress bars from SAM3's propagation
            import tqdm
            _orig_init = tqdm.tqdm.__init__
            def _quiet_init(self_tqdm, *args, **kwargs):
                kwargs['disable'] = True
                _orig_init(self_tqdm, *args, **kwargs)
            tqdm.tqdm.__init__ = _quiet_init

            results = {}
            try:
                if multi_class:
                    results = self._propagate_per_class(
                        state, class_terms, text_seed_frame,
                    )
                else:
                    seeds = self._seed_prompts_single_pass(
                        state, is_sam31, text_query, text_seed_frame,
                        frame_prompts,
                    )
                    for frame_idx, frame_results in \
                            self._video_predictor.propagate_in_video(state):
                        self._collect_frame_results(
                            results, frame_idx, frame_results,
                        )
                    if self._preserve_seed_ids and seeds:
                        seed_map = self._map_seed_ids_from_results(
                            results, seeds)
                        self._seeds_mapped += len(seed_map)
                        results = self._remap_seed_ids(results, seed_map)
            finally:
                tqdm.tqdm.__init__ = _orig_init

            # Free video state
            try:
                self._video_predictor.reset_state(state)
            except Exception:
                pass

        return results

    def _map_seed_ids_from_results(self, results, seeds):
        """Map SAM 3's own object ids onto the track ids that seeded them.

        SAM 3.1 takes no per-box obj_id: it numbers objects itself, skips seed
        boxes it declines to instantiate, and mixes in anything it detected on
        its own. Pairing its ids with the submitted boxes by position --
        which is what this did first -- therefore hands a track the mask of
        whichever unrelated object happened to land in the same slot.

        The reliable link is spatial: on the frame a seed was planted, the
        object covering that seed's box is the one it created. Objects
        matching no seed are SAM 3's own discoveries and are left out; the
        caller gives them ids that cannot collide with a real track.
        """
        if not results or not seeds:
            return {}

        # Propagated boxes, per frame, in image pixels.
        by_frame = {}
        for (oid, fidx), (_mask, box_xyxy, _score) in results.items():
            by_frame.setdefault(fidx, []).append((oid, box_xyxy))

        w, h = self._img_width, self._img_height

        pairs = []
        for si, (_det_id, fidx, box_rel) in enumerate(seeds):
            sbox = [box_rel[0] * w, box_rel[1] * h,
                    (box_rel[0] + box_rel[2]) * w,
                    (box_rel[1] + box_rel[3]) * h]
            for oid, obox in by_frame.get(fidx, ()):
                iou = _box_iou(sbox, obox)
                if iou > 0.0:
                    pairs.append((iou, si, oid))

        # Best-first greedy assignment, so a seed and the object over it are
        # paired before either can be claimed by a poorer overlap.
        pairs.sort(key=lambda p: (-p[0], p[1], p[2]))

        seed_map = {}
        used_seeds, used_oids = set(), set()
        for _iou, si, oid in pairs:
            if si in used_seeds or oid in used_oids:
                continue
            seed_map[oid] = seeds[si][0]
            used_seeds.add(si)
            used_oids.add(oid)

        return seed_map

    def _remap_seed_ids(self, results, seed_order):
        """Rename SAM 3's object ids back to the tracks that seeded them.

        ``seed_order`` maps a SAM 3 object id to the input track id whose
        detection seeded it. Anything absent from it is an object SAM 3 found
        on its own; it is given a negative id, unique across the whole run, so
        it can never be mistaken for a seeded track by the
        ``add_new_objects=False`` filter downstream.

        Ids are rewritten into a fresh dict rather than in place, because a
        seeded id and a discovered id can collide once the seeded ones are
        renamed.
        """
        remapped = {}
        collisions = 0
        discovered = {}
        for (oid, fidx), value in results.items():
            if oid in seed_order:
                new_oid = seed_order[oid]
            else:
                if oid not in discovered:
                    self._discovered_id_seq -= 1
                    discovered[oid] = self._discovered_id_seq
                new_oid = discovered[oid]
            key = (new_oid, fidx)
            if key in remapped:
                collisions += 1
                continue
            remapped[key] = value
            if new_oid != oid:
                cls = self._obj_id_to_class.get(new_oid)
                if cls is not None:
                    self._obj_id_to_class[new_oid] = cls
        if collisions:
            print("[SAM3] %d propagated results dropped on seed-id remap "
                  "(id collision)" % collisions)
        return remapped

    def _seed_prompts_single_pass(self, state, is_sam31, text_query,
                                  text_seed_frame, frame_prompts):
        """
        Add seed box / text prompts to a fresh video predictor state for a
        single-class (or single-pass) propagation. The SAM 3.1 multiplex
        resets state on each ``add_prompt`` call, so this path collapses
        everything into one call. SAM 3.0 accumulates state across calls
        and can accept per-frame seed boxes plus a single text prompt.
        """
        seeds = []
        if is_sam31:
            if frame_prompts:
                # add_prompt resets the whole inference state, so it can only
                # be called once per pass -- but it stores its boxes into
                # per-frame slots that propagation reads on every frame it
                # visits. So prompt with the earliest frame's boxes, then
                # plant the remaining frames' boxes into those same slots
                # directly.
                #
                # Piling every box onto the first frame instead -- which is
                # what this did first -- plants each object where it is not
                # yet, so SAM 3 either segments whatever else is at that spot
                # or instantiates nothing, and the detection is lost.
                seed_frame = min(frame_prompts.keys())
                first_boxes = [b for _oid, b in frame_prompts[seed_frame]]
                add_kwargs = dict(
                    frame_idx=seed_frame,
                    boxes_xywh=first_boxes,
                    box_labels=[1] * len(first_boxes),
                )
                if text_query is not None:
                    add_kwargs['text_str'] = text_query
                self._video_predictor.add_prompt(state, **add_kwargs)

                for fidx in sorted(frame_prompts.keys()):
                    if fidx != seed_frame:
                        self._plant_box_prompts(
                            state, fidx,
                            [b for _oid, b in frame_prompts[fidx]])
                    for _obj_id, box_rel_xywh in frame_prompts[fidx]:
                        seeds.append((_obj_id, fidx, box_rel_xywh))
                self._seeds_submitted += len(seeds)
            elif text_query is not None:
                self._video_predictor.add_prompt(
                    state,
                    frame_idx=text_seed_frame,
                    text_str=text_query,
                )
        else:
            # SAM 3.0 accepts an explicit obj_id per box, so identity is
            # preserved here already and no remap is needed.
            for frame_idx, prompts in frame_prompts.items():
                for obj_id, box_rel_xywh in prompts:
                    self._video_predictor.add_prompt(
                        state,
                        frame_idx=frame_idx,
                        boxes_xywh=[box_rel_xywh],
                        box_labels=[1],
                        obj_id=obj_id,
                    )
            if text_query is not None:
                self._video_predictor.add_prompt(
                    state,
                    frame_idx=text_seed_frame,
                    text_str=text_query,
                )

        return seeds

    def _plant_box_prompts(self, state, frame_idx, boxes_rel_xywh):
        """Add box prompts on a frame other than the one add_prompt was given.

        ``add_prompt`` resets the entire inference state on every call, so it
        cannot be used a second time to prompt another frame. What it does
        with the boxes, though, is write them into per-frame slots that
        propagation consults on each frame it reaches, so writing those same
        slots directly is how a pass gets prompts on more than one frame.
        Mirrors add_prompt's box branch, minus the reset and the backbone
        setup the first call already did.
        """
        import torch
        from sam3.model.box_ops import box_xywh_to_cxcywh

        model = getattr(self._video_predictor, '_model', self._video_predictor)
        boxes_xywh = torch.as_tensor(boxes_rel_xywh, dtype=torch.float32)
        box_labels = torch.ones(len(boxes_rel_xywh), dtype=torch.long)
        boxes_cxcywh = box_xywh_to_cxcywh(boxes_xywh)

        state['per_frame_raw_box_input'][frame_idx] = (
            boxes_cxcywh, box_labels)
        _b, _l, geometric_prompt = model._get_visual_prompt(
            state, frame_idx, boxes_cxcywh, box_labels)
        state['per_frame_geometric_prompt'][frame_idx] = geometric_prompt

    def _propagate_per_class(self, state, class_terms, text_seed_frame):
        """
        Run one full propagation pass per class term. Each pass gets an
        obj_id offset so IDs are unique across classes; each returned
        obj_id is tagged with its class term in ``_obj_id_to_class`` so
        ``_build_propagated_tracks`` emits the right label.
        """
        results = {}
        id_offset = 0
        for class_term in class_terms:
            # Clear any prior prompts on this state. SAM 3.1 resets
            # on add_prompt; SAM 3.0 needs an explicit reset.
            try:
                self._video_predictor.reset_state(state)
            except Exception:
                pass

            self._video_predictor.add_prompt(
                state,
                frame_idx=text_seed_frame,
                text_str=class_term,
            )

            max_out_id = -1
            for frame_idx, frame_results in \
                    self._video_predictor.propagate_in_video(state):
                obj_ids = np.asarray(frame_results['out_obj_ids']).astype(
                    np.int64)
                if len(obj_ids) == 0:
                    continue
                remapped_ids = obj_ids + id_offset
                if len(remapped_ids):
                    max_out_id = max(max_out_id, int(remapped_ids.max()))
                for oid in remapped_ids:
                    self._obj_id_to_class[int(oid)] = class_term
                remapped_fr = {
                    'out_obj_ids': remapped_ids,
                    'out_boxes_xywh': frame_results['out_boxes_xywh'],
                    'out_binary_masks': frame_results['out_binary_masks'],
                    'out_probs': frame_results.get('out_probs', []),
                }
                self._collect_frame_results(results, frame_idx, remapped_fr)

            if max_out_id >= 0:
                id_offset = max_out_id + 1

        # SAM 3.1's internal NMS only runs within a single class pass, so the
        # same physical object can get picked up independently by multiple
        # class queries and emerge as overlapping tracks. Drop duplicates
        # across class passes by track-level IoU — keep the track with the
        # higher mean detection score.
        return self._dedupe_tracks_by_iou(results, self._iou_threshold)

    def _dedupe_tracks_by_iou(self, results, iou_threshold):
        """
        Track-level NMS across per-class propagation passes, using
        SAM3's own ``apply_track_nms`` (vectorized, Numba-JIT'd). It
        aggregates total intersection / total union across every shared
        frame pair and suppresses lower-scored tracks whose track-IoU
        against a higher-scored track exceeds ``iou_threshold``. Returns
        a filtered results dict with losing tracks removed entirely.
        """
        if not results:
            return results

        from sam3.train.nms_helper import apply_track_nms

        # Collect per-track frame sequences
        per_track = {}
        for (oid, frame), (_mask, box, score) in results.items():
            per_track.setdefault(oid, []).append((frame, box, score))

        oids = list(per_track.keys())
        frame_indices = sorted({
            f for entries in per_track.values() for f, _, _ in entries
        })
        frame_to_col = {f: i for i, f in enumerate(frame_indices)}
        num_frames = len(frame_indices)

        # Build the dense [num_tracks, num_frames, 4] array SAM3 expects,
        # with NaN-padding for frames where each track has no detection.
        track_detections = []
        scores = np.zeros(len(oids), dtype=np.float32)
        for t_idx, oid in enumerate(oids):
            bboxes = np.full((num_frames, 4), np.nan, dtype=np.float32)
            entries = per_track[oid]
            track_score_sum = 0.0
            for frame, box, score in entries:
                bboxes[frame_to_col[frame]] = box
                track_score_sum += score
            track_detections.append({
                "track_idx": t_idx,
                "bboxes": bboxes,
                "score": track_score_sum / max(1, len(entries)),
            })
            scores[t_idx] = track_score_sum / max(1, len(entries))

        keep_idx = set(apply_track_nms(
            track_detections, scores, float(iou_threshold),
        ))
        kept_oids = {oids[i] for i in keep_idx}

        return {
            (oid, f): entry for (oid, f), entry in results.items()
            if oid in kept_oids
        }

    def _collect_frame_results(self, results, frame_idx, frame_results,
                               overwrite=True):
        """Extract per-object masks from a propagation frame result."""
        obj_ids = np.array(frame_results['out_obj_ids'])
        boxes_xywh = np.array(frame_results['out_boxes_xywh'])
        masks = np.array(frame_results['out_binary_masks'])
        probs = np.array(frame_results.get('out_probs', []))

        for i, oid in enumerate(obj_ids):
            oid = int(oid)
            key = (oid, frame_idx)
            if not overwrite and key in results:
                continue
            bx = boxes_xywh[i]
            ax1 = bx[0] * self._img_width
            ay1 = bx[1] * self._img_height
            ax2 = (bx[0] + bx[2]) * self._img_width
            ay2 = (bx[1] + bx[3]) * self._img_height
            score = float(probs[i]) if i < len(probs) else 1.0
            # Packed, not raw: these tuples are held for every object on
            # every frame until the whole video has been propagated.
            results[key] = (_PackedMask(masks[i]),
                            [ax1, ay1, ax2, ay2], score)

    # ------------------------------------------------------------------
    # Main refine method
    # ------------------------------------------------------------------

    @report_cuda_errors("SAM3 track refinement")
    def refine(self, ts, image_data, tracks):
        """
        Refine tracks for the current frame.

        When ``propagate_tracked`` is enabled, seed boxes are tracked
        across frames using SAM3's native video predictor with memory
        attention.  When disabled, each frame is processed independently.
        Args:
            ts: Timestamp for the current frame
            image_data: Image container for the current frame
            tracks: ObjectTrackSet containing tracks to refine

        Returns:
            ObjectTrackSet: Refined tracks
        """
        if not ts.has_valid_frame():
            raise RuntimeError("Frame timestamps must contain frame IDs")

        frame_id = ts.get_frame()
        img_np = image_to_rgb_numpy(image_data)

        # "Replace" mode: drop all pre-existing input annotations so only
        # newly text-detected objects are emitted. Applies to both the
        # per-frame and video-propagation paths (both derive their existing
        # seeds from ``tracks``).
        if self._replace_existing:
            tracks = ObjectTrackSet([])

        # Resolve each input track's ID to the ID we'll actually use. New
        # input IDs get registered; any that collide with an ID already
        # handed out to a refiner-created track get remapped to the next
        # unused ID. Subsequent frames reuse the mapping.
        for track in tracks.tracks():
            self._get_or_remap_input_id(track.id)

        # Extract current frame's track states, keyed by the (possibly
        # remapped) resolved ID so downstream output uses that ID too.
        track_states = {}  # resolved_id -> (track, state, detection)
        for track in tracks.tracks():
            resolved_id = self._input_id_remap[track.id]
            for state in track:
                if state.frame_id == frame_id:
                    detection = state.detection()
                    track_states[resolved_id] = (track, state, detection)
                    break

        if self._propagate_tracked:
            return self._refine_with_video_predictor(
                ts, frame_id, img_np, tracks, track_states
            )
        else:
            return self._refine_per_frame(
                ts, frame_id, img_np, tracks, track_states
            )

    # ------------------------------------------------------------------
    # Track-ID allocation helpers
    # ------------------------------------------------------------------

    def _allocate_next_id(self):
        """Return the next unused track ID and mark it as assigned."""
        while (self._next_track_id in self._assigned_ids
               or self._next_track_id in self._known_input_ids):
            self._next_track_id += 1
        result = self._next_track_id
        self._assigned_ids.add(result)
        self._next_track_id += 1
        return result

    def _get_or_remap_input_id(self, original_id):
        """
        Return the ID to use for an input track. If the input's original
        ID collides with an ID already handed out to a refiner-created
        track, remap it to a fresh unused ID. The remap is cached so the
        same input track keeps the same resolved ID across frames.
        """
        if original_id in self._input_id_remap:
            return self._input_id_remap[original_id]
        if original_id in self._assigned_ids:
            new_id = self._allocate_next_id()
            self._input_id_remap[original_id] = new_id
            self._known_input_ids.add(new_id)
            return new_id
        self._input_id_remap[original_id] = original_id
        self._known_input_ids.add(original_id)
        return original_id

    # ------------------------------------------------------------------
    # Video-predictor path (propagate_tracked=True)
    # ------------------------------------------------------------------

    def _refine_with_video_predictor(self, ts, frame_id, img_np,
                                     tracks, track_states):
        """
        Refine using SAM3 video predictor for native temporal tracking.

        Accumulates frames and seed box prompts.  On each call, runs
        SAM3 video propagation over the full buffer to produce masks
        for all tracked objects on all frames seen so far.  Only the
        current frame's results are used for track output; previous
        frames' results were already emitted.
        """
        # Convert to PIL and store
        pil = PILImage.fromarray(img_np)
        self._pil_frames.append(pil)
        if self._img_width == 0:
            self._img_width, self._img_height = pil.size

        local_frame_idx = len(self._pil_frames) - 1
        self._timestamps[local_frame_idx] = ts

        # An external detection set replaces the input tracks as the seed
        # source when one was supplied, and is matched by frame position.
        if self._external_dets:
            w, h = self._img_width, self._img_height

            # A detection that continues an object already being propagated
            # is a LINK, not a new object: SAM 3 is already carrying that
            # object forward and re-prompting it would start a second track
            # on the same animal. Only detections that match nothing alive
            # become new seeds.
            #
            # Prompting every detection instead -- which is what this did
            # first -- asks SAM 3 to hold one object per detection, so a
            # sequence with a few thousand detections grows without bound
            # and is killed long before it finishes.
            for oid in [o for o, (lf, _b) in self._active_seeds.items()
                        if local_frame_idx - lf > self._link_max_gap]:
                del self._active_seeds[oid]

            for det_id, (x1, y1, x2, y2), conf, cls in \
                    self._external_dets.get(local_frame_idx, []):
                if conf < self._detections_threshold:
                    self._below_threshold += 1
                    continue

                # Detector boxes routinely run past the image edge --
                # negative origins and corners beyond width/height are both
                # common in the shared detection sets. SAM 3 asserts its
                # prompts are normalised inside [0, 1], so clamp here and
                # drop anything with no area left afterwards rather than
                # letting the assertion take down the whole run.
                nx1 = min(max(x1 / w, 0.0), 1.0)
                ny1 = min(max(y1 / h, 0.0), 1.0)
                nx2 = min(max(x2 / w, 0.0), 1.0)
                ny2 = min(max(y2 / h, 0.0), 1.0)
                if nx2 <= nx1 or ny2 <= ny1:
                    self._clamped_out += 1
                    continue

                best_oid, best_iou = None, 0.0
                for oid, (_lf, last_box) in self._active_seeds.items():
                    overlap = _box_iou((x1, y1, x2, y2), last_box)
                    if overlap > best_iou:
                        best_oid, best_iou = oid, overlap

                if best_oid is not None and best_iou >= self._link_iou:
                    # Continuation of a live object; refresh its position so
                    # the next frame links against where it actually is.
                    self._active_seeds[best_oid] = (
                        local_frame_idx, (x1, y1, x2, y2))
                    self._linked += 1
                    continue

                if len(self._active_seeds) >= self._max_seed_objects:
                    # SAM 3 caps how many objects it will track at once, and
                    # the memory cost is per object. Drop the detection
                    # rather than exceed it, and say how many were dropped.
                    self._seed_capacity_drops += 1
                    continue

                box_rel = [nx1, ny1, nx2 - nx1, ny2 - ny1]
                self._frame_prompts.setdefault(local_frame_idx, []).append(
                    (det_id, box_rel)
                )
                self._obj_id_to_class[det_id] = cls
                self._active_seeds[det_id] = (
                    local_frame_idx, (x1, y1, x2, y2))
                self._seeded += 1
            return ObjectTrackSet([])

        # Collect seed box prompts from input tracks on this frame
        for tid, (track, state, det) in track_states.items():
            bbox = det.bounding_box
            x1, y1 = bbox.min_x(), bbox.min_y()
            x2, y2 = bbox.max_x(), bbox.max_y()
            w, h = self._img_width, self._img_height
            box_rel = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]
            self._frame_prompts.setdefault(local_frame_idx, []).append(
                (tid, box_rel)
            )
            class_name = ''
            try:
                class_name = det.type.get_most_likely_class()
            except Exception:
                pass
            self._obj_id_to_class[tid] = class_name

        # Store the text query for the video predictor.  SAM3's
        # add_prompt(text_str=...) applies globally to ALL frames
        # and resets the inference state each time it is called, so
        # we must only call it once (on the first frame).  The video
        # predictor's detector will run on every frame automatically
        # when a text prompt is set.
        if self._add_new_objects and self._text_query_list:
            if local_frame_idx == 0:
                text_query = ', '.join(self._text_query_list)
                self._text_prompt_frames[0] = text_query

        # Accumulate only — propagation runs once in finalize() after
        # all frames have been collected.
        return ObjectTrackSet([])

    @report_cuda_errors("SAM3 track refinement")
    def finalize(self):
        """
        Called by the pipeline after all frames have been processed.
        Runs SAM3 video propagation over the full accumulated buffer
        and returns the complete set of tracked objects.
        """
        if not self._propagate_tracked:
            return ObjectTrackSet([])

        try:
            self._run_finalize_propagation()
        except Exception as e:
            import sys, traceback
            sys.stderr.write(f"[SAM3 Refiner] ERROR in propagation: {e}\n")
            traceback.print_exc(file=sys.stderr)
            # Propagate so the job fails visibly (non-zero exit) instead of
            # silently exiting 0 with no tracks. The report_cuda_errors
            # decorator turns this into a DIVE-surfaced ERROR: line.
            raise
        finally:
            self._pil_frames.cleanup()

        # Split tracks that have large spatial jumps (identity switches)
        self._split_jumping_tracks()

        # Stitch tracks that end near where another begins (same object
        # that got different IDs across chunk boundaries)
        self._stitch_tracks()

        output_tracks = []
        for obj_id, history in self._propagated_tracks.items():
            if len(history) > 0:
                output_tracks.append(Track(obj_id, list(history)))

        return ObjectTrackSet(output_tracks)

    def _split_jumping_tracks(self):
        """
        Post-process propagated tracks to split any that jump
        unreasonably far between consecutive frames.  A jump is
        detected when the bounding-box center moves more than the
        diagonal of the box (i.e. more than its own size).
        """
        new_tracks = {}
        next_id = max(self._propagated_tracks.keys(), default=-1) + 1

        for oid, states in list(self._propagated_tracks.items()):
            states.sort(key=lambda s: s.frame_id)
            if len(states) <= 1:
                new_tracks[oid] = states
                continue

            segments = [[states[0]]]
            for i in range(1, len(states)):
                prev_det = segments[-1][-1].detection()
                curr_det = states[i].detection()
                pb = prev_det.bounding_box
                cb = curr_det.bounding_box

                # Center of previous and current boxes
                pcx = (pb.min_x() + pb.max_x()) / 2
                pcy = (pb.min_y() + pb.max_y()) / 2
                ccx = (cb.min_x() + cb.max_x()) / 2
                ccy = (cb.min_y() + cb.max_y()) / 2

                dist = ((ccx - pcx) ** 2 + (ccy - pcy) ** 2) ** 0.5

                # Max allowed jump: larger of the two box diagonals
                pw = pb.max_x() - pb.min_x()
                ph = pb.max_y() - pb.min_y()
                cw = cb.max_x() - cb.min_x()
                ch = cb.max_y() - cb.min_y()
                max_diag = max((pw**2 + ph**2) ** 0.5,
                               (cw**2 + ch**2) ** 0.5)

                # Threshold: 2.5x diagonal with a floor of 100px
                # (small objects need room) and a ceiling of 200px
                # (large objects shouldn't link across the frame).
                threshold = min(max(max_diag * 2.5, 100), 200)

                if dist > threshold:
                    # Start a new segment
                    segments.append([states[i]])
                else:
                    segments[-1].append(states[i])

            # Keep the first (longest existing ID) segment under the
            # original ID; assign new IDs to split-off segments
            best_idx = max(range(len(segments)), key=lambda j: len(segments[j]))
            for j, seg in enumerate(segments):
                if j == best_idx:
                    new_tracks[oid] = seg
                elif len(seg) >= 2:
                    new_tracks[next_id] = seg
                    next_id += 1
                # Drop single-frame split-off fragments

        self._propagated_tracks = new_tracks

    @staticmethod
    def _track_velocity(states, from_end=True, n_frames=5):
        """
        Estimate velocity (vx, vy) in pixels/frame from the last (or
        first) *n_frames* states of a track using linear regression of
        the bounding-box centres.
        """
        if len(states) < 2:
            return 0.0, 0.0
        if from_end:
            seg = states[-n_frames:]
        else:
            seg = states[:n_frames]
        frames = [s.frame_id for s in seg]
        cxs = [(s.detection().bounding_box.min_x() +
                s.detection().bounding_box.max_x()) / 2 for s in seg]
        cys = [(s.detection().bounding_box.min_y() +
                s.detection().bounding_box.max_y()) / 2 for s in seg]
        n = len(frames)
        if n < 2:
            return 0.0, 0.0
        fm = sum(frames) / n
        xm = sum(cxs) / n
        ym = sum(cys) / n
        denom = sum((f - fm) ** 2 for f in frames)
        if denom < 1e-9:
            return 0.0, 0.0
        vx = sum((f - fm) * (x - xm) for f, x in zip(frames, cxs)) / denom
        vy = sum((f - fm) * (y - ym) for f, y in zip(frames, cys)) / denom
        return vx, vy

    def _stitch_tracks(self, max_frame_gap=5):
        """
        Merge tracks where one ends near where another begins.

        Uses velocity-based position prediction: extrapolate the ending
        track's motion to estimate where it would be at the starting
        track's first frame, then compare that predicted position to
        the actual start.  This allows fast-moving objects to be linked
        across gaps that would be too large for a raw-distance check.

        Also falls back to raw distance for slow/stationary objects.
        """
        track_info = {}
        for oid, states in self._propagated_tracks.items():
            if not states:
                continue
            states.sort(key=lambda s: s.frame_id)
            sdet = states[0].detection()
            edet = states[-1].detection()
            sb = sdet.bounding_box
            eb = edet.bounding_box
            evx, evy = self._track_velocity(states, from_end=True)
            svx, svy = self._track_velocity(states, from_end=False)
            track_info[oid] = {
                'start_frame': states[0].frame_id,
                'end_frame': states[-1].frame_id,
                'start_cx': (sb.min_x() + sb.max_x()) / 2,
                'start_cy': (sb.min_y() + sb.max_y()) / 2,
                'start_diag': ((sb.max_x()-sb.min_x())**2 + (sb.max_y()-sb.min_y())**2) ** 0.5,
                'end_cx': (eb.min_x() + eb.max_x()) / 2,
                'end_cy': (eb.min_y() + eb.max_y()) / 2,
                'end_diag': ((eb.max_x()-eb.min_x())**2 + (eb.max_y()-eb.min_y())**2) ** 0.5,
                'end_vx': evx, 'end_vy': evy,
                'start_vx': svx, 'start_vy': svy,
            }

        merged = True
        while merged:
            merged = False
            candidates = []
            for oid_end, ie in track_info.items():
                for oid_start, ist in track_info.items():
                    if oid_end == oid_start:
                        continue
                    frame_gap = ist['start_frame'] - ie['end_frame']
                    if frame_gap < 1 or frame_gap > max_frame_gap:
                        continue

                    # Raw distance between endpoints
                    raw_dist = ((ie['end_cx'] - ist['start_cx'])**2 +
                                (ie['end_cy'] - ist['start_cy'])**2) ** 0.5

                    # Predicted position by extrapolating end velocity
                    pred_cx = ie['end_cx'] + ie['end_vx'] * frame_gap
                    pred_cy = ie['end_cy'] + ie['end_vy'] * frame_gap
                    pred_dist = ((pred_cx - ist['start_cx'])**2 +
                                 (pred_cy - ist['start_cy'])**2) ** 0.5

                    # Also try back-projecting the start velocity
                    bpred_cx = ist['start_cx'] - ist['start_vx'] * frame_gap
                    bpred_cy = ist['start_cy'] - ist['start_vy'] * frame_gap
                    bpred_dist = ((bpred_cx - ie['end_cx'])**2 +
                                  (bpred_cy - ie['end_cy'])**2) ** 0.5

                    # Use the best (smallest) of raw, forward-predicted,
                    # and backward-predicted distances
                    best_dist = min(raw_dist, pred_dist, bpred_dist)

                    max_diag = max(ie['end_diag'], ist['start_diag'], 10)
                    threshold = min(max(max_diag * 2.5, 100), 200)
                    if best_dist < threshold:
                        candidates.append((best_dist, oid_end, oid_start))

            if not candidates:
                break

            candidates.sort()
            used = set()
            for dist_val, oid_end, oid_start in candidates:
                if oid_end in used or oid_start in used:
                    continue
                self._propagated_tracks[oid_end].extend(
                    self._propagated_tracks[oid_start])
                self._propagated_tracks[oid_end].sort(
                    key=lambda s: s.frame_id)
                del self._propagated_tracks[oid_start]

                # Update track_info for merged track
                states = self._propagated_tracks[oid_end]
                edet = states[-1].detection()
                eb = edet.bounding_box
                evx, evy = self._track_velocity(states, from_end=True)
                track_info[oid_end]['end_frame'] = states[-1].frame_id
                track_info[oid_end]['end_cx'] = (eb.min_x() + eb.max_x()) / 2
                track_info[oid_end]['end_cy'] = (eb.min_y() + eb.max_y()) / 2
                track_info[oid_end]['end_diag'] = (
                    (eb.max_x()-eb.min_x())**2 + (eb.max_y()-eb.min_y())**2) ** 0.5
                track_info[oid_end]['end_vx'] = evx
                track_info[oid_end]['end_vy'] = evy
                del track_info[oid_start]

                used.add(oid_end)
                used.add(oid_start)
                merged = True

    def _run_finalize_propagation(self):
        """
        Run SAM3 video propagation on the full accumulated buffer.
        Called once after all frames have been collected.
        """
        if not self._pil_frames:
            return
        if not self._frame_prompts and not self._text_prompt_frames:
            return

        if self._external_dets:
            print('[SAM3] external detections: %d seeded, %d linked to a live '
                  'object, %d dropped at the %d-object cap, %d clamped out, '
                  '%d below the %.3f confidence gate'
                  % (self._seeded, self._linked, self._seed_capacity_drops,
                     self._max_seed_objects, self._clamped_out,
                     self._below_threshold, self._detections_threshold),
                  flush=True)

        all_results = self._run_video_propagation()

        if self._external_dets and self._seeds_submitted:
            print('[SAM3] seed prompts: %d submitted across chunks, %d picked '
                  'up by SAM 3 (%.1f%%)'
                  % (self._seeds_submitted, self._seeds_mapped,
                     100.0 * self._seeds_mapped / self._seeds_submitted),
                  flush=True)

        # Collect the set of object IDs we explicitly prompted
        prompted_ids = set()
        for prompts in self._frame_prompts.values():
            for obj_id, _ in prompts:
                prompted_ids.add(obj_id)

        self._propagated_tracks.clear()
        for (oid, fidx), result_tuple in all_results.items():
            mask, box_xyxy = result_tuple[0], result_tuple[1]
            score = result_tuple[2] if len(result_tuple) > 2 else 1.0

            # When add_new_objects is disabled (track selections), only
            # keep tracks the user explicitly seeded — drop SAM3
            # auto-discovered objects.
            if not self._add_new_objects and oid not in prompted_ids:
                continue

            mask_area = mask.area()
            if self._filter_by_quality and mask_area < self._min_mask_area:
                continue

            class_name = self._obj_id_to_class.get(oid, '')
            if not class_name and self._text_query_list:
                class_name = self._text_query_list[0]
            if not class_name:
                class_name = 'unknown'
            if mask_area > 0:
                bbox = BoundingBoxD(*mask.bbox)
            else:
                bbox = BoundingBoxD(box_xyxy[0], box_xyxy[1],
                                    box_xyxy[2], box_xyxy[3])

            confidence = float(score)
            dot = DetectedObjectType(class_name, confidence)
            det = DetectedObject(bbox, confidence, dot)

            # Set the full binary mask (cropped to bbox) on the
            # detection.  The CSV writer's mask_to_poly_points path
            # will convert this to multi-contour polygons with proper
            # (poly)/(hole) tags, handling disjoint mask regions that
            # a single flattened polygon cannot represent.
            _set_mask_on_detection(det, mask, bbox)

            if fidx not in self._timestamps:
                continue
            frame_ts = self._timestamps[fidx]

            new_state = ObjectTrackState(frame_ts, det)
            self._propagated_tracks.setdefault(oid, []).append(new_state)

        # Include input seed detections that SAM3 may not have yielded
        for local_idx, prompts in self._frame_prompts.items():
            if local_idx not in self._timestamps:
                continue
            frame_ts = self._timestamps[local_idx]
            for tid, box_rel in prompts:
                if (tid, local_idx) not in all_results:
                    w, h = self._img_width, self._img_height
                    ax1 = box_rel[0] * w
                    ay1 = box_rel[1] * h
                    ax2 = (box_rel[0] + box_rel[2]) * w
                    ay2 = (box_rel[1] + box_rel[3]) * h
                    bbox = BoundingBoxD(ax1, ay1, ax2, ay2)
                    class_name = self._obj_id_to_class.get(tid, 'unknown')
                    dot = DetectedObjectType(
                        _safe_class_name(class_name), 1.0)
                    det = DetectedObject(bbox, 1.0, dot)
                    state = ObjectTrackState(frame_ts, det)
                    self._propagated_tracks.setdefault(tid, []).append(state)

        for oid in self._propagated_tracks:
            self._propagated_tracks[oid].sort(key=lambda s: s.frame_id)

    # ------------------------------------------------------------------
    # Per-frame path (propagate_tracked=False, for text queries)
    # ------------------------------------------------------------------

    def _refine_per_frame(self, ts, frame_id, img_np, tracks, track_states):
        """
        Refine each frame independently using the image predictor.

        Two modes live here:

        * Text-query refinement (``add_new_objects=True``) — Grounding DINO
          detects new objects on every frame, independent of neighbors.
        * Box-propagation tracking (``track_new_objects=True``) — seed
          detections present on earlier frames are carried forward by
          re-segmenting their last-known box on each subsequent frame
          (SAM2-style). This is the right mode for user-selection tracking
          pipelines where there is no text query to drive a detector.
        """
        boxes_to_segment = []
        # Source tags:
        #   ('existing',   tid)                — input track state on this frame
        #   ('propagated', tid)                — prior-frame tracked object
        #   ('new',        score, class_name)  — text-detected new object
        box_sources = []

        # Re-segment existing input-track boxes
        if self._resegment_existing:
            for tid, (track, state, det) in track_states.items():
                bbox = det.bounding_box
                box = [bbox.min_x(), bbox.min_y(), bbox.max_x(), bbox.max_y()]
                boxes_to_segment.append(box)
                box_sources.append(('existing', tid))

        # Propagate previously-tracked objects that have no input state on
        # this frame — re-segment their last known box.
        if self._track_new_objects:
            for tid, tdata in self._tracked_objects.items():
                if tid not in track_states:
                    boxes_to_segment.append(list(tdata['last_box']))
                    box_sources.append(('propagated', tid))

        # Detect new objects with text query
        if self._add_new_objects:
            new_detections = self._model_manager.detect_with_text(
                img_np, self._text_query_list,
                self._detection_threshold, self._text_threshold,
            )
            suppress_boxes = [
                [det.bounding_box.min_x(), det.bounding_box.min_y(),
                 det.bounding_box.max_x(), det.bounding_box.max_y()]
                for _, (_, _, det) in track_states.items()
            ]
            if self._track_new_objects:
                for tdata in self._tracked_objects.values():
                    suppress_boxes.append(list(tdata['last_box']))
            for box, score, class_name in new_detections:
                overlaps = False
                for sb in suppress_boxes:
                    if compute_iou(box, sb) > self._iou_threshold:
                        overlaps = True
                        break
                new_count = len([s for s in box_sources if s[0] == 'new'])
                if not overlaps and new_count < self._max_new_objects:
                    boxes_to_segment.append(box)
                    box_sources.append(('new', score, class_name))

        # Segment all boxes with SAM image predictor
        if len(boxes_to_segment) > 0:
            masks = self._model_manager.segment_with_sam(img_np, boxes_to_segment)
        else:
            masks = []

        # Build output tracks
        output_tracks = []
        processed_track_ids = set()
        seen_tracked_ids = set()

        for i, (mask, source) in enumerate(zip(masks, box_sources)):
            mask_area = np.sum(mask)
            if self._filter_by_quality and mask_area < self._min_mask_area:
                if source[0] == 'existing':
                    processed_track_ids.add(source[1])
                elif source[0] == 'propagated':
                    # Mask failed quality check — don't update the tracker;
                    # it will accumulate a 'lost' count below.
                    pass
                continue

            if source[0] == 'existing':
                tid = source[1]
                track, old_state, old_det = track_states[tid]
                processed_track_ids.add(tid)

                new_det = self._create_refined_detection(
                    old_det, mask, self._adjust_boxes
                )
                new_state = ObjectTrackState(ts, new_det)

                new_history = []
                for state in track:
                    if state.frame_id == frame_id:
                        new_history.append(new_state)
                    else:
                        new_history.append(state)

                output_tracks.append(Track(tid, new_history))

                # Register/refresh this track in the propagation tracker so
                # its box carries forward on subsequent frames.
                if self._track_new_objects:
                    bbox = new_det.bounding_box
                    new_box = [bbox.min_x(), bbox.min_y(),
                               bbox.max_x(), bbox.max_y()]
                    class_name = ''
                    try:
                        class_name = old_det.type.get_most_likely_class()
                    except Exception:
                        pass
                    entry = self._tracked_objects.get(tid)
                    if entry is None:
                        self._tracked_objects[tid] = {
                            'class_name': class_name,
                            'last_box': new_box,
                            'lost': 0,
                            'history': [new_state],
                        }
                    else:
                        entry['last_box'] = new_box
                        entry['lost'] = 0
                        entry['history'].append(new_state)
                    seen_tracked_ids.add(tid)

            elif source[0] == 'propagated':
                tid = source[1]
                tdata = self._tracked_objects[tid]
                new_det = self._detection_from_mask(
                    mask, boxes_to_segment[i], tdata['class_name'], 1.0
                )
                if new_det is None:
                    continue
                new_state = ObjectTrackState(ts, new_det)
                tdata['history'].append(new_state)
                bbox = new_det.bounding_box
                tdata['last_box'] = [bbox.min_x(), bbox.min_y(),
                                     bbox.max_x(), bbox.max_y()]
                tdata['lost'] = 0
                seen_tracked_ids.add(tid)

            else:
                score, class_name = source[1], source[2]
                det = self._detection_from_mask(
                    mask, boxes_to_segment[i], class_name, score
                )
                if det is not None:
                    new_state = ObjectTrackState(ts, det)
                    tid = self._allocate_next_id()
                    output_tracks.append(Track(tid, [new_state]))
                    if self._track_new_objects:
                        bbox = det.bounding_box
                        self._tracked_objects[tid] = {
                            'class_name': class_name,
                            'last_box': [bbox.min_x(), bbox.min_y(),
                                         bbox.max_x(), bbox.max_y()],
                            'lost': 0,
                            'history': [new_state],
                        }
                        seen_tracked_ids.add(tid)

        # Age out tracked objects that weren't refreshed this frame. Emit
        # the full track history for anything still alive so downstream
        # writers see the accumulated states.
        if self._track_new_objects:
            expired = []
            for tid, tdata in self._tracked_objects.items():
                if tid not in seen_tracked_ids:
                    tdata['lost'] += 1
                if tdata['lost'] > self._lost_track_frames:
                    expired.append(tid)

            for tid, tdata in self._tracked_objects.items():
                if tid not in processed_track_ids and len(tdata['history']) > 0:
                    output_tracks.append(Track(tid, list(tdata['history'])))
                    processed_track_ids.add(tid)

            for tid in expired:
                del self._tracked_objects[tid]

        # Pass through unprocessed input tracks. Rebuild the Track with
        # the resolved ID when the input's original ID got remapped.
        if not self._resegment_existing:
            for tid, (track, state, det) in track_states.items():
                if tid not in processed_track_ids:
                    output_tracks.append(
                        track if track.id == tid else Track(tid, list(track))
                    )
                    processed_track_ids.add(tid)

        for track in tracks.tracks():
            tid = self._input_id_remap.get(track.id, track.id)
            if tid not in processed_track_ids and tid not in track_states:
                output_tracks.append(
                    track if track.id == tid else Track(tid, list(track))
                )

        return ObjectTrackSet(output_tracks)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _detection_from_mask(self, mask, det_box, class_name, score):
        """Create a DetectedObject from a mask, or None if invalid."""
        if not isinstance(mask, np.ndarray):
            import torch
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            else:
                mask = np.array(mask)

        if self._adjust_boxes:
            bbox = box_from_mask(mask)
            if bbox is None:
                return None
        else:
            bbox = BoundingBoxD(det_box[0], det_box[1], det_box[2], det_box[3])

        dot = DetectedObjectType(
            _safe_class_name(class_name, self._text_query_list[0]
                             if self._text_query_list else None), score)
        det = DetectedObject(bbox, score, dot)

        if self._output_type in ('polygon', 'both'):
            _set_polygon_on_detection(det, mask, self._polygon_simplification)

        return det

    def _create_refined_detection(self, old_det, mask, adjust_box):
        """Create a refined detection from an existing detection and new mask."""
        if adjust_box:
            bbox = box_from_mask(mask)
            if bbox is None:
                bbox = old_det.bounding_box
        else:
            bbox = old_det.bounding_box

        det_type = old_det.type
        confidence = old_det.confidence
        new_det = DetectedObject(bbox, confidence, det_type)

        if self._output_type in ('polygon', 'both'):
            _set_polygon_on_detection(new_det, mask, self._polygon_simplification)

        return new_det


class Sam3DetectionRefiner(RefineDetections):
    """
    SAM3-based Detection Refiner

    This refiner uses SAM3 to add segmentation masks to detections.
    It operates on DetectedObjectSet and adds polygon masks to each detection.

    Key features:
    - Re-segments detection bounding boxes with SAM3 for high-quality masks
    - Supports loading from local model files or HuggingFace
    - Generates polygon outputs from masks
    - Can optionally overwrite existing masks

    Example:
        >>> from viame.pytorch.sam3_refiner import Sam3DetectionRefiner
        >>> refiner = Sam3DetectionRefiner()
        >>> refiner.set_configuration({})
        >>> refined_dets = refiner.refine(image, detections)
    """

    def __init__(self):
        RefineDetections.__init__(self)

        self._kwiver_config = {
            'sam_model_id': 'facebook/sam2.1-hiera-large',
            'model_config': '',
            'grounding_model_id': '',
            'device': 'cuda',
            'replace_existing': 'False',
            'output_type': 'polygon',
            'polygon_simplification': '0.01',
            'text_query': '',
            'detection_threshold': '0.3',
            'text_threshold': '0.25',
            'iou_threshold': '0.5',
            'add_new_objects': 'False',
            'max_new_objects': '50',
            'min_mask_area': '10',
        }

        self._model_manager = SAM3ModelManager()

    def get_configuration(self):
        """Get the algorithm configuration."""
        cfg = super(RefineDetections, self).get_configuration()
        for key, value in self._kwiver_config.items():
            cfg.set_value(key, str(value))
        return cfg

    @report_cuda_errors("SAM3 detection refiner initialization")
    def set_configuration(self, cfg_in):
        """Set the algorithm configuration and initialize models."""
        cfg = self.get_configuration()
        vital_config_update(cfg, cfg_in)

        for key in self._kwiver_config.keys():
            self._kwiver_config[key] = str(cfg.get_value(key))

        # Create a minimal config object for model initialization
        class MinimalConfig:
            pass

        model_config = MinimalConfig()
        model_config.sam_model_id = self._kwiver_config['sam_model_id']
        model_config.model_config = self._kwiver_config.get('model_config', '')
        if model_config.model_config == '':
            model_config.model_config = None
        model_config.device = self._kwiver_config['device']

        # Load Grounding DINO if configured (for text-based detection)
        gid = self._kwiver_config.get('grounding_model_id', '')
        if gid and gid.lower() not in ('', 'none', 'false'):
            model_config.grounding_model_id = gid
        else:
            model_config.grounding_model_id = None

        self._model_manager.init_models(model_config, use_video_predictor=False)

        # Parse config values
        self._replace_existing = parse_bool(self._kwiver_config['replace_existing'])
        self._output_type = self._kwiver_config['output_type']
        self._polygon_simplification = float(self._kwiver_config['polygon_simplification'])
        self._add_new_objects = parse_bool(self._kwiver_config['add_new_objects'])
        self._max_new_objects = int(self._kwiver_config['max_new_objects'])
        self._min_mask_area = int(self._kwiver_config['min_mask_area'])
        self._detection_threshold = float(self._kwiver_config['detection_threshold'])
        self._text_threshold = float(self._kwiver_config['text_threshold'])
        self._iou_threshold = float(self._kwiver_config['iou_threshold'])

        tq = self._kwiver_config.get('text_query', '')
        self._text_query_list = [q.strip() for q in tq.split(',') if q.strip()] if tq else []

        return True

    def check_configuration(self, cfg):
        """Check if the configuration is valid."""
        return True

    @report_cuda_errors("SAM3 detection refinement")
    def refine(self, image_data, detections):
        """
        Refine detections by adding segmentation masks.  When
        ``add_new_objects`` is enabled and a ``text_query`` is set,
        also detects new objects via Grounding DINO before segmenting.

        Args:
            image_data: Image container
            detections: DetectedObjectSet to refine

        Returns:
            DetectedObjectSet: Refined detections with masks
        """
        import torch

        img_np = image_to_rgb_numpy(image_data)

        # "Replace" mode: drop all pre-existing input detections so only the
        # newly text-detected objects are emitted (mirrors SAM3Refiner). With
        # the input empty, the overlap-suppression below has nothing to
        # suppress against, so every new detection is kept.
        if self._replace_existing:
            detections = DetectedObjectSet()

        # Detect new objects with text query if configured
        if self._add_new_objects and self._text_query_list:
            new_dets = self._model_manager.detect_with_text(
                img_np, self._text_query_list,
                self._detection_threshold, self._text_threshold,
            )
            # Suppress new detections that overlap with existing ones
            suppress_boxes = []
            for det in detections:
                bb = det.bounding_box
                suppress_boxes.append(
                    [bb.min_x(), bb.min_y(), bb.max_x(), bb.max_y()])

            new_count = 0
            for box, score, class_name in new_dets:
                if new_count >= self._max_new_objects:
                    break
                overlaps = any(
                    compute_iou(box, sb) > self._iou_threshold
                    for sb in suppress_boxes)
                if not overlaps:
                    bbox = BoundingBoxD(box[0], box[1], box[2], box[3])
                    dot = DetectedObjectType(
                        _safe_class_name(class_name, self._text_query_list[0]
                                         if self._text_query_list else None),
                        score)
                    new_det = DetectedObject(bbox, score, dot)
                    detections.add(new_det)
                    suppress_boxes.append(list(box))
                    new_count += 1

        if len(detections) == 0:
            return DetectedObjectSet()

        # Collect boxes for segmentation
        boxes = []
        for det in detections:
            bbox = det.bounding_box
            boxes.append([bbox.min_x(), bbox.min_y(), bbox.max_x(), bbox.max_y()])

        # Segment all boxes with SAM
        masks = self._model_manager.segment_with_sam(img_np, boxes)

        # Create output detection set
        output = DetectedObjectSet()

        for det, mask in zip(detections, masks):
            mask_area = np.sum(mask)
            if mask_area < self._min_mask_area:
                continue

            # Set mask on detection for multi-contour polygon output
            _set_mask_on_detection(det, mask, det.bounding_box)

            output.add(det)

        return output



def __vital_algorithm_register__():
    from viame.core.vital_registration import register_vital_algorithm

    register_vital_algorithm(
        SAM3Refiner,
        "sam3",
        "SAM3 (Segment Anything Model 3) based track refiner with text queries",
    )
    register_vital_algorithm(
        Sam3DetectionRefiner,
        "sam3",
        "SAM3 (SAM 2.1) based detection refiner for adding segmentation masks",
    )