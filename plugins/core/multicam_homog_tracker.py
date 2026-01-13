# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from __future__ import division
from __future__ import print_function

from collections import namedtuple
import functools
import itertools
import logging

import numpy as np

from kwiver.sprokit.processes.kwiver_process import KwiverProcess
from kwiver.sprokit.pipeline import process
from kwiver.vital.types import ObjectTrackSet, ObjectTrackState, Track

# XXX Not ideal places to be importing things from
from .simple_homog_tracker import (
    DEFAULT_MIN_IOU,
    BBox, Homography, HomographyF2F, Transformer,
    add_declare_config, build_tracks, get_DetectedObject_bbox, ious,
    match_boxes_homog, min_track, optimize_iou_based_assignment,
    to_DetectedObject_list, transform_matrix_box, wrap_F2FHomography,
)
from .stabilize_many_images import (
    add_declare_input_port, add_declare_output_port,
)

logger = logging.getLogger(__name__)

class MultiBBox(object):
    """Captures bounding boxes for an object from multiple cameras

    The class has one attribute, "boxes", that gives the bounding
    boxes as a dict that maps a camera index to a BBox.

    """

    __slots__ = 'boxes',

    def __init__(self, boxes):
        """Create a MultiBBox with the provided value for the boxes attributes

        """
        self.boxes = boxes

class MultiHomographyF2F(object):
    """Captures homographies from each of a set of images to a reference frame

    Attributes:
    - homogs: A list of Homography objects, one per camera
    - from_id: Source frame identifier
    - to_id: Target frame identifier

    """

    __slots__ = 'homogs', 'from_id', 'to_id'

    def __init__(self, homogs, from_id, to_id):
        """Create a MultiHomographyF2F with the provided attribute values"""
        self.homogs = homogs
        self.from_id = from_id
        self.to_id = to_id

    @classmethod
    def from_homographyf2fs(cls, homogs):
        """Turn a non-empty iterable of HomographyF2Fs (one per
        camera) into a MultiHomographyF2F

        """
        ids = None
        hs = []
        for h in homogs:
            hs.append(h.homog)
            if ids is None:
                ids = h.from_id, h.to_id
            elif ids != (h.from_id, h.to_id):
                raise ValueError("Inconsistent IDs")
        return cls(hs, *ids)

    def __getitem__(self, x):
        """Get the xth item (possibly a slice) as a HomographyF2F (or a list
        thereof)

        """
        f, t = self.from_id, self.to_id
        if isinstance(x, slice):
            return [HomographyF2F(h, f, t) for h in self.homogs[x]]
        return HomographyF2F(self.homogs[x], f, t)

    def __len__(self):
        return len(self.homogs)

def arg_track_multiboxes(multihomog, box_lists, min_iou):
    """Turn a list of lists of BBoxes (one top-level list per camera) into
    a list of dicts mapping camera indices to BBox indices using the
    provided MultiHomographyF2F

    """
    matches = [
        match_boxes_homog(h, b, ph, pb, min_iou) for h, b, ph, pb
        in zip(multihomog, box_lists, multihomog[1:], box_lists[1:])
    ]
    matches.append(itertools.repeat(None, len(box_lists[-1])))
    # List of multibox dicts
    result = []
    # Partially maps a current-camera box index to an existing
    # multibox dict it should be added to.
    hooks = {}
    for i, ms in enumerate(matches):
        new_hooks = {}
        for j, m in enumerate(ms):
            mb = hooks.get(j, {})
            if not mb:
                result.append(mb)
            mb[i] = j
            if m is not None:
                new_hooks[m] = mb
        hooks = new_hooks
    return result

def create_track_multiboxes(box_lists, arg_dicts):
    """Given a list of dicts of indices created by arg_track_multiboxes,
    replace the index values with the corresponding box from the input
    lists.

    """
    return [{i: box_lists[i][j] for i, j in d.items()} for d in arg_dicts]

def track_multiboxes(multihomog, box_lists, min_iou):
    """Turn a list of lists of BBoxes (one top-level list per camera) into
    a list of MultiBBoxes using the provided MultiHomographyF2F,
    returning it in a pair whose second element is a list of dicts
    like MultiBBox.boxes but with indices into the corresponding input
    lists for values.

    """
    arg_dicts = arg_track_multiboxes(multihomog, box_lists, min_iou)
    box_dicts = create_track_multiboxes(box_lists, arg_dicts)
    return list(map(MultiBBox, box_dicts)), arg_dicts

def invert_permutation(seq):
    """Given a sequence that permutes indices, return a list describing
    the inverse permutation

    """
    result = [None] * len(seq)
    for i, j in enumerate(seq):
        result[j] = i
    if None in result:
        raise ValueError("Bad permutation")
    return result

ArrayifiedMultiboxes = namedtuple('ArrayifiedMultiboxes', [
    'boxes', 'cams', 'blocks', 'lengths', 'indices',
])

def arrayify_multiboxes(mbs):
    """Turn a non-empty list of N MultiBBoxes into an
    ArrayifiedMultiboxes containing the following five ndarrays:
    - boxes: An Mx2x2 array of bounding boxes (per BBox.matrix), where
      M is the total count of underlying BBoxes and each MultiBBox is
      represented by a contiguous range of rows
    - cams: An M-length array of the corresponding camera indices
    - blocks: An ordered Cx2 array of "blocks", where C is the number
      of distinct lengths of MultiBBox and each row is the start and
      stop indices of the block in the first two arrays
    - lengths: A C-length array of the length of MultiBBox represented
      in each block
    - indices: An N-length array giving the "index" in the output of
      each MultiBBox in the input

    So for instance,
    boxes[indices[i]*lengths[0] : (indices[i]+1)*lengths[0]]
    corresponds to mbs[i] when 0 <= indices[i] < blocks[0, 1]

    """
    if not mbs:
        raise ValueError
    def key(i_mb): return len(i_mb[1].boxes)
    indices, mbs = zip(*sorted(enumerate(mbs), key=key))
    boxes, cams = [], []
    blocks, i, lengths, cl = [], 0, [], None
    for mb in mbs:
        mb = mb.boxes
        for c, b in mb.items():
            boxes.append(b.matrix)
            cams.append(c)
        if len(mb) != cl:
            blocks.append(i)
            cl = len(mb)
            lengths.append(cl)
        i += cl
    blocks.append(i)
    # Unfortunately this needs to be done directly
    # (https://github.com/numpy/numpy/issues/7753)
    blocks = np.array(blocks)
    as_strided = np.lib.stride_tricks.as_strided
    blocks = as_strided(blocks, (len(blocks) - 1, 2), blocks.strides * 2)
    result = boxes, cams, blocks, lengths, invert_permutation(indices)
    return ArrayifiedMultiboxes(*map(np.asarray, result))

def norm_index(index, length):
    """Return a positive index such that seq[index] is equivalent to
    seq[norm_index(index, len(seq))]

    """
    if index < 0:
        index += length
    if index not in range(length):
        raise IndexError
    return index

def reblock_multiboxes(array, blocks, lengths, axis=None):
    """Given an Mx... array and arrays blocks and lengths describing how
    to split it up (see arrayify_multiboxes), yield XixLix... arrays
    where Xi is the number of elements in a block and Li is the length
    of the elements in that block

    If axis is provided, operate on that axis instead of the zeroth.

    """
    if axis is None: axis = 0
    axis = norm_index(axis, array.ndim)
    if array.shape[axis] != blocks[-1, 1]:
        raise ValueError('Blocks do not match array')
    shape_pre, shape_post = array.shape[:axis], array.shape[axis + 1:]
    for (start, stop), length in zip(blocks, lengths):
        shape = shape_pre + ((stop - start) // length, length) + shape_post
        slice_ = (slice(None),) * axis + (slice(start, stop),)
        yield array[slice_].reshape(shape)

def diff_homogs(source, dest):
    """Given a length-M source and length-N dest Homography list, return
    an MxN ndarray of array-format homographies

    """
    # Compute the transformations between current and previous cameras
    s = np.array([h.matrix for h in source])
    d = np.array([h.matrix for h in dest])
    return np.matmul(np.linalg.inv(d), s[:, np.newaxis])

def match_multiboxes_multihomog(
        multihomog, multiboxes, prev_multihomog, prev_multiboxes, min_iou,
):
    """Return a list of indices into prev_multiboxes, where each index
    identifies the entry of prev_multiboxes that matches the
    corresponding multibox in multiboxes.  A returned value may be
    None in the case of no match.

    multihomog and prev_multihomog are MultiHomographyF2Fs
    transforming the coordinates of multiboxes and prev_multiboxes,
    respectively, to reference-frame coordinates.

    IOUs between two multiboxes are approximated by taking the median
    of all the IOUs between the component detections of the
    multiboxes.  (Those IOUs are approximated by approximating one box
    in the other's coordinates and computing that IOU.)

    min_iou is the minimum IOU required for a match.

    """
    # Return early under certain circumstances
    if multihomog.to_id != prev_multihomog.to_id:
        logger.debug("Returning no matches due to break in homography stream")
        return [None] * len(multiboxes)
    if not (multiboxes and prev_multiboxes):
        return [None] * len(multiboxes)

    # homogs[i, j] is a transformation from current camera i
    # coordinates to previous camera j coordinates
    homogs = diff_homogs(multihomog.homogs, prev_multihomog.homogs)

    # Compute IOUs between current and previous multiboxes
    boxes, cams, blocks, lengths, indices = arrayify_multiboxes(multiboxes)
    prev = arrayify_multiboxes(prev_multiboxes)
    # trans_boxes[i, j] is the ith box in previous camera j coordinates
    trans_boxes = transform_matrix_box(homogs[cams], boxes[:, np.newaxis])
    trans_box_areas = BBox.matrix_area(trans_boxes)
    # box_iou[i, j] is the IOU of the ith box and the jth previous box
    # in the latter's coordinates
    box_iou = ious(trans_boxes[:, prev.cams], prev.boxes, trans_box_areas[:, prev.cams])
    # iou[i, j] is the approx. IOU of the ith multibox and jth
    # previous multibox (original indices)
    iou = np.block([
        [np.median(block, axis=(1, 3)) for block in reblock_multiboxes(
            row_blocks, prev.blocks, prev.lengths, axis=2,
        )] for row_blocks in reblock_multiboxes(box_iou, blocks, lengths)
    ])[np.ix_(indices, prev.indices)]

    return optimize_iou_based_assignment(iou, min_iou)

def core_multitrack(min_iou=None):
    # XXX doc me.  It's like core_track but with multiboxes and
    # multihomographies.
    if min_iou is None: min_iou = DEFAULT_MIN_IOU  # Default value
    match = functools.partial(match_multiboxes_multihomog, min_iou=min_iou)
    return min_track(match)

def to_ObjectTrackSet_list(tracks, ncam):
    """Create a list of ncam ObjectTrackSets from a dict of the form
    {track_id: [(timestamp, {camera_id: detected_object})]} (dicts and
    lists may contain multiple items).

    """
    result = [[] for _ in range(ncam)]
    for tid, states in tracks.items():
        t_out = [Track(id=tid) for _ in result]
        for ts, dos in states:
            for i, do in dos.items():
                ots = ObjectTrackState(ts.get_frame(), ts.get_time_usec(), do)
                if not t_out[i].append(ots):
                    raise ValueError("Unsorted input to to_ObjectTrackSet_list")
        for r, t in zip(result, t_out):
            if t:
                r.append(t)
    return list(map(ObjectTrackSet, result))

@Transformer.decorate
def multitrack(min_iou=None):
    """Create a Transformer that performs multi-camera tracking (only
    associating tracks with the given minimum IOU).  The .step call expects two arguments:
    - a list of pairs of a Kwiver DetectedObjectSet and a Kwiver
      F2FHomography, one per camera
    - a Kwiver timestamp

    and returns an equally long list of Kwiver ObjectTrackSets.  All
    received lists should be the same length.

    """
    ct = core_multitrack(min_iou)
    bt = build_tracks()

    output = None
    while True:
        cams, ts = yield output
        do_sets, homogs = zip(*cams)
        multihomog = MultiHomographyF2F.from_homographyf2fs(map(wrap_F2FHomography, homogs))
        do_lists = list(map(to_DetectedObject_list, do_sets))
        boxes = [list(map(get_DetectedObject_bbox, dos)) for dos in do_lists]
        multiboxes, ind = track_multiboxes(multihomog, boxes, min_iou)
        track_ids = ct.step(multiboxes, multihomog)
        multitracks = bt.step(track_ids, create_track_multiboxes(do_lists, ind), ts)
        output = to_ObjectTrackSet_list(multitracks, len(cams))

class MulticamHomogTracker(KwiverProcess):
    def __init__(self, config):
        KwiverProcess.__init__(self, config)

        add_declare_config(self, "min_iou", str(DEFAULT_MIN_IOU),
                           "Minimum IOU to associate a detection to a track")
        add_declare_config(self, 'n_input', '2', 'Number of inputs')

        optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)

        self.declare_input_port_using_trait('timestamp', required)

        # XXX work around insufficient wrapping
        self._n_input = int(self.config_value('n_input'))
        for i in range(1, self._n_input + 1):
            add_declare_input_port(self, 'det_objs_' + str(i), 'detected_object_set',
                                   required, 'Input detected object set #' + str(i))
            add_declare_input_port(self, 'homog' + str(i), 'homography_src_to_ref',
                                   required, 'Input homography (source-to-ref) #' + str(i))
            add_declare_output_port(self, 'obj_tracks_' + str(i), 'object_track_set',
                                    optional, 'Output object track set #' + str(i))

    def _configure(self):
        # XXX actually use this
        self._n_input = int(self.config_value('n_input'))
        self._tracker = multitrack(float(self.config_value('min_iou')))
        self._base_configure()

    def _step(self):
        ots_list = self._tracker.step([
            (self.grab_input_using_trait('det_objs_' + str(i)),
             self.grab_input_using_trait('homog' + str(i)))
            for i in range(1, self._n_input + 1)
        ], self.grab_input_using_trait('timestamp'))
        for i, ots in enumerate(ots_list, 1):
            self.push_to_port_using_trait('obj_tracks_' + str(i), ots)
        self._base_step()

def __sprokit_register__():
    from kwiver.sprokit.pipeline import process_factory
    module_name = 'python:viame.python.MulticamHomogTracker'
    if process_factory.is_process_module_loaded(module_name):
        return
    process_factory.add_process(
        'multicam_homog_tracker',
        'Multi-camera IOU-based tracker with homography support',
        MulticamHomogTracker,
    )
    process_factory.mark_process_module_as_loaded(module_name)
