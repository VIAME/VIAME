# ckwg +29
# Copyright 2019-2020 by Kitware, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither name of Kitware, Inc. nor the names of any contributors may be used
#    to endorse or promote products derived from this software without specific
#    prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import division
from __future__ import print_function

from collections import namedtuple
import functools
import itertools
import logging

import numpy as np
import scipy.optimize

from kwiver.sprokit.processes.kwiver_process import KwiverProcess
from kwiver.sprokit.pipeline import process
from kwiver.vital.types import ObjectTrackSet, ObjectTrackState, Track

logger = logging.getLogger(__name__)

_Homography = namedtuple('_Homography', [
    'matrix',  # A 3x3 numpy.ndarray transforming 2D homogeneous
               # coordinates (coordinate order is x, y, z)
])

class Homography(_Homography):
    def transform(self, array):
        """Given a ...x2xN numpy.ndarray of points (row order x, y)
        return a ...x2xN array of transformed points"""
        return self.matrix_transform(self.matrix, array)

    @staticmethod
    def matrix_transform(matrix, array):
        """Like Homography(matrix).transform(array), but broadcasts over
        matrix

        """
        ones = np.ones(array.shape[:-2] + (1, array.shape[-1]), dtype=array.dtype)
        # Add z coordinate
        array = np.concatenate((array, ones), axis=-2)
        result = np.matmul(matrix, array)
        # Remove z coordinate
        return result[..., :-1, :] / result[..., -1:, :]

HomographyF2F = namedtuple('HomographyF2F', [
    'homog',  # Homography instance
    'from_id',  # Source coordinate space ID
    'to_id',  # Destination coordinate space ID
])

_BBox = namedtuple('_BBox', ['xmin', 'ymin', 'xmax', 'ymax'])

class BBox(_BBox):
    @classmethod
    def from_matrix(cls, matrix):
        [[xmin, xmax], [ymin, ymax]] = matrix
        return cls(xmin, ymin, xmax, ymax)

    @property
    def matrix(self):
        """Return a 2x2 numpy.ndarray (column order min, max; row order x, y)"""
        return np.array([[self.xmin, self.xmax], [self.ymin, self.ymax]])

    @staticmethod
    def matrix_area(array):
        """Like BBox.from_matrix(array).area, but broadcasts"""
        return (array[..., 1] - array[..., 0]).prod(-1)

    @property
    def area(self):
        """Return the area of the bounding box"""
        return self.matrix_area(self.matrix)

    @staticmethod
    def matrix_from_points(array):
        """Like BBox.from_points(array).matrix, but broadcasts"""
        return np.stack([array.min(-1), array.max(-1)], axis=-1)

    @classmethod
    def from_points(cls, array):
        """Given a 2xN numpy.ndarray of points (row order x, y)
        return the smallest enclosing bounding box"""
        return cls.from_matrix(cls.matrix_from_points(array))

    @staticmethod
    def matrix_corners(array):
        """Like BBox.from_matrix(array).corners, but broadcasts"""
        if array.shape[-2:] != (2, 2):
            raise ValueError
        x, y, m, M = 0, 1, 0, 1  # x, y, min, and max
        return array[..., [[x], [y]], [[m, m, M, M], [m, M, m, M]]]

    @property
    def corners(self):
        """Return a 2x4 numpy.ndarray of the corner points (row order x, y)"""
        return self.matrix_corners(self.matrix)

class Transformer(object):
    """A Transformer is a stateful object that receives one tuple of
    values at a time and produces one value at a time in response.

    """
    __slots__ = '_gen',
    def __init__(self, gen):
        """Create a Transformer from a generator"""
        gen.send(None)  # Initialize generator
        self._gen = gen

    def step(self, *args):
        """Pass arguments through"""
        return self._gen.send(args)

    @classmethod
    def decorate(cls, f):
        """Have f return a Transformer instead of a generator.  The generator
        should yield to return a result and accept an argument
        tuple.

        """
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return cls(f(*args, **kwargs))
        return wrapper

def transform_box(homog, bbox):
    """Transform the bounding box with the given Homography,
    returning the smallest enclosing bounding box"""
    return BBox.from_matrix(transform_matrix_box(homog.matrix, bbox.matrix))

def transform_matrix_box(homog, bbox):
    """A broadcasting version of
    transform_box(Homography(homog), BBox.from_matrix(bbox)).matrix

    """
    tcorners = Homography.matrix_transform(homog, BBox.matrix_corners(bbox))
    return BBox.matrix_from_points(tcorners)

def ious(x, y, x_area=None, y_area=None):
    """Given two ...x2x2 numpy.ndarrays corresponding to bounding boxes
    (cf. BBox.matrix), return a ... array of IOU scores

    If provided, x_area and y_area should be BBox.matrix_area(x) and
    BBox.matrix_area(y) respectively.

    """
    maxmin = np.maximum(x[..., 0], y[..., 0])
    minmax = np.minimum(x[..., 1], y[..., 1])
    i = (minmax - maxmin).prod(-1)
    if x_area is None: x_area = BBox.matrix_area(x)
    if y_area is None: y_area = BBox.matrix_area(y)
    u = x_area + y_area - i
    return np.where((maxmin < minmax).all(-1), i / u, 0)

def optimize_iou_based_assignment(iou_array, min_iou):
    """Given an NxM numpy.ndarray of IOU scores between N source objects
    and M target objects, return an optimal assignment as an N-length
    list result such that the ith source object is assigned to the
    result[i]'th target object, or unassigned if result[i] is None.

    min_iou is the minimum IOU required for a match.

    """
    # linear_sum_assignment finds a minimum assignment, so subtract
    # IOUs from 1 (the max)
    weights = np.where(iou_array < min_iou, 1, 1 - iou_array)
    source_ind, target_ind = scipy.optimize.linear_sum_assignment(weights)
    result = [None] * len(iou_array)
    for si, ti in zip(source_ind, target_ind):
        if weights[si, ti] < 1:
            result[si] = ti
    return result

def match_boxes_homog(homog, boxes, prev_homog, prev_boxes, min_iou):
    """Return a list of indices into prev_boxes, where each index
    identifies the entry of prev_boxes that matches the
    corresponding box in boxes.  A returned value may be None in the
    case of no match.

    homog and prev_homog are HomographyF2Fs transforming the
    coordinates of boxes and prev_boxes, respectively, to
    reference-frame coordinates.

    min_iou is the minimum IOU required for a match."""
    if prev_homog.to_id != homog.to_id:
        logger.debug("Returning no matches due to break in homography stream")
        return [None] * len(boxes)
    if not (boxes and prev_boxes):
        return [None] * len(boxes)
    prev_homog_, homog_ = prev_homog.homog.matrix, homog.homog.matrix
    rel_homog = Homography(np.matmul(np.linalg.inv(prev_homog_), homog_))
    # Because aligned bounding boxes are easy to work with, we transform to
    # approximate aligned bounding boxes instead of an arbitrary quadrilateral
    boxes = [transform_box(rel_homog, b) for b in boxes]
    # Now boxes and prev_boxes are in the same coordinate system.
    # iou has shape (len(boxes), len(prev_boxes))
    iou = ious(
        np.array([[b.matrix] for b in boxes]),
        np.array([[pb.matrix for pb in prev_boxes]]),
    )
    return optimize_iou_based_assignment(iou, min_iou)

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
    matches.append(itertools.repeat(None))
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

    # Compute the transformations between current and previous cameras
    mh = np.array([h.matrix for h in multihomog.homogs])
    pmh = np.array([h.matrix for h in prev_multihomog.homogs])
    # homogs[i, j] is a transformation from current camera i
    # coordinates to previous camera j coordinates
    homogs = np.matmul(np.linalg.inv(pmh), mh[:, np.newaxis])

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

@Transformer.decorate
def min_track(match):
    """Create a Transformer that performs minimalistic tracking

    Arguments:
    - A matching function of signature (context1, seq1, context2,
      seq2, /) -> result, with:
      - seq1 and seq2 sequences of some specified type
      - context1 and context2 arbitrary context objects
      - result an iterable containing the index of the match in the
        seq2 for each element in seq1 (or None if no match)

    The .step call expects two arguments:
    - A sequence of the type expected by the matching function
    - An additional context object of the type expected by the
      matching function
    and returns:
    - a list of (integer) track IDs

    """
    def new_id(_c=itertools.count(1)): return next(_c)
    output = None
    prev_input = None
    while True:
        input_, state = yield output
        if prev_input is None:
            track_ids = [new_id() for _ in input_]
        else:
            mi = match(state, input_, prev_state, prev_input)
            track_ids = [new_id() if i is None else track_ids[i] for i in mi]
        prev_input, prev_state = input_, state
        output = track_ids

DEFAULT_MIN_IOU = 0.2

def core_track(min_iou=None):
    """Create a Transformer that performs tracking (only associating
    tracks with the given minimum IOU).  The .step call expects two
    arguments:
    - A list of BBoxes
    - A HomographyF2F from this-frame coordinates to reference-frame
      coordinates
    and returns:
    - a list of (integer) track IDs corresponding to the input boxes

    """
    if min_iou is None: min_iou = DEFAULT_MIN_IOU  # Default value
    return min_track(functools.partial(match_boxes_homog, min_iou=min_iou))

def core_multitrack(min_iou=None):
    # XXX doc me.  It's like core_track but with multiboxes and
    # multihomographies.
    if min_iou is None: min_iou = DEFAULT_MIN_IOU  # Default value
    match = functools.partial(match_multiboxes_multihomog, min_iou=min_iou)
    return min_track(match)

@Transformer.decorate
def build_tracks():
    """Create a Transformer that provides all input seen so far as a dict
    of tracks.  The .step call expects three arguments:
    - a list of track IDs
    - a corresponding list of arbitrary objects reflecting the track
      state
    - an arbitrary "timestamp" object indicating the current "time"
    and returns:
    - a dict that has track IDs for keys and lists of pairs for
      values, with each list consisting of pairs of a timestamp and a
      track state

    """
    tracks = {}
    output = None
    while True:
        track_ids, track_states, timestamp = yield output
        for tid, ts in zip(track_ids, track_states):
            tracks.setdefault(tid, []).append((timestamp, ts))
        # Return a copy
        output = {
            tid: tss[:] for tid, tss in tracks.items()
        }

# Converters to / from Kwiver types

def wrap_F2FHomography(h):
    """Convert Kwiver F2FHomography to HomographyF2F"""
    arr = np.array([
        [h.get(r, c) for c in range(3)] for r in range(3)
    ])
    return HomographyF2F(Homography(arr), h.from_id, h.to_id)

def to_DetectedObject_list(dos):
    """Get a list of the DetectedObjects in a Kwiver DetectedObjectSet"""
    return list(dos)

def get_DetectedObject_bbox(do):
    """Get the bounding box of a Kwiver DetectedObject as a BBox"""
    bbox = do.bounding_box
    return BBox(bbox.min_x(), bbox.min_y(), bbox.max_x(), bbox.max_y())

def to_ObjectTrackSet(tracks):
    """Create an ObjectTrackSet from a dict whose keys are track IDs
    and values are lists of pairs of Kwiver timestamps and Kwiver DetectedObjects"""
    # Modeled after similar code in srnn_tracker.py
    result = []
    for tid, states in tracks.items():
        t = Track(id=tid)
        for ts, do in states:
            ots = ObjectTrackState(ts.get_frame(), ts.get_time_usec(), do)
            if not t.append(ots):
                raise ValueError("Unsorted input to to_ObjectTrackSet")
        result.append(t)
    return ObjectTrackSet(result)

# Okay, here's our "not quite Kwiver" Transformer.  With the converters
# suitably defined, this will work on Kwiver types.  Then the only
# thing left to do is wrap it in a Sprokit process.

@Transformer.decorate
def track(min_iou=None):
    """Create a Transformer that performs tracking (only associating
    tracks with the given minimum IOU).  The .step call expects three
    arguments:
    - a Kwiver DetectedObjectSet
    - a Kwiver F2FHomography
    - a Kwiver timestamp
    and returns a Kwiver ObjectTrackSet"""
    ct = core_track(min_iou)
    bt = build_tracks()

    output = None
    while True:
        do_set, homog_s2r, ts = yield output
        dos = to_DetectedObject_list(do_set)
        boxes = [get_DetectedObject_bbox(do) for do in dos]
        track_ids = ct.step(boxes, wrap_F2FHomography(homog_s2r))
        tracks = bt.step(track_ids, dos, ts)
        output = to_ObjectTrackSet(tracks)

# The process itself

def add_declare_config(process, name_key, default, description):
    process.add_config_trait(name_key, name_key, default, description)
    process.declare_config_using_trait(name_key)

class SimpleHomogTracker(KwiverProcess):
    def __init__(self, config):
        KwiverProcess.__init__(self, config)

        add_declare_config(self, "min_iou", str(DEFAULT_MIN_IOU),
                           "Minimum IOU to associate a detection to a track")

        optional = process.PortFlags()
        required = process.PortFlags()
        required.add(self.flag_required)

        self.declare_input_port_using_trait('detected_object_set', required)
        self.declare_input_port_using_trait('timestamp', required)
        self.declare_input_port_using_trait('homography_src_to_ref', required)

        self.declare_output_port_using_trait('object_track_set', optional)

    def _configure(self):
        self._tracker = track(float(self.config_value('min_iou')))
        self._base_configure()

    def _step(self):
        ots = self._tracker.step(*map(
            self.grab_input_using_trait,
            ['detected_object_set', 'homography_src_to_ref', 'timestamp'],
        ))
        self.push_to_port_using_trait('object_track_set', ots)
        self._base_step()

def __sprokit_register__():
    from kwiver.sprokit.pipeline import process_factory
    module_name = 'python:kwiver.python.SimpleHomogTracker'
    if process_factory.is_process_module_loaded(module_name):
        return
    process_factory.add_process(
        'simple_homog_tracker',
        'Simple IOU-based tracker with homography support',
        SimpleHomogTracker,
    )
    process_factory.mark_process_module_as_loaded(module_name)
