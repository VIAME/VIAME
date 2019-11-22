# ckwg +29
# Copyright 2019 by Kitware, Inc.
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

from kwiver.kwiver_process import KwiverProcess
from sprokit.pipeline import process
from vital.types import ObjectTrackSet, ObjectTrackState, Track

logger = logging.getLogger(__name__)

_Homography = namedtuple('_Homography', [
    'matrix',  # A 3x3 numpy.ndarray transforming 2D homogeneous
               # coordinates (coordinate order is x, y, z)
])

class Homography(_Homography):
    def transform(self, array):
        """Given a ...x2xN numpy.ndarray of points (row order x, y)
        return a ...x2xN array of transformed points"""
        ones = np.ones(array.shape[:-2] + (1, array.shape[-1]), dtype=array.dtype)
        # Add z coordinate
        array = np.concatenate((array, ones), axis=-2)
        result = np.matmul(self.matrix, array)
        # Remove z coordinate
        return result[..., :-1, :] / result[..., -1:, :]

_BBox = namedtuple('_BBox', ['xmin', 'ymin', 'xmax', 'ymax'])

class BBox(_BBox):
    @classmethod
    def from_points(cls, array):
        """Given a 2xN numpy.ndarray of points (row order x, y)
        return the smallest enclosing bounding box"""
        xmin, ymin = array.min(1)
        xmax, ymax = array.max(1)
        return cls(xmin, ymin, xmax, ymax)

    @property
    def corners(self):
        """Return a 2x4 numpy.ndarray of the corner points (row order x, y)"""
        return np.array([
            [self.xmin, self.xmin, self.xmax, self.xmax],
            [self.ymin, self.ymax, self.ymin, self.ymax],
        ])

    @property
    def matrix(self):
        """Return a 2x2 numpy.ndarray (column order min, max; row order x, y)"""
        return np.array([
            [self.xmin, self.xmax],
            [self.ymin, self.ymax],
        ])

class Transformer(object):
    """A Transformer is a stateful object that receives one value at a time
    and produces one value at a time in response"""
    __slots__ = '_gen',
    def __init__(self, gen):
        """Create a Transformer from a generator"""
        gen.send(None)  # Initialize generator
        self._gen = gen

    def step(self, x):
        """Pass one value through"""
        return self._gen.send(x)

    def __call__(self, it):
        """Return an iterator of values produced by transforming the input it"""
        return map(self.step, it)

    @classmethod
    def decorate(cls, f):
        """Have f return a Transformer instead of a generator"""
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return cls(f(*args, **kwargs))
        return wrapper

def transform_box(homog, bbox):
    """Transform the bounding box with the given Homography,
    returning the smallest enclosing bounding box"""
    return BBox.from_points(homog.transform(bbox.corners))

def ious(x, y):
    """Given two ...x2x2 numpy.ndarrays corresponding to bounding boxes
    (cf. BBox.matrix), return a ... array of IOU scores"""
    maxmin = np.maximum(x[..., 0], y[..., 0])
    minmax = np.minimum(x[..., 1], y[..., 1])
    i = (minmax - maxmin).prod(-1)
    def area(x): return (x[..., 1] - x[..., 0]).prod(-1)
    u = area(x) + area(y) - i
    return np.where((maxmin < minmax).all(-1), i / u, 0)

def match_boxes_homog(homog, boxes, prev_boxes, min_iou):
    """Return a list of indices into prev_boxes, where each index
    identifies the entry of prev_boxes that matches the
    corresponding box in boxes.  A returned value may be None in the
    case of no match.

    homog is a Homography transforming the coordinates of boxes
    to those of prev_boxes.

    min_iou is the minimum IOU required for a match."""
    if not boxes:
        return []
    if not prev_boxes:
        return [None] * len(boxes)
    # Because aligned bounding boxes are easy to work with, we transform to
    # approximate aligned bounding boxes instead of an arbitrary quadrilateral
    boxes = [transform_box(homog, b) for b in boxes]
    # Now boxes and prev_boxes are in the same coordinate system.
    # iou has shape (len(boxes), len(prev_boxes))
    iou = ious(
        np.array([[b.matrix] for b in boxes]),
        np.array([[pb.matrix for pb in prev_boxes]]),
    )
    # linear_sum_assignment finds a minimum assignment, so subtract
    # IOUs from 1 (the max)
    weights = np.where(iou < min_iou, 1, 1 - iou)
    box_ind, prev_box_ind = scipy.optimize.linear_sum_assignment(weights)
    result = [None] * len(boxes)
    for bi, pbi in zip(box_ind, prev_box_ind):
        if weights[bi, pbi] < 1:
            result[bi] = pbi
    return result

# XXX Getting rid of all these *Input and *Output namedtuple types
# would be a good idea

CoreTrackInput = namedtuple('CoreTrackInput', [
    'boxes',  # List of BBoxes
    'homog',  # None or a Homography from this-frame
              # coordinates to previous-frame coordinates
])

CoreTrackOutput = namedtuple('CoreTrackOutput', [
    'track_ids',  # List of (integer) track IDs corresponding to input boxes
])

DEFAULT_MIN_IOU = 0.2

@Transformer.decorate
def core_track(min_iou=None):
    """Create a Transformer that performs tracking (only associating
    tracks with the given minimum IOU).  The .step call expects a
    CoreTrackInput and returns a CoreTrackOutput"""
    if min_iou is None: min_iou = DEFAULT_MIN_IOU  # Default value
    # new_id() gets a new track ID, starting from 1
    def new_id(_c=itertools.count(1)): return next(_c)
    output = None
    prev_boxes = None
    while True:
        ti = yield output
        if prev_boxes is not None and ti.homog is None:
            logger.debug("Breaking all tracks after break in homography stream")
            prev_boxes = None
        if prev_boxes is None:
            track_ids = [new_id() for _ in ti.boxes]
        else:
            mi = match_boxes_homog(ti.homog, ti.boxes, prev_boxes, min_iou)
            track_ids = [new_id() if i is None else track_ids[i] for i in mi]
        prev_boxes = ti.boxes
        output = CoreTrackOutput(track_ids)

HomographyF2F = namedtuple('HomographyF2F', [
    'homog',  # Homography instance
    'from_id',  # Source coordinate space ID
    'to_id',  # Destination coordinate space ID
])

ConvertHomInput = namedtuple('ConvertHomInput', [
    'homog_f2f',  # HomographyF2F instance (this frame to reference frame)
])
ConvertHomOutput = namedtuple('ConvertHomOutput', [
    'homog',  # Homography instance
])

@Transformer.decorate
def convert_homographies():
    """Create a Transformer that converts frame-to-frame homographies
    to homographies to the previous frame, or None if not possible.
    The .step call expects a ConvertHomInput and returns a ConvertHomOutput."""
    output = None
    prev = None
    while True:
        curr = (yield output).homog_f2f
        if prev is not None and prev.to_id == curr.to_id:
            homog = Homography(np.matmul(np.linalg.inv(prev.homog.matrix), curr.homog.matrix))
        else:
            homog = None
        prev = curr
        output = ConvertHomOutput(homog)

BuildTracksInput = namedtuple('BuildTracksInput', [
    'track_ids',  # List of track IDs
    'track_states',  # Corresponding list of arbitrary objects reflecting the track state
    'timestamp',  # Arbitrary object indicating the current "time"
])
BuildTracksOutput = namedtuple('BuildTracksOutput', [
    'tracks',  # Dict with track IDs for keys and lists of pairs
               # for values, each list consisting of pairs of
               # a timestamp and a track state
])

@Transformer.decorate
def build_tracks():
    """Create a Transformer that provides all input seen so far as a dict of tracks.
    The .step call expects a BuildTracksInput and returns a BuildTracksOutput."""
    tracks = {}
    output = None
    while True:
        bti = yield output
        for tid, ts in zip(bti.track_ids, bti.track_states):
            tracks.setdefault(tid, []).append((bti.timestamp, ts))
        # Return a copy
        output = BuildTracksOutput({
            tid: tss[:] for tid, tss in tracks.items()
        })

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
    bbox = do.bounding_box()
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

TrackInput = namedtuple('TrackInput', [
    'detected_object_set',  # Kwiver DetectedObjectSet
    'homography_src_to_ref',  # Kwiver F2FHomography
    'timestamp',  # Kwiver timestamp
])
TrackOutput = namedtuple('TrackOutput', [
    'object_track_set',  # Kwiver ObjectTrackSet
])

@Transformer.decorate
def track(min_iou=None):
    """Create a Transformer that performs tracking (only associating
    tracks with the given minimum IOU).  The .step call expects a
    TrackInput and returns a TrackOutput"""
    ch = convert_homographies()
    ct = core_track(min_iou)
    bt = build_tracks()

    output = None
    while True:
        ti = yield output
        homog = ch.step(ConvertHomInput(wrap_F2FHomography(ti.homography_src_to_ref))).homog
        dos = to_DetectedObject_list(ti.detected_object_set)
        boxes = [get_DetectedObject_bbox(do) for do in dos]
        track_ids = ct.step(CoreTrackInput(boxes, homog)).track_ids
        tracks = bt.step(BuildTracksInput(track_ids, dos, ti.timestamp)).tracks
        output = TrackOutput(to_ObjectTrackSet(tracks))

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
        ots = self._tracker.step(TrackInput(*map(
            self.grab_input_using_trait,
            ['detected_object_set', 'homography_src_to_ref', 'timestamp'],
        ))).object_track_set
        self.push_to_port_using_trait('object_track_set', ots)
        self._base_step()

def __sprokit_register__():
    from sprokit.pipeline import process_factory
    module_name = 'python:kwiver.python.SimpleHomogTracker'
    if process_factory.is_process_module_loaded(module_name):
        return
    process_factory.add_process(
        'simple_homog_tracker',
        'Simple IOU-based tracker with homography support',
        SimpleHomogTracker,
    )
    process_factory.mark_process_module_as_loaded(module_name)
