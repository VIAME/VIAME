# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Tube-IoU based fusion of object track sets from multiple trackers.

Tracks produced by two or more trackers over the same imagery are
associated by tube IoU -- the spatiotemporal overlap of their
per-frame bounding boxes -- using per-tracker-pair optimal assignment,
then each resulting cluster is fused into a single output track with
per-frame weighted box averaging and class probability fusion.

The clustering guarantees at most one track per input tracker per
cluster. Tracks seen by only some trackers are kept by default
(configurable via min_trackers), so trackers with complementary
failure modes (one fragments, one misses objects) reinforce rather
than suppress each other.

The sprokit process re-merges the cumulative track sets it receives on
each step, so its output on the final frame is the fused result of the
complete inputs.
"""

from __future__ import division
from __future__ import print_function

import ast

import numpy as np

from kwiver.sprokit.processes.kwiver_process import KwiverProcess
from kwiver.sprokit.pipeline import process
from kwiver.vital.types import (
    BoundingBoxD, DetectedObject, DetectedObjectType,
    ObjectTrackSet, ObjectTrackState, Track,
)

from .simple_homog_tracker import add_declare_config
from .stabilize_many_images import (
    add_declare_input_port, add_declare_output_port,
)

# ----------------------------- CORE MATH -------------------------------

def box_intersection_area( a, b ):
    width = min( a[2], b[2] ) - max( a[0], b[0] )
    height = min( a[3], b[3] ) - max( a[1], b[1] )
    return max( width, 0.0 ) * max( height, 0.0 )

def box_area( a ):
    return max( a[2] - a[0], 0.0 ) * max( a[3] - a[1], 0.0 )

def box_iou( a, b ):
    inter = box_intersection_area( a, b )
    if inter <= 0:
        return 0.0
    return inter / ( box_area( a ) + box_area( b ) - inter )

def tube_iou( frames_a, frames_b, mode='mean', min_overlap_frames=1 ):
    """Compute the tube IoU between two tracks given as dicts mapping
    frame id to a bounding box [x1, y1, x2, y2].

    mode 'mean' averages per-frame IoU over co-visible frames only
    (forgiving of fragmentation and length mismatch); mode 'tube' is
    the strict spatiotemporal IoU, where frames covered by only one
    track count fully against the union.
    """
    common = frames_a.keys() & frames_b.keys()
    if len( common ) < min_overlap_frames:
        return 0.0
    if mode == 'mean':
        return sum( box_iou( frames_a[f], frames_b[f] )
                    for f in common ) / len( common )
    inter_sum = sum( box_intersection_area( frames_a[f], frames_b[f] )
                     for f in common )
    union_sum = 0.0
    for f in frames_a.keys() | frames_b.keys():
        if f in common:
            union_sum += box_area( frames_a[f] ) + \
                         box_area( frames_b[f] ) - \
                         box_intersection_area( frames_a[f], frames_b[f] )
        elif f in frames_a:
            union_sum += box_area( frames_a[f] )
        else:
            union_sum += box_area( frames_b[f] )
    if union_sum <= 0:
        return 0.0
    return inter_sum / union_sum

def assign_pairs( iou_matrix, min_iou ):
    """Return a list of (row, column) assignments with IoU at or above
    the threshold, at most one per row and per column, maximizing total
    IoU (via scipy when available, greedy best-first otherwise).
    """
    if iou_matrix.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment( -iou_matrix )
        return [ ( int( r ), int( c ) ) for r, c in zip( rows, cols )
                 if iou_matrix[ r, c ] >= min_iou ]
    except ImportError:
        pairs = []
        used_rows = set()
        used_cols = set()
        order = np.dstack( np.unravel_index(
          np.argsort( -iou_matrix, axis=None ), iou_matrix.shape ) )[0]
        for r, c in order:
            if iou_matrix[ r, c ] < min_iou:
                break
            if r in used_rows or c in used_cols:
                continue
            pairs.append( ( int( r ), int( c ) ) )
            used_rows.add( int( r ) )
            used_cols.add( int( c ) )
        return pairs

def mean_frame_boxes( frame_dicts ):
    """Average a list of dicts mapping frame id to a box, returning a
    dict with the per-frame mean box over the dicts containing that
    frame.
    """
    sums = {}
    counts = {}
    for frame_dict in frame_dicts:
        for frame, box in frame_dict.items():
            box = np.asarray( box, dtype=float )
            if frame in sums:
                sums[ frame ] = sums[ frame ] + box
                counts[ frame ] += 1
            else:
                sums[ frame ] = box.copy()
                counts[ frame ] = 1
    return { f: sums[f] / counts[f] for f in sums }

def cluster_tracks( tracker_tracks, min_iou, mode='mean',
                    min_overlap_frames=1 ):
    """Cluster tracks across trackers by tube IoU.

    tracker_tracks is a list (one entry per tracker) of lists of track
    data dicts, each mapping frame id to a tuple whose first element is
    a bounding box. Returns a list of clusters, each a list of
    (tracker_index, track_index) pairs. A cluster holds at most one
    track per tracker per frame, but may absorb multiple temporally
    disjoint fragments from the same tracker.
    """
    clusters = []
    cluster_frames = []

    for t, tracks in enumerate( tracker_tracks ):
        track_frames = [ { f: np.asarray( e[0], dtype=float )
                           for f, e in td.items() } for td in tracks ]
        if not clusters:
            for i in range( len( tracks ) ):
                clusters.append( [ ( t, i ) ] )
                cluster_frames.append( track_frames[i] )
            continue
        iou_matrix = np.zeros( ( len( clusters ), len( tracks ) ) )
        for c in range( len( clusters ) ):
            for i in range( len( tracks ) ):
                iou_matrix[ c, i ] = tube_iou( cluster_frames[c],
                  track_frames[i], mode, min_overlap_frames )
        assigned = {}
        for c, i in assign_pairs( iou_matrix, min_iou ):
            clusters[c].append( ( t, i ) )
            cluster_frames[c] = mean_frame_boxes(
              [ cluster_frames[c], track_frames[i] ] )
            assigned[ i ] = c
        # Second chance: fragments of this tracker left unassigned may
        # join a cluster when temporally disjoint from any same-tracker
        # track already in it (handles fragmented trackers)
        for i in range( len( tracks ) ):
            if i in assigned:
                continue
            best_c = -1
            best_iou = min_iou
            for c in range( len( iou_matrix ) ):
                if iou_matrix[ c, i ] < best_iou:
                    continue
                conflict = any(
                  tj == t and track_frames[i].keys() & track_frames[j].keys()
                  for tj, j in clusters[c] )
                if not conflict:
                    best_c = c
                    best_iou = iou_matrix[ c, i ]
            if best_c >= 0:
                clusters[ best_c ].append( ( t, i ) )
                cluster_frames[ best_c ] = mean_frame_boxes(
                  [ cluster_frames[ best_c ], track_frames[i] ] )
            else:
                clusters.append( [ ( t, i ) ] )
                cluster_frames.append( track_frames[i] )

    return clusters

def fuse_cluster( track_datas, weights ):
    """Fuse member track data dicts (frame id -> (box, confidence,
    class distribution dict, time)) into one output track: a list of
    (frame, box, confidence, class distribution, time) tuples sorted
    by frame. Boxes are averaged weighted by tracker weight times
    detection confidence; class distributions are fused with the same
    weighting, normalized, then scaled so the top class carries the
    fused confidence.
    """
    all_frames = set()
    for track_data in track_datas:
        all_frames.update( track_data )

    output = []
    for frame in sorted( all_frames ):
        box_acc = np.zeros( 4 )
        box_wt = 0.0
        conf_acc = 0.0
        conf_wt = 0.0
        dist_acc = {}
        time = None
        for track_data, weight in zip( track_datas, weights ):
            if frame not in track_data:
                continue
            box, conf, dist, state_time = track_data[ frame ]
            wt = weight * max( conf, 1e-6 )
            box_acc += wt * np.asarray( box, dtype=float )
            box_wt += wt
            conf_acc += weight * conf
            conf_wt += weight
            for name, prob in dist.items():
                dist_acc[ name ] = dist_acc.get( name, 0.0 ) + wt * prob
            if time is None:
                time = state_time
        if box_wt <= 0:
            continue
        fused_box = box_acc / box_wt
        fused_conf = conf_acc / conf_wt if conf_wt > 0 else 0.0
        fused_dist = {}
        top = max( dist_acc.values() ) if dist_acc else 0.0
        if top > 0:
            fused_dist = { n: p * fused_conf / top
                           for n, p in dist_acc.items() }
        output.append( ( frame, fused_box, fused_conf, fused_dist, time ) )
    return output

def merge_track_sets( tracker_tracks, weights, min_iou, mode='mean',
                      min_overlap_frames=1, min_trackers=1 ):
    """Merge track data across trackers. tracker_tracks is a list per
    tracker of lists of track data dicts (frame id -> (box, confidence,
    class distribution, time)). Returns a list of fused tracks, each a
    list of (frame, box, confidence, class distribution, time) tuples
    sorted by frame.
    """
    clusters = cluster_tracks( tracker_tracks, min_iou, mode,
                               min_overlap_frames )
    output = []
    for members in clusters:
        if len( set( t for t, _ in members ) ) < min_trackers:
            continue
        member_datas = [ tracker_tracks[t][i] for t, i in members ]
        member_weights = [ weights[t] for t, _ in members ]
        fused = fuse_cluster( member_datas, member_weights )
        if fused:
            output.append( fused )
    return output

# --------------------------- KWIVER GLUE -------------------------------

def track_to_data( track ):
    """Convert a KWIVER Track of ObjectTrackStates into a track data
    dict mapping frame id to (box, confidence, class distribution
    dict, time).
    """
    output = {}
    for state in track:
        det = state.detection()
        if det is None:
            continue
        bbox = det.bounding_box
        box = np.array( [ bbox.min_x(), bbox.min_y(),
                          bbox.max_x(), bbox.max_y() ] )
        dist = {}
        if det.type is not None:
            for name in det.type.class_names():
                dist[ name ] = det.type.score( name )
        time = getattr( state, 'time_usec', state.frame_id )
        output[ state.frame_id ] = ( box, float( det.confidence ),
                                     dist, time )
    return output

def fused_to_ObjectTrackSet( fused_tracks ):
    tracks = []
    for track_id, states in enumerate( fused_tracks, 1 ):
        track = Track( id=track_id )
        for frame, box, conf, dist, time in states:
            dot = DetectedObjectType()
            for name, prob in dist.items():
                if prob > 1e-6:
                    dot.set_score( name, prob )
            det = DetectedObject(
              BoundingBoxD( box[0], box[1], box[2], box[3] ), conf, dot )
            track.append( ObjectTrackState( frame, time, det ) )
        tracks.append( track )
    return ObjectTrackSet( tracks )

class MergeTracksTubeIoU( KwiverProcess ):
    """Sprokit process fusing multiple object track set streams into one
    using tube-IoU track association.
    """
    def __init__( self, config ):
        KwiverProcess.__init__( self, config )

        add_declare_config( self, 'n_input', '2',
          'Number of input track sets' )
        add_declare_config( self, 'min_iou', '0.3',
          'Minimum tube IoU to associate tracks across trackers' )
        add_declare_config( self, 'iou_mode', 'mean',
          'Tube IoU flavor: "mean" averages per-frame IoU over co-visible '
          'frames; "tube" is strict spatiotemporal IoU over the union' )
        add_declare_config( self, 'min_overlap_frames', '1',
          'Minimum number of co-visible frames to associate tracks' )
        add_declare_config( self, 'weights', '[]',
          'Per-tracker fusion weights; empty for equal weighting' )
        add_declare_config( self, 'min_trackers', '1',
          'Minimum number of distinct trackers which must contribute to '
          'a fused track for it to be output' )

        optional = process.PortFlags()
        required = process.PortFlags()
        required.add( self.flag_required )

        self._n_input = int( self.config_value( 'n_input' ) )
        for i in range( 1, self._n_input + 1 ):
            add_declare_input_port( self, 'object_track_set' + str( i ),
              'object_track_set', required,
              'Input object track set #' + str( i ) )
        add_declare_output_port( self, 'object_track_set',
          'object_track_set', optional, 'Fused output object track set' )

    def _configure( self ):
        self._n_input = int( self.config_value( 'n_input' ) )
        self._min_iou = float( self.config_value( 'min_iou' ) )
        self._iou_mode = str( self.config_value( 'iou_mode' ) )
        self._min_overlap_frames = int(
          self.config_value( 'min_overlap_frames' ) )
        self._min_trackers = int( self.config_value( 'min_trackers' ) )
        weights = ast.literal_eval( self.config_value( 'weights' ) )
        if len( weights ) < self._n_input:
            weights = list( weights ) + \
                      [ 1.0 ] * ( self._n_input - len( weights ) )
        self._weights = weights
        self._base_configure()

    def _step( self ):
        tracker_tracks = []
        for i in range( 1, self._n_input + 1 ):
            track_set = self.grab_input_using_trait(
              'object_track_set' + str( i ) )
            tracks = track_set.tracks() if track_set is not None else []
            tracker_tracks.append(
              [ track_to_data( t ) for t in tracks ] )

        fused = merge_track_sets( tracker_tracks, self._weights,
          self._min_iou, self._iou_mode, self._min_overlap_frames,
          self._min_trackers )

        self.push_to_port_using_trait( 'object_track_set',
          fused_to_ObjectTrackSet( fused ) )
        self._base_step()

def __sprokit_register__():
    from kwiver.sprokit.pipeline import process_factory
    module_name = 'python:viame.python.MergeTracksTubeIoU'
    if process_factory.is_process_module_loaded( module_name ):
        return
    process_factory.add_process(
        'merge_track_sets_tube_iou',
        'Fusion of multiple object track sets via tube-IoU association',
        MergeTracksTubeIoU,
    )
    process_factory.mark_process_module_as_loaded( module_name )
