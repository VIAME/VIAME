# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Closed-loop parameter search for the detection-driven trackers.

The estimators in the tracker trainers fit their parameters from the shape of
the groundtruth: the association gate from the distribution of consecutive
frame IoU, the track buffer from how long the detector loses an animal, the
confidence thresholds from where matched and unmatched detections separate.
Each is a defensible proxy, and none of them measures tracking.

Measured against a sweep on the SEFSC test set, all of them erred the same
way -- conservative -- and the cost was not small:

    fitted by the estimators   HOTA 0.5111   track_pd 0.629
    best found by sweeping     HOTA 0.5256   track_pd 0.784

Two of the three tuned values sat exactly on the estimators' own clamp
floors, which is the signature of a proxy that is centred in the wrong place
rather than one that is merely noisy.

This module closes the loop: it runs the real tracker over the training
clips' own detections and scores the result, so the parameters are chosen by
the quantity they are meant to improve. It is affordable because these
trackers fit scalars rather than a network -- one clip evaluates in about a
second -- and it needs nothing the trainer does not already hold.

IDF1 is the objective rather than HOTA. HOTA needs the C++ scorer, while
IDF1 is available in process from the bundled motmetrics, and over 28
configurations measured on SEFSC the two agreed on rank with a Spearman
correlation of 0.992; selecting on IDF1 picked a configuration 0.0009 HOTA
from the true optimum. That is well inside the spread of the top group.

The search starts from the estimated values and only replaces them when a
candidate scores better on held-out data, so enabling it cannot do worse
than the estimators alone except by chance on the validation clips.
"""

import itertools
import os

import numpy as np

from viame.core.training_data import (
    build_sequence_maps,
    load_computed_detections,
    read_sequence_manifest,
)


# Candidates are searched one parameter at a time, in this order. The first
# entry of each list is a placeholder replaced by the estimated value, so a
# tie always resolves in favour of what the estimators produced.
DEFAULT_GRID = (
    ( 'high_thresh', ( 0.20, 0.30, 0.45, 0.60 ) ),
    ( 'low_thresh', ( 0.05, 0.10, 0.20 ) ),
    ( 'match_thresh', ( 0.80, 0.90, 0.95, 0.98 ) ),
    ( 'track_buffer', ( 10, 21, 35 ) ),
)

# match_thresh drives all three association gates. Sweeping them apart was
# measured and bought 0.0007 HOTA, which is noise, so they move together.
MATCH_KEYS = ( 'match_thresh', 'second_match_thresh', 'unconfirmed_match_thresh' )


def _groundtruth_by_frame( track_set ):
    """frame id -> list of ( x1, y1, x2, y2, track id )."""
    out = {}

    for track in track_set.tracks():
        for state in track:
            det = state.detection()

            if det is None:
                continue

            box = det.bounding_box
            out.setdefault( state.frame_id, [] ).append(
                ( box.min_x(), box.min_y(), box.max_x(), box.max_y(),
                  track.id ) )

    return out


def collect_sequences( track_sets, image_files, directory,
                       sequence_manifest="", max_sequences=0,
                       min_boxes=20 ):
    """Pair each clip's detections with its groundtruth.

    Clips are sampled evenly across the range of detection counts rather
    than taken in order, so a search on a subset is not decided by whichever
    clips happen to sort first.
    """
    if not directory:
        return []

    maps = names = None

    if sequence_manifest:
        maps, names = read_sequence_manifest(
            sequence_manifest, image_files, len( track_sets ) )

    if names is None:
        maps, names = build_sequence_maps( image_files, len( track_sets ),
                                           "parameter search" )

    found = []

    for seq_idx, track_set in enumerate( track_sets ):
        if track_set is None:
            continue

        name = names[ seq_idx ] if names else None

        if name is None:
            continue

        computed = load_computed_detections( directory, name )

        if not computed:
            continue

        truth = _groundtruth_by_frame( track_set )
        boxes = sum( len( v ) for v in truth.values() )

        if boxes < min_boxes:
            continue

        found.append( ( sum( len( v ) for v in computed.values() ),
                        computed, truth ) )

    found.sort( key=lambda entry: entry[ 0 ] )

    if max_sequences and len( found ) > max_sequences:
        step = len( found ) / float( max_sequences )
        found = [ found[ int( i * step ) ] for i in range( max_sequences ) ]

    return [ ( c, t ) for _n, c, t in found ]


def _run_tracker( implementation, params, detections, frame_rate ):
    """Drive the tracker over one clip's detections, without imagery.

    Returns frame id -> list of ( x1, y1, x2, y2, confidence, track id ).
    """
    from kwiver.vital.algo import TrackObjects
    from kwiver.vital.types import (
        BoundingBoxD, DetectedObject, DetectedObjectSet, DetectedObjectType,
        Timestamp,
    )

    algorithm = TrackObjects.create( implementation )
    config = algorithm.get_configuration()

    for key, value in params.items():
        config.set_value( key, str( value ) )

    algorithm.set_configuration( config )

    period = 1.0 / frame_rate if frame_rate else 0.2
    final = None

    for frame_id in sorted( detections ):
        detected = DetectedObjectSet()

        for x1, y1, x2, y2, confidence in detections[ frame_id ]:
            detected.add( DetectedObject(
                BoundingBoxD( x1, y1, x2, y2 ), confidence,
                DetectedObjectType( "object", confidence ) ) )

        stamp = Timestamp()
        stamp.set_frame( frame_id )
        stamp.set_time_seconds( frame_id * period )

        result = algorithm.track( stamp, None, detected )

        if result is not None:
            final = result

    if final is None:
        return {}

    tracks = {}

    for track in final.tracks():
        for state in track:
            det = state.detection()

            if det is None:
                continue

            box = det.bounding_box
            tracks.setdefault( state.frame_id, [] ).append(
                ( box.min_x(), box.min_y(), box.max_x(), box.max_y(),
                  det.confidence, track.id ) )

    return tracks


def _idf1( truth, hypothesis, confidence_threshold, iou_threshold=0.5 ):
    """IDF1 of one clip, over boxes at or above the confidence threshold."""
    import motmetrics as mm

    accumulator = mm.MOTAccumulator( auto_id=False )

    for frame_id in sorted( set( truth ) | set( hypothesis ) ):
        gt = truth.get( frame_id, [] )
        hyp = [ h for h in hypothesis.get( frame_id, [] )
                if h[ 4 ] >= confidence_threshold ]

        if gt and hyp:
            gt_boxes = np.array(
                [ [ g[0], g[1], g[2] - g[0], g[3] - g[1] ] for g in gt ] )
            hyp_boxes = np.array(
                [ [ h[0], h[1], h[2] - h[0], h[3] - h[1] ] for h in hyp ] )
            distance = mm.distances.iou_matrix( gt_boxes, hyp_boxes,
                                                max_iou=iou_threshold )
        else:
            distance = np.empty( ( len( gt ), len( hyp ) ) )

        accumulator.update( [ g[ 4 ] for g in gt ], [ h[ 5 ] for h in hyp ],
                            distance, frameid=frame_id )

    summary = mm.metrics.create().compute( accumulator, metrics=[ 'idf1' ],
                                           name='clip' )
    value = float( summary[ 'idf1' ].iloc[ 0 ] )

    return 0.0 if np.isnan( value ) else value


def search_parameters( implementation, seed_params, sequences,
                       confidence_threshold=0.0, frame_rate=5,
                       grid=DEFAULT_GRID, log=print ):
    """Coordinate descent from the estimated parameters.

    seed_params is replaced key by key, keeping a change only when it scores
    better than what is already held. Returns ( params, report ) where the
    report records every candidate so a run can be inspected afterwards.
    """
    try:
        import motmetrics  # noqa: F401
    except ImportError:
        log( "  motmetrics is unavailable, keeping the estimated parameters" )
        return dict( seed_params ), []

    if not sequences:
        log( "  no clip had both detections and groundtruth, keeping the "
             "estimated parameters" )
        return dict( seed_params ), []

    current = dict( seed_params )
    report = []

    def evaluate( params ):
        total = 0.0

        for detections, truth in sequences:
            tracks = _run_tracker( implementation, params, detections,
                                   frame_rate )
            total += _idf1( truth, tracks, confidence_threshold )

        return total / len( sequences )

    best = evaluate( current )
    report.append( ( 'seed', None, best ) )
    log( "  seed (estimated parameters): IDF1 {:.4f} over {} clips".format(
        best, len( sequences ) ) )

    for key, values in grid:
        if key not in current and key != 'match_thresh':
            continue

        for value in values:
            candidate = dict( current )

            if key == 'match_thresh':
                for match_key in MATCH_KEYS:
                    candidate[ match_key ] = value
            else:
                candidate[ key ] = value

            # low_thresh has to stay under high_thresh or the second
            # association stage is handed an empty band.
            if candidate.get( 'low_thresh', 0 ) >= candidate.get(
                    'high_thresh', 1 ):
                continue

            if candidate == current:
                continue

            score = evaluate( candidate )
            report.append( ( key, value, score ) )
            log( "    {} = {}: IDF1 {:.4f}".format( key, value, score ) )

            if score > best:
                best = score
                current = candidate

        held = ( current[ MATCH_KEYS[ 0 ] ] if key == 'match_thresh'
                 else current.get( key ) )
        log( "  -> {} = {}".format( key, held ) )

    log( "  search finished: IDF1 {:.4f}".format( best ) )

    return current, report
