# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Shared helpers for tracker trainers.

Two things every tracker trainer needs and none of them had:

Sequence grouping. train_tracker's add_data_from_disk is handed one flat list
of image files covering every clip, alongside one track set per clip. The
frame_id on a track state indexes the clip it came from, not that flat list,
so indexing the flat list with it is right for the first clip and wrong for
all the others. group_files_by_sequence splits the flat list back into per
clip lists, which also recovers the clip's name.

Computed detections. Every trainer sees perfect groundtruth: exact boxes, and
confidences that are 1.0 for every annotation. A tracker tuned on that has
never met the detector it will run behind. load_computed_detections reads what
a detector actually produced for the same clips, and match_to_groundtruth
pairs those with the groundtruth so a trainer can measure the detector's
localisation error and separate its true from its false positives.
"""

import os

from collections import OrderedDict


# ---------------------------------------------------------------------------
# Sequence grouping
# ---------------------------------------------------------------------------

def group_files_by_sequence( image_files, expected=None ):
    """Split a flat list of image files into one list per source clip.

    Frames extracted from a video land in a directory of their own, so the
    parent directory names the clip. The order of first appearance is the
    order the clips were read, which is the order of the track sets.

    Args:
        image_files: the flat list add_data_from_disk was handed
        expected: number of track sets, when known. Grouping that disagrees
            with it is not trusted, since acting on a bad split would pair
            annotations with another clip's pixels just as surely as not
            splitting at all.

    Returns:
        ( groups, names ), where groups is a list of lists of file paths and
        names is the corresponding clip names. ( None, None ) when the files
        cannot be split with confidence, which the caller should treat as a
        reason to fall back and warn rather than to guess.
    """
    if not image_files:
        return None, None

    groups = OrderedDict()

    for path in image_files:
        parent = os.path.dirname( path )
        groups.setdefault( parent, [] ).append( path )

    if len( groups ) <= 1:
        # Everything in one directory. Either there genuinely is one clip, or
        # this is a flat folder of images that cannot be told apart by path.
        if expected is not None and expected > 1:
            return None, None

    if expected is not None and len( groups ) != expected:
        return None, None

    names = [ os.path.splitext( os.path.basename( p.rstrip( '/' ) ) )[ 0 ]
              for p in groups.keys() ]

    return list( groups.values() ), names


def build_sequence_maps( image_files, track_set_count, label="training" ):
    """One frame id to file map per track set.

    Args:
        image_files: the flat list add_data_from_disk was handed
        track_set_count: how many track sets go with it
        label: named in the warning, to say which split fell back

    Returns:
        ( maps, names ). maps[ i ] belongs to track set i. On failure to split
        the files, every entry is the flat map the trainers used to build, so
        behaviour is no worse than before, and a warning says so.
    """
    groups, names = group_files_by_sequence( image_files,
                                             expected=track_set_count )

    if groups is None:
        print( "WARNING: could not split the {} images into one group per "
               "sequence ({} files, {} track sets). Frame ids are positions "
               "within their own sequence, so they will be resolved against "
               "the whole list and every sequence after the first will read "
               "another sequence's images. Extracted video frames land in a "
               "directory per clip, which is what this needs."
               .format( label, len( image_files ), track_set_count ) )

        flat = { i: path for i, path in enumerate( image_files ) }

        return [ flat ] * track_set_count, [ None ] * track_set_count

    return [ sequence_image_map( g ) for g in groups ], names


def sequence_image_map( sequence_files ):
    """Map a clip's frame ids to its files.

    The ids on a track state are positions within the clip, so the sorted
    file list is indexed directly. Sorting matters: the flat list arrives in
    read order, which is not always frame order.
    """
    ordered = sorted( sequence_files )

    return { i: path for i, path in enumerate( ordered ) }


# ---------------------------------------------------------------------------
# Computed detections
# ---------------------------------------------------------------------------

def load_computed_detections( directory, sequence_name ):
    """Read a detector's output for one clip.

    Looks for <sequence_name>.csv in directory, in VIAME CSV, which is what
    a detection or tracking pipeline writes. The name is matched against the
    clip name with and without an extension, since frames extracted from
    clip.mp4 land in a directory called clip.mp4.

    Returns:
        frame id -> list of ( x1, y1, x2, y2, confidence ), empty when there
        is no file for this clip.
    """
    if not directory or not sequence_name:
        return {}

    stem = os.path.splitext( sequence_name )[ 0 ]

    candidates = [ sequence_name + ".csv", stem + ".csv",
                   sequence_name + ".txt", stem + ".txt" ]

    path = None

    for candidate in candidates:
        trial = os.path.join( directory, candidate )

        if os.path.isfile( trial ):
            path = trial
            break

    if path is None:
        return {}

    detections = {}

    with open( path ) as handle:
        for line in handle:
            if not line.strip() or line.lstrip().startswith( '#' ):
                continue

            fields = line.rstrip( '\n' ).split( ',' )

            if len( fields ) < 8:
                continue

            try:
                frame = int( float( fields[ 2 ] ) )
                x1, y1, x2, y2 = ( float( f ) for f in fields[ 3:7 ] )
                confidence = float( fields[ 7 ] )
            except ValueError:
                continue

            if x2 <= x1 or y2 <= y1:
                continue

            detections.setdefault( frame, [] ).append(
                ( x1, y1, x2, y2, confidence ) )

    return detections


def detector_statistics( track_sets, image_files, directory,
                         iou_threshold=0.5 ):
    """Measure a detector against the groundtruth over the same clips.

    Frame ids here are positions within a clip, so the computed detections
    have to have been produced over the same frames the training is reading.
    Running the pipeline over the extracted frame directories does that;
    running it over the source videos does not, since the extraction takes
    every nth frame. A numbering that does not line up shows up as almost
    nothing matching, which the caller is expected to check rather than
    quietly treat as a detector that finds nothing.

    Returns a dict with:
        matched_confidences: confidence of detections that found a real
            object, so the scores a threshold must keep
        unmatched_confidences: confidence of detections that found nothing,
            so the scores a threshold must reject
        center_errors: centre offset of a matched box from its truth, in
            units of the truth box height
        scale_errors: |w - w_truth| / w_truth and the same for height
        gt_boxes, matched_boxes, frames_with_computed, frames_total
    """
    stats = {
        'matched_confidences': [],
        'unmatched_confidences': [],
        'center_errors': [],
        'scale_errors': [],
        'gt_boxes': 0,
        'matched_boxes': 0,
        'frames_with_computed': 0,
        'frames_total': 0,
    }

    if not directory:
        return stats

    maps, names = build_sequence_maps( image_files, len( track_sets ),
                                       "training" )

    for seq_idx, track_set in enumerate( track_sets ):
        if track_set is None:
            continue

        name = names[ seq_idx ] if names else None

        if name is None:
            continue

        computed = load_computed_detections( directory, name )

        if not computed:
            continue

        truth_by_frame = {}

        for track in track_set.tracks():
            for state in track:
                det = state.detection()

                if det is None:
                    continue

                box = det.bounding_box
                truth_by_frame.setdefault( state.frame_id, [] ).append(
                    ( box.min_x(), box.min_y(), box.max_x(), box.max_y(),
                      track.id ) )

        for frame_id, truth in truth_by_frame.items():
            stats[ 'frames_total' ] += 1
            stats[ 'gt_boxes' ] += len( truth )

            frame_computed = computed.get( frame_id, [] )

            if frame_computed:
                stats[ 'frames_with_computed' ] += 1

            matches, unmatched, _missed = match_to_groundtruth(
                frame_computed, truth, iou_threshold )

            stats[ 'matched_boxes' ] += len( matches )

            for c, t, _overlap in matches:
                stats[ 'matched_confidences' ].append( c[ 4 ] )

                t_w = t[ 2 ] - t[ 0 ]
                t_h = t[ 3 ] - t[ 1 ]

                if t_w <= 0 or t_h <= 0:
                    continue

                c_cx = ( c[ 0 ] + c[ 2 ] ) / 2
                c_cy = ( c[ 1 ] + c[ 3 ] ) / 2
                t_cx = ( t[ 0 ] + t[ 2 ] ) / 2
                t_cy = ( t[ 1 ] + t[ 3 ] ) / 2

                stats[ 'center_errors' ].append(
                    ( ( c_cx - t_cx ) ** 2 + ( c_cy - t_cy ) ** 2 ) ** 0.5
                    / t_h )
                stats[ 'scale_errors' ].append(
                    abs( ( c[ 2 ] - c[ 0 ] ) - t_w ) / t_w )
                stats[ 'scale_errors' ].append(
                    abs( ( c[ 3 ] - c[ 1 ] ) - t_h ) / t_h )

            for c in unmatched:
                stats[ 'unmatched_confidences' ].append( c[ 4 ] )

    return stats


def iou( a, b ):
    """Intersection over union of two ( x1, y1, x2, y2 ) boxes."""
    left = max( a[ 0 ], b[ 0 ] )
    top = max( a[ 1 ], b[ 1 ] )
    right = min( a[ 2 ], b[ 2 ] )
    bottom = min( a[ 3 ], b[ 3 ] )

    if right <= left or bottom <= top:
        return 0.0

    overlap = ( right - left ) * ( bottom - top )
    area_a = ( a[ 2 ] - a[ 0 ] ) * ( a[ 3 ] - a[ 1 ] )
    area_b = ( b[ 2 ] - b[ 0 ] ) * ( b[ 3 ] - b[ 1 ] )
    union = area_a + area_b - overlap

    if union <= 0:
        return 0.0

    return overlap / union


def match_to_groundtruth( computed, truth, threshold=0.5 ):
    """Pair one frame's computed detections with its groundtruth boxes.

    Greedy by descending overlap, one truth box to one detection, which is
    the usual detection metric convention.

    Args:
        computed: list of ( x1, y1, x2, y2, confidence )
        truth: list of ( x1, y1, x2, y2, identity ), identity being whatever
            the caller wants back for a match
        threshold: minimum overlap to call it the same object

    Returns:
        ( matches, unmatched_computed, missed ) where matches is a list of
        ( computed, truth, overlap ), unmatched_computed is the detections
        that hit nothing, being the detector's false positives, and missed is
        the truth boxes nothing found.
    """
    pairs = []

    for c_idx, c in enumerate( computed ):
        for t_idx, t in enumerate( truth ):
            overlap = iou( c[ :4 ], t[ :4 ] )

            if overlap >= threshold:
                pairs.append( ( overlap, c[ 4 ], c_idx, t_idx ) )

    # Best overlap first, and on a tie the more confident detection. Ordering
    # a tie by list position instead would hand the box to whichever detection
    # happened to be written first.
    pairs.sort( key=lambda p: ( -p[ 0 ], -p[ 1 ] ) )

    used_computed = set()
    used_truth = set()
    matches = []

    for overlap, _confidence, c_idx, t_idx in pairs:
        if c_idx in used_computed or t_idx in used_truth:
            continue

        used_computed.add( c_idx )
        used_truth.add( t_idx )
        matches.append( ( computed[ c_idx ], truth[ t_idx ], overlap ) )

    unmatched = [ c for i, c in enumerate( computed )
                  if i not in used_computed ]
    missed = [ t for i, t in enumerate( truth ) if i not in used_truth ]

    return matches, unmatched, missed
