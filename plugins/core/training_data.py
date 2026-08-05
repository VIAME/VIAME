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
import sys

from collections import OrderedDict


# Training runs embedded in the kwiver process, and its stdout is a pipe
# rather than a terminal, so python block buffers it. Nothing finalises the
# interpreter on the way out, so whatever is still in that buffer when the
# process exits is discarded -- which is always the tail, the part holding the
# final losses and the model that was selected. A three epoch run logs two
# epochs and stops mid sentence, and the run looks like it died where it in
# fact finished.
#
# Line buffering costs nothing at these volumes and makes the log match what
# actually ran. Done here because every tracker trainer imports this module.
try:
    sys.stdout.reconfigure( line_buffering=True )
    sys.stderr.reconfigure( line_buffering=True )
except ( AttributeError, ValueError ):
    # Not a reconfigurable stream, which is fine; this is a convenience
    pass


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

    # Split on each change of directory as the list is walked, rather than
    # collecting by directory name. The list arrives as one contiguous block
    # per input, in the order the inputs were read, so a run of files sharing
    # a directory is exactly one input.
    #
    # Collecting by name instead merged any two inputs that happened to sit in
    # the same directory and, worse, gave a group count that did not have to
    # equal the number of inputs. On a set that mixes videos, whose frames are
    # extracted into a directory each, with folders of images used where they
    # lie, the two counts disagreed and the whole thing fell back -- which is
    # to say it kept resolving frame ids against the wrong sequence.
    groups = []
    names = []
    previous = None

    for path in image_files:
        parent = os.path.dirname( path )

        if parent != previous:
            groups.append( [] )
            names.append( os.path.splitext(
                os.path.basename( parent.rstrip( '/' ) ) )[ 0 ] )
            previous = parent

        groups[ -1 ].append( path )

    if expected is not None and len( groups ) != expected:
        return None, None

    return groups, names


def build_sequence_maps( image_files, track_set_count, label="training",
                         frame_bounds=None ):
    """One frame id to file map per track set.

    Args:
        image_files: the flat list add_data_from_disk was handed
        track_set_count: how many track sets go with it
        label: named in the warning, to say which split fell back
        frame_bounds: per track set, the highest frame id it refers to, or
            None where it refers to none. Given these the alignment is
            checked rather than assumed, which matters because the two counts
            need not agree: a clip can contribute a track set and no images,
            and then there are fewer groups than track sets and no way to tell
            positionally which ones went missing.

    Returns:
        ( maps, names ). maps[ i ] belongs to track set i, and is empty for a
        track set with no images of its own. On failure to align, every entry
        is the flat map the trainers used to build, so behaviour is no worse
        than before, and a warning says so.
    """
    if not track_set_count or not image_files:
        return [], []

    groups, names = group_files_by_sequence( image_files )

    def flat_fallback( found ):
        print( "WARNING: could not split the {} images into one group per "
               "sequence: {} files fall into {} directories but there are {} "
               "track sets, and no alignment of the two was consistent with "
               "the frame ids. They will be resolved against the whole list "
               "instead, so every sequence after the first will read another "
               "sequence's images."
               .format( label, len( image_files ), found, track_set_count ) )

        # Enough to work out what the layout really is without another run
        if frame_bounds is not None and groups:
            empty = sum( 1 for b in frame_bounds if b is None )
            sizes = sorted( len( g ) for g in groups )
            print( "  {} of the {} track sets refer to no frames; "
                   "{} refer to some".format( empty, track_set_count,
                                              track_set_count - empty ) )
            print( "  directory sizes: min {} median {} max {}".format(
                sizes[ 0 ], sizes[ len( sizes ) // 2 ], sizes[ -1 ] ) )

            over = [ ( i, frame_bounds[ i ], len( groups[ i ] ) )
                     for i in range( min( len( groups ), track_set_count ) )
                     if frame_bounds[ i ] is not None
                     and frame_bounds[ i ] >= len( groups[ i ] ) ]
            print( "  straight alignment: {} of {} track sets ask for a frame "
                   "their directory does not have".format(
                       len( over ), track_set_count ) )
            for i, bound, size in over[ :5 ]:
                print( "     track set {} wants frame {} of a {} frame "
                       "directory".format( i, bound, size ) )

        flat = { i: path for i, path in enumerate( image_files ) }

        return [ flat ] * track_set_count, [ None ] * track_set_count

    if not groups:
        return flat_fallback( 0 )

    def consistent( assignment ):
        """Does every track set's highest frame id fit the group it was given?

        A misalignment shows up here as a track set asking for a frame its
        group does not have. It is not proof of correctness -- a short clip
        can fit inside the wrong group -- but it catches the case that
        matters, where the counts differ and the tail is shifted.
        """
        if frame_bounds is None:
            return len( groups ) == track_set_count

        for index, group_index in assignment.items():
            bound = frame_bounds[ index ]

            if bound is None:
                continue

            if group_index is None or bound >= len( groups[ group_index ] ):
                return False

        return True

    # One group per track set, in order
    straight = { i: ( i if i < len( groups ) else None )
                 for i in range( track_set_count ) }

    if len( groups ) == track_set_count and consistent( straight ):
        return ( [ sequence_image_map( groups[ i ] ) for i in
                   range( track_set_count ) ], list( names ) )

    # Otherwise assume the track sets that refer to no frames are the ones
    # that contributed no images, and give the groups to the rest in order
    if frame_bounds is not None:
        skipped = {}
        cursor = 0

        for i in range( track_set_count ):
            if frame_bounds[ i ] is None:
                skipped[ i ] = None
            else:
                skipped[ i ] = cursor if cursor < len( groups ) else None
                cursor += 1

        if cursor == len( groups ) and consistent( skipped ):
            print( "{} images: {} directories for {} track sets, and the {} "
                   "track sets that refer to no frames account for the "
                   "difference.".format( label.capitalize(), len( groups ),
                                         track_set_count,
                                         track_set_count - len( groups ) ) )

            maps, out_names = [], []

            for i in range( track_set_count ):
                g = skipped[ i ]
                maps.append( sequence_image_map( groups[ g ] ) if g is not None
                             else {} )
                out_names.append( names[ g ] if g is not None else None )

            return maps, out_names

    return flat_fallback( len( groups ) )


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

    # process_video.py writes <stem>_detections.csv and <stem>_tracks.csv, so
    # its output directory can be pointed at directly rather than renamed
    # first. Detections are preferred: the tracks file has been through a
    # tracker, and what is wanted here is what the detector said.
    candidates = []

    for base in ( sequence_name, stem ):
        candidates += [ base + "_detections.csv", base + ".csv",
                        base + "_tracks.csv", base + ".txt" ]

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
                         iou_threshold=0.5, sequence_manifest="" ):
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
        miss_runs: for each groundtruth track, the lengths of consecutive
            annotated frames over which the detector found nothing for it.
            This is the gap a lost track has to survive at inference, which
            is not the same quantity as a gap in the annotation itself
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
        'miss_runs': [],
    }

    if not directory:
        return stats

    # The manifest the training tool writes, when the caller passed it on.
    # The directory-layout heuristic below needs the image list to divide
    # into exactly one directory per track set, and it stops holding as soon
    # as anything else shares the tree -- an augmentation cache regrown with
    # more clips than the annotations cover turned every one of these
    # statistics into "0 of 0" for a trainer relying on the guess.
    maps = names = None

    if sequence_manifest:
        maps, names = read_sequence_manifest(
            sequence_manifest, image_files, len( track_sets ) )

    if names is None:
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

        hit_by_track = {}

        for frame_id, truth in sorted( truth_by_frame.items() ):
            stats[ 'frames_total' ] += 1
            stats[ 'gt_boxes' ] += len( truth )

            frame_computed = computed.get( frame_id, [] )

            if frame_computed:
                stats[ 'frames_with_computed' ] += 1

            matches, unmatched, _missed = match_to_groundtruth(
                frame_computed, truth, iou_threshold )

            stats[ 'matched_boxes' ] += len( matches )

            found = set( t[ 4 ] for _c, t, _o in matches )

            for t in truth:
                hit_by_track.setdefault( t[ 4 ], [] ).append(
                    ( frame_id, t[ 4 ] in found ) )

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

        for states in hit_by_track.values():
            run = 0

            for _frame_id, hit in sorted( states ):
                if hit:
                    if run:
                        stats[ 'miss_runs' ].append( run )
                    run = 0
                else:
                    run += 1

            if run:
                stats[ 'miss_runs' ].append( run )

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


def read_sequence_manifest( path, image_files, track_set_count ):
    """The frame to track set association, as the training tool recorded it.

    The tool builds the flat frame list and the track set list in two separate
    passes with their own filtering, so which frames belong to which track set
    cannot be recovered from either. It now writes the association out; this
    reads it.

    Returns ( maps, names ) as build_sequence_maps does, or ( None, None )
    when there is no usable manifest, leaving the caller to fall back to
    guessing from the directory layout.
    """
    if not path or not os.path.isfile( path ):
        return None, None

    maps = [ {} for _ in range( track_set_count ) ]
    names = [ None ] * track_set_count
    seen = 0

    with open( path ) as handle:
        for line in handle:
            if not line.strip() or line.lstrip().startswith( '#' ):
                continue

            fields = line.split( None, 3 )

            if len( fields ) < 4:
                continue

            try:
                index = int( fields[ 0 ] )
                first = int( fields[ 1 ] )
                count = int( fields[ 2 ] )
            except ValueError:
                continue

            source = fields[ 3 ].strip()

            if index >= track_set_count:
                continue

            files = image_files[ first:first + count ]
            maps[ index ] = sequence_image_map( files ) if files else {}
            names[ index ] = os.path.splitext(
                os.path.basename( source.rstrip( '/' ) ) )[ 0 ]
            seen += 1

    if not seen:
        return None, None

    print( "Sequence manifest: {} of {} track sets placed against {} frames"
           .format( seen, track_set_count, len( image_files ) ) )

    return maps, names


def split_validation( track_sets, maps, names, fraction=0.1 ):
    """Hold the last clips out as a validation split.

    Tracker training only receives a validation set when one is named
    explicitly, so a normal run hands the trainer every clip and nothing to
    measure against. A model is then chosen on training loss, which cannot
    say whether it has begun to memorise.

    The split is by clip and takes the tail rather than a sample: frames
    within a clip are far from independent, so holding out a slice of frames
    from clips that are also trained on measures very little. The tail is used
    rather than a random subset so the same clips are held out every run,
    which makes two runs comparable.

    Args:
        track_sets: one per clip, as add_data_from_disk was handed them
        maps: frame id to file, one per track set
        names: clip name per track set, may be None
        fraction: how much to hold back, 0 to disable

    Returns:
        ( train, validation ), each a tuple of ( track_sets, maps, names ).
        The validation part is empty when there is too little to split.
    """
    empty = ( [], [], [] )

    if not fraction or fraction <= 0 or len( track_sets ) < 2:
        return ( track_sets, maps, names ), empty

    usable = [ i for i, t in enumerate( track_sets ) if t is not None ]

    if len( usable ) < 2:
        return ( track_sets, maps, names ), empty

    holdout = max( 1, int( len( usable ) * fraction ) )

    # Never hold back so much that training has less than half the clips
    holdout = min( holdout, len( usable ) // 2 )

    if holdout < 1:
        return ( track_sets, maps, names ), empty

    held = set( usable[ -holdout: ] )

    def take( keep ):
        idx = [ i for i in range( len( track_sets ) ) if ( i in held ) == keep ]
        return (
            [ track_sets[ i ] for i in idx ],
            [ maps[ i ] for i in idx ] if maps else [],
            [ names[ i ] for i in idx ] if names else [],
        )

    train_part = take( False )
    valid_part = take( True )

    print( "Validation split: {} clips held out of {}, by clip rather than by "
           "frame".format( len( valid_part[ 0 ] ), len( usable ) ) )

    return train_part, valid_part
