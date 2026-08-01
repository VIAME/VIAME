# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
ByteTrack tracker training implementation.

Since ByteTrack uses Kalman filtering rather than learned models,
"training" consists of estimating optimal Kalman filter parameters
and thresholds from input track groundtruth data.

The estimated parameters include:
- _std_weight_position: Position uncertainty weight relative to bbox height
- _std_weight_velocity: Velocity uncertainty weight relative to bbox height
- high_thresh: Confidence threshold for first-stage (high-confidence) matching
- low_thresh: Confidence threshold for second-stage (low-confidence) matching
- match_thresh: IOU threshold for matching
- new_track_thresh: Minimum confidence to create new track
- track_buffer: Number of frames to keep lost tracks
"""

from kwiver.vital.algo import TrainTracker

from kwiver.vital.types import (
    CategoryHierarchy,
    ObjectTrackSet, ObjectTrackState,
    BoundingBoxD, DetectedObjectType
)

from distutils.util import strtobool
from shutil import copyfile

import os
import sys
import json
import numpy as np

from viame.core.training_data import detector_statistics


class ByteTrackTrainer( TrainTracker ):
    """
    Implementation of TrainTracker class for ByteTrack parameter estimation.

    ByteTrack doesn't have learned parameters, so this trainer analyzes
    input track groundtruth to estimate optimal Kalman filter parameters
    and detection thresholds.
    """
    def __init__( self ):
        TrainTracker.__init__( self )

        self._identifier = "viame-bytetrack-tracker"
        self._train_directory = "deep_training"
        self._output_directory = "category_models"
        self._output_prefix = "bytetrack_tracker"
        self._pipeline_template = ""
        self._threshold = "0.00"

        # Directory of detector output for the same clips, one VIAME CSV
        # per clip named after it. Left empty the estimates come from the
        # groundtruth alone, whose confidences are all 1.0, so the
        # confidence thresholds carry no information about the detector
        # this tracker will actually run behind.
        self._computed_detections = ""

        # Output parameter bounds (for clamping estimated values). The
        # velocity ceiling used to be 0.1 and the FishTrack23 fit landed
        # exactly on it -- the estimator railed rather than converged -- so
        # the default now leaves headroom above any value yet observed.
        self._min_std_weight_position = 0.01
        self._max_std_weight_position = 0.5
        self._min_std_weight_velocity = 0.001
        self._max_std_weight_velocity = 0.5

        # The association gate is fit from the groundtruth so that this
        # fraction of true consecutive-frame links clears it. 99.5 rather
        # than a tighter figure because assignment still picks the best
        # candidate among those admitted, so the gate is a safety net and
        # not a discriminator. At 97.5 the FishTrack train split yields a
        # gate of 0.238 against 0.051 for test -- the two splits agree in
        # the bulk (median link IoU 0.763 vs 0.755) and differ only in the
        # low tail, so a tight quantile does not transfer between them. The floor and
        # ceiling bound the gate itself (an IoU), not the config value
        # match_thresh, which is 1 - gate.
        self._match_gate_admit_percent = 99.5
        self._min_match_gate = 0.02
        self._max_match_gate = 0.5

        self._categories = []
        self._train_image_files = []
        self._train_tracks = []
        self._test_image_files = []
        self._test_tracks = []

    def get_configuration( self ):
        cfg = super( TrainTracker, self ).get_configuration()

        cfg.set_value( "identifier", self._identifier )
        cfg.set_value( "train_directory", self._train_directory )
        cfg.set_value( "output_directory", self._output_directory )
        cfg.set_value( "output_prefix", self._output_prefix )
        cfg.set_value( "pipeline_template", self._pipeline_template )
        cfg.set_value( "threshold", self._threshold )
        cfg.set_value( "computed_detections", self._computed_detections )
        cfg.set_value( "min_std_weight_position", str( self._min_std_weight_position ) )
        cfg.set_value( "max_std_weight_position", str( self._max_std_weight_position ) )
        cfg.set_value( "min_std_weight_velocity", str( self._min_std_weight_velocity ) )
        cfg.set_value( "max_std_weight_velocity", str( self._max_std_weight_velocity ) )
        cfg.set_value( "match_gate_admit_percent", str( self._match_gate_admit_percent ) )
        cfg.set_value( "min_match_gate", str( self._min_match_gate ) )
        cfg.set_value( "max_match_gate", str( self._max_match_gate ) )

        return cfg

    def set_configuration( self, cfg_in ):
        cfg = self.get_configuration()
        cfg.merge_config( cfg_in )

        self._identifier = str( cfg.get_value( "identifier" ) )
        self._train_directory = str( cfg.get_value( "train_directory" ) )
        self._output_directory = str( cfg.get_value( "output_directory" ) )
        self._output_prefix = str( cfg.get_value( "output_prefix" ) )
        self._pipeline_template = str( cfg.get_value( "pipeline_template" ) )
        self._threshold = str( cfg.get_value( "threshold" ) )
        self._computed_detections = str( cfg.get_value( "computed_detections" ) )
        self._min_std_weight_position = float( cfg.get_value( "min_std_weight_position" ) )
        self._max_std_weight_position = float( cfg.get_value( "max_std_weight_position" ) )
        self._min_std_weight_velocity = float( cfg.get_value( "min_std_weight_velocity" ) )
        self._max_std_weight_velocity = float( cfg.get_value( "max_std_weight_velocity" ) )
        self._match_gate_admit_percent = float( cfg.get_value( "match_gate_admit_percent" ) )
        self._min_match_gate = float( cfg.get_value( "min_match_gate" ) )
        self._max_match_gate = float( cfg.get_value( "max_match_gate" ) )

        # Create directories
        if self._train_directory:
            if not os.path.exists( self._train_directory ):
                os.makedirs( self._train_directory )

        if self._output_directory:
            if not os.path.exists( self._output_directory ):
                os.makedirs( self._output_directory )

        return True

    def check_configuration( self, cfg ):
        if not cfg.has_value( "identifier" ) or \
          len( cfg.get_value( "identifier") ) == 0:
            print( "A model identifier must be specified!" )
            return False
        return True

    def add_data_from_disk( self, categories, train_files, train_tracks,
                            test_files, test_tracks ):
        """
        Store track data for parameter estimation.
        """
        print( "Adding training data from disk..." )
        print( "  Training files: ", len( train_files ) )
        print( "  Training tracks: ", len( train_tracks ) )
        print( "  Test files: ", len( test_files ) )
        print( "  Test tracks: ", len( test_tracks ) )

        if categories is not None:
            self._categories = categories.all_class_names()
        else:
            self._categories = []

        self._train_image_files = list( train_files )
        self._train_tracks = list( train_tracks )
        self._test_image_files = list( test_files )
        self._test_tracks = list( test_tracks )

    def _extract_track_statistics( self ):
        """
        Extract statistics from track groundtruth for parameter estimation.

        Returns a dict with:
        - positions: list of (x, y, w, h) per detection
        - velocities: list of (vx, vy) between consecutive frames
        - confidences: list of detection confidences
        - track_lengths: list of track lengths
        - gap_lengths: list of gap lengths (frames without detection in tracks)
        """
        positions = []
        velocities = []
        confidences = []
        track_lengths = []
        gap_lengths = []
        link_ious = []

        def _iou( a, b ):
            x0 = max( a[0], b[0] )
            y0 = max( a[1], b[1] )
            x1 = min( a[2], b[2] )
            y1 = min( a[3], b[3] )
            if x1 <= x0 or y1 <= y0:
                return 0.0
            inter = ( x1 - x0 ) * ( y1 - y0 )
            area_a = ( a[2] - a[0] ) * ( a[3] - a[1] )
            area_b = ( b[2] - b[0] ) * ( b[3] - b[1] )
            return inter / ( area_a + area_b - inter )

        all_tracks = self._train_tracks + self._test_tracks

        for track_set in all_tracks:
            if track_set is None:
                continue

            for track in track_set.tracks():
                states = list( track )
                track_lengths.append( len( states ) )

                prev_frame = None
                prev_cx, prev_cy = None, None
                prev_box = None

                for state in states:
                    frame_id = state.frame_id
                    det = state.detection()

                    if det is None:
                        continue

                    bbox = det.bounding_box
                    x1 = bbox.min_x()
                    y1 = bbox.min_y()
                    x2 = bbox.max_x()
                    y2 = bbox.max_y()
                    w = x2 - x1
                    h = y2 - y1
                    cx = x1 + w / 2
                    cy = y1 + h / 2

                    positions.append( ( cx, cy, w, h ) )

                    if det.confidence is not None:
                        confidences.append( det.confidence )

                    # Compute velocity if we have previous position
                    if prev_frame is not None and prev_cx is not None:
                        dt = frame_id - prev_frame
                        if dt > 0:
                            vx = ( cx - prev_cx ) / dt
                            vy = ( cy - prev_cy ) / dt
                            velocities.append( ( vx, vy, h, dt ) )

                            # The IoU an association gate must admit for this
                            # track to survive one frame to the next. Only
                            # consecutive frames count: across a gap the
                            # filter has predicted forward and the overlap is
                            # not the quantity the gate sees.
                            if dt == 1 and prev_box is not None:
                                link_ious.append(
                                    _iou( prev_box, ( x1, y1, x2, y2 ) ) )

                            # Track gaps (missing frames)
                            if dt > 1:
                                gap_lengths.append( dt - 1 )

                    prev_frame = frame_id
                    prev_cx, prev_cy = cx, cy
                    prev_box = ( x1, y1, x2, y2 )

        return {
            'positions': positions,
            'velocities': velocities,
            'confidences': confidences,
            'track_lengths': track_lengths,
            'gap_lengths': gap_lengths,
            'link_ious': link_ious
        }

    def _estimate_kalman_parameters( self, stats ):
        """
        Estimate Kalman filter parameters from track statistics.

        The ByteTrack Kalman filter uses:
        - _std_weight_position: Controls position uncertainty, scaled by bbox height
        - _std_weight_velocity: Controls velocity uncertainty, scaled by bbox height

        These are estimated by analyzing the variance of position/velocity
        relative to bbox height in the training data.
        """
        positions = stats['positions']
        velocities = stats['velocities']

        if len( positions ) < 10:
            print( "Warning: Not enough position data, using defaults" )
            return 1.0 / 20, 1.0 / 160

        # Estimate position variance relative to height
        # For each detection, compute how much position varies from expected
        # This is approximated by looking at position changes within tracks
        pos_variances = []
        for vx, vy, h, dt in velocities:
            # Normalized position change per frame
            if h > 0 and dt == 1:  # Only use consecutive frames
                pos_var = np.sqrt( vx**2 + vy**2 ) / h
                pos_variances.append( pos_var )

        if len( pos_variances ) > 0:
            # Use median to be robust to outliers
            median_pos_var = np.median( pos_variances )
            # The std_weight_position is roughly the expected position std / height
            std_weight_position = np.clip(
                median_pos_var * 2,  # Factor of 2 for safety margin
                self._min_std_weight_position,
                self._max_std_weight_position
            )
        else:
            std_weight_position = 1.0 / 20

        # Estimate velocity variance
        # Compute acceleration (velocity changes) to estimate velocity uncertainty
        vel_variances = []
        prev_vx, prev_vy, prev_h = None, None, None
        for vx, vy, h, dt in velocities:
            if prev_vx is not None and dt == 1 and h > 0:
                # Acceleration (change in velocity)
                ax = vx - prev_vx
                ay = vy - prev_vy
                vel_var = np.sqrt( ax**2 + ay**2 ) / h
                vel_variances.append( vel_var )
            prev_vx, prev_vy, prev_h = vx, vy, h

        if len( vel_variances ) > 0:
            median_vel_var = np.median( vel_variances )
            std_weight_velocity = np.clip(
                median_vel_var * 2,
                self._min_std_weight_velocity,
                self._max_std_weight_velocity
            )
        else:
            std_weight_velocity = 1.0 / 160

        return std_weight_position, std_weight_velocity

    def _detector_stats( self ):
        """Measure the detector against the groundtruth, when one was given."""
        if not self._computed_detections:
            return None

        stats = detector_statistics(
            self._train_tracks + self._test_tracks,
            self._train_image_files + self._test_image_files,
            self._computed_detections )

        matched = len( stats[ 'matched_confidences' ] )
        unmatched = len( stats[ 'unmatched_confidences' ] )

        print( "Computed detections: {} matched a groundtruth box, {} did "
               "not, over {} of {} annotated frames".format(
                   matched, unmatched, stats[ 'frames_with_computed' ],
                   stats[ 'frames_total' ] ) )

        if stats[ 'frames_total' ] and \
                stats[ 'frames_with_computed' ] < 0.1 * stats[ 'frames_total' ]:
            print( "WARNING: almost no annotated frame has a computed "
                   "detection. Frame ids here are positions within a clip, so "
                   "the detections must come from a run over the same "
                   "extracted frames rather than over the source video. "
                   "Falling back to the groundtruth." )
            return None

        if matched < 10 or unmatched < 10:
            print( "Too few matched or unmatched detections to separate the "
                   "two, falling back to the groundtruth." )
            return None

        return stats

    def _estimate_thresholds_from_detector( self, stats ):
        """Thresholds that separate the detector's hits from its misfires.

        The groundtruth cannot answer this: every annotation is confidence
        1.0, so a percentile of it is a percentile of a constant and the
        result is whatever the clamp allows. What a threshold has to do is
        divide the scores of detections that found a real object from the
        scores of those that did not, and that needs the detector's own
        output.
        """
        matched = np.array( stats[ 'matched_confidences' ] )
        unmatched = np.array( stats[ 'unmatched_confidences' ] )

        # Keep most real detections in the high tier
        high_thresh = float( np.percentile( matched, 25 ) )

        # The low tier is ByteTrack's second chance: it should reach well
        # below the high tier without swallowing the bulk of the misfires
        low_thresh = float( max( np.percentile( matched, 2 ),
                                 np.percentile( unmatched, 60 ) ) )

        # Starting a track off a misfire costs more than missing one, so this
        # sits above the high tier
        new_track_thresh = float( np.percentile( matched, 40 ) )

        high_thresh = float( np.clip( high_thresh, 0.05, 0.95 ) )
        low_thresh = float( np.clip( low_thresh, 0.01, high_thresh - 0.05 ) )
        new_track_thresh = float( np.clip( new_track_thresh,
                                           high_thresh, 0.95 ) )

        print( "  thresholds from the detector: high {:.3f} low {:.3f} "
               "new_track {:.3f}".format( high_thresh, low_thresh,
                                          new_track_thresh ) )

        return high_thresh, low_thresh, new_track_thresh

    def _measurement_noise_from_detector( self, stats ):
        """Localisation error of the detector, for the Kalman update.

        Estimated from how far a detection sits from the box it matched,
        which is the quantity the filter's measurement noise describes. The
        groundtruth cannot supply it: measured against itself its error is
        zero.
        """
        errors = stats[ 'center_errors' ]

        if len( errors ) < 10:
            return None

        return float( np.median( errors ) )

    def _estimate_thresholds( self, stats ):
        """
        Estimate detection confidence thresholds from training data.

        Returns:
        - high_thresh: Threshold for high-confidence detections
        - low_thresh: Threshold for low-confidence detections
        - new_track_thresh: Minimum confidence to create new track
        """
        confidences = stats['confidences']

        if len( confidences ) < 10:
            print( "Warning: Not enough confidence data, using defaults" )
            return 0.6, 0.1, 0.6

        confidences = np.array( confidences )

        # high_thresh: Use 70th percentile (want to capture most good detections)
        high_thresh = np.percentile( confidences, 30 )

        # low_thresh: Use 10th percentile (want to capture almost all detections)
        low_thresh = np.percentile( confidences, 10 )

        # new_track_thresh: Same as high_thresh for creating new tracks
        new_track_thresh = high_thresh

        # Clamp to reasonable ranges
        high_thresh = np.clip( high_thresh, 0.3, 0.9 )
        low_thresh = np.clip( low_thresh, 0.05, high_thresh - 0.1 )
        new_track_thresh = np.clip( new_track_thresh, 0.3, 0.9 )

        return float( high_thresh ), float( low_thresh ), float( new_track_thresh )

    def _estimate_match_threshold( self, stats ):
        """The IoU association gate, fit from the groundtruth.

        The gate has one job: admit the overlap between where a track's
        animal was last frame and where it is now, while excluding overlap
        with other animals. The groundtruth gives the first distribution
        directly -- the consecutive-frame IoU of every annotated track --
        so the gate is set at the quantile that admits
        match_gate_admit_percent of those true links.

        On FishTrack23 the two distributions barely overlap (the 99th
        percentile of cross-animal IoU sits near the 5th percentile of
        same-animal IoU), and assignment still prefers the best match among
        admitted candidates, so erring loose costs little. Fast targets --
        an animal moving more than its own length per frame -- produce true
        links with near-zero IoU, which is why the floor matters and why a
        sweep on FishTrack23 favoured gates as low as 0.02.

        match_thresh is a distance bound in the tracker
        (linear_assignment on 1 - IoU), so the returned value is 1 - gate:
        LOOSENING the gate RAISES match_thresh.
        """
        link_ious = stats.get( 'link_ious', [] )

        if len( link_ious ) < 100:
            print( "Warning: too few consecutive-frame links "
                   "({}), keeping match_thresh default".format(
                       len( link_ious ) ) )
            return 0.8

        gate = float( np.percentile(
            np.array( link_ious ),
            100.0 - self._match_gate_admit_percent ) )
        gate = float( np.clip(
            gate, self._min_match_gate, self._max_match_gate ) )

        admitted = float( np.mean( np.array( link_ious ) >= gate ) )
        print( "  match gate IoU>={:.3f} admits {:.1f}% of {} true "
               "links -> match_thresh {:.3f}".format(
                   gate, 100 * admitted, len( link_ious ), 1.0 - gate ) )

        return round( 1.0 - gate, 3 )

    def _estimate_track_buffer( self, stats ):
        """
        Estimate track_buffer (frames to keep lost tracks) from gap statistics.
        """
        gap_lengths = stats['gap_lengths']

        if len( gap_lengths ) < 5:
            return 30  # Default

        # Use 90th percentile of gaps + some margin
        gap_90 = np.percentile( gap_lengths, 90 )
        track_buffer = int( gap_90 * 1.5 ) + 5

        # Clamp to reasonable range
        track_buffer = max( 10, min( 100, track_buffer ) )

        return track_buffer

    def update_model( self ):
        """
        Analyze track groundtruth and estimate ByteTrack parameters.

        Returns:
            dict: Map of template replacements and file copies
        """
        print( "Starting ByteTrack parameter estimation..." )

        # Extract statistics from tracks
        print( "Extracting track statistics..." )
        stats = self._extract_track_statistics()

        print( f"  Found {len(stats['positions'])} detections" )
        print( f"  Found {len(stats['velocities'])} velocity measurements" )
        print( f"  Found {len(stats['track_lengths'])} tracks" )
        print( f"  Found {len(stats['gap_lengths'])} gaps" )

        # Estimate Kalman filter parameters
        print( "Estimating Kalman filter parameters..." )
        std_weight_position, std_weight_velocity = self._estimate_kalman_parameters( stats )
        print( f"  std_weight_position: {std_weight_position:.6f}" )
        print( f"  std_weight_velocity: {std_weight_velocity:.6f}" )

        # Estimate thresholds
        print( "Estimating detection thresholds..." )
        high_thresh, low_thresh, new_track_thresh = self._estimate_thresholds( stats )

        # A detector's own output, when supplied, answers what the
        # groundtruth cannot: where its boxes land relative to the truth, and
        # which of its scores correspond to real objects
        detector = self._detector_stats()

        if detector is not None:
            high_thresh, low_thresh, new_track_thresh = \
                self._estimate_thresholds_from_detector( detector )

            measured = self._measurement_noise_from_detector( detector )

            if measured is not None:
                std_weight_position = float( np.clip(
                    measured,
                    self._min_std_weight_position,
                    self._max_std_weight_position ) )
                print( "  measurement noise from the detector: "
                       "std_weight_position {:.4f}".format(
                           std_weight_position ) )

        print( f"  high_thresh: {high_thresh:.3f}" )
        print( f"  low_thresh: {low_thresh:.3f}" )
        print( f"  new_track_thresh: {new_track_thresh:.3f}" )

        # Estimate track buffer
        print( "Estimating track buffer..." )
        track_buffer = self._estimate_track_buffer( stats )
        print( f"  track_buffer: {track_buffer}" )

        # IOU association gate, from the groundtruth's own link overlaps
        print( "Estimating match threshold..." )
        match_thresh = self._estimate_match_threshold( stats )

        # Save parameters to JSON file in train directory (will be copied by caller)
        params = {
            'std_weight_position': std_weight_position,
            'std_weight_velocity': std_weight_velocity,
            'high_thresh': high_thresh,
            'low_thresh': low_thresh,
            'match_thresh': match_thresh,
            'new_track_thresh': new_track_thresh,
            'track_buffer': track_buffer
        }

        params_file = os.path.join( self._train_directory, "bytetrack_params.json" )
        with open( params_file, 'w' ) as f:
            json.dump( params, f, indent=2 )
        print( f"Saved parameters to {params_file}" )

        # Build output map
        output = self._get_output_map( params, params_file )

        print( "\nByteTrack parameter estimation complete!\n" )

        return output

    def _get_output_map( self, params, params_file ):
        """
        Build output map with template replacements and file copies.

        Returns:
            dict: Map where file paths are file copies, other values are template replacements
        """
        output = {}
        algo = "bytetrack"

        output["type"] = algo

        # Config keys matching bytetrack inference config
        output[algo + ":high_thresh"] = f"{params['high_thresh']:.3f}"
        output[algo + ":low_thresh"] = f"{params['low_thresh']:.3f}"
        output[algo + ":match_thresh"] = f"{params['match_thresh']:.3f}"
        output[algo + ":track_buffer"] = str( params['track_buffer'] )
        output[algo + ":new_track_thresh"] = f"{params['new_track_thresh']:.3f}"

        # File copies
        output["bytetrack_params.json"] = params_file

        return output


def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    implementation_name = "bytetrack"

    if algorithm_factory.has_algorithm_impl_name(
        ByteTrackTrainer.static_type_name(), implementation_name ):
        return

    algorithm_factory.add_algorithm(
        implementation_name,
        "ByteTrack parameter estimation from track groundtruth",
        ByteTrackTrainer
    )

    algorithm_factory.mark_algorithm_as_loaded( implementation_name )
