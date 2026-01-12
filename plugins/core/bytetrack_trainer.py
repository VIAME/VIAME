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

        # Output parameter bounds (for clamping estimated values)
        self._min_std_weight_position = 0.01
        self._max_std_weight_position = 0.5
        self._min_std_weight_velocity = 0.001
        self._max_std_weight_velocity = 0.1

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
        cfg.set_value( "min_std_weight_position", str( self._min_std_weight_position ) )
        cfg.set_value( "max_std_weight_position", str( self._max_std_weight_position ) )
        cfg.set_value( "min_std_weight_velocity", str( self._min_std_weight_velocity ) )
        cfg.set_value( "max_std_weight_velocity", str( self._max_std_weight_velocity ) )

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
        self._min_std_weight_position = float( cfg.get_value( "min_std_weight_position" ) )
        self._max_std_weight_position = float( cfg.get_value( "max_std_weight_position" ) )
        self._min_std_weight_velocity = float( cfg.get_value( "min_std_weight_velocity" ) )
        self._max_std_weight_velocity = float( cfg.get_value( "max_std_weight_velocity" ) )

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

        all_tracks = self._train_tracks + self._test_tracks

        for track_set in all_tracks:
            if track_set is None:
                continue

            for track in track_set.tracks():
                states = list( track )
                track_lengths.append( len( states ) )

                prev_frame = None
                prev_cx, prev_cy = None, None

                for state in states:
                    frame_id = state.frame()
                    det = state.detection()

                    if det is None:
                        continue

                    bbox = det.bounding_box()
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

                            # Track gaps (missing frames)
                            if dt > 1:
                                gap_lengths.append( dt - 1 )

                    prev_frame = frame_id
                    prev_cx, prev_cy = cx, cy

        return {
            'positions': positions,
            'velocities': velocities,
            'confidences': confidences,
            'track_lengths': track_lengths,
            'gap_lengths': gap_lengths
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
        print( f"  high_thresh: {high_thresh:.3f}" )
        print( f"  low_thresh: {low_thresh:.3f}" )
        print( f"  new_track_thresh: {new_track_thresh:.3f}" )

        # Estimate track buffer
        print( "Estimating track buffer..." )
        track_buffer = self._estimate_track_buffer( stats )
        print( f"  track_buffer: {track_buffer}" )

        # IOU match threshold (use default, hard to estimate from GT)
        match_thresh = 0.8

        # Save parameters to JSON file
        params = {
            'std_weight_position': std_weight_position,
            'std_weight_velocity': std_weight_velocity,
            'high_thresh': high_thresh,
            'low_thresh': low_thresh,
            'match_thresh': match_thresh,
            'new_track_thresh': new_track_thresh,
            'track_buffer': track_buffer
        }

        params_file = os.path.join( self._output_directory, "bytetrack_params.json" )
        with open( params_file, 'w' ) as f:
            json.dump( params, f, indent=2 )
        print( f"Saved parameters to {params_file}" )

        # Generate pipeline file from template if provided
        self._save_final_config( params )

        print( "\nByteTrack parameter estimation complete!\n" )

    def _save_final_config( self, params ):
        """
        Generate pipeline configuration file with estimated parameters.
        """
        if not self._pipeline_template:
            # Generate a default pipeline snippet
            config_content = f"""# ByteTrack tracker configuration (estimated from training data)
# Generated by bytetrack_trainer

process tracker
  :: bytetrack_tracker
  high_thresh = {params['high_thresh']:.3f}
  low_thresh = {params['low_thresh']:.3f}
  match_thresh = {params['match_thresh']:.3f}
  track_buffer = {params['track_buffer']}
  new_track_thresh = {params['new_track_thresh']:.3f}

# Kalman filter parameters (modify bytetrack_tracker.py to use these)
# std_weight_position = {params['std_weight_position']:.6f}
# std_weight_velocity = {params['std_weight_velocity']:.6f}
"""
            output_pipeline = os.path.join(
                self._output_directory, "bytetrack_tracker.pipe"
            )
            with open( output_pipeline, 'w' ) as f:
                f.write( config_content )
            print( f"Generated pipeline file: {output_pipeline}" )
            return

        # Use template if provided
        if os.path.exists( self._pipeline_template ):
            with open( self._pipeline_template, 'r' ) as fin:
                template_content = fin.read()

            # Replace placeholders
            pipeline_content = template_content
            pipeline_content = pipeline_content.replace(
                "[-HIGH-THRESH-]", f"{params['high_thresh']:.3f}"
            )
            pipeline_content = pipeline_content.replace(
                "[-LOW-THRESH-]", f"{params['low_thresh']:.3f}"
            )
            pipeline_content = pipeline_content.replace(
                "[-MATCH-THRESH-]", f"{params['match_thresh']:.3f}"
            )
            pipeline_content = pipeline_content.replace(
                "[-TRACK-BUFFER-]", str( params['track_buffer'] )
            )
            pipeline_content = pipeline_content.replace(
                "[-NEW-TRACK-THRESH-]", f"{params['new_track_thresh']:.3f}"
            )

            output_pipeline = os.path.join(
                self._output_directory, "tracker.pipe"
            )

            with open( output_pipeline, 'w' ) as fout:
                fout.write( pipeline_content )

            print( f"Generated pipeline file: {output_pipeline}" )


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
