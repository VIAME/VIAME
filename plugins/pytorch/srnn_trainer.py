# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
SRNN (Siamese RNN) tracker training implementation.

This trainer wraps the existing SRNN training pipeline which consists of:
1. Data preparation - Convert tracks to KW18 format and generate training data
2. Siamese model training - Train appearance feature extractor
3. Feature extraction - Extract features using trained Siamese
4. Individual LSTM training - Train A/I/M/B LSTMs
5. Combined SRNN training - Train final TargetLSTM model

The existing training scripts in srnn/ folder are orchestrated by this trainer.
"""

from kwiver.vital.algo import TrainTracker

from kwiver.vital.types import (
    CategoryHierarchy,
    ObjectTrackSet, ObjectTrackState,
    BoundingBoxD, DetectedObjectType
)

from distutils.util import strtobool
from pathlib import Path

import os
import sys
import shutil
import subprocess
import signal
import time
import threading
from viame.pytorch.utilities import report_cuda_errors
from viame.core.training_data import (build_sequence_maps,
    read_sequence_manifest,
    load_computed_detections, match_to_groundtruth)
from viame.pytorch.srnn.generate_training_files import BoundingBox


def _frame_bounds( track_sets ):
    """Highest frame id each track set refers to, or None where it refers to
    none. build_sequence_maps checks its alignment against these, since the
    number of track sets and the number of image directories need not agree.
    """
    bounds = []

    for track_set in track_sets:
        highest = None

        if track_set is not None:
            for track in track_set.tracks():
                for state in track:
                    if state.detection() is None:
                        continue
                    if highest is None or state.frame_id > highest:
                        highest = state.frame_id

        bounds.append( highest )

    return bounds


class SRNNTrainer( TrainTracker ):
    """
    Implementation of TrainTracker class for SRNN tracker training.

    Wraps the existing SRNN training pipeline to train Siamese CNN
    and LSTM components for multi-object tracking.
    """
    def __init__( self ):
        TrainTracker.__init__( self )

        self._identifier = "viame-srnn-tracker"

        # Directory of detector output for the same clips, one VIAME
        # CSV per clip. Optional: left empty everything comes from the
        # groundtruth exactly as before.
        self._computed_detections = ""

        # Written by the training tool: which frames of the flat list
        # belong to which track set. Empty falls back to inferring it
        # from the directory layout, which this dataset defeats.
        self._sequence_manifest = ""
        self._train_directory = "deep_training"
        self._gpu_count = -1
        self._threshold = "0.00"
        self._timeout = "604800"  # 1 week default

        # SRNN-specific parameters
        self._stabilized = False
        self._grid_num = 15
        self._siamese_img_sample_rate = 8
        self._siamese_pos_sample_rate = 10
        self._rnn_component = "AIM"  # Which LSTM components to use
        self._resume = False
        # How many of the eight component LSTMs to train at once, and how many
        # data loading workers each gets. Kept low by default: on Python 3.14
        # loader workers come from a forkserver, so several trainings at once
        # with many workers each exhausted a two device node.
        self._lstm_concurrency = 1
        self._lstm_loader_workers = 2

        self._categories = []
        self._train_image_files = []
        self._train_tracks = []
        self._test_image_files = []
        self._test_tracks = []

    def get_configuration( self ):
        cfg = super( TrainTracker, self ).get_configuration()

        cfg.set_value( "identifier", self._identifier )
        cfg.set_value( "computed_detections", self._computed_detections )
        cfg.set_value( "sequence_manifest", self._sequence_manifest )
        cfg.set_value( "train_directory", self._train_directory )
        cfg.set_value( "gpu_count", str( self._gpu_count ) )
        cfg.set_value( "threshold", self._threshold )
        cfg.set_value( "timeout", self._timeout )
        cfg.set_value( "stabilized", str( self._stabilized ) )
        cfg.set_value( "grid_num", str( self._grid_num ) )
        cfg.set_value( "siamese_img_sample_rate", str( self._siamese_img_sample_rate ) )
        cfg.set_value( "siamese_pos_sample_rate", str( self._siamese_pos_sample_rate ) )
        cfg.set_value( "rnn_component", self._rnn_component )
        cfg.set_value( "resume", str( self._resume ) )
        cfg.set_value( "lstm_concurrency", str( self._lstm_concurrency ) )
        cfg.set_value( "lstm_loader_workers", str( self._lstm_loader_workers ) )

        return cfg

    @report_cuda_errors("SRNNTrainer initialization")
    def set_configuration( self, cfg_in ):
        cfg = self.get_configuration()
        cfg.merge_config( cfg_in )

        self._identifier = str( cfg.get_value( "identifier" ) )
        self._computed_detections = str( cfg.get_value( "computed_detections" ) )
        self._sequence_manifest = str( cfg.get_value( "sequence_manifest" ) )
        self._train_directory = str( cfg.get_value( "train_directory" ) )
        self._gpu_count = int( cfg.get_value( "gpu_count" ) )
        self._threshold = str( cfg.get_value( "threshold" ) )
        self._timeout = str( cfg.get_value( "timeout" ) )
        self._stabilized = strtobool( cfg.get_value( "stabilized" ) )
        self._grid_num = int( cfg.get_value( "grid_num" ) )
        self._siamese_img_sample_rate = int( cfg.get_value( "siamese_img_sample_rate" ) )
        self._siamese_pos_sample_rate = int( cfg.get_value( "siamese_pos_sample_rate" ) )
        self._rnn_component = str( cfg.get_value( "rnn_component" ) )
        self._resume = strtobool( cfg.get_value( "resume" ) )
        self._lstm_concurrency = int( cfg.get_value( "lstm_concurrency" ) )
        self._lstm_loader_workers = int( cfg.get_value( "lstm_loader_workers" ) )

        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                if self._gpu_count < 0:
                    self._gpu_count = torch.cuda.device_count()
        except ImportError:
            print( "PyTorch not available, defaulting to 1 GPU" )
            if self._gpu_count < 0:
                self._gpu_count = 1

        # Create directories
        if self._train_directory:
            if not os.path.exists( self._train_directory ):
                os.makedirs( self._train_directory )

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
        Store track data for later processing during update_model.

        The track data will be converted to KW18 format for SRNN training.
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

    def _prepare_training_data( self ):
        """
        Convert track data to KW18 format expected by SRNN training pipeline.

        Creates directory structure:
        - data_root/train/sequence_XXX/img1/  (images)
        - data_root/test/sequence_XXX/img1/
        """
        data_root = Path( self._train_directory ) / "srnn_data"
        if data_root.exists():
            shutil.rmtree( data_root )

        train_dir = data_root / "train"
        test_dir = data_root / "test"
        train_dir.mkdir( parents=True )
        test_dir.mkdir( parents=True )

        print( "Preparing training data for SRNN..." )

        train_tracks = self._train_tracks
        test_tracks = self._test_tracks
        test_image_files = self._test_image_files

        # Tracker training only populates a validation set when one is given
        # explicitly (validation.txt or --validation-dir), so test_tracks is
        # normally empty. The Siamese stage then finds no test images at all and
        # aborts with "Found 0 images in data", which takes the whole SRNN run
        # down. Hold a slice of the training sequences out instead, so the split
        # is always non-empty.
        if not test_tracks and len( train_tracks ) > 1:
            holdout = max( 1, int( len( train_tracks ) * 0.1 ) )

            test_tracks = train_tracks[ -holdout: ]
            train_tracks = train_tracks[ :-holdout ]

            # The held out tracks still index into the training image list, so
            # the test split has to be given that same list. Handing it the
            # empty _test_image_files would write groundtruth with no imagery
            # beside it, which fails the same way as having no test set at all.
            test_image_files = self._train_image_files

            print( f"  No validation set supplied, holding out {holdout} of "
                   f"{len( self._train_tracks )} sequences for test" )

        # Process training data
        tracks = {}

        tracks[ "train" ] = self._prepare_split_data(
            train_tracks, self._train_image_files, train_dir, "train"
        )

        # Process test data
        tracks[ "test" ] = self._prepare_split_data(
            test_tracks, test_image_files, test_dir, "test"
        )

        return data_root, tracks

    def _load_computed_by_sequence( self, names, track_sets ):
        """Detector output per sequence, keyed by track set index."""
        if not self._computed_detections:
            return None

        if not names or all( n is None for n in names ):
            print( "WARNING: computed_detections was given but the images "
                   "could not be split per sequence, so there is no clip name "
                   "to look a detection file up by. Using the groundtruth." )
            return None

        loaded = {}

        for seq_idx, name in enumerate( names ):
            if name is None or seq_idx >= len( track_sets ):
                continue

            detections = load_computed_detections( self._computed_detections,
                                                   name )

            if detections:
                loaded[ seq_idx ] = detections

        print( f"  computed detections found for {len(loaded)} of "
               f"{len(track_sets)} sequences" )

        return loaded or None

    @staticmethod
    def _substitute_computed( frame_annotations, computed, counters,
                              iou_threshold=0.5 ):
        """Swap groundtruth boxes for the detector boxes that matched them."""
        replaced = {}

        for frame_id, truth in frame_annotations.items():
            frame_computed = computed.get( frame_id, [] )

            if not frame_computed:
                continue

            matches, unmatched, missed = match_to_groundtruth(
                frame_computed,
                [ ( t[ 1 ], t[ 2 ], t[ 3 ], t[ 4 ], t[ 0 ] ) for t in truth ],
                iou_threshold )

            counters[ 'matched' ] += len( matches )
            counters[ 'false_positives' ] += len( unmatched )
            counters[ 'missed' ] += len( missed )

            rows = []

            for c, t, _overlap in matches:
                x1, y1, x2, y2 = ( int( c[ 0 ] ), int( c[ 1 ] ),
                                   int( c[ 2 ] ), int( c[ 3 ] ) )

                if x2 <= x1 or y2 <= y1:
                    continue

                rows.append( ( t[ 4 ], x1, y1, x2, y2 ) )

            if rows:
                replaced[ frame_id ] = rows

        return replaced

    def _prepare_split_data( self, track_sets, image_files, output_dir, split_name ):
        """
        Prepare data for one split (train or test).

        Each track_set represents a sequence. We create:
        - sequence_XXX/img1/ with symlinks to images
        """
        print( f"  Processing {split_name} split: {len(track_sets)} track sets" )

        sequence_tracks = {}

        # One image map per sequence. A frame id is a position within its own
        # sequence, so resolving it against the flat list of every sequence's
        # images only ever worked for the first one.
        image_maps, names = read_sequence_manifest(
            self._sequence_manifest, image_files, len( track_sets ) )

        if image_maps is None:
            image_maps, names = build_sequence_maps(
                image_files, len( track_sets ), split_name,
                _frame_bounds( track_sets ) )

        computed_by_sequence = self._load_computed_by_sequence(
            names, track_sets )
        counters = { 'matched': 0, 'false_positives': 0, 'missed': 0 }

        for seq_idx, track_set in enumerate( track_sets ):
            if track_set is None:
                continue

            image_map = image_maps[ seq_idx ]

            seq_name = f"sequence_{seq_idx:04d}"
            seq_dir = output_dir / seq_name
            img_dir = seq_dir / "img1"

            # Collect all frames and annotations for this sequence
            frame_annotations = {}  # frame_id -> [(track_id, x1, y1, x2, y2)]
            all_frame_ids = set()

            for track in track_set.tracks():
                track_id = track.id

                for state in track:
                    frame_id = state.frame_id
                    det = state.detection()

                    if det is None:
                        continue

                    all_frame_ids.add( frame_id )

                    bbox = det.bounding_box
                    x1 = int( bbox.min_x() )
                    y1 = int( bbox.min_y() )
                    x2 = int( bbox.max_x() )
                    y2 = int( bbox.max_y() )

                    # Zero area boxes carry no information and are rejected by
                    # generate_training_files with "Width and height must
                    # be positive", which aborts the whole run over a single
                    # bad annotation
                    if x2 <= x1 or y2 <= y1:
                        continue

                    if frame_id not in frame_annotations:
                        frame_annotations[ frame_id ] = []
                    frame_annotations[ frame_id ].append(
                        ( track_id, x1, y1, x2, y2 )
                    )

            # Where a detector's own output is supplied, learn from its boxes
            # rather than the groundtruth's. It matters more here than for a
            # re-ID crop: the motion LSTM trained on groundtruth sees
            # trajectories that are smooth by construction, and at inference
            # it is handed the detector's jitter.
            if computed_by_sequence and seq_idx in computed_by_sequence:
                frame_annotations = self._substitute_computed(
                    frame_annotations, computed_by_sequence[ seq_idx ],
                    counters )
                all_frame_ids = set( frame_annotations )

            # A clip with no annotations gets no sequence directory at all.
            # The directory used to be created before this point, leaving a
            # sequence_NNNN/img1 holding frames that nothing described, and
            # feature generation walks every sequence directory it finds.
            if not frame_annotations:
                print( f"    {seq_name}: no annotations, skipping" )
                continue

            img_dir.mkdir( parents=True )

            # Symlink the frames this sequence annotates and build its track
            # states in the same order. Feature generation zips its sorted
            # image list against this list positionally, so both are built
            # together here. Indexing by frame id instead, as this once did,
            # silently misaligns any clip not starting at frame zero.
            frame_states = []

            for frame_id in sorted( all_frame_ids ):
                dst_path = img_dir / f"frame_{frame_id:06d}.jpg"
                if frame_id in image_map and os.path.exists( image_map[ frame_id ] ):
                    src_path = Path( image_map[ frame_id ] ).resolve()
                    dst_path.symlink_to( src_path )

                frame_states.append( [
                    ( track_id, BoundingBox.from_corners( x1, y1, x2, y2 ) )
                    for track_id, x1, y1, x2, y2
                    in frame_annotations.get( frame_id, [] )
                ] )

            sequence_tracks[ seq_name ] = frame_states

            print( f"    {seq_name}: {len(frame_annotations)} frames, "
                   f"{len(set(t for anns in frame_annotations.values() for t,_,_,_,_ in anns))} tracks" )

        return sequence_tracks

    @report_cuda_errors("SRNNTrainer training")
    def update_model( self ):
        """
        Run the SRNN training pipeline.
        """
        print( "Starting SRNN training..." )

        # Lay out the image sequences and collect their track states
        data_root, tracks = self._prepare_training_data()

        # Output directory for SRNN training
        srnn_output = Path( self._train_directory ) / "srnn_output"

        # Never delete previous results silently. The generated training data
        # and extracted features under here are the bulk of a run's wall clock
        # and are what a resume exists to reuse, so a run started without
        # resume set stops rather than destroying them. Move or remove the
        # directory by hand to start over.
        if srnn_output.exists() and not self._resume:
            existing = [ p for p in srnn_output.rglob( "*" ) if p.is_file() ]

            if existing:
                raise RuntimeError(
                    "{} already holds {} files from an earlier run. Set "
                    "resume to carry on from them, or move the directory "
                    "aside to start fresh; it will not be deleted "
                    "automatically.".format( srnn_output, len( existing ) ) )

            shutil.rmtree( srnn_output )

        # Handle interrupt signals
        if threading.current_thread().__class__.__name__ == '_MainThread':
            signal.signal( signal.SIGINT, lambda sig, frame: self._interrupt_handler() )
            signal.signal( signal.SIGTERM, lambda sig, frame: self._interrupt_handler() )

        # The pipeline is driven in process so the track states can be passed
        # as objects. Only the first stage needs them; the model training
        # stages it runs still shell out, as they read the files that stage
        # produces rather than any groundtruth.
        from viame.pytorch.srnn.train_everything import main as run_srnn_pipeline

        print( "Running SRNN pipeline over " + str( data_root ) )

        run_srnn_pipeline(
            data_root=Path( data_root ),
            output_dir=srnn_output,
            stabilized=bool( self._stabilized ),
            tracks=tracks,
            resume=bool( self._resume ),
            lstm_concurrency=int( self._lstm_concurrency ),
            lstm_loader_workers=int( self._lstm_loader_workers ),
        )

        output = self._get_output_map( srnn_output )

        print( "\nSRNN training complete!" )

        return output

    def _interrupt_handler( self ):
        self.proc.send_signal( signal.SIGINT )
        timeout = 0
        while self.proc.poll() is None:
            time.sleep( 0.1 )
            timeout += 0.1
            if timeout > 5:
                self.proc.kill()
                break
        sys.exit( 0 )

    def _get_output_map( self, srnn_output ):
        """Build output map for process_trainer_output."""
        output = {}

        srnn_output = Path( srnn_output )

        # Find best models
        siamese_model = srnn_output / "siamese" / "best_model.pt"
        target_lstm_F = srnn_output / "target_lstm" / "best_F_model.pt"
        target_lstm_V = srnn_output / "target_lstm" / "best_V_model.pt"

        found_any = False

        algo = "srnn"
        output["type"] = algo

        if siamese_model.exists():
            output[algo + ":siamese_model"] = "siamese_model.pt"
            output["siamese_model.pt"] = str( siamese_model )
            found_any = True
            print( f"Found Siamese model: {siamese_model}" )

        if target_lstm_F.exists():
            output[algo + ":target_lstm_fixed"] = "target_lstm_F.pt"
            output["target_lstm_F.pt"] = str( target_lstm_F )
            found_any = True
            print( f"Found Target LSTM (fixed) model: {target_lstm_F}" )

        if target_lstm_V.exists():
            output[algo + ":target_lstm_variable"] = "target_lstm_V.pt"
            output["target_lstm_V.pt"] = str( target_lstm_V )
            found_any = True
            print( f"Found Target LSTM (variable) model: {target_lstm_V}" )

        if not found_any:
            print( "\nNo trained models found, training may have failed" )
            return output

        print( f"\nThe {self._train_directory} directory can now be deleted, "
               "unless you want to review training metrics first." )

        return output


def __vital_algorithm_register__():
    from viame.core.vital_registration import register_vital_algorithm

    register_vital_algorithm(
        SRNNTrainer,
        "srnn",
        "PyTorch SRNN tracker training routine",
    )
