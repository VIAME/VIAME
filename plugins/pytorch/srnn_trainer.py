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
from shutil import copyfile
from pathlib import Path

import os
import sys
import shutil
import subprocess
import signal
import time
import threading


class SRNNTrainer( TrainTracker ):
    """
    Implementation of TrainTracker class for SRNN tracker training.

    Wraps the existing SRNN training pipeline to train Siamese CNN
    and LSTM components for multi-object tracking.
    """
    def __init__( self ):
        TrainTracker.__init__( self )

        self._identifier = "viame-srnn-tracker"
        self._train_directory = "deep_training"
        self._output_directory = "category_models"
        self._output_prefix = "srnn_tracker"
        self._pipeline_template = ""
        self._gpu_count = -1
        self._threshold = "0.00"
        self._timeout = "604800"  # 1 week default

        # SRNN-specific parameters
        self._stabilized = False
        self._grid_num = 15
        self._siamese_img_sample_rate = 8
        self._siamese_pos_sample_rate = 10
        self._rnn_component = "AIM"  # Which LSTM components to use

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
        cfg.set_value( "gpu_count", str( self._gpu_count ) )
        cfg.set_value( "threshold", self._threshold )
        cfg.set_value( "timeout", self._timeout )
        cfg.set_value( "stabilized", str( self._stabilized ) )
        cfg.set_value( "grid_num", str( self._grid_num ) )
        cfg.set_value( "siamese_img_sample_rate", str( self._siamese_img_sample_rate ) )
        cfg.set_value( "siamese_pos_sample_rate", str( self._siamese_pos_sample_rate ) )
        cfg.set_value( "rnn_component", self._rnn_component )

        return cfg

    def set_configuration( self, cfg_in ):
        cfg = self.get_configuration()
        cfg.merge_config( cfg_in )

        self._identifier = str( cfg.get_value( "identifier" ) )
        self._train_directory = str( cfg.get_value( "train_directory" ) )
        self._output_directory = str( cfg.get_value( "output_directory" ) )
        self._output_prefix = str( cfg.get_value( "output_prefix" ) )
        self._pipeline_template = str( cfg.get_value( "pipeline_template" ) )
        self._gpu_count = int( cfg.get_value( "gpu_count" ) )
        self._threshold = str( cfg.get_value( "threshold" ) )
        self._timeout = str( cfg.get_value( "timeout" ) )
        self._stabilized = strtobool( cfg.get_value( "stabilized" ) )
        self._grid_num = int( cfg.get_value( "grid_num" ) )
        self._siamese_img_sample_rate = int( cfg.get_value( "siamese_img_sample_rate" ) )
        self._siamese_pos_sample_rate = int( cfg.get_value( "siamese_pos_sample_rate" ) )
        self._rnn_component = str( cfg.get_value( "rnn_component" ) )

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
        - data_root/train/sequence_XXX/gt.kw18 (groundtruth)
        - data_root/test/sequence_XXX/img1/
        - data_root/test/sequence_XXX/gt.kw18
        """
        data_root = Path( self._train_directory ) / "srnn_data"
        if data_root.exists():
            shutil.rmtree( data_root )

        train_dir = data_root / "train"
        test_dir = data_root / "test"
        train_dir.mkdir( parents=True )
        test_dir.mkdir( parents=True )

        print( "Preparing training data for SRNN..." )

        # Process training data
        self._prepare_split_data(
            self._train_tracks, self._train_image_files, train_dir, "train"
        )

        # Process test data
        self._prepare_split_data(
            self._test_tracks, self._test_image_files, test_dir, "test"
        )

        return data_root

    def _prepare_split_data( self, track_sets, image_files, output_dir, split_name ):
        """
        Prepare data for one split (train or test).

        Each track_set represents a sequence. We create:
        - sequence_XXX/img1/ with symlinks to images
        - sequence_XXX/gt.kw18 with track annotations
        """
        print( f"  Processing {split_name} split: {len(track_sets)} track sets" )

        # Build mapping from frame indices to image files if available
        image_map = {}
        for i, img_file in enumerate( image_files ):
            image_map[ i ] = img_file

        for seq_idx, track_set in enumerate( track_sets ):
            if track_set is None:
                continue

            seq_name = f"sequence_{seq_idx:04d}"
            seq_dir = output_dir / seq_name
            img_dir = seq_dir / "img1"
            img_dir.mkdir( parents=True )

            # Collect all frames and annotations for this sequence
            frame_annotations = {}  # frame_id -> [(track_id, x1, y1, x2, y2)]
            all_frame_ids = set()

            for track in track_set.tracks():
                track_id = track.id()

                for state in track:
                    frame_id = state.frame()
                    det = state.detection()

                    if det is None:
                        continue

                    all_frame_ids.add( frame_id )

                    bbox = det.bounding_box()
                    x1 = int( bbox.min_x() )
                    y1 = int( bbox.min_y() )
                    x2 = int( bbox.max_x() )
                    y2 = int( bbox.max_y() )

                    if frame_id not in frame_annotations:
                        frame_annotations[ frame_id ] = []
                    frame_annotations[ frame_id ].append(
                        ( track_id, x1, y1, x2, y2 )
                    )

            if not frame_annotations:
                continue

            # Create symlinks to images (or placeholder if not available)
            for frame_id in sorted( all_frame_ids ):
                dst_path = img_dir / f"frame_{frame_id:06d}.jpg"
                if frame_id in image_map and os.path.exists( image_map[ frame_id ] ):
                    src_path = Path( image_map[ frame_id ] ).resolve()
                    dst_path.symlink_to( src_path )
                # If no image available, training will still work with gt.kw18

            # Write gt.kw18 file
            # KW18 format: track_id frame obj_id len_frames x y w h world_x world_y ts conf
            # Simplified format used by SRNN: track_id ? frame_id ? ? ? ? ? ? x1 y1 x2 y2
            gt_file = seq_dir / "gt.kw18"
            with open( gt_file, 'w' ) as f:
                for frame_id in sorted( frame_annotations.keys() ):
                    for track_id, x1, y1, x2, y2 in frame_annotations[ frame_id ]:
                        # Format matches what process_gt_file expects:
                        # track_id at [0], frame_id at [2], bbox at [9:13]
                        # Fields: track_id, obj_len, frame, tracking_plane_loc_x/y,
                        #         velocity_x/y, image_loc_x/y, img_bbox (x1,y1,x2,y2), ...
                        line = f"{track_id} 0 {frame_id} 0 0 0 0 0 0 {x1} {y1} {x2} {y2} 0 0 0\n"
                        f.write( line )

            print( f"    {seq_name}: {len(frame_annotations)} frames, "
                   f"{len(set(t for anns in frame_annotations.values() for t,_,_,_,_ in anns))} tracks" )

    def update_model( self ):
        """
        Run the SRNN training pipeline.
        """
        print( "Starting SRNN training..." )

        # Prepare training data in KW18 format
        data_root = self._prepare_training_data()

        # Output directory for SRNN training
        srnn_output = Path( self._train_directory ) / "srnn_output"
        if srnn_output.exists():
            shutil.rmtree( srnn_output )

        # Build training command
        python_exe = "python.exe" if os.name == 'nt' else "python"

        cmd = [
            python_exe, "-m",
            "viame.pytorch.srnn.train_everything",
            str( data_root ),
            str( srnn_output ),
        ]

        if self._stabilized:
            cmd.append( "--stabilized" )

        print( "Running command: " + " ".join( cmd ) )

        # Handle interrupt signals
        if threading.current_thread().__class__.__name__ == '_MainThread':
            signal.signal( signal.SIGINT, lambda sig, frame: self._interrupt_handler() )
            signal.signal( signal.SIGTERM, lambda sig, frame: self._interrupt_handler() )

        self.proc = subprocess.Popen( cmd )
        self.proc.wait()

        if self.proc.returncode != 0:
            print( f"Warning: Training process exited with code {self.proc.returncode}" )

        self._save_final_model( srnn_output )

        print( "\nSRNN training complete!\n" )

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

    def _save_final_model( self, srnn_output ):
        """
        Copy trained models to output directory and generate pipeline file.
        """
        srnn_output = Path( srnn_output )

        # Find best models
        model_files = []

        # Siamese model
        siamese_model = srnn_output / "siamese" / "best_model.pt"
        if siamese_model.exists():
            dst = Path( self._output_directory ) / "siamese_model.pt"
            copyfile( siamese_model, dst )
            model_files.append( ( "siamese", dst ) )
            print( f"Copied Siamese model to {dst}" )

        # Target LSTM models (fixed and variable length)
        for fix_letter in ['F', 'V']:
            lstm_model = srnn_output / "target_lstm" / f"best_{fix_letter}_model.pt"
            if lstm_model.exists():
                dst = Path( self._output_directory ) / f"target_lstm_{fix_letter}.pt"
                copyfile( lstm_model, dst )
                model_files.append( ( f"target_lstm_{fix_letter}", dst ) )
                print( f"Copied Target LSTM model to {dst}" )

        if not model_files:
            print( "Warning: No trained models found" )
            return

        # Generate pipeline file from template if provided
        if self._pipeline_template and os.path.exists( self._pipeline_template ):
            with open( self._pipeline_template, 'r' ) as fin:
                template_content = fin.read()

            # Replace model path placeholders
            pipeline_content = template_content
            for model_name, model_path in model_files:
                placeholder = f"[-{model_name.upper().replace('_', '-')}-MODEL-]"
                pipeline_content = pipeline_content.replace(
                    placeholder, model_path.name
                )

            output_pipeline = os.path.join(
                self._output_directory, "tracker.pipe"
            )

            with open( output_pipeline, 'w' ) as fout:
                fout.write( pipeline_content )

            print( f"Generated pipeline file: {output_pipeline}" )
        else:
            # Generate a simple config snippet
            config_content = f"""# SRNN tracker configuration
# Generated by srnn_trainer

# Siamese appearance model
siamese_model = siamese_model.pt

# Target LSTM models
target_lstm_fixed = target_lstm_F.pt
target_lstm_variable = target_lstm_V.pt

# RNN components used: {self._rnn_component}
"""
            config_file = os.path.join( self._output_directory, "srnn_config.txt" )
            with open( config_file, 'w' ) as f:
                f.write( config_content )
            print( f"Generated config file: {config_file}" )


def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    implementation_name = "srnn"

    if algorithm_factory.has_algorithm_impl_name(
        SRNNTrainer.static_type_name(), implementation_name ):
        return

    algorithm_factory.add_algorithm(
        implementation_name,
        "PyTorch SRNN tracker training routine",
        SRNNTrainer
    )

    algorithm_factory.mark_algorithm_as_loaded( implementation_name )
