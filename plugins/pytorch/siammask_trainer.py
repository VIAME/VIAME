# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

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
import shutil
import subprocess
import signal
import time
import threading


class SiamMaskTrainer( TrainTracker ):
    """
    Implementation of TrainTracker class for SiamMask tracker training
    """
    def __init__( self ):
        TrainTracker.__init__( self )

        self._identifier = "viame-siammask-tracker"
        self._config_file = ""
        self._seed_model = ""
        self._train_directory = "deep_training"
        self._output_directory = "category_models"
        self._output_prefix = "siammask_tracker"
        self._pipeline_template = ""
        self._gpu_count = -1
        self._max_epochs = "20"
        self._batch_size = "auto"
        self._crop_size = 511
        self._threshold = "0.00"
        self._skip_crop = False
        self._timeout = "1209600"

        self._categories = []
        self._train_image_files = []
        self._train_tracks = []
        self._test_image_files = []
        self._test_tracks = []

    def get_configuration( self ):
        cfg = super( TrainTracker, self ).get_configuration()

        cfg.set_value( "identifier", self._identifier )
        cfg.set_value( "config_file", self._config_file )
        cfg.set_value( "seed_model", self._seed_model )
        cfg.set_value( "train_directory", self._train_directory )
        cfg.set_value( "output_directory", self._output_directory )
        cfg.set_value( "output_prefix", self._output_prefix )
        cfg.set_value( "pipeline_template", self._pipeline_template )
        cfg.set_value( "gpu_count", str( self._gpu_count ) )
        cfg.set_value( "max_epochs", str( self._max_epochs ) )
        cfg.set_value( "batch_size", self._batch_size )
        cfg.set_value( "crop_size", str( self._crop_size ) )
        cfg.set_value( "threshold", self._threshold )
        cfg.set_value( "skip_crop", str( self._skip_crop ) )
        cfg.set_value( "timeout", self._timeout )

        return cfg

    def set_configuration( self, cfg_in ):
        cfg = self.get_configuration()
        cfg.merge_config( cfg_in )

        self._identifier = str( cfg.get_value( "identifier" ) )
        self._config_file = str( cfg.get_value( "config_file" ) )
        self._seed_model = str( cfg.get_value( "seed_model" ) )
        self._train_directory = str( cfg.get_value( "train_directory" ) )
        self._output_directory = str( cfg.get_value( "output_directory" ) )
        self._output_prefix = str( cfg.get_value( "output_prefix" ) )
        self._pipeline_template = str( cfg.get_value( "pipeline_template" ) )
        self._gpu_count = int( cfg.get_value( "gpu_count" ) )
        self._max_epochs = str( cfg.get_value( "max_epochs" ) )
        self._batch_size = str( cfg.get_value( "batch_size" ) )
        self._crop_size = int( cfg.get_value( "crop_size" ) )
        self._threshold = str( cfg.get_value( "threshold" ) )
        self._skip_crop = strtobool( cfg.get_value( "skip_crop" ) )
        self._timeout = str( cfg.get_value( "timeout" ) )

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

        The track data will be converted to SiamMask training format
        (cropped image pairs) when update_model is called.
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
        Convert track data to SiamMask training format.

        Each track_set in _train_tracks represents a sequence/video and contains
        all tracks for that sequence. Each track spans multiple frames with
        consistent track IDs.

        This creates:
        - crop_{crop_size}/ directory with cropped images
        - dataset.json with track annotations in SiamMask format
        """
        import json

        crop_dir = os.path.join( self._train_directory, f"crop{self._crop_size}" )
        if os.path.exists( crop_dir ):
            shutil.rmtree( crop_dir )
        os.makedirs( crop_dir )

        dataset = {}

        # Process training track sets
        # Each track_set is a sequence containing tracks that span multiple frames
        print( "Preparing training data for SiamMask..." )
        print( f"  Processing {len(self._train_tracks)} track sets" )

        for seq_idx, track_set in enumerate( self._train_tracks ):
            if track_set is None:
                continue

            seq_name = f"sequence_{seq_idx:04d}"
            video_data = {}

            # Get all tracks in this sequence
            tracks = track_set.tracks()
            print( f"  Sequence {seq_idx}: {len(tracks)} tracks" )

            for track in tracks:
                track_id = track.id()
                track_key = f"{track_id:06d}"

                if track_key not in video_data:
                    video_data[ track_key ] = {}

                # Iterate through all states (frames) in this track
                for state in track:
                    frame_id = state.frame()
                    det = state.detection()

                    if det is None:
                        continue

                    bbox = det.bounding_box()
                    x1 = int( bbox.min_x() )
                    y1 = int( bbox.min_y() )
                    x2 = int( bbox.max_x() )
                    y2 = int( bbox.max_y() )
                    w = x2 - x1
                    h = y2 - y1
                    cx = x1 + w // 2
                    cy = y1 + h // 2

                    frame_key = f"{frame_id:06d}"
                    video_data[ track_key ][ frame_key ] = [ cx, cy, w, h ]

            if video_data:
                dataset[ seq_name ] = video_data
                total_frames = sum( len( v ) for v in video_data.values() )
                print( f"    Added {len(video_data)} tracks, {total_frames} total annotations" )

        # Write dataset.json
        dataset_file = os.path.join( self._train_directory, "dataset.json" )
        with open( dataset_file, 'w' ) as f:
            json.dump( dataset, f, indent=2 )

        total_tracks = sum( len( v ) for v in dataset.values() )
        print( f"Created dataset.json with {len(dataset)} sequences, {total_tracks} total tracks" )

        return dataset_file

    def update_model( self ):
        """
        Run the SiamMask training process.
        """
        print( "Starting SiamMask training..." )

        # Prepare training data
        dataset_file = self._prepare_training_data()

        # Build training command
        python_exe = "python.exe" if os.name == 'nt' else "python"

        cmd = [
            python_exe, "-m",
            "viame.pytorch.siammask.siammask_trainer",
            "-i", self._train_directory,
            "-s", self._train_directory,
            "-t", self._threshold,
        ]

        if self._config_file:
            cmd.extend( [ "-c", self._config_file ] )
        else:
            # Use default config
            default_config = os.path.join(
                os.path.dirname( os.path.realpath( __file__ ) ),
                "siammask", "experiments", "siammask_r50_l3.yaml"
            )
            if os.path.exists( default_config ):
                cmd.extend( [ "-c", default_config ] )

        if self._skip_crop:
            cmd.append( "--skip-crop" )

        print( "Running command: " + " ".join( cmd ) )

        # Handle interrupt signals
        if threading.current_thread().__class__.__name__ == '_MainThread':
            signal.signal( signal.SIGINT, lambda sig, frame: self._interrupt_handler() )
            signal.signal( signal.SIGTERM, lambda sig, frame: self._interrupt_handler() )

        self.proc = subprocess.Popen( cmd )
        self.proc.wait()

        self._save_final_model()

        print( "\nSiamMask training complete!\n" )

        return {"type": "siammask"}

    def _interrupt_handler( self ):
        self.proc.send_signal( signal.SIGINT )
        timeout = 0
        while self.proc.poll() is None:
            time.sleep( 0.1 )
            timeout += 0.1
            if timeout > 5:
                self.proc.kill()
                break
        self._save_final_model()
        sys.exit( 0 )

    def _save_final_model( self ):
        """
        Copy trained model to output directory and generate pipeline file.
        """
        if not self._pipeline_template:
            return

        # Find the latest checkpoint
        snapshot_dir = os.path.join( self._train_directory, "snapshot" )
        if not os.path.exists( snapshot_dir ):
            print( "No snapshot directory found" )
            return

        checkpoints = sorted( [
            f for f in os.listdir( snapshot_dir )
            if f.startswith( "checkpoint_e" ) and f.endswith( ".pth" )
        ] )

        if not checkpoints:
            print( "No checkpoints found" )
            return

        latest_checkpoint = checkpoints[ -1 ]
        src_model = os.path.join( snapshot_dir, latest_checkpoint )
        output_model_name = "trained_tracker.pth"
        dst_model = os.path.join( self._output_directory, output_model_name )

        copyfile( src_model, dst_model )
        print( f"Copied model to {dst_model}" )

        # Generate pipeline file from template
        if os.path.exists( self._pipeline_template ):
            with open( self._pipeline_template, 'r' ) as fin:
                template_content = fin.read()

            pipeline_content = template_content.replace(
                "[-MODEL-FILE-]", output_model_name
            )

            output_pipeline = os.path.join(
                self._output_directory, "tracker.pipe"
            )

            with open( output_pipeline, 'w' ) as fout:
                fout.write( pipeline_content )

            print( f"Generated pipeline file: {output_pipeline}" )


def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    implementation_name = "siammask"

    if algorithm_factory.has_algorithm_impl_name(
        SiamMaskTrainer.static_type_name(), implementation_name ):
        return

    algorithm_factory.add_algorithm(
        implementation_name,
        "PyTorch SiamMask tracker training routine",
        SiamMaskTrainer
    )

    algorithm_factory.mark_algorithm_as_loaded( implementation_name )
