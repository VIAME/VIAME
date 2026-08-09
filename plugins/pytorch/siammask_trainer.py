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

import cv2
import numpy as np

import os
import socket
import sys
import shutil
from glob import glob
import subprocess
import signal
import time
import threading
from viame.pytorch.utilities import report_cuda_errors
from viame.core.training_data import ( build_sequence_maps,
    read_sequence_manifest, split_validation, seed_everything )
from viame.pytorch.siammask import ( VALIDATION_RECORD,
    VALIDATION_SEQUENCES )


def _frame_bounds( track_sets ):
    """Highest frame id each track set refers to, or None where it refers to
    none, for build_sequence_maps to check its alignment against.
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
        self._samples_per_sequence = 6000
        self._resume_model = ""
        self._backbone_seed = ""

        # How much of the training data to hold back to choose the epoch on,
        # as deepsort and botsort already do. Tracker training is handed no
        # validation set unless one is named, so without this the shipped
        # model is the last epoch trained, whether or not it is the best one.
        # 0 disables it and restores that behaviour.
        self._validation_fraction = 0.1

        # Seed for every generator this trainer and its ranks draw from.
        # Distinct from backbone_seed above, which names pretrained weights.
        # Exported by seed_everything so each worker inherits it; the worker
        # offsets by rank. Negative restores the previous behaviour.
        self._random_seed = "42"

        # Written by the training tool: which frames of the flat list
        # belong to which track set.
        self._sequence_manifest = ""
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
        cfg.set_value( "samples_per_sequence", str( self._samples_per_sequence ) )
        cfg.set_value( "resume_model", self._resume_model )
        cfg.set_value( "backbone_seed", self._backbone_seed )
        cfg.set_value( "validation_fraction", str( self._validation_fraction ) )
        cfg.set_value( "random_seed", self._random_seed )
        cfg.set_value( "sequence_manifest", self._sequence_manifest )
        cfg.set_value( "timeout", self._timeout )

        return cfg

    @report_cuda_errors("SiamMaskTrainer initialization")
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
        self._samples_per_sequence = int( cfg.get_value( "samples_per_sequence" ) )
        self._resume_model = str( cfg.get_value( "resume_model" ) )
        self._backbone_seed = str( cfg.get_value( "backbone_seed" ) )
        self._validation_fraction = float(
            cfg.get_value( "validation_fraction" ) )
        self._random_seed = str( cfg.get_value( "random_seed" ) )
        self._sequence_manifest = str( cfg.get_value( "sequence_manifest" ) )
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

    def _visible_gpu_count( self ):
        """Devices this training may use, at least one.

        CUDA_VISIBLE_DEVICES is what actually bounds the process, and the
        gpu_count setting caps it further when it is set to a positive number.
        """
        visible = os.environ.get( "CUDA_VISIBLE_DEVICES" )

        if visible is not None:
            count = len( [ d for d in visible.split( "," ) if d.strip() ] )
        else:
            try:
                import torch
                count = torch.cuda.device_count()
            except Exception:
                count = 1

        if self._gpu_count and self._gpu_count > 0:
            count = min( count, self._gpu_count )

        return max( count, 1 )

    def _prepare_training_data( self ):
        """
        Lay out training data in the form par_crop and gen_json consume.

        Per sequence, under the training directory:
        - sequence_XXXX/frame_NNNNNN.jpg  symlinks to the annotated frames
        - sequence_XXXX/groundtruth.csv   VIAME CSV for those frames
        - sequence_XXXX/masks/*.png       per detection masks, where the
                                          groundtruth segments rather than
                                          just boxes

        par_crop then writes crop511/ from that, and gen_json writes
        dataset.json from the crops. Neither is produced here. This used to
        write a dataset.json directly and create an empty crop511, but
        gen_json always runs and overwrote it with {} because there was no
        image and CSV layout for it to read, so training loaded no data at
        all.

        A mask is written whenever the detection carries one, which requires
        poly_to_mask on the track reader for polygon groundtruth. kwiver's
        convention is that the mask is already cropped to the bounding box, so
        it is stored that way and par_crop places it back into the frame. Its
        file name goes in a tenth CSV column, which the readers downstream
        ignore since they index the first seven. A detection without one
        leaves that column empty and trains boxes only, so a mixed dataset
        needs no special handling.
        """
        # One image map per sequence. A frame id is a position within its own
        # sequence, so resolving it against the flat list of every sequence's
        # images only ever worked for the first one.
        image_maps, _names = read_sequence_manifest(
            self._sequence_manifest, self._train_image_files,
            len( self._train_tracks ) )

        if image_maps is None:
            image_maps, _names = build_sequence_maps(
                self._train_image_files, len( self._train_tracks ), "training",
                _frame_bounds( self._train_tracks ) )

        print( "Preparing training data for SiamMask..." )
        print( f"  Processing {len(self._train_tracks)} track sets" )

        # Clear sequences left by an earlier run so they cannot be picked up
        for stale in glob( os.path.join( self._train_directory, "sequence_*" ) ):
            if os.path.isdir( stale ):
                shutil.rmtree( stale )

        sequence_count = 0
        annotation_count = 0
        mask_count = 0

        # Sequence directories that really got written, in track set order.
        # A track set can be dropped here for holding no usable box, and one
        # that was dropped must not end up in the validation list.
        written = []

        for seq_idx, track_set in enumerate( self._train_tracks ):
            if track_set is None:
                continue

            seq_name = f"sequence_{seq_idx:04d}"
            seq_dir = os.path.join( self._train_directory, seq_name )
            mask_dir = None
            image_map = image_maps[ seq_idx ]

            # frame id -> [ ( track_id, x1, y1, x2, y2, mask ) ]
            frame_annotations = {}

            for track in track_set.tracks():
                for state in track:
                    det = state.detection()

                    if det is None:
                        continue

                    bbox = det.bounding_box
                    x1 = int( bbox.min_x() )
                    y1 = int( bbox.min_y() )
                    x2 = int( bbox.max_x() )
                    y2 = int( bbox.max_y() )

                    # Zero area boxes crop to nothing
                    if x2 <= x1 or y2 <= y1:
                        continue

                    frame_annotations.setdefault( state.frame_id, [] ).append(
                        ( track.id, x1, y1, x2, y2,
                          self._detection_mask( det ) )
                    )

            if not frame_annotations:
                continue

            os.makedirs( seq_dir )

            # Frames are numbered by position rather than by their id in the
            # source clip, so the CSV frame column doubles as an index into the
            # sorted image list, which is what crop_video falls back to when it
            # cannot resolve the file name directly
            rows = []

            for position, frame_id in enumerate( sorted( frame_annotations ) ):
                src = image_map.get( frame_id )

                if src is None or not os.path.exists( src ):
                    continue

                image_name = f"frame_{position:06d}.jpg"
                os.symlink( os.path.realpath( src ),
                            os.path.join( seq_dir, image_name ) )

                for track_id, x1, y1, x2, y2, mask in \
                        frame_annotations[ frame_id ]:
                    mask_name = ""

                    if mask is not None:
                        if mask_dir is None:
                            mask_dir = os.path.join( seq_dir, "masks" )
                            os.makedirs( mask_dir )

                        mask_name = "masks/{:06d}_{}.png".format(
                            position, track_id )

                        cv2.imwrite( os.path.join( seq_dir, mask_name ),
                                     mask * 255 )
                        mask_count += 1

                    rows.append( "{},{},{},{},{},{},{},1.0,-1,{}\n".format(
                        track_id, image_name, position,
                        x1, y1, x2, y2, mask_name ) )

            if not rows:
                shutil.rmtree( seq_dir )
                continue

            with open( os.path.join( seq_dir, "groundtruth.csv" ), 'w' ) as f:
                f.writelines( rows )

            sequence_count += 1
            annotation_count += len( rows )
            written.append( seq_name )

            print( f"    {seq_name}: {len(frame_annotations)} frames, "
                   f"{len(rows)} annotations" )

        print( f"Prepared {sequence_count} sequences, "
               f"{annotation_count} annotations for cropping, "
               f"{mask_count} of them with a mask" )

        if not mask_count:
            print( "  No masks in the groundtruth. The mask head will keep "
                   "whatever weights it was seeded with and only the box "
                   "branches will train. Polygon groundtruth reaches here "
                   "only with poly_to_mask set on the track reader." )

        self._plan_validation( written )

        return self._train_directory

    def _plan_validation( self, written ):
        """Name the clips to hold out of training, for the entry point to read.

        The split is by clip and is decided here because this is the last
        point at which a clip is still a track set, which is what
        split_validation takes -- the same call, and so the same tail of
        clips, that deepsort and botsort hold out. Past here a clip is a
        directory of crops and then a key in dataset.json.

        Only the names are handed over. Every sequence is still written and
        cropped, held out or not, because measuring a held out clip needs its
        crop511 pyramid exactly as training does; what the entry point does
        with this list is partition dataset.json, which costs nothing and
        keeps one crop pass for both splits.

        The tail is deterministic, so a resumed run holds out the same clips
        as the run it continues and the two runs' validation numbers can be
        compared against each other.
        """
        listing = os.path.join( self._train_directory, VALIDATION_SEQUENCES )
        record = os.path.join( self._train_directory, VALIDATION_RECORD )

        # A list left by an earlier run describes that run's clips, and would
        # hold out the wrong ones here if this run writes no list of its own
        if os.path.exists( listing ):
            os.remove( listing )

        # Only a run continuing an earlier one inherits its measurements,
        # which is what keeping them in a file is for: its own early epochs
        # may have been measured in a previous sitting. A run starting over is
        # about to overwrite those checkpoints, so what was measured of them
        # no longer describes what is on disk.
        if not self._resume_model and os.path.exists( record ):
            os.remove( record )

        held = []

        # Under three clips there is nothing to hold back that still leaves a
        # training set worth the name -- split_validation would refuse below
        # two anyway -- so such a dataset trains on everything and keeps the
        # last epoch, which is what it did before validation existed.
        if self._validation_fraction > 0 and len( written ) >= 3:
            names = [ "sequence_{:04d}".format( i )
                      for i in range( len( self._train_tracks ) ) ]

            _train_part, valid_part = split_validation(
                self._train_tracks, None, names, self._validation_fraction )

            held = [ n for n in valid_part[ 2 ] if n in written ]

        if not held:
            if self._validation_fraction > 0:
                print( "Validation: {} usable clip(s) is too few to hold any "
                       "back, so every clip is trained on and the last epoch "
                       "is the one shipped.".format( len( written ) ) )
            else:
                print( "Validation: disabled by validation_fraction, so the "
                       "last epoch is the one shipped." )

            # Nothing will be measured to supersede an inherited record, and
            # selecting on it would ship an epoch of a different run
            if os.path.exists( record ):
                os.remove( record )

            return

        with open( listing, 'w' ) as handle:
            handle.writelines( name + "\n" for name in held )

        print( "Validation: holding out {} of {} clips ({}), training on the "
               "rest".format( len( held ), len( written ),
                              ", ".join( held ) ) )

    @staticmethod
    def _detection_mask( det ):
        """This detection's mask as a uint8 0/1 array, or None.

        kwiver hands it over cropped to the bounding box rather than the size
        of the frame, which is the form it is stored in.
        """
        try:
            container = det.mask

            if container is None:
                return None

            mask = container.image().asarray()
        except Exception:
            return None

        if mask is None:
            return None

        mask = np.asarray( mask )

        if mask.size == 0:
            return None

        if mask.ndim == 3:
            mask = mask[ ..., 0 ]

        return ( mask > 0 ).astype( np.uint8 )

    @report_cuda_errors("SiamMaskTrainer training")
    def update_model( self ):
        """
        Run the SiamMask training process.
        """
        print( "Starting SiamMask training..." )

        # Before the dataset is laid out and before any worker is spawned.
        # seed_everything exports the seed, and train_env below is a copy of
        # this environment, so every rank inherits it.
        if seed_everything( self._random_seed ):
            print( "  seeded with " + str( self._random_seed ) )
        else:
            print( "  unseeded: run to run variation is expected" )

        # Prepare training data
        dataset_file = self._prepare_training_data()

        # The non-zero ranks wait on this file for rank zero to finish
        # cropping. One left behind by an interrupted run would release them
        # immediately, against a crop511 that is half built or absent.
        stale_flag = os.path.join( self._train_directory, ".prep_complete" )

        if os.path.exists( stale_flag ):
            os.remove( stale_flag )

        # Build training command
        python_exe = "python.exe" if os.name == 'nt' else "python"

        # SiamMask training is genuinely distributed -- it wraps the model in
        # DistModule, reduces gradients across ranks and switches to a
        # DistributedSampler once the world size exceeds one. Launched as a
        # plain "python -m" it only ever forms a single rank process group and
        # uses one device no matter how many are visible, so hand it to
        # torch.distributed.run when there is more than one to use.
        train_gpus = self._visible_gpu_count()

        cmd = [ python_exe, "-m" ]

        if train_gpus > 1:
            cmd = [
                python_exe, "-m", "torch.distributed.run",
                "--standalone",
                "--nproc-per-node", str( train_gpus ),
                "-m",
            ]

        cmd += [
            "viame.pytorch.siammask.siammask_trainer",
            "-i", self._train_directory,
            "-s", self._train_directory,
            "-t", self._threshold,
        ]

        # The architecture config. Required by the training entry point, so
        # say what is missing here rather than letting it exit on an argparse
        # usage dump that names neither the option nor the file it wanted.
        if self._config_file:
            if not os.path.exists( self._config_file ):
                raise RuntimeError(
                    "siammask config_file does not exist: {}".format(
                        self._config_file ) )

            config_file = self._config_file
        else:
            config_file = os.path.join(
                os.path.dirname( os.path.realpath( __file__ ) ),
                "siammask", "experiments", "siammask_r50_l3.yaml"
            )

            if not os.path.exists( config_file ):
                raise RuntimeError(
                    "no siammask architecture config. Set "
                    "tracker_trainer:siammask:config_file in the settings "
                    "file; the standard training config points it at "
                    "models/siammask_default.yaml. The built in fallback "
                    "{} is not present in this install.".format( config_file ) )

        cmd.extend( [ "-c", config_file ] )

        if self._skip_crop:
            cmd.append( "--skip-crop" )

        cmd.append( "--samples-per-sequence={}".format(
            self._samples_per_sequence ) )

        # max_epochs was read from the settings and then never reached the
        # training, which took TRAIN.EPOCH from the architecture yaml -- 20 by
        # default, whatever the setting said. The default here is the same 20,
        # so nothing changes for a run that leaves it alone.
        cmd.append( "--epochs={}".format( self._max_epochs ) )

        # The held out clips, where there are any. Absent, the entry point
        # trains on everything and measures nothing.
        listing = os.path.join( self._train_directory, VALIDATION_SEQUENCES )

        if os.path.exists( listing ):
            cmd.append( "--validation-sequences=" + listing )

        if self._resume_model:
            cmd.append( "--resume=" + self._resume_model )

        if self._backbone_seed:
            cmd.append( "--backbone-pretrained=" + self._backbone_seed )

        # seed_model was read from the config but never reached training, so
        # fine tuning silently started from scratch. It is loaded over the
        # whole network via TRAIN.PRETRAINED.
        if self._seed_model:
            cmd.extend( [ "--pretrained", self._seed_model ] )

        print( "Running command: " + " ".join( cmd ) )

        # Handle interrupt signals
        if threading.current_thread().__class__.__name__ == '_MainThread':
            signal.signal( signal.SIGINT, lambda sig, frame: self._interrupt_handler() )
            signal.signal( signal.SIGTERM, lambda sig, frame: self._interrupt_handler() )

        # dist_init() reads RANK and then init_process_group. Under
        # torch.distributed.run those variables are set per worker, so they are
        # only described here for the single device launch, which would
        # otherwise die with KeyError: 'RANK'.
        train_env = os.environ.copy()

        if train_gpus <= 1:
            train_env.setdefault( "RANK", "0" )
            train_env.setdefault( "LOCAL_RANK", "0" )
            train_env.setdefault( "WORLD_SIZE", "1" )
            train_env.setdefault( "MASTER_ADDR", "127.0.0.1" )

            if "MASTER_PORT" not in train_env:
                sock = socket.socket( socket.AF_INET, socket.SOCK_STREAM )
                sock.bind( ( "", 0 ) )
                train_env[ "MASTER_PORT" ] = str( sock.getsockname()[1] )
                sock.close()

        self.proc = subprocess.Popen( cmd, env=train_env )
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
        # The model is copied whether or not a pipeline template was given.
        # Returning early on an empty template used to skip the copy as well,
        # so a run that trained for nineteen hours and reported "training
        # completed successfully" left category_models empty and was marked
        # FAILED for having no model in it.

        # Find the latest checkpoint
        snapshot_dir = os.path.join( self._train_directory, "snapshot" )
        if not os.path.exists( snapshot_dir ):
            print( "No snapshot directory found" )
            return

        def epoch_of( name ):
            return int( name[ len( "checkpoint_e" ):-len( ".pth" ) ] )

        checkpoints = [
            f for f in os.listdir( snapshot_dir )
            if f.startswith( "checkpoint_e" ) and f.endswith( ".pth" )
        ]

        # By epoch, not by name. Sorted as strings, checkpoint_e9 comes after
        # checkpoint_e20, so a twenty epoch run shipped the ninth epoch. The
        # same holds of the lookup below, which is keyed by epoch number for
        # the same reason.
        try:
            by_epoch = { epoch_of( f ): f for f in checkpoints }
            checkpoints.sort( key=epoch_of )
        except ValueError:
            by_epoch = {}
            checkpoints.sort()

        if not checkpoints:
            print( "No checkpoints found" )
            return

        selected = self._best_checkpoint( by_epoch )

        if selected is None:
            selected = checkpoints[ -1 ]
            print( "Selected {} (the last epoch trained); there were no "
                   "validation losses to choose on".format( selected ) )

        src_model = os.path.join( snapshot_dir, selected )
        output_model_name = "trained_tracker.pth"
        dst_model = os.path.join( self._output_directory, output_model_name )

        copyfile( src_model, dst_model )
        print( f"Copied model to {dst_model}" )

        # Generate pipeline file from template, where one was given
        if self._pipeline_template and os.path.exists( self._pipeline_template ):
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

    def _best_checkpoint( self, by_epoch ):
        """The checkpoint with the lowest validation loss, or None.

        None where nothing was measured -- validation off, too few clips,
        every epoch's validation failed -- leaving the caller to keep the last
        epoch, as this did for every run before validation existed. Losing the
        record should cost the choice between checkpoints and not the run.

        The record is a file rather than a scrape of the log because a resumed
        run has to choose across every epoch of every sitting, including ones
        whose log was rotated away, in the same way the SRNN stage does.

        A plain global minimum: the epoch with the lowest val_total wins,
        wherever it falls. Worth knowing that the loss landscape is not
        stationary across the run -- at BACKBONE.TRAIN_EPOCH the backbone
        unfreezes and the losses shift, so epochs either side of it are not
        strictly comparable -- but a rule that preferred the later regime
        would be guessing, and the measurement is on held out clips either
        way, so the argmin is taken over all of them.
        """
        record = os.path.join( self._train_directory, VALIDATION_RECORD )

        if not os.path.isfile( record ):
            return None

        # Keyed by epoch rather than appended in order. A resumed run writes
        # its epochs across more than one sitting and does not start from
        # zero, and a re-run over the same directory measures an epoch that
        # was already recorded; the later record of an epoch is the one that
        # describes the checkpoint now on disk, so it wins.
        losses = {}

        with open( record ) as handle:
            for line in handle:
                if not line.strip() or line.lstrip().startswith( '#' ):
                    continue

                fields = line.split()

                if len( fields ) < 2:
                    continue

                try:
                    losses[ int( fields[ 0 ] ) ] = float( fields[ 1 ] )
                except ValueError:
                    continue

        # Only epochs whose checkpoint is actually there
        available = { epoch: loss for epoch, loss in losses.items()
                      if epoch in by_epoch }

        if not available:
            return None

        best = min( available, key=lambda epoch: available[ epoch ] )

        print( "Selected epoch {} by validation loss ({:.5f}), out of the {} "
               "epoch(s) measured; the last epoch trained was {}".format(
                   best, available[ best ], len( available ),
                   max( by_epoch ) ) )

        return by_epoch[ best ]


def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    # The same trainer drives both architectures; which one is built is decided
    # by MASK and REFINE in the yaml each config points at. Registering the two
    # names separately is what lets a training pick one by tracker name and get
    # the matching config.
    variants = [
        ( "siammask", "PyTorch SiamMask tracker training routine" ),
        ( "siamrpn", "PyTorch SiamRPN++ tracker training routine" ),
    ]

    for implementation_name, description in variants:
        if algorithm_factory.has_algorithm_impl_name(
            SiamMaskTrainer.static_type_name(), implementation_name ):
            continue

        algorithm_factory.add_algorithm(
            implementation_name,
            description,
            SiamMaskTrainer
        )

        algorithm_factory.mark_algorithm_as_loaded( implementation_name )
