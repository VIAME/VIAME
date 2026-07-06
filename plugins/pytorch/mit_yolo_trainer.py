# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #
from __future__ import annotations

from kwiver.vital.algo import (
    DetectedObjectSetOutput,
    ImageObjectDetector,
    TrainDetector
)

# from distutils.util import strtobool

# import numpy as np
# import torch
import os
import signal
import sys
import time
import types
# import math

from .kwcoco_train_detector import KWCocoTrainDetector
from .kwcoco_train_detector import KWCocoTrainDetectorConfig
from viame.pytorch.utilities import vital_config_update, report_cuda_errors

import scriptconfig as scfg
import ubelt as ub
from hydra import compose, initialize_config_dir
from yolo.lazy import main


if types.TYPE_CHECKING:
    import kwcoco


class MITYoloConfig(KWCocoTrainDetectorConfig):
    """
    The configuration for :class:`MITYoloTrainer`.
    """
    identifier = "viame-mit-yolo-detector"
    train_directory = "deep_training"
    seed_model = ""

    tmp_training_file = "training_truth.json"
    tmp_validation_file = "validation_truth.json"
    accelerator = scfg.Value('auto', help='lightning accelerator. Can be cpu, gpu, or auto')
    model = scfg.Value('v9-c', help='the model archictecture')

    out_path = scfg.Value('allow', help=ub.paragraph(
        '''
        Where to save results
        '''))

    categories = []

    max_epochs = scfg.Value(500, help='Maximum number of epochs to train for')
    batch_size = scfg.Value(4, help='Number of chips per batch.')
    learning_rate = scfg.Value(3e-4, help='Learning rate for gradient update steps.')
    timeout = scfg.Value('1209600', help='Max training time in seconds (default=1209600, two weeks)')

    def __post_init__(self):
        super().__post_init__()


class MITYoloTrainer( KWCocoTrainDetector ):
    """
    Implementation of TrainDetector class
    """
    def __init__( self ):
        TrainDetector.__init__( self )
        self._config = MITYoloConfig()

    def get_configuration(self):
        # Inherit from the base class
        print('[MITYoloTrainer] get_configuration')
        cfg = super().get_configuration()
        for key, value in self._config.items():
            cfg.set_value(key, str(value))
        return cfg

    @report_cuda_errors("MITYoloTrainer initialization")
    def set_configuration(self, cfg_in):
        print('[MITYoloTrainer] set_configuration')
        cfg = self.get_configuration()
        vital_config_update(cfg, cfg_in)
        for key in self._config.keys():
            self._config[key] = str(cfg.get_value(key))
        self._config.__post_init__()

        # Hack in the previous style configuration as class variables prefixed
        # with an underscore. Ideally we would just access these via the config
        # object as a single source of truth
        for key, value in self._config.items():
            setattr(self, "_" + key, value)

        self._post_config_set()
        return True

    def _post_config_set(self):
        print('[MITYoloTrainer] _post_config_set')
        assert self._config['mode'] == "detector"

        # Make required directories and file streams
        if self._train_directory is not None:
            if not os.path.exists( self._train_directory ):
                os.mkdir( self._train_directory )
            self._training_file = os.path.join(
                self._train_directory, self._tmp_training_file )
            self._validation_file = os.path.join(
                self._train_directory, self._tmp_validation_file )
            self._chip_directory = os.path.join(
                self._train_directory, "image_chips" )
        else:
            self._training_file = self._tmp_training_file
            self._validation_file = self._tmp_validation_file

        from kwiver.vital.modules import load_known_modules
        load_known_modules()

        if not self._no_format:
            self._training_writer = DetectedObjectSetOutput.create( "coco" )
            self._validation_writer = DetectedObjectSetOutput.create( "coco" )

            writer_conf = self._training_writer.get_configuration()
            # writer_conf.set_value( "aux_image_labels", self._aux_image_labels )
            # writer_conf.set_value( "aux_image_extensions", self._aux_image_extensions )
            self._training_writer.set_configuration( writer_conf )

            writer_conf = self._validation_writer.get_configuration()
            # writer_conf.set_value( "aux_image_labels", self._aux_image_labels )
            # writer_conf.set_value( "aux_image_extensions", self._aux_image_extensions )
            self._validation_writer.set_configuration( writer_conf )

            self._training_writer.open( self._training_file )
            self._validation_writer.open( self._validation_file )

        if self._mode == "detection_refiner" and not os.path.exists( self._chip_directory ):
            os.mkdir( self._chip_directory )

        # Load object detector if enabled
        if self._detector_model:
            self._detector = ImageObjectDetector.create( "yolo" )
            detector_config = self._detector.get_configuration()
            detector_config.set_value( "deployed", self._detector_model )
            if not self._detector.set_configuration( detector_config ):
                print( "Unable to configure detector" )
                return False

        # QUESTION: Do we need to handle "scale type file"?

        # Other misc setting adjustments
        if self._chip_extension and self._chip_extension[0] != '.':
            self._chip_extension = '.' + self._chip_extension

        if int( self._chip_height ) <= 0:
            self._chip_height = self._chip_width
        if int( self._chip_width ) <= 0:
            self._chip_width = self._chip_height

        # Initialize persistent variables
        self._training_data = []
        self._validation_data = []
        self._sample_count = 0

    def _ensure_format_writers(self):
        if not self._no_format:
            self._training_writer.complete()
            self._validation_writer.complete()

            class_list = list(self._categories)
            if not class_list:
                raise ValueError("MIT-YOLO categories must be initialized before writing kwcoco files")

            import kwcoco
            train_fpath = ub.Path(self._training_file)
            if train_fpath.exists():
                print("[MITYoloTrainer] Conforming training dataset")
                train_dset = kwcoco.CocoDataset(train_fpath)
                train_dset.conform()
                _align_kwcoco_categories(train_dset, class_list)
                train_dset.dump()  # write back to the same place we read

                val_fpath = ub.Path(self._validation_file)
                if val_fpath.exists():
                    print("[MITYoloTrainer] Conforming validation dataset")
                    val_dset = kwcoco.CocoDataset(val_fpath)
                    val_dset.conform()
                    _align_kwcoco_categories(val_dset, class_list)
                    val_dset.dump()  # write back to the same place we read

    def check_configuration( self, cfg ):
        if not cfg.has_value( "identifier" ) or len( cfg.get_value( "identifier") ) == 0:
            print( "A model identifier must be specified!" )
            return False
        return True

    @report_cuda_errors("MITYoloTrainer training")
    def update_model( self ):
        self._ensure_format_writers()
        self._accelerator = 'auto'

        # On Windows, reconfigure stdout/stderr to use UTF-8 encoding.
        # Third-party libraries (rich, lightning, yolo) use emoji characters
        # that cannot be encoded in the default cp1252 Windows codepage.
        if sys.platform == 'win32':
            for stream in (sys.stdout, sys.stderr):
                if hasattr(stream, 'reconfigure'):
                    stream.reconfigure(encoding='utf-8', errors='replace')

        # Initialize hydra config
        import yolo.config
        config_dir = ub.Path(yolo.config.__file__).parent
        hydra_overrides = [
            "task=train",
            f"model={self._config.model}",
            "use_wandb=False",
            "cpu_num=4",
            "dataset=coco",
            "dataset.auto_download=",
            f"name={self._identifier}",
            f"dataset.path={os.fspath('.')}",
            f"dataset.train={os.fspath(self._training_file)}",
            f"dataset.validation={os.fspath(self._validation_file) if self._validation_file else None}",
            f"dataset.class_list={self._categories}",
            f"dataset.class_num={len(self._categories)}",
            f"out_path={self._train_directory}",
            f"accelerator={self._accelerator}",
            f"task.data.batch_size={self._batch_size}",
            f"task.optimizer.args.lr={self._learning_rate}",
            f"task.epoch={self._max_epochs}",
            f"+timeout={self._timeout}",
            "++save_best=True"
        ]
        if len(self._seed_model) > 0:
            hydra_overrides += [f"weight={self._seed_model}"]
        with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
            cfg = compose(config_name="config.yaml", overrides=hydra_overrides)

        # Launch training
        main(cfg)

        output = self._get_output_map()

        print( "\nModel training complete!\n" )

        return output

    def interupt_handler( self ):
        self.proc.send_signal( signal.SIGINT )
        timeout = 0
        while self.proc.poll() is None:
            time.sleep( 0.1 )
            timeout += 0.1
            if timeout > 5:
                self.proc.kill()
                break
        sys.exit( 0 )

    def _get_output_map( self ):
        output_model_name = "trained_mit_yolo_checkpoint.ckpt"

        train_dpath = ub.Path(self._train_directory)
        yolo_train_dpath = train_dpath / 'train' / self._identifier
        checkpoint_dpath = yolo_train_dpath / 'checkpoints'
        candiate_checkpoints = sorted(checkpoint_dpath.glob('*'))
        if len(candiate_checkpoints) == 0:
            print( "\nNo checkpoints found, model may have failed to train" )
            return {"type": "mit_yolo"}

        # Prefer best checkpoint, fall back to last checkpoint
        best_checkpoints = sorted(checkpoint_dpath.glob('best-*.ckpt'))
        if best_checkpoints:
            final_ckpt = best_checkpoints[-1]
            print( "\nBest model found at " + str(final_ckpt) )
        else:
            final_ckpt = candiate_checkpoints[-1]
            print( "\nModel found at " + str(final_ckpt) )

        print( "\nThe " + self._train_directory + " directory can now be deleted, "
               "unless you want to review training metrics or generated plots in "
               "there first." )

        output = {}
        output["type"] = "mit_yolo"
        output["mit_yolo:weight"] = output_model_name
        output["mit_yolo:model"] = self._config.model
        output[output_model_name] = str(final_ckpt)

        # The detector needs train_config.yaml next to the checkpoint
        # to introspect model architecture and class list.
        train_config_fpath = yolo_train_dpath / 'train_config.yaml'
        if train_config_fpath.exists():
            output["train_config.yaml"] = str(train_config_fpath)

        return output


def _align_kwcoco_categories(
        dset: kwcoco.CocoDataset,
        class_list: list[str]
     ) -> kwcoco.CocoDataset:
    """
    Modify (inplace) the CocoDataset so categories are in a specified order,
    and handle the case of unexpected categories.

    Example:
        >>> import pytest
        >>> dset = kwcoco.CocoDataset.demo('vidshapes1')
        >>> class_list = ['superstar', 'star', 'eff']
        >>> _align_kwcoco_categories(dset.copy(), class_list)
        >>> #
        >>> # Case: dset has category we can't align with, should error
        >>> with pytest.raises(ValueError):
        >>>     class_list = ['star', 'eff']
        >>>     _align_kwcoco_categories(dset.copy(), class_list)
        >>> #
        >>> # Case: extra category, should be fine
        >>> class_list = ['superstar', 'star', 'eff', 'longcat']
        >>> _align_kwcoco_categories(dset.copy(), class_list)
    """
    existing_names = set(dset.categories().lookup('name'))
    expected_names = set(class_list)

    if len(class_list) != len(expected_names):
        raise ValueError('class list should not contain duplicates')

    unexpected_names = existing_names - expected_names
    if unexpected_names:
        used_cat_ids = set(dset.annots().lookup('category_id', None))
        used_catnames = {
            None if cid is None else dset.index.cats[cid].get('name')
            for cid in used_cat_ids
        }
        used_unexpected = used_catnames & unexpected_names
        if used_unexpected:
            raise ValueError(
                f"{dset} contains annotations outside the configured "
                f"training categories: {sorted(used_unexpected)}"
            )
        # Normalize to category ids in case CocoDataset API changes
        unexpected_cids = [dset.index.name_to_cat[name]['id']
                           for name in unexpected_names]
        # Note: keep annots is irrelevant because we error if there are any
        # annots to remove. However, if we relax that to a warning removing the
        # annots (or modifying their categories) is the right thing to do, but
        # that should be handled before it ever gets to this point.
        dset.remove_categories(unexpected_cids, keep_annots=False)

    for name in class_list:
        dset.ensure_category(name)

    dset.normalize_category_ids(start_id=1, order=class_list)
    return dset


def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    # Register Algorithm
    implementation_name = "mit_yolo"

    if algorithm_factory.has_algorithm_impl_name( MITYoloTrainer.static_type_name(), implementation_name ):
        return

    algorithm_factory.add_algorithm( implementation_name, "PyTorch MIT YOLO detection training routine", MITYoloTrainer )

    algorithm_factory.mark_algorithm_as_loaded( implementation_name )
