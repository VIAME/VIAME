# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from kwiver.vital.algo import (
    DetectedObjectSetOutput,
    TrainDetector
)

import os
import json
import shutil
import sys
import subprocess

from .kwcoco_train_detector import KWCocoTrainDetector
from .kwcoco_train_detector import KWCocoTrainDetectorConfig

import scriptconfig as scfg
import ubelt as ub

from viame.pytorch.utilities import (
    report_cuda_errors,
    vital_config_update,
    register_vital_algorithm,
    TrainingInterruptHandler,
)


class LitDetTrainerConfig(KWCocoTrainDetectorConfig):
    identifier = "viame-litdet-detector"
    train_directory = "deep_training"
    seed_model = ""

    tmp_training_file = "training_truth.json"
    tmp_validation_file = "validation_truth.json"

    config_file = scfg.Value('', help='Path to the user Hydra YAML configuration file')
    categories = []


class LitDetTrainer(KWCocoTrainDetector):
    def __init__(self):
        TrainDetector.__init__(self)
        self._config = LitDetTrainerConfig()

    def get_configuration(self):
        print('[LitDetTrainer] get_configuration')
        cfg = super().get_configuration()
        for key, value in self._config.items():
            cfg.set_value(key, str(value))
        return cfg

    @report_cuda_errors("LitDetTrainer initialization")
    def set_configuration(self, cfg_in):
        print('[LitDetTrainer] set_configuration')
        cfg = self.get_configuration()
        vital_config_update(cfg, cfg_in)
        for key in self._config.keys():
            self._config[key] = str(cfg.get_value(key))

        for key, value in self._config.items():
            setattr(self, "_" + key, value)

        self._post_config_set()
        return True

    def _post_config_set(self):
        print('[LitDetTrainer] _post_config_set')
        assert self._config['mode'] == "detector"

        if self._train_directory is not None:
            if not os.path.exists(self._train_directory):
                os.mkdir(self._train_directory)
            self._training_file = os.path.join(self._train_directory, self._tmp_training_file)
            self._validation_file = os.path.join(self._train_directory, self._tmp_validation_file)
        else:
            self._training_file = self._tmp_training_file
            self._validation_file = self._tmp_validation_file

        from kwiver.vital.modules import load_known_modules
        load_known_modules()

        if not self._no_format:
            self._training_writer = DetectedObjectSetOutput.create("coco")
            self._validation_writer = DetectedObjectSetOutput.create("coco")

            writer_conf = self._training_writer.get_configuration()
            self._training_writer.set_configuration(writer_conf)

            writer_conf = self._validation_writer.get_configuration()
            self._validation_writer.set_configuration(writer_conf)

            self._training_writer.open(self._training_file)
            self._validation_writer.open(self._validation_file)

    def _ensure_format_writers(self):
        if not self._no_format:
            self._training_writer.complete()
            self._validation_writer.complete()

            import kwcoco
            paths_to_fix = [self._training_file, self._validation_file]
            for fpath in paths_to_fix:
                fpath = ub.Path(fpath)
                if fpath.exists():
                    dset = kwcoco.CocoDataset(fpath)
                    dset.conform()
                    dset.dump()

    def check_configuration(self, cfg):
        if not cfg.has_value("identifier") or len(cfg.get_value("identifier")) == 0:
            print("A model identifier must be specified!")
            return False
        return True

    def _prepare_litdet_data_structure(self, train_coco_path, val_coco_path, output_dir):
        output_dir = ub.Path(output_dir)

        images_train_dir = output_dir / "COCO" / "images" / "train"
        images_valid_dir = output_dir / "COCO" / "images" / "valid"
        images_test_dir = output_dir / "COCO" / "images" / "test"
        labels_train_dir = output_dir / "COCO" / "labels" / "train"
        labels_valid_dir = output_dir / "COCO" / "labels" / "valid"

        images_train_dir.ensuredir()
        images_valid_dir.ensuredir()
        images_test_dir.ensuredir()
        labels_train_dir.ensuredir()
        labels_valid_dir.ensuredir()

        def process_split(coco_path, images_dir, labels_dir, split_name):
            coco_path = ub.Path(coco_path)
            if not coco_path.exists():
                print(f"[LitDetTrainer] Warning: {coco_path} does not exist")
                return 0, []

            with open(coco_path, 'r') as f:
                coco_data = json.load(f)

            new_images = []
            for img in coco_data.get('images', []):
                old_path = ub.Path(img['file_name'])
                if not old_path.is_absolute():
                    old_path = coco_path.parent / old_path

                if old_path.exists():
                    new_path = images_dir / old_path.name
                    if not new_path.exists():
                        try:
                            new_path.symlink_to(old_path.resolve())
                        except OSError:
                            shutil.copy2(old_path, new_path)

                    img['file_name'] = old_path.name
                    new_images.append(img)

            coco_data['images'] = new_images

            categories = []
            for cat in coco_data.get('categories', []):
                if 'supercategory' not in cat:
                    cat['supercategory'] = cat.get('name', 'object')
                categories.append(cat)
            coco_data['categories'] = categories

            annotations_path = labels_dir / f"instances_{split_name}.json"
            with open(annotations_path, 'w') as f:
                json.dump(coco_data, f)

            print(f"[LitDetTrainer] Prepared {len(new_images)} images for {split_name}")
            return len(new_images), categories

        _, categories = process_split(train_coco_path, images_train_dir, labels_train_dir, "train")
        num_valid, _ = process_split(val_coco_path, images_valid_dir, labels_valid_dir, "valid")

        if num_valid > 0:
            val_ann_path = labels_valid_dir / "instances_valid.json"
            if val_ann_path.exists():
                with open(val_ann_path, 'r') as f:
                    val_data = json.load(f)

                for img in val_data.get('images', []):
                    src = images_valid_dir / img['file_name']
                    dst = images_test_dir / img['file_name']
                    if src.exists() and not dst.exists():
                        try:
                            dst.symlink_to(src.resolve())
                        except OSError:
                            shutil.copy2(src, dst)

        return output_dir, len(categories)

    @report_cuda_errors("LitDetTrainer training")
    def update_model(self):
        self._ensure_format_writers()
        print("[LitDetTrainer] Starting LitDet training via CLI")

        dataset_dir = ub.Path(self._train_directory) / "litdet_dataset"
        dataset_dir.ensuredir()
        output_dir = ub.Path(self._train_directory) / "litdet_output"
        output_dir.ensuredir()

        data_dir, num_classes = self._prepare_litdet_data_structure(
            self._training_file,
            self._validation_file,
            dataset_dir
        )
        print(f"[LitDetTrainer] Prepared data with {num_classes} classes")

        cmd = [sys.executable, "-m", "litdet.train"]

        if self._config_file and os.path.exists(self._config_file):
            hydra_custom_dir = output_dir / "custom_hydra_config"
            exp_dir = hydra_custom_dir / "experiment"
            exp_dir.ensuredir()
            shutil.copy2(self._config_file, exp_dir / "viame_custom.yaml")
            cmd.append(f"--config-dir={hydra_custom_dir}")
            cmd.append("+experiment=viame_custom")
        else:
            raise ValueError("No valid config file provided. Using LitDet defaults.")

        viame_overrides = [
            f"paths.data_dir={data_dir}",
            f"paths.output_dir={output_dir}",
            f"paths.log_dir={output_dir}/logs",
            f"task.model.num_classes={num_classes + 1}",
            f"callbacks.model_checkpoint.dirpath={output_dir}/checkpoints",
            f"callbacks.model_best.dirpath={output_dir}/saved_models",
            f"callbacks.model_best.filename=best",
        ]
        cmd.extend(viame_overrides)

        print(f"\n[LitDetTrainer] Executing command:\n{' '.join(cmd)}\n")

        with TrainingInterruptHandler("LitDetTrainer") as handler:
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[LitDetTrainer] ERROR: LitDet training failed with exit code {e.returncode}")
                return {}
            except KeyboardInterrupt:
                print("[LitDetTrainer] Training interrupted by user")

            self._interrupted = handler.interrupted

        output = self._get_output_map(output_dir)
        print("\n[LitDetTrainer] Model training complete!")

        return output

    def _get_output_map(self, output_dir):
        output = {}
        output_model_name = "model.pth"

        if output_dir is not None:
            output_dir = ub.Path(output_dir)
            weights_path = output_dir / "saved_models" / "best.pth"

            if not weights_path.is_file():
                print("[LitDetTrainer] No weights file found")
                return output
        else:
            print("[LitDetTrainer] No output directory specified")
            return output

        algo = "litdet"
        output["type"] = algo
        output[algo + ":deployed"] = output_model_name
        output[output_model_name] = str(weights_path)
        output['config.yaml'] = str(self._config_file)

        print(f"\n[LitDetTrainer] Model found at: {weights_path}")
        print(f"\n[LitDetTrainer] The {self._train_directory} directory can now be deleted, "
              "unless you want to review training metrics first.")

        return output


def __vital_algorithm_register__():
    register_vital_algorithm(
        LitDetTrainer, "litdet", "PyTorch LitDet detection training routine"
    )