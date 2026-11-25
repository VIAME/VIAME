# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import json
import os
import socket
import subprocess
import sys
import time

from kwiver.vital.algo import TrainDetector

import scriptconfig as scfg
import ubelt as ub

from viame.pytorch.utilities import (
    report_cuda_errors,
    vital_config_update,
    resolve_device_str,
    parse_bool,
    parse_resolution,
    format_resolution,
    register_vital_algorithm,
    ddp_available,
    find_python_interpreter,
    parse_temporal_offsets,
    temporal_offsets_to_frames,
)
from viame.pytorch.ptseformer_common import (
    SUPPORTED_BACKBONES,
    default_arch,
    natural_sort_key,
    parse_reference_window,
    reference_window_to_frames,
    sequence_frame_rate,
)


class PTSEFormerTrainerConfig(scfg.DataConfig):
    """
    The configuration for :class:`PTSEFormerTrainer`.
    """
    identifier = "viame-ptseformer-detector"
    train_directory = "deep_training"
    seed_model = scfg.Value('', help=(
        'Local weights to start from; nothing is downloaded. Accepts a '
        'PTSEFormer checkpoint (the released ImageNet-VID model or an earlier '
        'run), a Deformable-DETR checkpoint, or a torchvision ImageNet ResNet '
        'state dict (backbone only). Tensors whose shape does not match, such '
        'as the class head, stay at their initial values. Empty trains from '
        'random initialisation, which is only useful for testing.'))

    backbone = scfg.Value('resnet101', help=(
        'torchvision backbone: ' + ', '.join(SUPPORTED_BACKBONES) + '. Must '
        'match the seed_model when that carries more than a backbone.'))
    num_queries = scfg.Value(100, help=(
        'Object queries per frame. The decoder sees this many for the key '
        'frame plus this many for every reference frame.'))
    resolution = scfg.Value('960x1728', help=(
        'Network input as HEIGHTxWIDTH (or one number for a square). Every '
        'frame is resized to exactly this size; upstream instead keeps the '
        'aspect ratio at a short side of 800.'))

    temporal_offsets = scfg.Value('', help=(
        'Fixed reference frames: comma-separated offsets from each training '
        'frame, in frames ("-1,-2") or seconds with an "s" suffix '
        '("-1s,-2s", converted per sequence with its frame rate, see '
        'temporal_fallback_rate). Empty uses PTSEFormer\'s own sampling: '
        'num_reference_frames drawn at random from reference_window for each '
        'training sample. References are found by sorted file name in the '
        'training frame\'s own folder, so frames must be on disk in temporal '
        'order and windowing must be disabled; at a sequence edge the nearest '
        'frame is used. The detector only has past frames, so use negative '
        'offsets for a model that will be deployed.'))
    reference_window = scfg.Value('-12,0', help=(
        'Offset range "min,max" the random references are drawn from, in '
        'frames or seconds ("-0.6s,0s"). Upstream uses -12,12 on 25-30 fps '
        'video; the default here keeps its past half. Validation and the '
        'detector spread num_reference_frames evenly over the past part of '
        'this window. Unused when temporal_offsets is set.'))
    num_reference_frames = scfg.Value(2, help=(
        'Reference frames per training sample when temporal_offsets is empty. '
        'Memory and time grow with each one.'))
    temporal_fallback_rate = scfg.Value(5.0, help=(
        'Frame rate (frames/s) used to turn second offsets into frame offsets '
        'for a sequence with no known rate. Frames extracted from a video use '
        'the extraction rate; an image folder uses the fps in its groundtruth '
        'csv metadata when present; anything else uses this value.'))
    reference_grad = scfg.Value(False, help=(
        'Backpropagate through the backbone and encoder for the reference '
        'frames as well as the key frame, as upstream does. Off, they learn '
        'through the key frame alone and memory stays near the single-frame '
        'cost; the enhancement stages are trained either way.'))
    gradient_checkpointing = scfg.Value(True, help=(
        'Recompute the temporal enhancement stages in the backward pass '
        'instead of storing their activations. Same gradients, roughly half '
        'the training memory, somewhat slower.'))
    amp = scfg.Value(False, help=(
        'Mixed-precision training. Saves memory and time, but upstream was '
        'trained in full precision and this path is less tested.'))

    device = scfg.Value('auto', help='Device to train on: auto, cpu, cuda, or cuda:N')
    devices = scfg.Value('auto', help=(
        'GPUs to train on: "auto" uses every visible GPU (one process per GPU, '
        'gradients synchronised with DistributedDataParallel), a number uses '
        'that many, and a comma list names them ("cuda:0,cuda:2"). Pinning '
        'device = cuda:N uses that one GPU.'))

    max_epochs = scfg.Value(20, help='Number of training epochs')
    batch_size = scfg.Value(8, help=(
        'Frames per optimizer step, over all GPUs. The network takes one key '
        'frame per forward pass, so this is reached by accumulating gradients '
        'and does not change memory. Upstream trains at 8 (8 GPUs x 1).'))
    learning_rate = scfg.Value(1e-4, help='Learning rate (upstream r101: 1e-4)')
    learning_rate_backbone = scfg.Value(1e-5, help='Backbone learning rate')
    lr_linear_proj_mult = scfg.Value(0.1, help=(
        'Learning-rate factor for the deformable-attention sampling offsets '
        'and the query reference points.'))
    lr_drop = scfg.Value(0, help=(
        'Epoch at which the learning rate drops by 10x. 0 places it at 80% '
        'of max_epochs, the proportion upstream uses (40 of 50).'))
    weight_decay = scfg.Value(1e-4, help='AdamW weight decay')
    clip_max_norm = scfg.Value(0.1, help='Gradient-norm clip (0 disables)')
    horizontal_flip = scfg.Value(0.5, help=(
        'Probability of mirroring a training sample (key and reference frames '
        'together). Upstream trains without any augmentation; 0 matches it.'))

    num_workers = scfg.Value(4, help='DataLoader workers per GPU')
    val_subsample = scfg.Value(2000, help=(
        'Cap on the validation frames scored after each epoch, taken evenly '
        'across the validation set. 0 scores all of them.'))
    eval_interval = scfg.Value(1, help='Validate every N epochs')
    eval_max_dets = scfg.Value(100, help='Detections per frame kept for validation AP')
    print_interval = scfg.Value(50, help='Training steps between progress lines')
    seed = scfg.Value(42, help='Random seed')
    resume = scfg.Value(False, help=(
        'Continue from the last epoch checkpoint in train_directory when one '
        'exists, instead of starting from seed_model.'))
    timeout = scfg.Value(1209600, help=(
        'Max training time in seconds (default=1209600, two weeks); the best '
        'checkpoint so far is kept. 0 disables.'))

    def __post_init__(self):
        super().__post_init__()


class PTSEFormerTrainer(TrainDetector):
    """
    Implementation of TrainDetector for PTSEFormer (ECCV 2022), a video object
    detector that enhances a Deformable-DETR key frame with reference frames.
    """
    def __init__(self):
        TrainDetector.__init__(self)
        self._config = PTSEFormerTrainerConfig()
        self._class_names = []
        self._train_image_files = []
        self._train_detections = []
        self._test_image_files = []
        self._test_detections = []

    def get_configuration(self):
        cfg = super(TrainDetector, self).get_configuration()
        for key, value in self._config.items():
            cfg.set_value(key, str(value))
        return cfg

    @report_cuda_errors("PTSEFormerTrainer initialization")
    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        vital_config_update(cfg, cfg_in)
        for key in self._config.keys():
            self._config[key] = str(cfg.get_value(key))
        self._config.__post_init__()

        for key, value in self._config.items():
            setattr(self, "_" + key, value)

        if self._train_directory is not None:
            if not os.path.exists(self._train_directory):
                os.mkdir(self._train_directory)

        return True

    def check_configuration(self, cfg):
        if not cfg.has_value("identifier") or len(cfg.get_value("identifier")) == 0:
            print("A model identifier must be specified!")
            return False
        return True

    def add_data_from_disk(self, categories, train_files, train_dets,
                           test_files, test_dets):
        print("[PTSEFormerTrainer] Adding training data from disk")
        print("  Training files: ", len(train_files))
        print("  Training detections: ", len(train_dets))
        print("  Test files: ", len(test_files))
        print("  Test detections: ", len(test_dets))

        if categories is not None:
            self._class_names = categories.all_class_names()
        else:
            self._class_names = []

        self._train_image_files = list(train_files)
        self._train_detections = list(train_dets)
        self._test_image_files = list(test_files)
        self._test_detections = list(test_dets)

    def _reference_spec(self):
        """Parsed (temporal_offsets, reference_window, num_reference_frames)."""
        offsets = parse_temporal_offsets(self._temporal_offsets)
        window = parse_reference_window(self._reference_window)
        count = len(offsets) if offsets else int(self._num_reference_frames)
        if count < 1:
            raise ValueError(
                "PTSEFormer needs at least one reference frame: set "
                "num_reference_frames >= 1 or temporal_offsets")
        return offsets, window, count

    def _collect_split(self, image_files, detection_sets, sequences, folder_index):
        """One split's frames with their boxes and the sequence each frame sits in."""
        offsets, window, _ = self._reference_spec()
        uses_seconds = any(isinstance(v, str) for v in list(offsets) + list(window))
        cat_name_to_id = {name: idx for idx, name in enumerate(self._class_names)}
        stats = {"in_sequence": 0, "boxes": 0}
        images = []

        def sequence_of(img_path):
            """(sequence id, position) of a frame among its folder's sorted frames."""
            folder, name = os.path.split(img_path)
            if folder not in folder_index:
                ext = os.path.splitext(name)[1].lower()
                try:
                    names = sorted(
                        (n for n in os.listdir(folder)
                         if os.path.splitext(n)[1].lower() == ext),
                        key=natural_sort_key)
                except OSError:
                    names = []
                rate, source = (sequence_frame_rate(folder) if uses_seconds
                                else (None, "frames"))
                if rate is None and uses_seconds:
                    rate, source = float(self._temporal_fallback_rate), "fallback"
                self._rate_sources[source] = self._rate_sources.get(source, 0) + 1
                entry = {"folder": folder, "frames": names}
                if offsets:
                    entry["offsets"] = temporal_offsets_to_frames(offsets, rate or 1.0)
                else:
                    entry["window"] = list(reference_window_to_frames(window, rate or 1.0))
                sequences.append(entry)
                folder_index[folder] = (
                    len(sequences) - 1, {n: i for i, n in enumerate(names)})
            seq_id, positions = folder_index[folder]
            return (seq_id, positions[name]) if name in positions else (-1, 0)

        for img_idx in range(min(len(image_files), len(detection_sets))):
            abs_path = os.path.abspath(str(image_files[img_idx]))
            det_set = detection_sets[img_idx]
            seq_id, position = sequence_of(abs_path)
            stats["in_sequence"] += seq_id >= 0

            boxes = []
            for det in (det_set if det_set is not None else []):
                bbox = det.bounding_box
                x1, y1 = bbox.min_x(), bbox.min_y()
                x2, y2 = bbox.max_x(), bbox.max_y()
                if x2 - x1 <= 0 or y2 - y1 <= 0 or det.type is None:
                    continue
                class_name = det.type.get_most_likely_class()
                if class_name not in cat_name_to_id:
                    continue
                boxes.append([x1, y1, x2, y2, cat_name_to_id[class_name]])
            stats["boxes"] += len(boxes)
            images.append({"file": abs_path, "seq": seq_id, "index": position,
                           "boxes": boxes})
        return images, stats

    def _prepare_dataset(self):
        """Write the train/validation JSON files the launcher reads."""
        dataset_dir = ub.Path(self._train_directory) / "ptseformer_dataset"
        dataset_dir.ensuredir()
        self._rate_sources = {}
        sequences, folder_index = [], {}

        train_images, stats = self._collect_split(
            self._train_image_files, self._train_detections, sequences, folder_index)
        print(f"[PTSEFormerTrainer] Training set: {len(train_images)} frames, "
              f"{stats['boxes']} boxes")
        if not train_images:
            raise RuntimeError("PTSEFormer training needs at least one training frame")

        offsets, window, count = self._reference_spec()
        described = (f"fixed offsets {offsets}" if offsets else
                     f"{count} random frame(s) from window {window}")
        print(f"[PTSEFormerTrainer] Reference frames: {described}; frame rates "
              f"from {self._rate_sources}")
        lengths = [len(seq["frames"]) for seq in sequences]
        if stats["in_sequence"] == 0 or max(lengths, default=0) < 2:
            print("[PTSEFormerTrainer] WARNING: no training frame has another "
                  "frame beside it on disk, so every reference is the key frame "
                  "itself. Keep windowing disabled and train from per-sequence "
                  "folders in temporal order.")

        val_images, _ = self._collect_split(
            self._test_image_files, self._test_detections, sequences, folder_index)
        print(f"[PTSEFormerTrainer] Validation set: {len(val_images)} frames; "
              f"{len(sequences)} sequence folder(s) in all")

        # Both files carry the full sequence table their frames index into.
        paths = []
        for name, images in (("train.json", train_images), ("val.json", val_images)):
            paths.append(str(dataset_dir / name) if images else '')
            if images:
                with open(paths[-1], 'w') as fh:
                    json.dump({"class_names": list(self._class_names),
                               "sequences": sequences, "images": images}, fh)
        return paths[0], paths[1]

    def _resolve_devices(self):
        """List of device strings, one training process each."""
        import torch

        # scriptconfig hands a comma list over as a list, not text.
        requested = self._devices
        if isinstance(requested, (list, tuple)):
            requested = ','.join(str(d) for d in requested)
        requested = str(requested).strip().lower()
        if ',' in requested or requested.startswith('cuda:'):
            devices = [d.strip() for d in requested.split(',') if d.strip()]
        else:
            device = resolve_device_str(self._device)
            if device.startswith('cpu'):
                return ['cpu']
            if ':' in device:
                return [device]
            count = torch.cuda.device_count()
            if requested not in ('auto', ''):
                count = min(count, max(1, int(requested)))
            devices = [f"cuda:{i}" for i in range(max(1, count))]
        if len(devices) > 1 and not ddp_available():
            print(f"[PTSEFormerTrainer] {len(devices)} GPUs requested, but this "
                  "PyTorch build cannot run distributed training on GPUs. "
                  "Training on one GPU.", flush=True)
            devices = devices[:1]
        return devices

    def _subprocess_env(self):
        """Environment that lets a plain interpreter import what this process does."""
        env = dict(os.environ)
        # Vendored trees stay out; see the same step in rf_detr_trainer.py.
        extra_paths = [p for p in sys.path
                       if "_vendor" not in p.replace("\\", "/").split("/")]
        for mod_name in ("torch", "torchvision", "viame"):
            try:
                mod = __import__(mod_name)
                extra_paths.append(
                    os.path.dirname(os.path.dirname(os.path.abspath(mod.__file__))))
            except Exception:
                pass
        if env.get("PYTHONPATH"):
            extra_paths.append(env["PYTHONPATH"])
        ordered = []
        for p in extra_paths:
            if p and p not in ordered:
                ordered.append(p)
        env["PYTHONPATH"] = os.pathsep.join(ordered)
        return env

    def _launch(self, params, params_path, devices):
        """Run the launcher: one subprocess per device, or in-process as a fallback."""
        from viame.pytorch import ptseformer_launcher as launcher

        python = find_python_interpreter()
        if not python:
            if len(devices) > 1:
                raise RuntimeError(
                    "Multi-GPU PTSEFormer training needs a standalone Python "
                    "interpreter to launch one process per GPU, and none was "
                    "found. Pin a single GPU with device = cuda:0.")
            print("[PTSEFormerTrainer] No standalone interpreter found; "
                  "training in-process")
            launcher.train(params)
            return

        env = self._subprocess_env()
        env[launcher.WORLD_ENV] = str(len(devices))
        if len(devices) > 1:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                env[launcher.PORT_ENV] = str(sock.getsockname()[1])
        script = os.path.abspath(launcher.__file__)
        print(f"[PTSEFormerTrainer] Launching training on {', '.join(devices)}: "
              f"{python} {script}", flush=True)
        procs = []
        for rank in range(len(devices)):
            rank_env = dict(env)
            rank_env[launcher.RANK_ENV] = str(rank)
            procs.append(subprocess.Popen([python, script, params_path], env=rank_env))

        failed = None
        try:
            while failed is None and any(p.poll() is None for p in procs):
                time.sleep(1.0)
                failed = next((p for p in procs if p.poll() not in (None, 0)), None)
        except KeyboardInterrupt:
            # The ranks received the same signal and are saving; wait for them.
            for proc in procs:
                proc.wait()
        failed = failed or next((p for p in procs if p.poll() not in (None, 0)), None)
        if failed is not None:
            for proc in procs:
                if proc.poll() is None:
                    proc.terminate()
            if failed.returncode == -9:
                raise RuntimeError(
                    "A PTSEFormer training process was killed with SIGKILL, "
                    "which usually means the host (not the GPU) ran out of "
                    "memory. Lower num_workers or ask for more memory.")
            raise RuntimeError(
                f"PTSEFormer training failed (exit code {failed.returncode}); "
                "the output above carries the reason.")

    @report_cuda_errors("PTSEFormerTrainer training")
    def update_model(self):
        print("[PTSEFormerTrainer] Starting PTSEFormer training")

        backbone = str(self._backbone).strip().lower()
        if backbone not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"backbone = {self._backbone!r} is not supported; use one of "
                f"{', '.join(SUPPORTED_BACKBONES)}")
        resolution = parse_resolution(self._resolution)
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        if min(resolution) < 64:
            raise ValueError(f"resolution = {self._resolution!r} is too small")

        seed_model = str(self._seed_model).strip()
        if seed_model and not os.path.isfile(seed_model):
            raise FileNotFoundError(
                f"seed_model = {seed_model} does not exist. PTSEFormer weights "
                "are not downloaded at training time; see the PTSEFormer "
                "section of examples/object_detector_training/README.rst.")
        if not seed_model:
            print("[PTSEFormerTrainer] WARNING: no seed_model; training from "
                  "random initialisation")

        train_json, val_json = self._prepare_dataset()
        offsets, window, num_refs = self._reference_spec()
        devices = self._resolve_devices()

        arch = default_arch()
        arch.update(backbone=backbone, num_queries=int(self._num_queries),
                    num_classes=max(1, len(self._class_names)))

        output_dir = ub.Path(self._train_directory) / "ptseformer_output"
        output_dir.ensuredir()
        resume_from = ''
        if parse_bool(self._resume) and (output_dir / "checkpoint.pth").exists():
            resume_from = str(output_dir / "checkpoint.pth")

        epochs = int(self._max_epochs)
        lr_drop = int(self._lr_drop) or max(1, int(round(0.8 * epochs)))
        batch_size = int(self._batch_size)
        per_gpu = max(1, int(round(batch_size / float(len(devices)))))
        print(f"[PTSEFormerTrainer] {len(devices)} device(s), {per_gpu} frame(s) "
              f"accumulated per device per step (effective batch "
              f"{per_gpu * len(devices)}), lr drop at epoch {lr_drop}")

        params = dict(
            train_json=train_json,
            val_json=val_json,
            output_dir=str(output_dir),
            devices=devices,
            arch=arch,
            resolution=list(resolution),
            num_reference_frames=num_refs,
            seed_model=seed_model,
            resume_from=resume_from,
            reference_grad=parse_bool(self._reference_grad),
            gradient_checkpointing=parse_bool(self._gradient_checkpointing),
            amp=parse_bool(self._amp),
            epochs=epochs,
            batch_size=per_gpu,
            lr=float(self._learning_rate),
            lr_backbone=float(self._learning_rate_backbone),
            lr_linear_proj_mult=float(self._lr_linear_proj_mult),
            lr_drop=lr_drop,
            weight_decay=float(self._weight_decay),
            clip_max_norm=float(self._clip_max_norm),
            horizontal_flip=float(self._horizontal_flip),
            num_workers=int(self._num_workers),
            val_subsample=int(self._val_subsample),
            eval_interval=int(self._eval_interval),
            eval_max_dets=int(self._eval_max_dets),
            print_interval=int(self._print_interval),
            seed=int(self._seed),
            timeout=float(self._timeout),
            # Stored in every checkpoint; the detector rebuilds the model from it.
            checkpoint_args=dict(
                arch=arch,
                class_names=list(self._class_names),
                resolution=list(resolution),
                temporal_offsets=offsets,
                reference_window=window,
                num_reference_frames=num_refs,
            ),
        )
        params_path = os.path.join(self._train_directory, "ptseformer_params.json")
        with open(params_path, 'w') as fh:
            json.dump(params, fh, indent=1)

        self._launch(params, params_path, devices)

        output = self._get_output_map(output_dir)
        print("\n[PTSEFormerTrainer] Model training complete!")
        return output

    def _get_output_map(self, output_dir):
        """Template replacements and file copies for the generated detector pipeline."""
        output_model_name = "trained_detector.pth"
        final_ckpt = ub.Path(output_dir) / "checkpoint_best.pth"
        if not final_ckpt.exists():
            raise RuntimeError(
                f"PTSEFormer training produced no checkpoint in {output_dir}. "
                "The trainer output above carries the reason; no model was written.")

        algo = "ptseformer"
        output = {}
        output["type"] = algo
        output[algo + ":weight"] = output_model_name
        output[algo + ":resolution"] = format_resolution(
            parse_resolution(self._resolution))

        # File copy entry (key=destination filename, value=source path)
        output[output_model_name] = str(final_ckpt)

        print(f"\nModel found at: {final_ckpt}")
        print(f"\nThe {self._train_directory} directory can now be deleted, "
              "unless you want to review training metrics first.")
        return output


def __vital_algorithm_register__():
    register_vital_algorithm(
        PTSEFormerTrainer, "ptseformer", "PyTorch PTSEFormer video detection training routine"
    )
