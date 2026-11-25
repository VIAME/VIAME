# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import json
import os
import re
import signal
import subprocess
import sys

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
    resolution_is_set,
    register_vital_algorithm,
    TrainingInterruptHandler,
    find_python_interpreter,
    parse_temporal_offsets,
    temporal_offsets_to_frames,
)
from viame.pytorch.yolov_common import (
    MODEL_SIZES,
    SEED_URLS,
    natural_sort_key,
    video_head_config,
)


def sequence_frame_rate(folder):
    """(frames/s, source) for a folder of frames, or (None, None) when unknown.

    Same sources as the RF-DETR trainer: a "<folder>_frame_rate.txt" sidecar
    written at extraction, else "fps: N" in a groundtruth csv header.
    """
    folder = folder.rstrip(os.sep)
    try:
        with open(folder + "_frame_rate.txt") as fh:
            rate = float(fh.read().split()[0])
        if rate > 0:
            return rate, "extraction"
    except (OSError, ValueError, IndexError):
        pass
    try:
        csvs = [n for n in os.listdir(folder) if n.lower().endswith('.csv')]
    except OSError:
        csvs = []
    for csv_name in sorted(csvs):
        try:
            with open(os.path.join(folder, csv_name)) as fh:
                for line in fh:
                    if not line.startswith('#'):
                        break
                    match = re.search(r'fps[:=]\s*([0-9.]+)', line)
                    if match and float(match.group(1)) > 0:
                        return float(match.group(1)), "csv metadata"
        except (OSError, ValueError):
            continue
    return None, None


class YOLOVTrainerConfig(scfg.DataConfig):
    """
    The configuration for :class:`YOLOVTrainer`.
    """
    identifier = "viame-yolov-detector"
    train_directory = "deep_training"
    seed_model = scfg.Value('', help=(
        'Local path of the COCO-pretrained YOLOX weights (yolox_s/m/l/x.pth, '
        'matching model_size) the base detector is fine-tuned from. Nothing '
        'is downloaded at train time; empty trains the base from random '
        'weights, which upstream reports as much worse.'))
    base_model = scfg.Value('', help=(
        'An already fine-tuned YOLOX base detector for these classes (the '
        'base_best.pth of an earlier run, or an upstream YOLOX checkpoint). '
        'When set, stage 1 is skipped and only the YOLOV++ head is trained.'))

    # Model
    model_size = scfg.Value('l', help=(
        'YOLOX variant the detector is built on: s, m, l or x. Upstream '
        'publishes YOLOV++ configurations for s and l.'))
    device = scfg.Value('auto', help='Device to train on: auto, cuda, or cuda:N (one GPU)')
    resolution = scfg.Value('640', help=(
        'Network input: a square side ("640") or height x width ("960x1728"). '
        'Frames are letterboxed into it, and each dim is rounded to a multiple '
        'of 32.'))
    temporal_offsets = scfg.Value('', help=(
        'Reference frames the YOLOV++ head aggregates each frame with. Empty '
        'uses the method\'s own sampling: random labelled frames of the same '
        'sequence in training, a random draw from the frames seen so far at '
        'inference. Otherwise comma-separated offsets relative to each frame, '
        'in frames ("-1,-2") or seconds ("-1s,-2s", converted per sequence '
        'with its frame rate, see temporal_fallback_rate); the frames are '
        'found by sorted file name in the training frame\'s folder and need '
        'no labels. Use negative offsets: the detector only has past frames.'))
    temporal_fallback_rate = scfg.Value(5.0, help=(
        'Frame rate (frames/s) used to turn second offsets into frame offsets '
        'for a sequence with no known rate.'))
    reference_frames = scfg.Value(15, help=(
        'Past frames the detector aggregates each frame with when '
        'temporal_offsets is empty. Recorded in the model for the detector.'))
    memory_frames = scfg.Value(120, help=(
        'Past frames the detector keeps to draw reference_frames from when '
        'temporal_offsets is empty. Recorded in the model for the detector.'))
    min_proposals = scfg.Value(-1, help=(
        'Proposals every frame hands the aggregation head at least. -1 keeps '
        'the upstream value (50 for l/x, 0 for s/m).'))
    max_proposals = scfg.Value(200, help=(
        'Cap on the proposals per frame handed to the aggregation head, by '
        'score. Attention memory grows with the square of proposals x frames, '
        'and upstream leaves it unbounded (0), which does not fit crowded '
        'high-resolution frames. Also caps the detections per frame.'))

    # Schedule
    base_epochs = scfg.Value(14, help='Epochs of stage 1, the YOLOX base detector')
    video_epochs = scfg.Value(6, help=(
        'Epochs of stage 2, the YOLOV++ head on the frozen base. 0 stops '
        'after stage 1 and outputs the single-frame base detector.'))
    base_batch_size = scfg.Value(4, help='Frames per batch in stage 1')
    video_frames_per_batch = scfg.Value(16, help=(
        'Frames per batch in stage 2. With temporal_offsets empty they are '
        'all from one sequence and aggregated together; with offsets a batch '
        'holds as many frame-plus-references groups as fit.'))
    base_lr_per_image = scfg.Value(0.01 / 64.0, help=(
        'Stage 1 learning rate per image; multiplied by base_batch_size.'))
    video_lr_per_image = scfg.Value(0.002 / 64.0, help=(
        'Stage 2 learning rate per image; multiplied by video_frames_per_batch.'))
    base_warmup_epochs = scfg.Value(1.0, help='Stage 1 warmup epochs')
    video_warmup_epochs = scfg.Value(0.0, help='Stage 2 warmup epochs')
    base_no_aug_epochs = scfg.Value(2, help=(
        'Final stage 1 epochs without mosaic/mixup and with the L1 box loss.'))
    video_no_aug_epochs = scfg.Value(2, help=(
        'Final stage 2 epochs held at the minimum learning rate.'))
    weight_decay = scfg.Value(5e-4, help='SGD weight decay')
    momentum = scfg.Value(0.9, help='SGD momentum')
    use_ema = scfg.Value(True, help='Keep and export an exponential moving average of the weights')
    amp = scfg.Value(True, help='Mixed-precision training')
    freeze_bn = scfg.Value(True, help=(
        'Keep the BatchNorm statistics of the frozen base detector fixed in '
        'stage 2, so the base inside the video model stays the stage 1 model. '
        'Upstream lets them drift on the single-sequence batches (false).'))

    # Augmentation
    mosaic_prob = scfg.Value(1.0, help='Stage 1 mosaic probability')
    mixup_prob = scfg.Value(1.0, help='Stage 1 mixup probability (applied after mosaic)')
    mosaic_scale = scfg.Value('0.1,2', help='Stage 1 mosaic random scale range "min,max"')
    degrees = scfg.Value(10.0, help='Stage 1 mosaic rotation range in degrees')
    translate = scfg.Value(0.1, help='Stage 1 mosaic translation range')
    shear = scfg.Value(2.0, help='Stage 1 mosaic shear range in degrees')
    hsv_prob = scfg.Value(1.0, help='HSV jitter probability (both stages)')
    flip_prob = scfg.Value(0.5, help='Horizontal flip probability (both stages)')
    base_multiscale_range = scfg.Value(5, help=(
        'Stage 1 multi-scale: every 10 iterations the batch is resized to the '
        'input height +- this many steps of 32 px. 0 disables.'))
    video_multiscale_range = scfg.Value(5, help='The same for stage 2')
    max_labels = scfg.Value(200, help='Groundtruth boxes kept per frame')

    # Validation
    eval_interval = scfg.Value(1, help='Run validation every N epochs')
    val_max_frames = scfg.Value(0, help=(
        'If > 0, validate on about this many frames: the leading part of '
        'every validation sequence, so the frames stay in temporal order.'))
    selection_metric = scfg.Value('ap50', help=(
        'Validation metric the best checkpoint of each stage is chosen by: '
        'ap50 or ap50_95.'))
    nms_thresh = scfg.Value(0.5, help='NMS IoU of the YOLOV++ output')
    base_nms_thresh = scfg.Value(0.65, help='NMS IoU of the base detector')

    num_workers = scfg.Value(4, help='DataLoader worker processes')
    print_interval = scfg.Value(10, help='Iterations between progress lines')
    random_seed = scfg.Value(0, help='Seed of the sampling and augmentation RNGs')

    # Timeout
    timeout = scfg.Value('1209600', help='Max training time in seconds (default=1209600, two weeks)')

    categories = []

    def __post_init__(self):
        super().__post_init__()


class YOLOVTrainer(TrainDetector):
    """
    Implementation of TrainDetector for YOLOV++ video object detectors.

    Runs both stages of the method in one call: the still-image YOLOX base
    detector, then the feature-aggregation head over frames of a sequence.
    """
    def __init__(self):
        TrainDetector.__init__(self)
        self._config = YOLOVTrainerConfig()
        self._class_names = []
        self._train_image_files = []
        self._train_detections = []
        self._test_image_files = []
        self._test_detections = []

    def get_configuration(self):
        print('[YOLOVTrainer] get_configuration')
        cfg = super(TrainDetector, self).get_configuration()
        for key, value in self._config.items():
            cfg.set_value(key, str(value))
        return cfg

    @report_cuda_errors("YOLOVTrainer initialization")
    def set_configuration(self, cfg_in):
        print('[YOLOVTrainer] set_configuration')
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
        print("[YOLOVTrainer] Adding training data from disk")
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

    def _input_size(self):
        """(height, width) of the network input, each a multiple of 32."""
        resolution = parse_resolution(self._resolution)
        if not resolution_is_set(resolution):
            resolution = 640
        height, width = (resolution if isinstance(resolution, (list, tuple))
                         else (resolution, resolution))
        snapped = [max(32, int(round(dim / 32.0)) * 32) for dim in (height, width)]
        if snapped != [height, width]:
            print(f"[YOLOVTrainer] resolution {format_resolution(resolution)} is "
                  f"not a multiple of 32; training at {snapped[0]}x{snapped[1]}")
        return snapped

    def _prepare_dataset(self):
        """Write the frames, boxes and sequence layout the launcher trains from.

        Returns (path of the dataset json, parsed temporal offsets).
        """
        from PIL import Image

        class_names = list(self._class_names)
        cat_name_to_id = {name: idx for idx, name in enumerate(class_names)}
        unreadable = []

        def build_records(image_files, detection_sets):
            records = []
            num_boxes = 0
            for img_idx in range(min(len(image_files), len(detection_sets))):
                img_path = os.path.abspath(str(image_files[img_idx]))
                try:
                    with Image.open(img_path) as im:
                        width, height = im.size
                except Exception as exc:
                    unreadable.append((img_path, f"{type(exc).__name__}: {exc}"))
                    continue

                boxes = []
                for det in (detection_sets[img_idx] or []):
                    bbox = det.bounding_box
                    x1, y1 = max(0.0, bbox.min_x()), max(0.0, bbox.min_y())
                    x2, y2 = min(width, bbox.max_x()), min(height, bbox.max_y())
                    if x2 - x1 <= 0 or y2 - y1 <= 0 or det.type is None:
                        continue
                    class_name = det.type.get_most_likely_class()
                    if class_name not in cat_name_to_id:
                        continue
                    boxes.append([x1, y1, x2, y2, cat_name_to_id[class_name]])
                num_boxes += len(boxes)
                records.append({"file": img_path, "width": width,
                                "height": height, "boxes": boxes})
            return records, num_boxes

        train, n_train = build_records(self._train_image_files, self._train_detections)
        val, n_val = build_records(self._test_image_files, self._test_detections)
        print(f"[YOLOVTrainer] Train: {len(train)} frames, {n_train} boxes in "
              f"{len({os.path.dirname(r['file']) for r in train})} sequence folders")
        print(f"[YOLOVTrainer] Valid: {len(val)} frames, {n_val} boxes in "
              f"{len({os.path.dirname(r['file']) for r in val})} sequence folders")
        if not train or n_train == 0:
            raise RuntimeError("No training frame with a usable box was provided.")

        # Fixed-offset references: list every frame of each folder, in order.
        temporal_offsets = parse_temporal_offsets(self._temporal_offsets)
        sequences = {}
        if temporal_offsets:
            uses_seconds = any(isinstance(off, str) for off in temporal_offsets)
            rate_sources = {}
            for record in train + val:
                folder, name = os.path.split(record["file"])
                if folder in sequences:
                    continue
                ext = os.path.splitext(name)[1].lower()
                names = sorted(
                    (n for n in os.listdir(folder)
                     if os.path.splitext(n)[1].lower() == ext),
                    key=natural_sort_key)
                rate, source = (sequence_frame_rate(folder) if uses_seconds
                                else (None, "frames"))
                if rate is None and uses_seconds:
                    rate, source = float(self._temporal_fallback_rate), "fallback"
                rate_sources[source] = rate_sources.get(source, 0) + 1
                sequences[folder] = {
                    "names": names,
                    "offsets": temporal_offsets_to_frames(temporal_offsets, rate or 1.0)}

            found = 0
            for record in train:
                folder, name = os.path.split(record["file"])
                info = sequences[folder]
                pos = info["names"].index(name) if name in info["names"] else -1
                found += sum(1 for off in info["offsets"]
                             if pos >= 0 and 0 <= pos + off < len(info["names"]))
            print(f"[YOLOVTrainer] Reference offsets {temporal_offsets}: {found} "
                  f"reference frame(s) found beside {len(train)} training frames"
                  + (f"; frame rate from {rate_sources}" if uses_seconds else ""))
            if found == 0:
                raise RuntimeError(
                    "temporal_offsets is set but no training frame has a "
                    "neighbouring frame beside it on disk. The frames must sit "
                    "in per-sequence folders in temporal order.")

        if unreadable:
            report = ub.Path(self._train_directory) / "unreadable_images.txt"
            with open(report, "w") as f:
                for path, reason in unreadable:
                    f.write(f"{path}\t{reason}\n")
            print(f"[YOLOVTrainer] WARNING: excluded {len(unreadable)} unreadable "
                  f"image(s) from training; full list in {report}")

        dataset_path = os.path.join(self._train_directory, "yolov_dataset.json")
        with open(dataset_path, "w") as f:
            json.dump({"classes": class_names, "train": train, "val": val,
                       "sequences": sequences}, f)
        return dataset_path, temporal_offsets

    def _launch_params(self, dataset_path, temporal_offsets, output_dir):
        model_size = str(self._model_size).strip().lower()
        if model_size not in MODEL_SIZES:
            raise ValueError(f"Unknown YOLOV model size: {self._model_size}. "
                             f"Expected one of: {', '.join(MODEL_SIZES)}")

        seed_model = str(self._seed_model).strip()
        base_model = str(self._base_model).strip()
        for label, path in (("seed_model", seed_model), ("base_model", base_model)):
            if path and not os.path.isfile(path):
                raise RuntimeError(
                    f"{label} = {path} does not exist. Weights are not downloaded "
                    f"at train time; the COCO-pretrained YOLOX-{model_size} "
                    f"weights are at {SEED_URLS[model_size]}")
        if not seed_model and not base_model:
            print(f"[YOLOVTrainer] WARNING: no seed_model, so the base detector "
                  f"trains from random weights. COCO-pretrained YOLOX-{model_size} "
                  f"weights: {SEED_URLS[model_size]}")

        metric = str(self._selection_metric).strip().lower()
        if metric not in ('ap50', 'ap50_95'):
            raise ValueError(f"selection_metric must be ap50 or ap50_95, not {metric}")

        input_size = self._input_size()
        head_config = video_head_config(
            model_size, int(self._min_proposals), int(self._max_proposals))
        mosaic_scale = [float(v) for v in str(self._mosaic_scale).strip('()[] ').split(',')]

        # Everything the detector needs to rebuild and run the model.
        meta = {
            "model_size": model_size,
            "class_names": list(self._class_names),
            "input_size": input_size,
            "temporal_offsets": temporal_offsets,
            "reference_frames": int(self._reference_frames),
            "memory_frames": int(self._memory_frames),
            "nms_thresh": float(self._nms_thresh),
            "base_nms_thresh": float(self._base_nms_thresh),
        }
        return {
            "dataset": os.path.abspath(dataset_path),
            "output_dir": os.path.abspath(str(output_dir)),
            "model_size": model_size,
            "class_names": list(self._class_names),
            "input_size": input_size,
            "seed_model": os.path.abspath(seed_model) if seed_model else "",
            "base_model": os.path.abspath(base_model) if base_model else "",
            "device": resolve_device_str(self._device),
            "head_config": head_config,
            "meta": meta,
            "temporal_offsets": temporal_offsets,
            "reference_frames": int(self._reference_frames),
            "memory_frames": int(self._memory_frames),
            "nms_thresh": float(self._nms_thresh),
            "base_nms_thresh": float(self._base_nms_thresh),
            "momentum": float(self._momentum),
            "weight_decay": float(self._weight_decay),
            "ema": parse_bool(self._use_ema),
            "amp": parse_bool(self._amp),
            "freeze_bn": parse_bool(self._freeze_bn),
            "hsv_prob": float(self._hsv_prob),
            "flip_prob": float(self._flip_prob),
            "max_labels": int(self._max_labels),
            "eval_interval": max(1, int(self._eval_interval)),
            "val_max_frames": int(self._val_max_frames),
            "selection_metric": metric,
            "num_workers": int(self._num_workers),
            "print_interval": max(1, int(self._print_interval)),
            "seed": int(self._random_seed),
            "timeout": float(self._timeout),
            "base": {
                "epochs": int(self._base_epochs),
                "batch_size": int(self._base_batch_size),
                "lr_per_image": float(self._base_lr_per_image),
                "warmup_epochs": float(self._base_warmup_epochs),
                "no_aug_epochs": int(self._base_no_aug_epochs),
                "min_lr_ratio": 0.05,
                "multiscale_range": int(self._base_multiscale_range),
                "mosaic_prob": float(self._mosaic_prob),
                "mixup_prob": float(self._mixup_prob),
                "mosaic_scale": mosaic_scale,
                "degrees": float(self._degrees),
                "translate": float(self._translate),
                "shear": float(self._shear),
            },
            "video": {
                "epochs": int(self._video_epochs),
                "frames_per_batch": int(self._video_frames_per_batch),
                "lr_per_image": float(self._video_lr_per_image),
                "warmup_epochs": float(self._video_warmup_epochs),
                "no_aug_epochs": int(self._video_no_aug_epochs),
                "min_lr_ratio": 0.1,
                "multiscale_range": int(self._video_multiscale_range),
            },
        }

    @report_cuda_errors("YOLOVTrainer training")
    def update_model(self):
        dataset_path, temporal_offsets = self._prepare_dataset()

        output_dir = ub.Path(self._train_directory) / "yolov_output"
        output_dir.ensuredir()
        # A stale result would be packaged if this run died before writing its own.
        (output_dir / "result.json").delete()

        params = self._launch_params(dataset_path, temporal_offsets, output_dir)
        params_path = os.path.join(self._train_directory, "yolov_params.json")
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=2)

        # A plain child process: DataLoader workers and two model builds stay out
        # of the embedded interpreter, and the GPU is released between stages.
        python = find_python_interpreter()
        if not python:
            raise RuntimeError(
                "YOLOV++ training needs a standalone Python interpreter to "
                "launch its training process, and none was found beside this "
                "install or on PATH.")
        impl = os.path.join(os.path.dirname(__file__), "yolov_launcher.py")

        # Hand the child the import paths of this process, minus vendored trees
        # (see rf_detr_trainer.py for why those must not be propagated).
        env = dict(os.environ)
        extra_paths = [p for p in sys.path
                       if "_vendor" not in p.replace("\\", "/").split("/")]
        for mod_name in ("torch", "torchvision", "yolox", "viame"):
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

        print(f"[YOLOVTrainer] Launching training: {python} {impl}", flush=True)
        proc = subprocess.Popen([python, impl, params_path], env=env)

        def stop_child():
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)

        with TrainingInterruptHandler("YOLOVTrainer", on_interrupt=stop_child) as handler:
            returncode = proc.wait()

        if returncode != 0 and not handler.interrupted:
            raise RuntimeError(
                f"YOLOV++ training exited with code {returncode}. The trainer "
                f"output above carries the reason; no model was written.")

        output = self._get_output_map(output_dir)
        print("\n[YOLOVTrainer] Model training complete!")
        return output

    def _get_output_map(self, output_dir):
        """
        Build and return output map containing template replacements and
        file copies. The C++ process_trainer_output handles copying files
        to the output directory and generating the pipeline from a template.
        """
        output_dir = ub.Path(output_dir)
        final_ckpt = None

        result_path = output_dir / "result.json"
        if result_path.exists():
            with open(result_path) as f:
                final_ckpt = json.load(f).get("final")

        # An interrupted run leaves no result file; use the furthest stage on disk.
        if not final_ckpt:
            for name in ("video_best.pth", "base_best.pth"):
                if (output_dir / name).exists():
                    final_ckpt = str(output_dir / name)
                    print(f"[YOLOVTrainer] Training did not finish; using {name}")
                    break

        if not final_ckpt or not os.path.exists(final_ckpt):
            raise RuntimeError(
                "YOLOV++ training produced no checkpoint in {}. The trainer "
                "output above carries the reason; no model was written.".format(
                    output_dir))

        algo = "yolov"
        output_model_name = "trained_detector.pth"

        output = {}
        output["type"] = algo

        # Config key matching the yolov detector; the checkpoint itself carries
        # the architecture, classes, input size and reference-frame settings.
        output[algo + ":weight"] = output_model_name

        # File copy entry (key=destination filename, value=source path)
        output[output_model_name] = str(final_ckpt)

        print(f"\nModel found at: {final_ckpt}")
        print(f"\nThe {self._train_directory} directory can now be deleted, "
              "unless you want to review training metrics first.")

        return output


def __vital_algorithm_register__():
    register_vital_algorithm(
        YOLOVTrainer, "yolov", "PyTorch YOLOV++ video detection training routine"
    )
