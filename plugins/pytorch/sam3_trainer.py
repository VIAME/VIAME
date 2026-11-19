# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
SAM 3.1 fine-tuning trainers.

SAM 3.1 is two networks sharing one vision backbone: a detector, which
finds every instance of a text query in a frame, and a multiplex tracker,
which carries each object's mask from frame to frame. Both trainers keep
the backbone frozen, so either network can be fine-tuned without
invalidating the other, and both write a full SAM 3.1 checkpoint that drops
in wherever sam3.1_weights.pt is used.

1. SAM3Trainer (train_detector, "sam3") - fine-tunes the detector on
   per-image annotations with SAM 3's own training objective (sam3.train:
   Hungarian matching, box/GIoU, presence-aware focal and mask losses). The
   engine is sam3_detector_training.py. The output pipeline runs the
   detector per image through the sam3 image_object_detector.

2. SAM3TrackerTrainer (train_tracker, "sam3") - fine-tunes the tracker on
   ground-truth tracks with the model's SAM2-style video objective (engine:
   sam3_tracker_training.py). With train_detector (the default) it first
   fine-tunes the detector on the same annotations, so the generated
   text-query tracking pipeline gets both. With train_detector = false only
   the tracker changes, for use with detections from another detector.
"""

import json
import os
import subprocess
import sys

from kwiver.vital.algo import TrainDetector, TrainTracker

from viame.pytorch.utilities import report_cuda_errors


def _resolve_model_path(path):
    """Resolve a shipped model file to an absolute path.

    Training runs from a per-model working directory, so a relative default
    like "models/sam3.1_weights.pt" will not resolve against the current
    directory. Fall back to the install tree, which is where VIAME puts the
    checkpoints that ship with it, and which is what the shipped pipelines
    reference.
    """
    if not path or os.path.isabs(path):
        return path

    if os.path.exists(path):
        return os.path.abspath(path)

    viame_install = os.environ.get("VIAME_INSTALL")

    if viame_install:
        candidate = os.path.join(viame_install, "configs", "pipelines", path)

        if os.path.exists(candidate):
            return candidate

    return path


def _parse_bool(value):
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _parse_timeout(value):
    value = str(value).strip().lower()
    return None if value in ("", "none", "default") else float(value)


def _detection_polygons(det):
    """A detection's polygons as flat [x, y, ...] lists (may be empty)."""
    try:
        return [list(map(float, p)) for p in det.get_flattened_polygons()
                if len(p) >= 6]
    except (AttributeError, RuntimeError):
        return []


def _run_engine(module_name, params, n_gpus, label):
    """
    Run a training engine: in-process on one GPU, else as torch.distributed
    ranks in a child interpreter (the engines import nothing from kwiver so
    they can run as plain scripts).
    """
    from viame.pytorch.utilities import ddp_available, find_python_interpreter

    if n_gpus > 1 and not ddp_available():
        print(f"[{label}] {n_gpus} GPUs requested, but this PyTorch build has "
              "no NCCL. Training on one GPU.")
        n_gpus = 1

    work = params["work_dir"]
    os.makedirs(work, exist_ok=True)
    params_path = os.path.join(work, "params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=1)

    if n_gpus <= 1:
        import importlib
        importlib.import_module("viame.pytorch." + module_name).run(params)
        return

    python = find_python_interpreter()
    if not python:
        raise RuntimeError(
            f"Multi-GPU {label} training needs a standalone Python "
            "interpreter to launch the DDP ranks, and none was found. Set "
            "gpu_count = 1 to train in-process.")

    impl = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        module_name + ".py")

    # Carry this interpreter's import paths (minus vendored trees, see
    # rf_detr_trainer) so the ranks import what this process does.
    env = dict(os.environ)
    paths = [p for p in sys.path
             if "_vendor" not in p.replace("\\", "/").split("/")]
    for mod_name in ("torch", "sam3", "viame"):
        try:
            mod = __import__(mod_name)
            paths.append(os.path.dirname(os.path.dirname(
                os.path.abspath(mod.__file__))))
        except Exception:
            pass
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    seen, ordered = set(), []
    for p in paths:
        if p and p not in seen:
            seen.add(p)
            ordered.append(p)
    env["PYTHONPATH"] = os.pathsep.join(ordered)

    cmd = [python, "-m", "torch.distributed.run", "--standalone",
           "--nproc_per_node", str(n_gpus), impl, params_path]
    print(f"[{label}] Launching {n_gpus}-GPU training: " + " ".join(cmd),
          flush=True)
    subprocess.run(cmd, check=True, env=env)


def _gpu_count(requested):
    try:
        import torch
        if torch.cuda.is_available() and requested < 0:
            requested = torch.cuda.device_count()
    except ImportError:
        pass
    return max(1, requested)


def _config_path(sam_model_id, name):
    """The 3.1 model config that ships next to the weights, if any."""
    path = os.path.join(os.path.dirname(sam_model_id), name)
    return path if os.path.exists(path) else None


# =============================================================================
# Detector
# =============================================================================

def _deploy_threshold(detector_work, default=0.3):
    """
    Detection threshold for a tuned detector: the one that maximized
    validation F1 (sam3_detector/best.json). Fine-tuning shifts scores well
    above the stock model's, so the stock 0.3 floods the output with false
    positives.
    """
    threshold = default
    best = os.path.join(detector_work, "best.json")
    if os.path.exists(best):
        with open(best) as f:
            threshold = json.load(f).get("f1_threshold", threshold)
    return min(max(float(threshold), 0.05), 0.9)


class SAM3Trainer(TrainDetector):
    """
    Implementation of TrainDetector for SAM 3.1 detector fine-tuning.

    Trains the detector's transformer, scorer, geometry encoder and mask
    head on per-image boxes (and polygons, where present) for one text
    query, with the vision backbone and text encoder frozen. See
    sam3_detector_training.py for the method.

    The output is a full SAM 3.1 checkpoint (only the detector differs from
    the source) and a pipeline running it through the sam3
    image_object_detector, with the detection threshold set to the one that
    maximized F1 on the validation images.
    """

    OUTPUT_MODEL_NAME = "sam3.1_detector_finetuned.pt"

    KEYS = ("max_iterations", "learning_rate", "weight_decay",
            "warmup_iterations", "batch_size", "eval_every", "eval_images",
            "validation_fraction", "num_workers", "text_query", "amp",
            "timeout", "seed")

    def __init__(self):
        TrainDetector.__init__(self)

        self._identifier = "viame-sam3-detector"
        self._sam_model_id = "models/sam3.1_weights.pt"
        self._train_directory = "deep_training"
        self._gpu_count = -1
        self._max_iterations = "3000"
        self._learning_rate = "8e-5"
        self._weight_decay = "0.1"
        self._warmup_iterations = "200"
        self._batch_size = "4"
        self._eval_every = "500"
        self._eval_images = "500"
        self._validation_fraction = "0.1"
        self._num_workers = "4"
        self._text_query = "fish"
        self._amp = "true"
        self._timeout = "604800"
        self._seed = "0"

        self._train_image_files = []
        self._train_detections = []
        self._test_image_files = []
        self._test_detections = []

    def get_configuration(self):
        cfg = super(TrainDetector, self).get_configuration()
        cfg.set_value("identifier", self._identifier)
        cfg.set_value("sam_model_id", self._sam_model_id)
        cfg.set_value("train_directory", self._train_directory)
        cfg.set_value("gpu_count", str(self._gpu_count))
        for key in self.KEYS:
            cfg.set_value(key, getattr(self, "_" + key))
        return cfg

    @report_cuda_errors("SAM3Trainer initialization")
    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)

        self._identifier = str(cfg.get_value("identifier"))
        self._sam_model_id = _resolve_model_path(
            str(cfg.get_value("sam_model_id")))
        self._train_directory = str(cfg.get_value("train_directory"))
        self._gpu_count = _gpu_count(int(cfg.get_value("gpu_count")))
        for key in self.KEYS:
            setattr(self, "_" + key, str(cfg.get_value(key)))

        if self._train_directory and not os.path.exists(self._train_directory):
            os.makedirs(self._train_directory)
        return True

    def check_configuration(self, cfg):
        if not cfg.has_value("identifier") or len(cfg.get_value("identifier")) == 0:
            print("A model identifier must be specified!")
            return False
        return True

    def add_data_from_disk(
        self, categories, train_files, train_dets, test_files, test_dets
    ):
        print("Adding training data from disk...")
        print("  Training files: ", len(train_files))
        print("  Training detections: ", len(train_dets))
        print("  Test files: ", len(test_files))
        print("  Test detections: ", len(test_dets))

        self._train_image_files = list(train_files)
        self._train_detections = list(train_dets)
        self._test_image_files = list(test_files)
        self._test_detections = list(test_dets)

    @staticmethod
    def _image_items(files, det_sets):
        items = []
        for path, dets in zip(files, det_sets):
            objects = []
            if dets is not None:
                for det in dets:
                    bbox = det.bounding_box
                    objects.append({
                        "bbox": [bbox.min_x(), bbox.min_y(),
                                 bbox.max_x(), bbox.max_y()],
                        "polys": _detection_polygons(det)})
            items.append({"image": path, "group": os.path.dirname(path),
                          "objects": objects})
        return items

    def _build_specs(self):
        train = self._image_items(self._train_image_files,
                                  self._train_detections)
        if self._test_image_files:
            return train, self._image_items(self._test_image_files,
                                            self._test_detections)
        return split_by_group(train, float(self._validation_fraction))

    @report_cuda_errors("SAM3Trainer training")
    def update_model(self):
        print("Starting SAM3 detector fine-tuning...")
        train, val = self._build_specs()
        print(f"  Training images: {len(train)}")
        print(f"  Validation images: {len(val)}")
        if not any(it["objects"] for it in train):
            raise RuntimeError("No annotated training images were found; the "
                               "trainer output above carries the reason.")

        work = os.path.abspath(os.path.join(self._train_directory,
                                            "sam3_detector"))
        output_model = run_detector_stage(
            train, val, work, self._sam_model_id, self._gpu_count,
            self._detector_params(), self.OUTPUT_MODEL_NAME)
        output = self._get_output_map(output_model, work)
        print("\nSAM3 detector training complete!")
        return output

    def _detector_params(self):
        return dict(
            max_iterations=int(self._max_iterations),
            learning_rate=float(self._learning_rate),
            weight_decay=float(self._weight_decay),
            warmup_iterations=int(self._warmup_iterations),
            batch_size=int(self._batch_size),
            eval_every=int(self._eval_every),
            eval_images=int(self._eval_images),
            num_workers=int(self._num_workers),
            text_query=self._text_query,
            amp=_parse_bool(self._amp),
            timeout=_parse_timeout(self._timeout),
            seed=int(self._seed))

    def _get_output_map(self, output_model, work):
        if not os.path.exists(output_model):
            raise RuntimeError(
                "SAM3 detector training produced no model at {}. The trainer "
                "output above carries the reason.".format(output_model))

        threshold = _deploy_threshold(work)

        output = {
            "type": "sam3",
            "sam3:sam_model_id": self.OUTPUT_MODEL_NAME,
            "sam3:text_query": self._text_query,
            "sam3:threshold": f"{threshold:.3f}",
            self.OUTPUT_MODEL_NAME: output_model,
        }
        print(f"\nModel written to: {output_model}")
        print(f"\nThe {self._train_directory} directory can now be deleted, "
              "unless you want to review training metrics first "
              "(sam3_detector/eval.jsonl).")
        return output


def split_by_group(items, fraction):
    """
    Hold out whole groups (videos / image folders) for validation, since
    neighbouring frames are near duplicates. The last groups in sorted order
    are taken so runs are comparable. Falls back to every 10th image when
    there is only one group.
    """
    if not fraction or fraction <= 0 or len(items) < 2:
        return items, []
    groups = sorted({it["group"] for it in items})
    if len(groups) >= 2:
        n = min(max(1, int(round(len(groups) * fraction))), len(groups) // 2)
        held = set(groups[-n:])
        return ([it for it in items if it["group"] not in held],
                [it for it in items if it["group"] in held])
    step = max(2, int(round(1 / fraction)))
    return ([it for i, it in enumerate(items) if i % step],
            [it for i, it in enumerate(items) if not i % step])


def run_detector_stage(train, val, work, sam_model_id, n_gpus, params,
                       output_name):
    os.makedirs(work, exist_ok=True)
    train_path = os.path.join(work, "train_images.json")
    val_path = os.path.join(work, "val_images.json")
    with open(train_path, "w") as f:
        json.dump(train, f)
    with open(val_path, "w") as f:
        json.dump(val, f)
    output_model = os.path.join(work, output_name)
    p = dict(params)
    p.update(train_spec=train_path, val_spec=val_path,
             sam_model_id=os.path.abspath(sam_model_id), work_dir=work,
             output_model=output_model)
    _run_engine("sam3_detector_training", p, n_gpus, "SAM3Trainer")
    return output_model


# =============================================================================
# Tracker (and optionally detector)
# =============================================================================

class SAM3TrackerTrainer(TrainTracker):
    """
    Implementation of TrainTracker for SAM 3.1.

    Fine-tunes the SAM 3.1 multiplex tracker -- memory attention, memory
    encoder, mask decoders and object-pointer heads -- on ground-truth tracks
    with polygon annotations (sam3_tracker_training.py).

    With train_detector (default) the detector is fine-tuned first on every
    annotated frame of the same tracks (sam3_detector_training.py) and the
    tracker is then trained from that checkpoint, so the output carries both
    and the generated text-query tracking pipeline uses both. With
    train_detector = false the detector weights are left exactly as in the
    source checkpoint, for running the tuned tracker on detections from
    another detector.

    Both stages keep the shared vision backbone frozen, so they are
    independent. The output is a full SAM 3.1 checkpoint plus a pipeline from
    templates/tracker_sam3.pipe.
    """

    OUTPUT_MODEL_NAME = "sam3.1_tracker_finetuned.pt"
    DETECTOR_MODEL_NAME = "sam3.1_detector_finetuned.pt"
    CONFIG_NAME = "sam3.1_config.json"

    KEYS = ("max_iterations", "learning_rate", "weight_decay",
            "warmup_iterations", "clip_length", "max_objects", "eval_every",
            "eval_clips", "validation_fraction", "num_workers", "box_targets",
            "text_query", "timeout", "seed", "sequence_manifest",
            "validation_manifest", "train_detector",
            "detector_max_iterations", "detector_learning_rate",
            "detector_batch_size", "detector_eval_images")

    def __init__(self):
        TrainTracker.__init__(self)
        self._detection_threshold = 0.3

        self._identifier = "viame-sam3-tracker"
        self._sam_model_id = "models/sam3.1_weights.pt"
        self._train_directory = "deep_training"
        self._gpu_count = -1
        self._max_iterations = "4000"
        self._learning_rate = "2e-5"
        self._weight_decay = "0.05"
        self._warmup_iterations = "300"
        self._clip_length = "8"
        self._max_objects = "8"
        self._eval_every = "500"
        self._eval_clips = "300"
        self._validation_fraction = "0.1"
        self._num_workers = "4"
        self._box_targets = "false"
        self._text_query = "fish"
        self._timeout = "604800"
        self._seed = "0"
        self._sequence_manifest = ""
        self._validation_manifest = ""
        self._train_detector = "true"
        self._detector_max_iterations = "3000"
        self._detector_learning_rate = "8e-5"
        self._detector_batch_size = "4"
        self._detector_eval_images = "500"

        self._train_image_files = []
        self._train_tracks = []
        self._test_image_files = []
        self._test_tracks = []

    def get_configuration(self):
        cfg = super(TrainTracker, self).get_configuration()
        cfg.set_value("identifier", self._identifier)
        cfg.set_value("sam_model_id", self._sam_model_id)
        cfg.set_value("train_directory", self._train_directory)
        cfg.set_value("gpu_count", str(self._gpu_count))
        for key in self.KEYS:
            cfg.set_value(key, getattr(self, "_" + key))
        return cfg

    @report_cuda_errors("SAM3TrackerTrainer initialization")
    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config(cfg_in)

        self._identifier = str(cfg.get_value("identifier"))
        self._sam_model_id = _resolve_model_path(
            str(cfg.get_value("sam_model_id")))
        self._train_directory = str(cfg.get_value("train_directory"))
        self._gpu_count = _gpu_count(int(cfg.get_value("gpu_count")))
        for key in self.KEYS:
            setattr(self, "_" + key, str(cfg.get_value(key)))

        if self._train_directory and not os.path.exists(self._train_directory):
            os.makedirs(self._train_directory)
        return True

    def check_configuration(self, cfg):
        if not cfg.has_value("identifier") or len(cfg.get_value("identifier")) == 0:
            print("A model identifier must be specified!")
            return False
        return True

    def add_data_from_disk(
        self, categories, train_files, train_tracks, test_files, test_tracks
    ):
        print("Adding training data from disk...")
        print("  Training files: ", len(train_files))
        print("  Training tracks: ", len(train_tracks))
        print("  Test files: ", len(test_files))
        print("  Test tracks: ", len(test_tracks))

        self._train_image_files = list(train_files)
        self._train_tracks = list(train_tracks)
        self._test_image_files = list(test_files)
        self._test_tracks = list(test_tracks)

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------

    @staticmethod
    def _sequence_maps(image_files, track_sets, manifest, label):
        """
        One frame-id -> image map per track set. Track frame ids are local to
        their sequence while the image list is flat, so the training tool's
        manifest is used when present, else the directory layout.
        """
        from viame.core.training_data import (
            build_sequence_maps, read_sequence_manifest)

        maps = names = None
        if manifest:
            maps, names = read_sequence_manifest(
                manifest, image_files, len(track_sets))
        if maps is None:
            maps, names = build_sequence_maps(
                image_files, len(track_sets), label)
        return maps, names

    @staticmethod
    def _track_set_spec(track_set, frame_map, name):
        """JSON-serializable frames/objects spec for one sequence."""
        frames, objects = {}, {}
        for track in track_set.tracks():
            for state in track:
                det = state.detection()
                if det is None:
                    continue
                fid = state.frame_id
                path = frame_map.get(fid)
                if path is None:
                    continue
                bbox = det.bounding_box
                x1, y1 = bbox.min_x(), bbox.min_y()
                x2, y2 = bbox.max_x(), bbox.max_y()
                if x2 - x1 <= 0 or y2 - y1 <= 0:
                    continue
                frames[str(fid)] = path
                objects.setdefault(str(fid), []).append(
                    {"tid": int(track.id), "bbox": [x1, y1, x2, y2],
                     "polys": _detection_polygons(det)})
        return {"name": name, "frames": frames, "objects": objects}

    def _build_specs(self):
        from viame.core.training_data import split_validation

        train_maps, train_names = self._sequence_maps(
            self._train_image_files, self._train_tracks,
            self._sequence_manifest, "training")
        train = (self._train_tracks, train_maps, train_names)

        if self._test_tracks:
            val_maps, val_names = self._sequence_maps(
                self._test_image_files, self._test_tracks,
                self._validation_manifest, "validation")
            val = (self._test_tracks, val_maps, val_names)
        else:
            # Hold out whole sequences, so model selection is not done on
            # frames from clips that are also trained on.
            train, val = split_validation(
                self._train_tracks, train_maps, train_names,
                float(self._validation_fraction))

        def to_spec(part, prefix):
            sets, maps, names = part
            out = []
            for i, ts in enumerate(sets):
                if ts is None or not maps or not maps[i]:
                    continue
                name = (names[i] if names and names[i] else f"{prefix}{i}")
                spec = self._track_set_spec(ts, maps[i], name)
                if spec["objects"]:
                    out.append(spec)
            return out

        return to_spec(train, "train"), to_spec(val, "val")

    @staticmethod
    def _detector_items(specs):
        """Every annotated frame of the sequences, as detector images."""
        items = []
        for s in specs:
            for fid, objs in s["objects"].items():
                items.append({"image": s["frames"][fid], "group": s["name"],
                              "objects": [{"bbox": o["bbox"],
                                           "polys": o["polys"]}
                                          for o in objs]})
        return items

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    @report_cuda_errors("SAM3TrackerTrainer training")
    def update_model(self):
        train_detector = _parse_bool(self._train_detector)
        print("Starting SAM3 " + ("detector and tracker" if train_detector
                                  else "tracker") + " fine-tuning...")

        train_spec, val_spec = self._build_specs()
        n_train = sum(len(s["objects"]) for s in train_spec)
        n_val = sum(len(s["objects"]) for s in val_spec)
        print(f"  Training sequences: {len(train_spec)} ({n_train} frames)")
        print(f"  Validation sequences: {len(val_spec)} ({n_val} frames)")

        if not train_spec:
            raise RuntimeError(
                "No training frames could be matched to track annotations; "
                "the trainer output above carries the reason.")

        root = os.path.abspath(self._train_directory)
        source = os.path.abspath(self._sam_model_id)

        self._detection_threshold = 0.3
        if train_detector:
            print("\n[SAM3TrackerTrainer] Stage 1 of 2: detector")
            source = run_detector_stage(
                self._detector_items(train_spec),
                self._detector_items(val_spec),
                os.path.join(root, "sam3_detector"), source, self._gpu_count,
                dict(max_iterations=int(self._detector_max_iterations),
                     learning_rate=float(self._detector_learning_rate),
                     batch_size=int(self._detector_batch_size),
                     eval_every=int(self._eval_every),
                     eval_images=int(self._detector_eval_images),
                     num_workers=int(self._num_workers),
                     text_query=self._text_query,
                     timeout=_parse_timeout(self._timeout),
                     seed=int(self._seed)),
                self.DETECTOR_MODEL_NAME)
            self._detection_threshold = _deploy_threshold(
                os.path.join(root, "sam3_detector"))
            print("\n[SAM3TrackerTrainer] Stage 2 of 2: tracker")

        work = os.path.join(root, "sam3_tracker")
        os.makedirs(work, exist_ok=True)
        train_path = os.path.join(work, "train_spec.json")
        val_path = os.path.join(work, "val_spec.json")
        with open(train_path, "w") as f:
            json.dump(train_spec, f)
        with open(val_path, "w") as f:
            json.dump(val_spec, f)

        output_model = os.path.join(work, self.OUTPUT_MODEL_NAME)
        params = dict(
            train_spec=train_path, val_spec=val_path,
            # The tracker export copies every non-tracker weight from here,
            # so a tuned detector from stage 1 carries through.
            sam_model_id=source,
            work_dir=work, output_model=output_model,
            max_iterations=int(self._max_iterations),
            learning_rate=float(self._learning_rate),
            weight_decay=float(self._weight_decay),
            warmup_iterations=int(self._warmup_iterations),
            clip_length=int(self._clip_length),
            max_objects=int(self._max_objects),
            eval_every=int(self._eval_every),
            eval_clips=int(self._eval_clips),
            num_workers=int(self._num_workers),
            box_targets=_parse_bool(self._box_targets),
            timeout=_parse_timeout(self._timeout),
            seed=int(self._seed),
        )
        _run_engine("sam3_tracker_training", params, self._gpu_count,
                    "SAM3TrackerTrainer")

        output = self._get_output_map(output_model)
        print("\nSAM3 tracker training complete!")
        return output

    def _get_output_map(self, output_model):
        """
        Output map for process_trainer_output. Values that are files are
        copied next to the generated pipeline; the rest fill [-KEY-]
        placeholders in templates/tracker_sam3.pipe.
        """
        if not os.path.exists(output_model):
            raise RuntimeError(
                "SAM3 tracker training produced no model at {}. The trainer "
                "output above carries the reason.".format(output_model))

        output = {}
        output["sam3:sam_model_id"] = self.OUTPUT_MODEL_NAME
        output["sam3:model_config"] = self.CONFIG_NAME
        output["sam3:text_query"] = self._text_query
        output["sam3:video_detection_threshold"] = \
            f"{self._detection_threshold:.3f}"
        output[self.OUTPUT_MODEL_NAME] = output_model

        # The 3.1 config travels with the weights so the pipeline resolves it
        # relative to itself.
        config = _config_path(self._sam_model_id, self.CONFIG_NAME)
        if config:
            output[self.CONFIG_NAME] = config

        print(f"\nModel written to: {output_model}")
        print(f"\nThe {self._train_directory} directory can now be deleted, "
              "unless you want to review training metrics first "
              "(sam3_detector/eval.jsonl, sam3_tracker/eval.jsonl).")

        return output


def __vital_algorithm_register__():
    from viame.core.vital_registration import register_vital_algorithm

    register_vital_algorithm(
        SAM3Trainer,
        "sam3",
        "SAM 3.1 detector fine-tuning with SAM 3's training objective",
    )
    register_vital_algorithm(
        SAM3TrackerTrainer,
        "sam3",
        "SAM 3.1 tracker (and optionally detector) fine-tuning",
    )
