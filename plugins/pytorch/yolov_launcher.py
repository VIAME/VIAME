# This file is part of VIAME, and is distributed under an OSI-approved        #
# BSD 3-Clause License. See top-level LICENSE.txt or                          #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.           #
"""Standalone YOLOV++ training entrypoint.

yolov_trainer.py writes the dataset and a parameter file, then runs this module
as a subprocess (argv[1] = the parameter JSON). It trains the two stages the
method needs in one go: the still-image YOLOX base detector, then the YOLOV++
aggregation head on top of the frozen base, and records the checkpoints it
produced in <output_dir>/result.json."""
import json
import math
import os
import random
import sys
import time


def log(message):
    print(f"[yolov_launcher] {message}", flush=True)


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

def read_bgr(path):
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"cannot read image {path}")
    return img


def record_boxes(record):
    """[n, 5] float32 (x1, y1, x2, y2, class) of a dataset record."""
    import numpy as np
    boxes = np.asarray(record.get("boxes") or [], dtype=np.float32)
    return boxes.reshape(-1, 5)


def group_by_folder(records):
    """{folder: [records in temporal order]}, folders in sorted order."""
    from viame.pytorch.yolov_common import natural_sort_key
    groups = {}
    for record in records:
        groups.setdefault(os.path.dirname(record["file"]), []).append(record)
    for folder in groups:
        groups[folder].sort(key=lambda r: natural_sort_key(os.path.basename(r["file"])))
    return dict(sorted(groups.items()))


class FrameDataset:
    """Still frames in the form upstream's MosaicDetection wrapper pulls them."""

    def __init__(self, records, input_size):
        self.records = records
        self.input_dim = tuple(input_size)

    def __len__(self):
        return len(self.records)

    def _ratio(self, record):
        return min(self.input_dim[0] / record["height"], self.input_dim[1] / record["width"])

    def load_anno(self, index):
        record = self.records[index]
        labels = record_boxes(record).copy()
        labels[:, :4] *= self._ratio(record)
        return labels

    def pull_item(self, index):
        import cv2
        record = self.records[index]
        img = read_bgr(record["file"])
        ratio = min(self.input_dim[0] / img.shape[0], self.input_dim[1] / img.shape[1])
        resized = cv2.resize(
            img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR)
        labels = record_boxes(record).copy()
        labels[:, :4] *= ratio
        return resized, labels, img.shape[:2], index


class EvalFrameDataset:
    """Letterboxed validation frames, each with the reference frames it needs."""

    def __init__(self, items, input_size):
        # items: (record index, frame path, [reference paths], first-of-sequence flag)
        self.items = items
        self.input_size = tuple(input_size)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        import torch
        from viame.pytorch.yolov_common import preprocess_frame
        record_index, path, references, first = self.items[index]
        img, ratio = preprocess_frame(read_bgr(path), self.input_size)
        refs = [torch.from_numpy(preprocess_frame(read_bgr(p), self.input_size)[0])
                for p in references]
        return record_index, torch.from_numpy(img), ratio, references, refs, first


class VideoBatchDataset:
    """One YOLOV++ training batch per item: groups of frames aggregated together."""

    def __init__(self, batches, input_size, transform):
        # batches: list of groups; a group is a list of (record or None, path)
        self.batches = batches
        self.input_size = tuple(input_size)
        self.transform = transform

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        import numpy as np
        import torch
        imgs, targets, supervised, groups = [], [], [], []
        for group in self.batches[index]:
            start = len(imgs)
            for record, path in group:
                boxes = (np.zeros((0, 5), dtype=np.float32) if record is None
                         else record_boxes(record).copy())
                img, target = self.transform(read_bgr(path), boxes, self.input_size)
                imgs.append(torch.from_numpy(np.ascontiguousarray(img)))
                targets.append(torch.from_numpy(target))
                supervised.append(record is not None)
            groups.append((start, len(imgs)))
        return (torch.stack(imgs), torch.stack(targets),
                torch.tensor(supervised, dtype=torch.bool), groups)


def native_batches(train_groups, frames_per_batch, rng):
    """Upstream's global sampling: shuffled frames of one video, split into batches."""
    batches = []
    for records in train_groups.values():
        records = list(records)
        rng.shuffle(records)
        for start in range(0, len(records), frames_per_batch):
            chunk = records[start:start + frames_per_batch]
            # Upstream drops what does not fill a batch; short sequences are kept here.
            if len(chunk) >= 2:
                batches.append([[(r, r["file"]) for r in chunk]])
    rng.shuffle(batches)
    return batches


def reference_paths(path, sequences):
    """Frames at the configured offsets of path that exist in its folder."""
    folder, name = os.path.split(path)
    info = sequences.get(folder)
    if not info:
        return []
    index = info.setdefault("_index", {n: i for i, n in enumerate(info["names"])})
    pos = index.get(name, -1)
    refs = []
    for offset in info["offsets"]:
        other = pos + offset if pos >= 0 else -1
        if 0 <= other < len(info["names"]):
            refs.append(os.path.join(folder, info["names"][other]))
    return refs


def offset_batches(train_records, sequences, frames_per_batch, rng):
    """Each labelled frame with its offset neighbours; several such groups per batch."""
    by_path = {r["file"]: r for r in train_records}
    groups = []
    for record in train_records:
        refs = reference_paths(record["file"], sequences)
        groups.append([(record, record["file"])] + [(by_path.get(p), p) for p in refs])
    rng.shuffle(groups)
    largest = max(len(g) for g in groups)
    per_batch = max(1, frames_per_batch // largest)
    return [groups[i:i + per_batch] for i in range(0, len(groups), per_batch)]


def subsample_validation(val_groups, max_frames):
    """Keep the leading part of every sequence so at most ~max_frames remain."""
    total = sum(len(v) for v in val_groups.values())
    if max_frames <= 0 or total <= max_frames:
        return val_groups
    fraction = max_frames / float(total)
    return {folder: records[:max(1, int(math.ceil(len(records) * fraction)))]
            for folder, records in val_groups.items()}


def seed_worker(worker_id):
    # Forked workers share the parent's RNG state; give each its own stream.
    import numpy as np
    import torch
    seed = (torch.initial_seed() + worker_id) % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def box_iou_matrix(a, b):
    import numpy as np
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def average_precision(detections, groundtruth, num_classes, max_dets=300):
    """(AP50, AP50:95) averaged over the classes that have groundtruth.

    detections / groundtruth: per image [n, 6] (x1, y1, x2, y2, score, class)
    and [m, 5] (x1, y1, x2, y2, class) arrays in the same pixel space.
    """
    import numpy as np
    thresholds = np.arange(0.5, 0.96, 0.05)
    aps = []
    for cls in range(num_classes):
        scores, matched, num_gt = [], [], 0
        for dets, gts in zip(detections, groundtruth):
            gts = gts[gts[:, 4] == cls]
            dets = dets[dets[:, 5] == cls]
            dets = dets[np.argsort(-dets[:, 4])][:max_dets]
            num_gt += len(gts)
            if len(dets) == 0:
                continue
            hits = np.zeros((len(dets), len(thresholds)), dtype=bool)
            if len(gts):
                ious = box_iou_matrix(dets[:, :4], gts[:, :4])
                for t, thresh in enumerate(thresholds):
                    taken = np.zeros(len(gts), dtype=bool)
                    for d in range(len(dets)):
                        row = np.where(taken, -1.0, ious[d])
                        best = int(row.argmax())
                        if row[best] >= thresh:
                            taken[best] = True
                            hits[d, t] = True
            scores.append(dets[:, 4])
            matched.append(hits)
        if num_gt == 0:
            continue
        if not scores:
            aps.append(np.zeros(len(thresholds)))
            continue
        order = np.argsort(-np.concatenate(scores))
        hits = np.concatenate(matched)[order]
        per_thresh = []
        for t in range(len(thresholds)):
            tp = np.cumsum(hits[:, t])
            fp = np.cumsum(~hits[:, t])
            recall = np.concatenate([[0.0], tp / num_gt, [1.0]])
            precision = np.concatenate([[1.0], tp / np.maximum(tp + fp, 1), [0.0]])
            precision = np.maximum.accumulate(precision[::-1])[::-1]
            steps = np.where(recall[1:] != recall[:-1])[0]
            per_thresh.append(float(((recall[steps + 1] - recall[steps]) * precision[steps + 1]).sum()))
        aps.append(np.asarray(per_thresh))
    if not aps:
        return 0.0, 0.0
    aps = np.stack(aps)
    return float(aps[:, 0].mean()), float(aps.mean())


def evaluate(model, video, params, val_groups, sequences, device):
    """AP of a model on the validation frames, run the way the detector runs it."""
    import numpy as np
    import torch
    from viame.pytorch.yolov_common import (
        OnlineVideoDetector, aggregate, base_detect, detections_to_numpy,
        extract_proposals)

    records, items = [], []
    for folder, group in val_groups.items():
        for position, record in enumerate(group):
            refs = reference_paths(record["file"], sequences) if (video and sequences) else []
            items.append((len(records), record["file"], refs, position == 0))
            records.append(record)
    if not records:
        return None

    loader = torch.utils.data.DataLoader(
        EvalFrameDataset(items, params["input_size"]), batch_size=None,
        shuffle=False, num_workers=int(params["num_workers"]))
    online = OnlineVideoDetector(
        model, params["input_size"], nms_thresh=params["nms_thresh"],
        reference_frames=params["reference_frames"],
        memory_frames=params["memory_frames"], seed=params["seed"])

    was_training = model.training
    model.eval()
    detections = [None] * len(records)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=bool(params["amp"])):
        for record_index, img, ratio, ref_paths, refs, first in loader:
            img = img.unsqueeze(0).to(device)
            if not video:
                result = base_detect(model, img, 0.001, params["base_nms_thresh"])[0]
            elif sequences:
                # Fixed offsets: aggregate against exactly those frames from disk.
                frames = extract_proposals(model, torch.stack([img[0]] + [r.to(device) for r in refs]))
                result = None
                if frames[0] is not None:
                    result = aggregate(model, [f for f in frames if f is not None],
                                       params["nms_thresh"], keys=[0])[0]
            else:
                if first:
                    online.reset()
                result = online.step_tensor(img)
            detections[record_index] = detections_to_numpy(result, ratio)
    model.train(was_training)

    groundtruth = [record_boxes(r) for r in records]
    ap50, ap5095 = average_precision(detections, groundtruth, len(params["class_names"]))
    return {"ap50": ap50, "ap50_95": ap5095, "frames": len(records),
            "boxes": int(sum(len(g) for g in groundtruth)),
            "detections": int(sum(len(d) for d in detections))}


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def build_optimizer(model, lr, momentum, weight_decay):
    """YOLOX's SGD groups: norm weights without decay, other weights with, biases."""
    import torch
    import torch.nn as nn
    pg0, pg1, pg2 = [], [], []
    for name, module in model.named_modules():
        bias = getattr(module, "bias", None)
        weight = getattr(module, "weight", None)
        if isinstance(bias, nn.Parameter) and bias.requires_grad:
            pg2.append(bias)
        if not isinstance(weight, nn.Parameter) or not weight.requires_grad:
            continue
        if isinstance(module, nn.BatchNorm2d) or "bn" in name:
            pg0.append(weight)
        else:
            pg1.append(weight)
    groups = [{"params": pg0}, {"params": pg1, "weight_decay": weight_decay}, {"params": pg2}]
    groups = [g for g in groups if g["params"]]
    return torch.optim.SGD(groups, lr=lr, momentum=momentum, nesterov=True)


def random_input_size(input_size, multiscale_range):
    """A training size within +-multiscale_range * 32 px of the height, same aspect."""
    if multiscale_range <= 0:
        return tuple(input_size)
    base = int(input_size[0] / 32)
    size = random.randint(max(1, base - multiscale_range), base + multiscale_range)
    return int(32 * size), 32 * int(size * input_size[1] / float(input_size[0]))


def resize_batch(imgs, targets, size, input_size):
    import torch.nn.functional as F
    if tuple(size) == tuple(input_size):
        return imgs, targets
    scale_y = size[0] / input_size[0]
    scale_x = size[1] / input_size[1]
    imgs = F.interpolate(imgs, size=tuple(size), mode="bilinear", align_corners=False)
    targets = targets.clone()
    targets[..., 1::2] = targets[..., 1::2] * scale_x
    targets[..., 2::2] = targets[..., 2::2] * scale_y
    return imgs, targets


def save_checkpoint(path, model, meta, attempts=3):
    import torch
    tmp = path + ".tmp"
    for attempt in range(attempts):
        try:
            torch.save({"model": model.state_dict(), "meta": meta}, tmp)
            break
        except (OSError, RuntimeError) as exc:
            # A failed write on a busy filesystem should not cost the whole run.
            if attempt + 1 == attempts:
                raise
            log(f"writing {path} failed ({exc}); retrying")
            time.sleep(5)
    os.replace(tmp, path)


class StageRunner:
    """Bookkeeping shared by both stages: schedule, EMA, validation and checkpoints."""

    def __init__(self, name, model, params, stage, lr, iters_per_epoch, ema_decay,
                 meta, validate, deadline):
        import torch
        from yolox.utils import LRScheduler, ModelEMA

        self.name = name
        self.model = model
        self.params = params
        self.stage = stage
        self.epochs = int(stage["epochs"])
        self.iters_per_epoch = iters_per_epoch
        self.meta = meta
        self.validate = validate
        self.deadline = deadline
        # As upstream: the first step runs at the warmup start when there is a warmup.
        self.optimizer = build_optimizer(
            model, 0.0 if float(stage["warmup_epochs"]) > 0 else lr,
            float(params["momentum"]), float(params["weight_decay"]))
        self.scheduler = LRScheduler(
            "yoloxwarmcos", lr, iters_per_epoch, self.epochs,
            warmup_epochs=float(stage["warmup_epochs"]), warmup_lr_start=0,
            no_aug_epochs=int(stage["no_aug_epochs"]),
            min_lr_ratio=float(stage["min_lr_ratio"]))
        self.scaler = torch.amp.GradScaler("cuda", enabled=bool(params["amp"]))
        self.ema = ModelEMA(model, ema_decay) if params["ema"] else None
        self.metric = params["selection_metric"]
        self.best = None
        self.best_path = os.path.join(params["output_dir"], f"{name}_best.pth")
        self.last_path = os.path.join(params["output_dir"], f"{name}_last.pth")
        # Checkpoints of an earlier run in this directory must not pass for this one's.
        for stale in (self.best_path, self.last_path):
            if os.path.exists(stale):
                os.remove(stale)
        self.history = []
        self.iteration = 0
        self.input_size = tuple(params["input_size"])
        self.started = time.time()
        torch.cuda.reset_peak_memory_stats()

    @property
    def eval_model(self):
        return self.ema.ema if self.ema is not None else self.model

    def out_of_time(self):
        return self.deadline is not None and time.time() > self.deadline

    def step(self, loss, epoch, log_values):
        """Backward, optimizer step, EMA and LR update for one iteration."""
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.ema is not None:
            self.ema.update(self.model)
        self.after_iteration(epoch, log_values)

    def after_iteration(self, epoch, log_values):
        import torch
        self.iteration += 1
        lr = self.scheduler.update_lr(self.iteration)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        in_epoch = (self.iteration - 1) % self.iters_per_epoch + 1
        if in_epoch % int(self.params["print_interval"]) == 0 or in_epoch == self.iters_per_epoch:
            values = ", ".join(f"{k}: {float(v):.3f}" for k, v in log_values.items())
            log(f"{self.name} epoch {epoch + 1}/{self.epochs} iter {in_epoch}/"
                f"{self.iters_per_epoch}, {values}, lr: {lr:.3e}, size: "
                f"{self.input_size[0]}x{self.input_size[1]}, mem: "
                f"{torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MiB, "
                f"elapsed: {time.time() - self.started:.0f}s")
        if self.iteration % 10 == 0:
            self.input_size = random_input_size(
                self.params["input_size"], int(self.stage["multiscale_range"]))

    def end_epoch(self, epoch):
        """Validate, keep the best checkpoint; returns False when training should stop."""
        last = epoch + 1 >= self.epochs or self.out_of_time()
        evaluated = (epoch + 1) % int(self.params["eval_interval"]) == 0 or last
        metrics = self.validate(self.eval_model) if evaluated else None
        save_checkpoint(self.last_path, self.eval_model, self.meta)
        if evaluated and metrics is None:
            # No validation data: the most recent weights are the best known.
            save_checkpoint(self.best_path, self.eval_model, self.meta)
        elif metrics is not None:
            self.history.append(dict(metrics, epoch=epoch + 1))
            score = metrics[self.metric]
            better = self.best is None or score > self.best
            log(f"{self.name} epoch {epoch + 1}: AP50 {metrics['ap50']:.4f}, "
                f"AP50:95 {metrics['ap50_95']:.4f} over {metrics['frames']} frames"
                f"{' (best)' if better else ''}")
            if better:
                self.best = score
                save_checkpoint(self.best_path, self.eval_model, self.meta)
        if not os.path.exists(self.best_path):
            save_checkpoint(self.best_path, self.eval_model, self.meta)
        if self.out_of_time() and not epoch + 1 >= self.epochs:
            log(f"{self.name}: time limit reached, stopping after epoch {epoch + 1}")
            return False
        return True


def train_base(params, data, device, deadline):
    """Stage 1: fine-tune the YOLOX base detector on still frames."""
    import torch
    from yolox.data.data_augment import TrainTransform
    from yolox.data.datasets.mosaicdetection import MosaicDetection
    from viame.pytorch.yolov_common import (
        build_base_detector, load_matching_weights, read_checkpoint)

    stage = params["base"]
    num_classes = len(params["class_names"])
    model = build_base_detector(params["model_size"], num_classes)
    if params.get("seed_model"):
        state_dict, _ = read_checkpoint(params["seed_model"])
        loaded, skipped = load_matching_weights(model, state_dict)
        log(f"base: seeded {len(loaded)} tensors from {params['seed_model']}, "
            f"{len(skipped)} left at their initial values "
            f"({', '.join(skipped[:4])}{'...' if len(skipped) > 4 else ''})")
    else:
        log("base: no seed_model, training YOLOX from random weights")
    model.to(device).train()

    input_size = tuple(params["input_size"])
    dataset = MosaicDetection(
        FrameDataset(data["train"], input_size), img_size=input_size, mosaic=True,
        preproc=TrainTransform(max_labels=int(params["max_labels"]),
                               flip_prob=float(params["flip_prob"]),
                               hsv_prob=float(params["hsv_prob"])),
        degrees=float(stage["degrees"]), translate=float(stage["translate"]),
        mosaic_scale=tuple(stage["mosaic_scale"]), mixup_scale=(0.5, 1.5),
        shear=float(stage["shear"]), enable_mixup=float(stage["mixup_prob"]) > 0,
        mosaic_prob=float(stage["mosaic_prob"]), mixup_prob=float(stage["mixup_prob"]))

    batch_size = int(stage["batch_size"])
    drop_last = len(dataset) >= batch_size

    def make_loader():
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last,
            num_workers=int(params["num_workers"]), pin_memory=True,
            worker_init_fn=seed_worker)

    iters_per_epoch = len(make_loader())
    val_groups = subsample_validation(
        group_by_folder(data["val"]), int(params["val_max_frames"]))
    meta = dict(params["meta"], stage="base")
    runner = StageRunner(
        "base", model, params, stage, float(stage["lr_per_image"]) * batch_size,
        iters_per_epoch, 0.9999, meta,
        lambda m: evaluate(m, False, params, val_groups, None, device), deadline)
    log(f"base: {len(dataset)} frames, batch {batch_size}, {iters_per_epoch} "
        f"iterations/epoch, {runner.epochs} epochs")

    for epoch in range(runner.epochs):
        if epoch + 1 > runner.epochs - int(stage["no_aug_epochs"]):
            # YOLOX closes mosaic and adds the L1 box loss for the last epochs.
            dataset.enable_mosaic = False
            model.head.use_l1 = True
        for imgs, targets, _, _ in make_loader():
            imgs = imgs.to(device, non_blocking=True).float()
            targets = targets.to(device, non_blocking=True).float()
            imgs, targets = resize_batch(imgs, targets, runner.input_size, input_size)
            with torch.amp.autocast("cuda", enabled=bool(params["amp"])):
                outputs = model(imgs, targets)
            runner.step(outputs["total_loss"], epoch, {
                "loss": outputs["total_loss"], "iou": outputs["iou_loss"],
                "obj": outputs["conf_loss"], "cls": outputs["cls_loss"]})
            if runner.out_of_time():
                break
        if not runner.end_epoch(epoch):
            break
    return runner


def train_video(params, data, base_path, device, deadline):
    """Stage 2: train the YOLOV++ aggregation head on the frozen base detector."""
    import torch
    from yolox.data.data_augment import TrainTransform
    from viame.pytorch.yolov_common import (
        build_video_detector, checkpoint_num_classes, frozen_video_modules,
        load_matching_weights, read_checkpoint)

    stage = params["video"]
    num_classes = len(params["class_names"])
    state_dict, _ = read_checkpoint(base_path)
    if checkpoint_num_classes(state_dict) != num_classes:
        raise RuntimeError(
            f"{base_path} predicts {checkpoint_num_classes(state_dict)} classes "
            f"but the training data has {num_classes}. The YOLOV++ stage keeps "
            f"the base detector fixed, so it needs a base fine-tuned on these "
            f"classes: let stage 1 run instead of setting base_model.")

    model = build_video_detector(params["model_size"], num_classes, params["head_config"])
    loaded, skipped = load_matching_weights(model, state_dict)
    log(f"video: loaded {len(loaded)} base tensors from {base_path}; "
        f"{len(skipped)} aggregation-head tensors start untrained")
    for module in frozen_video_modules(model):
        for parameter in module.parameters():
            parameter.requires_grad = False
    model.to(device)

    def set_train():
        model.train()
        if params["freeze_bn"]:
            for module in frozen_video_modules(model):
                module.eval()

    set_train()
    input_size = tuple(params["input_size"])
    sequences = data.get("sequences") or {}
    frames_per_batch = int(stage["frames_per_batch"])
    train_groups = group_by_folder(data["train"])
    rng = random.Random(int(params["seed"]))
    transform = TrainTransform(max_labels=int(params["max_labels"]),
                               flip_prob=float(params["flip_prob"]),
                               hsv_prob=float(params["hsv_prob"]))

    def make_loader():
        if sequences:
            batches = offset_batches(data["train"], sequences, frames_per_batch, rng)
        else:
            batches = native_batches(train_groups, frames_per_batch, rng)
        return torch.utils.data.DataLoader(
            VideoBatchDataset(batches, input_size, transform), batch_size=None,
            shuffle=False, num_workers=int(params["num_workers"]), pin_memory=True,
            worker_init_fn=seed_worker)

    iters_per_epoch = len(make_loader())
    if iters_per_epoch == 0:
        raise RuntimeError(
            "No YOLOV++ training batch could be formed: every sequence folder "
            "holds fewer than two training frames.")
    val_groups = subsample_validation(
        group_by_folder(data["val"]), int(params["val_max_frames"]))
    meta = dict(params["meta"], stage="video", head_config=params["head_config"])
    runner = StageRunner(
        "video", model, params, stage,
        float(stage["lr_per_image"]) * frames_per_batch, iters_per_epoch, 0.9998,
        meta, lambda m: evaluate(m, True, params, val_groups, sequences, device),
        deadline)
    sampling = (f"offsets {params['temporal_offsets']}" if sequences
                else "random frames of the same sequence")
    log(f"video: {len(data['train'])} frames in {len(train_groups)} sequences, "
        f"{frames_per_batch} frames/batch ({sampling}), {iters_per_epoch} "
        f"iterations/epoch, {runner.epochs} epochs")

    head = model.head
    for epoch in range(runner.epochs):
        for imgs, targets, supervised, groups in make_loader():
            imgs = imgs.to(device, non_blocking=True).float()
            targets = targets.to(device, non_blocking=True).float()
            supervised = supervised.to(device)
            imgs, targets = resize_batch(imgs, targets, runner.input_size, input_size)
            losses, values = [], {}
            with torch.amp.autocast("cuda", enabled=bool(params["amp"])):
                with torch.no_grad():
                    fpn_outs = model.backbone(imgs)
                for start, end in groups:
                    head.supervised = supervised[start:end]
                    out = head([f[start:end] for f in fpn_outs], targets[start:end],
                               imgs[start:end], lframe=0, gframe=end - start)
                    # The head returns a constant when no frame yields a proposal.
                    if not (torch.is_tensor(out[0]) and out[0].requires_grad):
                        continue
                    losses.append(out[0])
                    for key, value in zip(("obj", "cls", "ref_cls", "ref_obj"),
                                          (out[2], out[3], out[6], out[8])):
                        values[key] = values.get(key, 0.0) + float(value) / len(groups)
            head.supervised = None
            if losses:
                loss = sum(losses) / len(losses)
                runner.step(loss, epoch, dict({"loss": loss}, **values))
            else:
                runner.after_iteration(epoch, {"loss": 0.0})
            if runner.out_of_time():
                break
        keep_going = runner.end_epoch(epoch)
        set_train()
        if not keep_going:
            break
    return runner


def run(params):
    import numpy as np
    import torch
    from viame.pytorch.yolov_common import quiet_upstream_logging, select_cuda_device

    quiet_upstream_logging()
    os.makedirs(params["output_dir"], exist_ok=True)
    with open(params["dataset"]) as f:
        data = json.load(f)

    seed = int(params["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Off, unlike upstream: its workspace probes spike by several GB per input size.
    torch.backends.cudnn.benchmark = False
    device = select_cuda_device(params["device"])

    timeout = float(params.get("timeout") or 0)
    deadline = time.time() + timeout if timeout > 0 else None
    result = {"base": None, "video": None, "history": {}}

    base_path = params.get("base_model") or None
    if base_path:
        log(f"Stage 1 skipped: using the fine-tuned base detector {base_path}")
    elif int(params["base"]["epochs"]) > 0:
        runner = train_base(params, data, device, deadline)
        base_path = runner.best_path
        result["base"] = base_path
        result["history"]["base"] = runner.history
        del runner
        torch.cuda.empty_cache()
    else:
        raise RuntimeError(
            "base_epochs = 0 skips the base detector stage, which needs "
            "base_model set to an already fine-tuned YOLOX checkpoint.")

    if int(params["video"]["epochs"]) <= 0:
        log("Stage 2 skipped: video_epochs = 0, the base detector is the result")
    elif deadline is not None and time.time() > deadline:
        log("Stage 2 skipped: the time limit ran out during stage 1")
    else:
        runner = train_video(params, data, base_path, device, deadline)
        result["video"] = runner.best_path
        result["history"]["video"] = runner.history

    result["final"] = result["video"] or result["base"]
    if not result["final"]:
        raise RuntimeError("Neither stage ran, so there is no model to write.")
    with open(os.path.join(params["output_dir"], "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    log(f"Training complete: {result['final']}")


if __name__ == "__main__":
    # viame/pytorch/ holds helper packages named torchvision and netharn; with
    # the script directory on sys.path they would shadow the installed ones.
    _here = {os.path.dirname(os.path.abspath(__file__)),
             os.path.dirname(os.path.realpath(__file__))}
    sys.path[:] = [p for p in sys.path if p and os.path.abspath(p) not in _here]
    with open(sys.argv[1]) as _f:
        run(json.load(_f))
