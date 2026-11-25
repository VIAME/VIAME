# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Standalone PTSEFormer training entrypoint.

ptseformer_trainer.py writes the dataset and a parameter file, then runs this
module as a subprocess, once per GPU (argv[1] is the parameter JSON; rank and
world size arrive in the environment). It replaces upstream's tools/train.py
and ILSVRC dataset layer; the model, matcher and criterion are upstream's.
"""

import json
import math
import os
import random
import signal
import sys
import time

RANK_ENV = "PTSEFORMER_RANK"
WORLD_ENV = "PTSEFORMER_WORLD_SIZE"
PORT_ENV = "PTSEFORMER_MASTER_PORT"


class SequenceDataset:
    """Key frame + reference frames from the same sequence, as upstream samples them.

    Training draws the references at random from the sequence's offset window
    (or uses its fixed offsets); validation uses the detector's causal offsets.
    """

    def __init__(self, json_path, resolution, num_references, train,
                 horizontal_flip=0.0, max_images=0):
        with open(json_path) as fh:
            data = json.load(fh)
        self.sequences = data["sequences"]
        self.images = data["images"]
        if max_images and len(self.images) > max_images:
            step = len(self.images) / float(max_images)
            self.images = [self.images[int(i * step)] for i in range(max_images)]
        self.resolution = tuple(resolution)
        self.num_references = int(num_references)
        self.train = train
        self.horizontal_flip = float(horizontal_flip)

    def __len__(self):
        return len(self.images)

    def _reference_offsets(self, seq):
        from viame.pytorch.ptseformer_common import causal_reference_offsets

        if seq.get("offsets"):
            return list(seq["offsets"])
        low, high = seq["window"]
        if not self.train:
            return causal_reference_offsets(low, high, self.num_references)
        span = list(range(low, high + 1))
        if len(span) >= self.num_references:
            return random.sample(span, self.num_references)
        return [random.choice(span) for _ in range(self.num_references)]

    @staticmethod
    def _read(path):
        import numpy as np
        from PIL import Image

        with Image.open(path) as img:
            return np.array(img.convert("RGB"))

    def __getitem__(self, idx):
        import torch
        from viame.pytorch.ptseformer_common import preprocess_frame

        # An unreadable frame is replaced by the next readable one, so every rank
        # still takes the same number of steps.
        cur = None
        for attempt in range(min(10, len(self.images))):
            entry = self.images[(idx + attempt) % len(self.images)]
            try:
                cur = self._read(entry["file"])
                break
            except Exception as exc:
                print(f"[PTSEFormer] Skipping unreadable image {entry['file']}: {exc}",
                      flush=True)
        if cur is None:
            raise RuntimeError("PTSEFormer: ten consecutive training images are unreadable")
        height, width = cur.shape[:2]
        flip = self.train and random.random() < self.horizontal_flip

        refs = []
        if entry["seq"] >= 0:
            seq = self.sequences[entry["seq"]]
            frames = seq["frames"]
            for offset in self._reference_offsets(seq):
                # Sequence edges clamp to the first/last frame, as upstream does.
                pos = min(max(entry["index"] + offset, 0), len(frames) - 1)
                ref = cur
                if pos != entry["index"]:
                    try:
                        ref = self._read(os.path.join(seq["folder"], frames[pos]))
                    except Exception:
                        ref = cur
                if ref.shape != cur.shape:
                    ref = cur
                refs.append(ref)
        else:
            # No sequence on disk: the frame is its own reference (upstream's DET images).
            refs = [cur] * self.num_references

        boxes = torch.tensor(
            [b[:4] for b in entry["boxes"]], dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor([int(b[4]) for b in entry["boxes"]], dtype=torch.int64)
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height)
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, labels = boxes[keep], labels[keep]
        if flip:
            boxes = torch.stack([width - boxes[:, 2], boxes[:, 1],
                                 width - boxes[:, 0], boxes[:, 3]], dim=1)
        scale = torch.tensor([width, height, width, height], dtype=torch.float32)
        xyxy = boxes / scale
        cxcywh = torch.cat([(xyxy[:, :2] + xyxy[:, 2:]) / 2, xyxy[:, 2:] - xyxy[:, :2]], 1)

        return {
            "cur": preprocess_frame(cur, self.resolution, "cpu", flip)[0],
            "refs": [preprocess_frame(r, self.resolution, "cpu", flip)[0] for r in refs],
            "target": {"boxes": cxcywh, "labels": labels},
            "xyxy": xyxy,
        }


class RankSampler:
    """Shuffled indices for one rank, padded so every rank steps equally often."""

    def __init__(self, length, rank, world, seed):
        self.length, self.rank, self.world, self.seed = length, rank, world, seed
        self.epoch = 0
        self.count = int(math.ceil(length / float(world)))

    def __len__(self):
        return self.count

    def __iter__(self):
        order = list(range(self.length))
        random.Random(self.seed + self.epoch).shuffle(order)
        order += order[:self.count * self.world - self.length]
        return iter(order[self.rank::self.world])


def first_item(batch):
    return batch[0]


def average_precision(detections, truths, iou_thresh=0.5):
    """Mean AP at one IoU over the classes that have groundtruth.

    detections: (image, class, score, xyxy) tuples; truths: {(image, class): [xyxy]}.
    """
    import numpy as np

    classes = sorted({key[1] for key, boxes in truths.items() if len(boxes)})
    aps = []
    for cls in classes:
        dets = sorted((d for d in detections if d[1] == cls), key=lambda d: -d[2])
        total = sum(len(b) for (_, c), b in truths.items() if c == cls)
        used = {key: np.zeros(len(b), dtype=bool)
                for key, b in truths.items() if key[1] == cls}
        hits = np.zeros(len(dets))
        for i, (image, _, _, box) in enumerate(dets):
            gt = truths.get((image, cls))
            if gt is None or not len(gt):
                continue
            gt = np.asarray(gt, dtype=np.float64)
            lt = np.maximum(gt[:, :2], box[:2])
            rb = np.minimum(gt[:, 2:], box[2:])
            inter = np.prod(np.clip(rb - lt, 0, None), axis=1)
            union = (np.prod(gt[:, 2:] - gt[:, :2], axis=1)
                     + (box[2] - box[0]) * (box[3] - box[1]) - inter)
            iou = np.where(used[(image, cls)], -1.0, inter / np.maximum(union, 1e-12))
            best = int(iou.argmax())
            if iou[best] >= iou_thresh:
                used[(image, cls)][best] = True
                hits[i] = 1
        tp = np.cumsum(hits)
        recall = tp / max(total, 1)
        precision = tp / np.maximum(np.arange(1, len(dets) + 1), 1)
        # Area under the monotone precision envelope.
        envelope = np.concatenate([[0.0], precision, [0.0]])
        for i in range(len(envelope) - 2, -1, -1):
            envelope[i] = max(envelope[i], envelope[i + 1])
        steps = np.concatenate([[0.0], recall, [recall[-1] if len(recall) else 0.0]])
        aps.append(float(np.sum((steps[1:] - steps[:-1]) * envelope[1:])))
    return float(np.mean(aps)) if aps else 0.0


def train(params):
    import numpy as np
    import torch

    from viame.pytorch import ptseformer_common as common

    rank = int(os.environ.get(RANK_ENV, "0"))
    world = int(os.environ.get(WORLD_ENV, "1"))
    main_rank = rank == 0
    devices = params["devices"]
    device = torch.device(devices[rank % len(devices)])
    output_dir = params["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    def log(message):
        if main_rank:
            print(f"[PTSEFormer] {message}", flush=True)

    seed = int(params["seed"]) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if world > 1:
        import datetime
        # NCCL needs one GPU per rank; anything else synchronises over gloo. The
        # long timeout covers ranks waiting while rank 0 validates.
        distinct = len(set(devices)) == len(devices) and device.type == "cuda"
        backend = "nccl" if distinct and torch.distributed.is_nccl_available() else "gloo"
        torch.distributed.init_process_group(
            backend, init_method=f"tcp://127.0.0.1:{os.environ[PORT_ENV]}",
            rank=rank, world_size=world, timeout=datetime.timedelta(hours=12))
        log(f"Distributed training: {world} ranks, backend {backend}")

    arch = params["arch"]
    resolution = tuple(params["resolution"])
    num_refs = int(params["num_reference_frames"])
    model = common.build_model(arch, lr_backbone=float(params["lr_backbone"]))
    if params.get("seed_model") and not params.get("resume_from"):
        log(common.seed_model(model, params["seed_model"]))
    elif not params.get("resume_from"):
        log("No seed_model: training from random initialisation")
    model.stage_checkpointing = bool(params["gradient_checkpointing"])
    model.to(device)
    criterion = common.build_criterion(arch["num_classes"]).to(device)
    weight_dict = criterion.weight_dict

    def in_group(name, keywords):
        return any(key in name for key in keywords)

    # Upstream's three learning-rate groups.
    backbone_names, proj_names = ["backbone.0"], ["reference_points", "sampling_offsets"]
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    lr = float(params["lr"])
    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in named if not in_group(n, backbone_names)
                    and not in_group(n, proj_names)], "lr": lr},
        {"params": [p for n, p in named if in_group(n, backbone_names)],
         "lr": float(params["lr_backbone"])},
        {"params": [p for n, p in named if in_group(n, proj_names)
                    and not in_group(n, backbone_names)],
         "lr": lr * float(params["lr_linear_proj_mult"])},
    ], lr=lr, weight_decay=float(params["weight_decay"]))
    epochs = int(params["epochs"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(params["lr_drop"]))

    start_epoch, best_ap, best_epoch, scaler_state = 0, -1.0, -1, None
    last_path = os.path.join(output_dir, "checkpoint.pth")
    best_path = os.path.join(output_dir, "checkpoint_best.pth")
    if params.get("resume_from"):
        state = common.read_checkpoint(params["resume_from"])
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["lr_scheduler"])
        scaler_state = state.get("scaler")
        start_epoch = int(state["epoch"]) + 1
        best_ap, best_epoch = float(state["best_ap"]), int(state["best_epoch"])
        log(f"Resumed from {params['resume_from']} at epoch {start_epoch}")

    train_model = model
    if world > 1:
        train_model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True)

    train_set = SequenceDataset(
        params["train_json"], resolution, num_refs, train=True,
        horizontal_flip=float(params["horizontal_flip"]))
    val_set = None
    if main_rank and params.get("val_json"):
        val_set = SequenceDataset(
            params["val_json"], resolution, num_refs, train=False,
            max_images=int(params["val_subsample"]))
        if not len(val_set):
            val_set = None
    workers = int(params["num_workers"])
    sampler = RankSampler(len(train_set), rank, world, int(params["seed"]))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=1, sampler=sampler, num_workers=workers,
        collate_fn=first_item, pin_memory=device.type == "cuda")
    log(f"{len(train_set)} training frames, "
        f"{len(val_set) if val_set else 0} validation frames, resolution "
        f"{resolution[0]}x{resolution[1]}, {num_refs} reference frame(s)")

    amp = bool(params["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    if amp and scaler_state:
        scaler.load_state_dict(scaler_state)
    accum = max(1, int(params["batch_size"]))
    reference_grad = bool(params["reference_grad"])
    clip = float(params["clip_max_norm"])
    print_interval = max(1, int(params["print_interval"]))
    deadline = time.time() + float(params["timeout"]) if float(params["timeout"]) > 0 else None

    stop = {"flag": False}

    def request_stop(signum, frame):
        stop["flag"] = True
        log("Interrupt received; finishing the current step and saving")

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, request_stop)
        except ValueError:
            pass

    def should_stop():
        local = stop["flag"] or (deadline is not None and time.time() > deadline)
        if world == 1:
            return local
        # Every rank must agree, or the others block in the next gradient sync.
        flag = torch.tensor([1.0 if local else 0.0], device=device)
        torch.distributed.all_reduce(flag)
        return flag.item() > 0

    def to_samples(item):
        return {
            "cur": common.to_nested(item["cur"][None].to(device, non_blocking=True)),
            "ref_l": [common.to_nested(r[None].to(device, non_blocking=True))
                      for r in item["refs"]],
        }

    def run_validation():
        model.eval()
        loader = torch.utils.data.DataLoader(
            val_set, batch_size=1, shuffle=False, num_workers=workers,
            collate_fn=first_item)
        detections, truths, losses = [], {}, []
        with torch.no_grad():
            for image_id, item in enumerate(loader):
                with torch.autocast(device.type, enabled=amp):
                    outputs = model(to_samples(item))
                outputs = {k: v.float() for k, v in outputs.items() if k != "aux_outputs"}
                target = {k: v.to(device) for k, v in item["target"].items()}
                loss_dict = criterion.local_losses(outputs, [target])
                losses.append(sum(loss_dict[k].item() * weight_dict[k]
                                  for k in weight_dict))
                boxes, scores = common.decode_outputs(
                    outputs, int(params["eval_max_dets"]))
                boxes, scores = boxes.cpu().numpy(), scores.cpu().numpy()
                for box, score in zip(boxes, scores):
                    cls = int(score.argmax())
                    detections.append((image_id, cls, float(score[cls]), box))
                for box, label in zip(item["xyxy"].numpy(), item["target"]["labels"]):
                    truths.setdefault((image_id, int(label)), []).append(box)
        return average_precision(detections, truths), float(np.mean(losses or [0.0]))

    def checkpoint(epoch, full):
        state = {"model": model.state_dict(), "args": params["checkpoint_args"]}
        if full:
            state.update(optimizer=optimizer.state_dict(),
                         lr_scheduler=scheduler.state_dict(),
                         scaler=scaler.state_dict(), epoch=epoch,
                         best_ap=best_ap, best_epoch=best_epoch)
        return state

    log(f"Training for {epochs} epoch(s), {len(train_loader)} frames per epoch"
        f"{' per GPU' if world > 1 else ''}, {accum} frame(s) per optimizer step")
    stopped = False
    for epoch in range(start_epoch, epochs):
        train_model.train()
        criterion.train()
        sampler.epoch = epoch
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        running, seen, skipped, tic = 0.0, 0, 0, time.time()
        total = len(train_loader)
        for step, item in enumerate(train_loader):
            last = step + 1 == total
            target = {k: v.to(device) for k, v in item["target"].items()}
            with torch.autocast(device.type, enabled=amp):
                outputs = train_model(to_samples(item), reference_grad=reference_grad)
            outputs = {k: v.float() for k, v in outputs.items() if k != "aux_outputs"}
            loss_dict = criterion(outputs, [target])
            loss = sum(loss_dict[k] * weight_dict[k] for k in weight_dict)
            if not math.isfinite(loss.item()):
                raise RuntimeError(
                    f"PTSEFormer loss is {loss.item()} at epoch {epoch} step "
                    f"{step}; lower the learning rate or turn amp off")
            scaler.scale(loss / accum).backward()
            running += loss.item()
            seen += 1
            if (step + 1) % accum == 0 or last:
                if clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # The scaler drops a step whose gradients overflowed and lowers its scale.
                skipped += scaler.get_scale() < scale
                optimizer.zero_grad(set_to_none=True)
                if should_stop():
                    stopped = True
                    break
            if (step + 1) % print_interval == 0 or last:
                mem = (torch.cuda.max_memory_allocated(device) / 2 ** 30
                       if device.type == "cuda" else 0.0)
                log(f"epoch {epoch + 1}/{epochs} step {step + 1}/{total} "
                    f"loss {running / max(seen, 1):.4f} "
                    f"lr {optimizer.param_groups[0]['lr']:.2e} "
                    f"{(step + 1) / (time.time() - tic):.2f} it/s "
                    f"peak mem {mem:.1f} GB")
        scheduler.step()
        if skipped:
            log(f"epoch {epoch + 1}/{epochs}: {skipped} optimizer step(s) skipped "
                "for non-finite mixed-precision gradients")

        if main_rank:
            entry = {"epoch": epoch, "train_loss": running / max(seen, 1)}
            evaluate = val_set is not None and (
                (epoch + 1) % max(1, int(params["eval_interval"])) == 0
                or epoch + 1 == epochs or stopped)
            if evaluate:
                ap50, val_loss = run_validation()
                entry.update(val_ap50=ap50, val_loss=val_loss)
                log(f"epoch {epoch + 1}/{epochs} validation AP50 {ap50:.4f} "
                    f"loss {val_loss:.4f}")
                if ap50 >= best_ap:
                    best_ap, best_epoch = ap50, epoch
                    torch.save(checkpoint(epoch, full=False), best_path)
                    log(f"New best checkpoint (epoch {epoch + 1}, AP50 {ap50:.4f})")
            elif val_set is None:
                # No validation data: the latest epoch is the model.
                best_epoch = epoch
                torch.save(checkpoint(epoch, full=False), best_path)
            torch.save(checkpoint(epoch, full=True), last_path)
            with open(os.path.join(output_dir, "log.txt"), "a") as fh:
                fh.write(json.dumps(entry) + "\n")
        if stopped:
            log("Stopping early (interrupt or timeout)")
            break

    log(f"Training finished; best epoch {best_epoch + 1}"
        + (f" (validation AP50 {best_ap:.4f})" if best_ap >= 0 else ""))
    if world > 1:
        torch.distributed.destroy_process_group()


def main(argv):
    # As a script this folder leads sys.path, where the torchvision/ plugin
    # package would shadow the real torchvision.
    here = {os.path.realpath(os.path.dirname(__file__)),
            os.path.dirname(os.path.realpath(__file__))}
    sys.path[:] = [p for p in sys.path if os.path.realpath(p or os.getcwd()) not in here]
    with open(argv[1]) as fh:
        params = json.load(fh)
    train(params)


if __name__ == "__main__":
    main(sys.argv)
