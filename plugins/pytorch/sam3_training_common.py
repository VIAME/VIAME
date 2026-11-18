# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Pieces shared by the SAM 3.1 fine-tuning engines (sam3_detector_training.py
and sam3_tracker_training.py): checkpoint I/O, the pre-Ampere attention
workaround, and the distributed step logic.

Like the engines, this has no kwiver imports, so DDP ranks launched as plain
scripts can use it.
"""

import contextlib
import math
import os
import random
import time

import numpy as np

MODEL_SIZE = 1008                 # SAM 3.1 input resolution


# =============================================================================
# Checkpoints
# =============================================================================

def load_full_checkpoint(path):
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    return ckpt


def is_sam31_checkpoint(state):
    return any(k.startswith("tracker.model.") for k in state)


def require_sam31(state, path, what):
    if not is_sam31_checkpoint(state):
        raise RuntimeError(
            f"{path} is not a SAM 3.1 checkpoint (no tracker.model.* keys). "
            f"The SAM3 {what} trainer fine-tunes SAM 3.1; point sam_model_id "
            "at sam3.1_weights.pt.")


def write_into_checkpoint(source_ckpt, out_path, prefix, state, skip=()):
    """
    Save a copy of ``source_ckpt`` with ``state`` written under ``prefix``
    (keys starting with any of ``skip`` are left alone), and check that no
    weight outside ``prefix`` changed.
    """
    import torch
    base = load_full_checkpoint(source_ckpt)
    full = dict(base)
    for k, v in state.items():
        if any(k.startswith(s) for s in skip):
            continue
        fk = prefix + k
        if fk not in full or full[fk].shape != v.shape:
            raise ValueError(f"{fk} does not match the source checkpoint")
        full[fk] = v.detach().to(full[fk].dtype).cpu().contiguous()
    for k in base:
        if not k.startswith(prefix) and not torch.equal(full[k], base[k]):
            raise RuntimeError(f"weight {k} outside {prefix} changed on export")
    atomic_save(full, out_path)
    return out_path


def atomic_save(obj, path):
    import torch
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# =============================================================================
# Model helpers
# =============================================================================

def patch_sdpa_for_pre_ampere():
    """
    sam3/model/decoder.py forces sdpa_kernel(FLASH_ATTENTION), which has no
    kernel before Ampere. Same workaround as
    sam3_utilities._patch_sam3_sdpa_for_pre_ampere, kept here so ranks need
    no kwiver imports.
    """
    import torch
    try:
        if torch.cuda.get_device_capability(0)[0] >= 8:
            return
    except Exception:
        return
    from sam3.model import decoder as _dec

    @contextlib.contextmanager
    def _noop(*args, **kwargs):
        yield
    _dec.sdpa_kernel = _noop


def to_float(x):
    """Cast fp16 tensors (in nested containers) back to fp32."""
    import torch
    if torch.is_tensor(x) and x.dtype in (torch.float16, torch.bfloat16):
        return x.float()
    if isinstance(x, (tuple, list)):
        return type(x)(to_float(v) for v in x)
    if isinstance(x, dict):
        return {k: to_float(v) for k, v in x.items()}
    return x


# =============================================================================
# Distributed training loop pieces
# =============================================================================

def setup_distributed(seed):
    """Init the process group when launched by torch.distributed.run."""
    import torch
    import torch.distributed as dist
    distributed = "RANK" in os.environ and int(
        os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank, world, local = 0, 1, 0
    torch.cuda.set_device(local)
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    return distributed, rank, world, torch.device("cuda", local)


def cosine_lr(i, lr0, warmup, iters, floor=0.05):
    if i < warmup:
        return lr0 * (i + 1) / warmup
    frac = (i - warmup) / max(1, iters - warmup)
    return lr0 * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * frac)))


def adamw(params, lr, weight_decay):
    """AdamW with no decay on biases and norms."""
    import torch
    return torch.optim.AdamW(
        [{"params": [q for q in params if q.ndim > 1],
          "weight_decay": weight_decay},
         {"params": [q for q in params if q.ndim <= 1], "weight_decay": 0.0}],
        lr=lr)


def sync_and_step(params, opt, scaler, skip, distributed, world, clip,
                  device):
    """
    Average gradients across ranks and step.

    The *scaled* gradients are all-reduced before ``unscale_`` so every rank
    makes the same skip-on-inf decision; unscaling first lets a rank step on
    another rank's inf gradients and poison the weights. ``skip`` (this
    rank ran out of memory) rides along in the same all-reduce, so all
    ranks skip together. Returns (stepped, grad_norm).
    """
    import torch
    import torch.distributed as dist
    for q in params:
        if q.grad is None:
            q.grad = torch.zeros_like(q)
    if distributed:
        flat = torch.cat([q.grad.flatten() for q in params] +
                         [torch.tensor([float(skip)], device=device)])
        dist.all_reduce(flat)
        skip = float(flat[-1])
        flat /= world
        off = 0
        for q in params:
            n = q.numel()
            q.grad.copy_(flat[off:off + n].view_as(q))
            off += n
    if skip:
        opt.zero_grad(set_to_none=True)
        return False, float("nan")
    scaler.unscale_(opt)
    gn = torch.nn.utils.clip_grad_norm_(params, clip)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    return True, float(gn)


def any_rank(flag, distributed, device):
    """True on every rank if ``flag`` is true on any rank."""
    if not distributed:
        return bool(flag)
    import torch
    import torch.distributed as dist
    t = torch.tensor([float(bool(flag))], device=device)
    dist.all_reduce(t)
    return t.item() > 0


def out_of_time(t_start, timeout, distributed, device):
    late = timeout is not None and time.time() - t_start > timeout
    return any_rank(late, distributed, device)


def all_mean(values, distributed, device, weights=None):
    """
    Average a list of floats across ranks, each weighted by ``weights``
    (e.g. sample counts). NaNs count as zero.
    """
    if not distributed:
        return list(values)
    import torch
    import torch.distributed as dist
    n = len(values)
    w = weights if weights is not None else [1.0] * n
    v = torch.tensor([x * wi for x, wi in zip(values, w)] + list(w),
                     device=device, dtype=torch.float64)
    v = torch.nan_to_num(v)
    dist.all_reduce(v)
    return [(v[i] / v[n + i].clamp(min=1e-12)).item() for i in range(n)]
