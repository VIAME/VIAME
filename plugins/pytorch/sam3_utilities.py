# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Shared utilities for SAM3 (SAM 2.1) based algorithms.

This module provides common functionality used by sam3_tracker,
sam3_refiner, sam3_segmenter, and sam3_text_query, including:
- Shared model cache for memory efficiency
- Model initialization (SAM2, Grounding DINO)
- Text-based object detection
- SAM segmentation with box prompts
- Mask to polygon/points conversion
- IoU computation
- Configuration management
- Inference context helpers
"""

import os
import sys
import json
import shutil
import tempfile
import threading
from typing import Optional, Tuple, Any

import scriptconfig as scfg
import numpy as np

# Import shared utilities from the base utilities module
# These are re-exported here for backward compatibility with sam3_* modules
from viame.pytorch.utilities import (
    mask_to_polygon,
    box_from_mask,
    image_to_rgb_numpy,
    get_autocast_context,
    parse_bool,
)


# =============================================================================
# SAM3 version detection (sam3 / sam3.1)
# =============================================================================

_SAM3_SDPA_PATCHED = False


def _flash_sdpa_runnable():
    """Probe whether torch's FLASH_ATTENTION SDPA backend is actually usable.

    Returns True only if a tiny SDPA call runs to completion under an
    explicit ``sdpa_kernel([FLASH_ATTENTION])`` context.  Returns False for
    pre-Ampere GPUs, CPU-only builds, and Windows PyTorch wheels that ship
    without a Flash kernel (where the forced context raises "No available
    kernel").
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        cap = torch.cuda.get_device_capability(0)
        if cap[0] < 8:
            return False
        from torch.nn.attention import sdpa_kernel, SDPBackend
        q = torch.randn(1, 2, 8, 16, device='cuda', dtype=torch.float16)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            torch.nn.functional.scaled_dot_product_attention(q, q, q)
        return True
    except Exception:
        return False


def _patch_sam3_sdpa_for_pre_ampere():
    """
    Work around a hard-coded ``sdpa_kernel(SDPBackend.FLASH_ATTENTION)`` in
    ``sam3/model/decoder.py::functional_attention``.

    Flash Attention requires Ampere (compute capability 8.0+) and a torch
    build that actually includes the Flash kernel.  Pre-Ampere GPUs (e.g.
    Turing RTX 5000) and Windows PyTorch wheels (which frequently omit
    Flash) both raise ``RuntimeError: No available kernel`` when the
    decoder forces Flash, even when callers have set up a fallback via
    their own ``sdpa_kernel`` context.

    This shim replaces ``sam3.model.decoder.sdpa_kernel`` with a no-op
    context manager whenever Flash isn't runnable, so the decoder falls
    back to whichever backend the caller (or PyTorch's autoselect) chooses.
    Systems where Flash works keep the original behavior.
    """
    global _SAM3_SDPA_PATCHED
    if _SAM3_SDPA_PATCHED:
        return
    try:
        if _flash_sdpa_runnable():
            _SAM3_SDPA_PATCHED = True
            return
        import contextlib
        from sam3.model import decoder as _sam3_decoder

        @contextlib.contextmanager
        def _noop_sdpa_kernel(*args, **kwargs):
            yield
        _sam3_decoder.sdpa_kernel = _noop_sdpa_kernel
        _SAM3_SDPA_PATCHED = True
    except Exception:
        pass


def detect_sam3_version(checkpoint_path: Optional[str] = None,
                        model_config_path: Optional[str] = None,
                        explicit: Optional[str] = None) -> str:
    """
    Decide whether a set of model artifacts are SAM 3.0 or SAM 3.1.

    SAM 3.1 introduces Object Multiplex (``sam3.1_multiplex.pt``) and uses a
    checkpoint layout that is *not* drop-in compatible with the 3.0 loaders in
    ``build_sam3_image_model`` / ``build_sam3_video_model``.  This helper picks
    the version so upstream code can dispatch to the right builder.

    Resolution order:
    1. ``explicit`` argument if it is ``'sam3'`` or ``'sam3.1'``.
    2. Hints in the checkpoint filename (``3.1``, ``3p1``, ``multiplex``).
    3. Hints in the sidecar ``config.json`` (``sam3.1`` substring).
    4. Peek at the checkpoint state-dict: 3.0 has top-level ``backbone.``
       keys, 3.1 wraps them under ``detector.`` / ``tracker.``.
    5. Default to ``'sam3'``.
    """
    if explicit and str(explicit).lower() not in ('', 'auto', 'none'):
        v = str(explicit).lower().replace('_', '.').replace('-', '.')
        if v in ('sam3', '3', '3.0', 'sam3.0'):
            return 'sam3'
        if v in ('sam3.1', '3.1', 'sam31'):
            return 'sam3.1'

    name = os.path.basename(str(checkpoint_path or '')).lower()
    if '3.1' in name or '3p1' in name or 'multiplex' in name:
        return 'sam3.1'

    if model_config_path and os.path.exists(str(model_config_path)):
        try:
            with open(model_config_path, 'r') as f:
                txt = f.read()
            if 'sam3.1' in txt.lower() or 'multiplex' in txt.lower():
                return 'sam3.1'
        except Exception:
            pass

    if checkpoint_path and os.path.exists(str(checkpoint_path)):
        try:
            import torch
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            if isinstance(ckpt, dict) and 'model' in ckpt and isinstance(ckpt['model'], dict):
                ckpt = ckpt['model']
            if isinstance(ckpt, dict):
                has_detector_prefix = any(k.startswith('detector.') for k in ckpt)
                has_tracker_prefix = any(k.startswith('tracker.') for k in ckpt)
                if has_detector_prefix and has_tracker_prefix:
                    # Both 3.0 and 3.1 have detector./tracker. prefixes.
                    # 3.1 multiplex wraps the tracker in a PredictorWrapper,
                    # adding a .model. level: tracker.model.backbone.* etc.
                    # 3.0 has tracker.backbone.* directly (no .model. level).
                    has_wrapper = any(
                        k.startswith('tracker.model.') for k in ckpt
                    )
                    if has_wrapper:
                        return 'sam3.1'
                    else:
                        return 'sam3'
        except Exception:
            pass

    return 'sam3'


# =============================================================================
# SAM 3.1 compatibility adapters
# =============================================================================
#
# The SAM 3.1 release replaced the SAM2-style interactive image predictor with
# the "Object Multiplex" video predictor (``Sam3MultiplexVideoPredictor``).  It
# only exposes a session-based ``handle_request`` API that expects a directory
# of frames on disk.  The existing VIAME plugins were written against the 3.0
# APIs (``predictor.set_image`` / ``predictor.predict`` for per-image
# segmentation, and the classic ``init_state`` / ``add_prompt`` /
# ``propagate_in_video`` / ``reset_state`` chain for video refinement).
#
# Rather than rewriting every plugin, we wrap the 3.1 model in two adapters
# that expose the 3.0 API shape.  The image adapter drives a fresh 1-frame
# session per ``set_image`` call via a scratch directory; the video adapter
# materializes a list of PIL frames to a scratch directory and routes the
# prompt/propagate calls through the underlying multiplex demo model (whose
# ``add_prompt`` / ``propagate_in_video`` / ``reset_state`` methods match the
# 3.0 signatures closely enough to pass through).


class _Sam3p1ImagePredictorAdapter:
    """
    Thin adapter that makes a ``Sam3MultiplexVideoPredictor`` look like the
    SAM2-style interactive image predictor the 3.0 plugins were written for.

    Contract preserved from the 3.0 predictor:

    * ``set_image(np_image)`` — load an image; subsequent ``predict`` calls
      apply to that image.
    * ``predict(box=..., point_coords=..., point_labels=..., mask_input=...,
      multimask_output=...)`` returns ``(masks, scores, low_res_masks)`` where
      ``masks`` is ``[N, H, W]`` boolean, ``scores`` is ``[N]`` float, and
      ``low_res_masks`` is ``None`` (the multiplex predictor does not expose
      the low-res logits, so multi-frame mask priming is unavailable in 3.1).

    Each ``set_image`` starts a fresh session on a temp directory holding the
    single frame.  Sessions are closed on the next ``set_image``, on
    ``close``, or on garbage collection.
    """

    def __init__(self, multiplex_predictor, device):
        self._p = multiplex_predictor
        self.device = device
        self.model = multiplex_predictor  # so code that does .model.parameters() works
        self._session_id = None
        self._tmp_dir = None
        self._orig_hw = None

    def _inference_ctx(self):
        """Context stack with autocast + SDPA fallback kernel selection.

        Mirrors ``_Sam3p1VideoPredictorAdapter._autocast``; see its docstring
        for the rationale.
        """
        import contextlib
        cm = contextlib.ExitStack()
        try:
            cm.enter_context(get_autocast_context(str(self.device)))
        except Exception:
            pass
        try:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            cm.enter_context(sdpa_kernel(
                [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
            ))
        except Exception:
            pass
        return cm

    def _close_session(self):
        if self._session_id is not None:
            try:
                self._p.handle_request({
                    "type": "close_session",
                    "session_id": self._session_id,
                })
            except Exception:
                pass
            self._session_id = None
        if self._tmp_dir is not None:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = None

    def set_image(self, image_np):
        from PIL import Image
        self._close_session()
        self._tmp_dir = tempfile.mkdtemp(prefix="sam3p1_img_")
        # SAM 3.1's image-folder loader expects zero-padded JPEG filenames
        # starting from 00000.jpg.
        Image.fromarray(image_np).save(
            os.path.join(self._tmp_dir, "00000.jpg"), quality=95
        )
        self._orig_hw = image_np.shape[:2]
        with self._inference_ctx():
            resp = self._p.handle_request({
                "type": "start_session",
                "resource_path": self._tmp_dir,
                "offload_video_to_cpu": False,
            })
        self._session_id = resp["session_id"]

    def predict(self, point_coords=None, point_labels=None, box=None,
                mask_input=None, multimask_output=False):
        if self._session_id is None:
            raise RuntimeError(
                "_Sam3p1ImagePredictorAdapter.predict called before set_image"
            )
        if mask_input is not None:
            # 3.1 multiplex add_prompt has no low-res-mask prior hook.  This
            # loses a bit of tracking consistency for the refiner, but the
            # box prompt still gives a usable mask.
            pass

        H, W = self._orig_hw
        boxes = None
        if box is not None:
            b = np.asarray(box, dtype=np.float32)
            if b.ndim == 1:
                b = b[None, :]
            elif b.ndim == 3:
                b = b.reshape(-1, 4)
            # 3.0 predictor accepts absolute xyxy; multiplex expects
            # *relative* xywh in add_prompt (boxes_xywh).
            x1 = b[:, 0] / float(W)
            y1 = b[:, 1] / float(H)
            x2 = b[:, 2] / float(W)
            y2 = b[:, 3] / float(H)
            boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=-1).tolist()

        points = None
        plabels = None
        if point_coords is not None:
            p = np.asarray(point_coords, dtype=np.float32)
            if p.ndim == 1:
                p = p[None, :]
            elif p.ndim == 3:
                p = p.reshape(-1, 2)
            px = p[:, 0] / float(W)
            py = p[:, 1] / float(H)
            points = np.stack([px, py], axis=-1).tolist()
            if point_labels is None:
                plabels = [1] * len(points)
            else:
                plabels = np.asarray(point_labels, dtype=np.int32).flatten().tolist()

        # Preserve 3.0 shape convention: a 1-D ``box`` argument yields a 3-D
        # ``[1, H, W]`` mask (consumers expect ``masks[0]``), while a 2-D
        # ``box`` / multi-box argument yields a 4-D ``[N, 1, H, W]`` tensor
        # (consumers iterate ``masks[i, 0]``).
        single_shot = (
            box is not None
            and np.asarray(box).ndim == 1
            and (point_coords is None or np.asarray(point_coords).ndim == 1)
        )

        n_prompts = max(
            len(boxes) if boxes is not None else 0,
            len(points) if points is not None else 0,
            1,
        )

        all_masks = []
        for i in range(n_prompts):
            req = {
                "type": "add_prompt",
                "session_id": self._session_id,
                "frame_index": 0,
                "obj_id": i + 1,
                "clear_old_points": True,
                "clear_old_boxes": True,
            }
            if boxes is not None:
                req["bounding_boxes"] = [boxes[i]]
                req["bounding_box_labels"] = [1]
            if points is not None:
                # One point per object in this simple adapter.
                req["points"] = [points[i]]
                req["point_labels"] = [plabels[i]]
            with self._inference_ctx():
                resp = self._p.handle_request(req)
            outputs = resp.get("outputs", {}) if isinstance(resp, dict) else {}

            masks_out = outputs.get("out_binary_masks", None)
            if masks_out is None:
                masks_out = np.zeros((1, H, W), dtype=bool)
            else:
                if hasattr(masks_out, "cpu"):
                    masks_out = masks_out.cpu().numpy()
                masks_out = np.asarray(masks_out).astype(bool)
                if masks_out.ndim == 2:
                    masks_out = masks_out[None]
                # pick the mask corresponding to this obj_id if multiple returned
                obj_ids = outputs.get("out_obj_ids", None)
                if obj_ids is not None:
                    if hasattr(obj_ids, "cpu"):
                        obj_ids = obj_ids.cpu().numpy()
                    obj_ids = np.asarray(obj_ids).astype(np.int64).tolist()
                    target = i + 1
                    if target in obj_ids:
                        idx = obj_ids.index(target)
                        masks_out = masks_out[idx:idx + 1]
                    else:
                        masks_out = np.zeros((1, H, W), dtype=bool)
                else:
                    masks_out = masks_out[:1]

            all_masks.append(masks_out[0])

        if single_shot:
            if all_masks:
                masks = np.stack(all_masks, axis=0)  # [1, H, W]
            else:
                masks = np.zeros((1, H, W), dtype=bool)
        else:
            if all_masks:
                masks = np.stack(all_masks, axis=0)[:, None, :, :]  # [N, 1, H, W]
            else:
                masks = np.zeros((0, 1, H, W), dtype=bool)
        scores = np.ones(len(masks), dtype=np.float32)
        return masks, scores, None

    def close(self):
        self._close_session()

    def __del__(self):
        try:
            self._close_session()
        except Exception:
            pass


class _Sam3p1VideoPredictorAdapter:
    """
    Exposes a 3.0-style ``init_state`` / ``add_prompt`` / ``propagate_in_video``
    / ``reset_state`` interface on top of the SAM 3.1 multiplex demo model.

    The 3.0 refiner code passes a list of PIL frames to ``init_state`` and
    then drives the predictor with per-frame prompts.  The 3.1 demo model
    only knows how to load frames from a directory, so this adapter writes
    the frames to a scratch directory during ``init_state`` and cleans it up
    on ``reset_state`` / GC.

    ``add_prompt`` and ``propagate_in_video`` are thin pass-throughs — the
    3.1 demo model accepts the same kwargs (``frame_idx``, ``text_str``,
    ``boxes_xywh``, ``box_labels``, ``obj_id``, ...) and yields outputs with
    the same ``out_obj_ids`` / ``out_boxes_xywh`` / ``out_binary_masks``
    keys the 3.0 caller already consumes.
    """

    def __init__(self, multiplex_predictor, device):
        self._p = multiplex_predictor
        self._model = multiplex_predictor.model  # Sam3MultiplexTrackingWithInteractivity
        self.device = device

    def _autocast(self):
        """
        Context manager stack that wraps an inference call with the shims
        SAM 3.1 needs on non-Ampere GPUs:

        1. ``torch.inference_mode`` — the multiplex model does in-place ops
           that assume inference mode.
        2. Autocast fp16 — matches the image predictor path and keeps
           intermediate tensors small.
        3. ``sdpa_kernel([MATH, EFFICIENT_ATTENTION])`` — forces a
           fallback SDPA backend.  On pre-Ampere GPUs the 3.1 decoder's
           ``scaled_dot_product_attention`` call fails with "No available
           kernel" when flash/cuDNN paths are attempted first; explicitly
           picking math / mem-efficient resolves it.
        """
        import contextlib, torch
        cm = contextlib.ExitStack()
        cm.enter_context(torch.inference_mode())
        try:
            cm.enter_context(get_autocast_context(str(self.device)))
        except Exception:
            pass
        try:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            cm.enter_context(sdpa_kernel(
                [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
            ))
        except Exception:
            pass
        return cm

    def init_state(self, resource, offload_video_to_cpu=False, **kwargs):
        """
        Initialize an inference state.  ``resource`` can be a directory path,
        a single image path, or a list of PIL Images.  The underlying
        ``Sam3MultiplexTrackingWithInteractivity.init_state`` accepts all of
        these via ``resource_path`` (its loader dispatches on the type), so
        this is a thin pass-through.
        """
        import inspect
        init_kwargs = dict(
            resource_path=resource,
            offload_video_to_cpu=offload_video_to_cpu,
        )
        sig = inspect.signature(self._model.init_state)
        for k, v in list(kwargs.items()):
            if k in sig.parameters:
                init_kwargs[k] = v
        with self._autocast():
            return self._model.init_state(**init_kwargs)

    def add_prompt(self, inference_state, frame_idx, **kwargs):
        import inspect
        sig = inspect.signature(self._model.add_prompt)
        filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
        with self._autocast():
            return self._model.add_prompt(
                inference_state=inference_state,
                frame_idx=frame_idx,
                **filtered,
            )

    def propagate_in_video(self, inference_state, **kwargs):
        import inspect
        sig = inspect.signature(self._model.propagate_in_video)
        filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
        with self._autocast():
            for item in self._model.propagate_in_video(
                inference_state=inference_state,
                **filtered,
            ):
                yield item

    def reset_state(self, inference_state):
        import torch
        with torch.inference_mode():
            self._model.reset_state(inference_state)


# =============================================================================
# Shared Model Cache for SAM3 Algorithms
# =============================================================================

class SharedSAM3ModelCache:
    """
    Thread-safe cache for SAM3 models to avoid loading duplicates.

    When both SAM3Segmenter and SAM3TextQuery are configured with the same
    checkpoint and device, they will share the same model instance.

    Usage:
        model, predictor = SharedSAM3ModelCache.get_or_create(
            checkpoint="/path/to/model.pt",
            model_config=None,
            device="cuda",
            logger=print
        )

        # Use the model/predictor with the lock:
        with SharedSAM3ModelCache.get_lock(checkpoint, model_config, device):
            predictor.set_image(image)
            masks, scores, _ = predictor.predict(...)
    """

    _cache = {}  # Key: (checkpoint, model_config, device) -> (model, predictor)
    _locks = {}  # Key: (checkpoint, model_config, device) -> threading.RLock
    _global_lock = threading.Lock()

    @classmethod
    def _make_key(cls, checkpoint: Optional[str], model_config: Optional[str],
                  device: str) -> Tuple[str, str, str]:
        """Create a cache key from configuration parameters."""
        return (checkpoint or "", model_config or "", str(device))

    @classmethod
    def get_lock(cls, checkpoint: Optional[str] = None,
                 model_config: Optional[str] = None,
                 device: str = "cuda") -> threading.RLock:
        """
        Get the lock for a specific model configuration.

        Use this lock when performing inference to ensure thread safety.

        Args:
            checkpoint: Path to model checkpoint
            model_config: Path to model config JSON
            device: Device string

        Returns:
            threading.RLock for the model configuration
        """
        key = cls._make_key(checkpoint, model_config, device)
        with cls._global_lock:
            if key not in cls._locks:
                cls._locks[key] = threading.RLock()
            return cls._locks[key]

    @classmethod
    def get_or_create(
        cls,
        checkpoint: Optional[str] = None,
        model_config: Optional[str] = None,
        device: str = "cuda",
        logger=None,
    ) -> Tuple[Any, Any]:
        """
        Get or create a shared SAM3 model instance.

        If a model with the same configuration already exists in the cache,
        return it. Otherwise, create a new one and cache it.

        Args:
            checkpoint: Path to model checkpoint (or HuggingFace model ID)
            model_config: Path to model config JSON (optional)
            device: Device to run on ('cuda', 'cpu', 'auto')
            logger: Optional logging function (e.g., print)

        Returns:
            Tuple of (model, predictor)
        """
        key = cls._make_key(checkpoint, model_config, device)

        with cls._global_lock:
            if key in cls._cache:
                if logger:
                    logger(f"Using cached SAM3 model for {key}")
                return cls._cache[key]

        # Load model outside global lock (loading can take time)
        if logger:
            logger(f"Loading new SAM3 model for {key}")

        model, predictor = cls._load_model(checkpoint, model_config, device, logger)

        with cls._global_lock:
            # Double-check in case another thread loaded it while we were loading
            if key not in cls._cache:
                cls._cache[key] = (model, predictor)
                if key not in cls._locks:
                    cls._locks[key] = threading.RLock()
            return cls._cache[key]

    @classmethod
    def _load_model(
        cls,
        checkpoint: Optional[str],
        model_config: Optional[str],
        device: str,
        logger=None,
    ) -> Tuple[Any, Any]:
        """
        Load the SAM3 model.

        Returns:
            Tuple of (model, predictor)
        """
        def log(msg):
            if logger:
                logger(msg)

        # Check if using local model files
        is_local = (
            (checkpoint and os.path.exists(checkpoint)) or
            (model_config and os.path.exists(model_config))
        )

        if is_local:
            return cls._load_local_model(checkpoint, model_config, device, log)
        else:
            return cls._load_hf_model(checkpoint, device, log)

    @classmethod
    def _load_local_model(cls, checkpoint, model_config, device, log):
        """Load model from local files."""
        # Determine model directory and paths
        if checkpoint and os.path.isdir(checkpoint):
            model_dir = checkpoint
            checkpoint = os.path.join(model_dir, 'model_weights.pt')
        elif checkpoint:
            model_dir = os.path.dirname(checkpoint)
        else:
            model_dir = os.path.dirname(model_config) if model_config else None

        log(f"  Loading from local: {model_dir or checkpoint}")

        version = detect_sam3_version(checkpoint, model_config)
        log(f"  Detected SAM3 version: {version}")

        try:
            if version == 'sam3.1':
                from sam3.model_builder import build_sam3_predictor
                _patch_sam3_sdpa_for_pre_ampere()

                # For SAM 3.1 the multiplex video predictor is the only model
                # we need: it powers interactive segmentation (via the image
                # adapter) and video refinement (via the video adapter).  We
                # deliberately do NOT also build the 3.0-style image model
                # here — the two would occupy roughly 2x the GPU memory and
                # the image model's interactive predictor weights are not
                # present in the 3.1 checkpoint anyway.
                multiplex = build_sam3_predictor(
                    checkpoint_path=checkpoint,
                    version='sam3.1',
                    compile=False,
                    use_fa3=False,
                    use_rope_real=False,
                )
                predictor = _Sam3p1ImagePredictorAdapter(multiplex, device)
                # The first tuple slot is historically the Sam3Image model,
                # consumed by callers for ``.parameters()`` / optional
                # ``mask_generator`` hooks.  Return the multiplex predictor
                # in its place — it exposes ``.model`` whose ``.parameters``
                # satisfies the device-probing code in the plugins.
                log("  Loaded via sam3 module (sam3.1 multiplex)")
                return multiplex, predictor

            # Default: SAM 3.0
            from sam3.model_builder import build_sam3_image_model

            model = build_sam3_image_model(
                checkpoint_path=checkpoint,
                device=device,
                eval_mode=True,
                load_from_HF=False,
                enable_segmentation=True,
                enable_inst_interactivity=True,
                compile=False,
            )

            predictor = model.inst_interactive_predictor
            if predictor is None:
                raise RuntimeError("Model does not have instance interactive predictor")
            # The predictor's internal tracker model may not share
            # the backbone with the parent model; fix this reference
            if (hasattr(predictor, 'model') and
                    hasattr(predictor.model, 'backbone') and
                    predictor.model.backbone is None and
                    hasattr(model, 'backbone') and
                    model.backbone is not None):
                predictor.model.backbone = model.backbone
            log("  Loaded via sam3 module (sam3.0)")
            return model, predictor
        except ImportError as e:
            raise RuntimeError(f"sam3 module not available: {e}. Install sam3 package and its dependencies (including decord).")
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM3 model: {e}")

    @classmethod
    def _load_hf_model(cls, checkpoint, device, log):
        """Load model from HuggingFace."""
        try:
            from sam3.model_builder import build_sam3_image_model

            model = build_sam3_image_model(
                checkpoint_path=checkpoint,
                device=device,
                eval_mode=True,
                load_from_HF=checkpoint is None,
                enable_segmentation=True,
                enable_inst_interactivity=True,
                compile=False,
            )

            predictor = model.inst_interactive_predictor
            if predictor is None:
                raise RuntimeError("Model does not have instance interactive predictor")
            log("  Loaded via sam3 module")
            return model, predictor
        except ImportError:
            pass

        # Fallback to transformers
        from transformers import Sam2Model, Sam2Processor

        model_id = checkpoint or "facebook/sam2.1-hiera-large"
        processor = Sam2Processor.from_pretrained(model_id)
        model = Sam2Model.from_pretrained(model_id).to(device)
        model.eval()

        # Create wrapper predictor
        predictor = _SharedSAM3PredictorWrapper(model, processor, device)
        log("  Loaded via transformers")
        return model, predictor

    @classmethod
    def clear(cls):
        """Clear all cached models. Useful for testing or memory cleanup."""
        with cls._global_lock:
            cls._cache.clear()
            cls._locks.clear()


class _SharedSAM3PredictorWrapper:
    """
    Wrapper to provide a SAM2-like predictor interface for HuggingFace SAM models.

    This is the shared version used by SharedSAM3ModelCache.
    """

    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device
        self._image_embeddings = None
        self._original_size = None
        self._inputs = None

    def set_image(self, image):
        """Set the image for prediction."""
        import torch
        from PIL import Image

        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        self._original_size = pil_image.size[::-1]  # (H, W)

        self._inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            self._image_embeddings = self.model.get_image_embeddings(self._inputs.pixel_values)

    def predict(self, point_coords=None, point_labels=None, box=None,
                mask_input=None, multimask_output=True):
        """Run prediction with the given prompts."""
        import torch

        if self._image_embeddings is None:
            raise RuntimeError("Must call set_image before predict")

        # Prepare inputs
        input_points = None
        input_labels = None

        if point_coords is not None:
            input_points = torch.tensor(point_coords, dtype=torch.float32, device=self.device)
            if input_points.ndim == 2:
                input_points = input_points.unsqueeze(0)
        if point_labels is not None:
            input_labels = torch.tensor(point_labels, dtype=torch.int64, device=self.device)
            if input_labels.ndim == 1:
                input_labels = input_labels.unsqueeze(0)

        input_boxes = None
        if box is not None:
            input_boxes = torch.tensor(box, dtype=torch.float32, device=self.device)
            if input_boxes.ndim == 1:
                input_boxes = input_boxes.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(
                image_embeddings=self._image_embeddings,
                input_points=input_points,
                input_labels=input_labels,
                input_boxes=input_boxes,
                multimask_output=multimask_output,
            )

        masks = outputs.pred_masks.squeeze(0).cpu().numpy()
        scores = outputs.iou_scores.squeeze(0).cpu().numpy()

        # Resize masks to original size if needed
        if masks.shape[-2:] != self._original_size:
            import cv2
            resized_masks = []
            for m in masks:
                resized = cv2.resize(
                    m.astype(np.float32),
                    (self._original_size[1], self._original_size[0]),
                    interpolation=cv2.INTER_LINEAR
                )
                resized_masks.append(resized > 0.5)
            masks = np.array(resized_masks)

        low_res_masks = masks

        return masks, scores, low_res_masks


class SAM3BaseConfig(scfg.DataConfig):
    """
    Base configuration for SAM3-based algorithms.

    Contains shared configuration parameters for SAM2 and Grounding DINO models.
    """
    # Model configuration
    sam_model_id = scfg.Value(
        "facebook/sam2.1-hiera-large",
        help='SAM model ID from HuggingFace, local path to weights (.pt), or directory'
    )
    model_config = scfg.Value(
        None,
        help='Path to SAM3 config JSON file (for local model loading)'
    )
    sam3_version = scfg.Value(
        'auto',
        help='SAM3 model version: "auto" (detect from checkpoint/config), '
             '"sam3" (3.0), or "sam3.1" (Object Multiplex)'
    )
    grounding_model_id = scfg.Value(
        "IDEA-Research/grounding-dino-tiny",
        help='Grounding DINO model ID for text-based detection'
    )

    # Device configuration
    device = scfg.Value('cuda', help='Device to run models on (cuda, cpu, auto)')

    # Text query configuration
    text_query = scfg.Value(
        'object',
        help='Text query describing objects. Can be comma-separated for multiple classes.'
    )

    # Detection thresholds
    detection_threshold = scfg.Value(
        0.3,
        help='Confidence threshold for text-based detections'
    )
    text_threshold = scfg.Value(
        0.25,
        help='Text matching threshold for grounding detection'
    )

    # Output configuration
    output_type = scfg.Value(
        'polygon',
        help='Type of output: "polygon" for mask contours, "points" for centroid/keypoints, "both"'
    )
    polygon_simplification = scfg.Value(
        0.01,
        help='Douglas-Peucker simplification epsilon (relative to perimeter). 0 to disable.'
    )
    num_points = scfg.Value(
        5,
        help='Number of points to output when output_type includes points'
    )

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.text_query, str):
            self.text_query_list = [q.strip() for q in self.text_query.split(',')]
        elif isinstance(self.text_query, (list, tuple)):
            # scriptconfig may convert comma-separated strings to lists
            self.text_query_list = [str(q).strip() for q in self.text_query]
        else:
            self.text_query_list = [str(self.text_query)]


class SAM3ModelManager:
    """
    Manages SAM2 and Grounding DINO model initialization and inference.

    This class provides a shared interface for model operations used by
    both sam3_tracker and sam3_refiner.
    """

    def __init__(self):
        self._sam_predictor = None
        self._sam_model = None
        self._sam_processor = None
        self._grounding_processor = None
        self._grounding_model = None
        self._video_predictor = None
        self._device = None

    @property
    def device(self):
        return self._device

    def init_models(self, config, use_video_predictor=False):
        """
        Initialize SAM and Grounding DINO models.

        Args:
            config: Configuration object with model paths and device settings
                    (should have sam_model_id, grounding_model_id, device attributes)
            use_video_predictor: If True, initialize SAM2 video predictor
                                 instead of image predictor
        """
        from viame.pytorch.utilities import resolve_device

        self._device = resolve_device(config.device)

        # Initialize Grounding DINO (if model ID provided and not disabled)
        grounding_model_id = getattr(config, 'grounding_model_id', None)
        if grounding_model_id and str(grounding_model_id).lower() not in ('', 'none', 'false'):
            self._init_grounding_dino(grounding_model_id)

        # Check if using local SAM3 model files
        model_config = getattr(config, 'model_config', None)
        sam_model_id = config.sam_model_id

        # Determine if this is a local model (path to .pt file or directory with config)
        is_local = self._is_local_model(sam_model_id, model_config)

        if is_local:
            self._init_sam3_local(sam_model_id, model_config, use_video_predictor)
        elif use_video_predictor:
            self._init_sam2_video(sam_model_id)
        else:
            self._init_sam2_image(sam_model_id)

    def _is_local_model(self, model_id, model_config):
        """Check if the model should be loaded from local files."""
        import os
        if model_config and os.path.exists(str(model_config)):
            return True
        if model_id and os.path.exists(str(model_id)):
            return True
        return False

    def _init_sam3_local(self, weights_path, config_path, use_video_predictor=False):
        """
        Initialize SAM3 from local model files.

        Args:
            weights_path: Path to sam3_weights.pt or model directory
            config_path: Path to sam3_config.json
            use_video_predictor: If True, initialize for video prediction
        """
        import os
        import json
        import torch

        print(f"[SAM3] Loading SAM3 from local files...")
        print(f"[SAM3]   Weights: {weights_path}")
        print(f"[SAM3]   Config: {config_path}")

        # Determine model directory
        if os.path.isdir(weights_path):
            model_dir = weights_path
            weights_path = os.path.join(model_dir, 'sam3_weights.pt')
        else:
            model_dir = os.path.dirname(weights_path)

        # If config_path not provided, look for it in model directory
        if not config_path:
            config_path = os.path.join(model_dir, 'sam3_config.json')

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"SAM3 weights not found: {weights_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"SAM3 config not found: {config_path}")

        # Try to load using transformers library first, then fall back to
        # native sam3 module.  Only print errors if ALL methods fail.
        transformers_err = None
        native_err = None

        try:
            self._init_sam3_transformers(model_dir, weights_path, config_path, use_video_predictor)
            return
        except Exception as e:
            transformers_err = e

        try:
            self._init_sam3_native(weights_path, config_path, use_video_predictor)
            return
        except Exception as e:
            native_err = e

        print(f"[SAM3] Could not load via transformers: {transformers_err}")
        print(f"[SAM3] Could not load via native sam3: {native_err}")
        raise RuntimeError("Failed to load SAM3 model from local files")

    def _init_sam3_transformers(self, model_dir, weights_path, config_path, use_video_predictor):
        """Load SAM3 using HuggingFace transformers library."""
        import os
        import json
        import torch

        # Load config to determine model type
        with open(config_path, 'r') as f:
            config_data = json.load(f)

        model_type = config_data.get('model_type', 'sam3_video')

        # Check if model_type is sam3_* which requires custom transformers
        if model_type.startswith('sam3'):
            # Try loading via a registered Sam3 model first
            try:
                from transformers import AutoModel, AutoConfig
                model_config = AutoConfig.from_pretrained(
                    model_dir,
                    local_files_only=True
                )
                self._sam_model = AutoModel.from_pretrained(
                    model_dir,
                    config=model_config,
                    local_files_only=True
                ).to(self._device)
                self._sam_model.eval()
                print(f"[SAM3] Successfully loaded SAM3 via transformers AutoModel")
                self._setup_predictor_interface(use_video_predictor)
                return
            except ValueError:
                pass  # model_type not registered — try fallbacks below

        # Fallback: Try loading as Sam2 model (for sam2_* model types or as fallback)
        try:
            from transformers import Sam2Model, Sam2Processor

            processor_config_path = os.path.join(model_dir, 'sam3_processor_config.json')
            if os.path.exists(processor_config_path):
                try:
                    self._sam_processor = Sam2Processor.from_pretrained(
                        model_dir,
                        local_files_only=True
                    )
                except Exception:
                    pass

            self._sam_model = Sam2Model.from_pretrained(
                model_dir,
                local_files_only=True
            ).to(self._device)
            self._sam_model.eval()
            print(f"[SAM3] Successfully loaded model via Sam2Model")
            self._setup_predictor_interface(use_video_predictor)
            return
        except Exception:
            pass

        # Try standard SAM2 Video model for video predictor
        if use_video_predictor:
            try:
                from transformers import Sam2VideoModel, Sam2VideoProcessor
                self._sam_processor = Sam2VideoProcessor.from_pretrained(
                    model_dir,
                    local_files_only=True
                )
                self._sam_model = Sam2VideoModel.from_pretrained(
                    model_dir,
                    local_files_only=True
                ).to(self._device)
                self._sam_model.eval()
                print(f"[SAM3] Successfully loaded model via Sam2VideoModel")
                self._setup_predictor_interface(use_video_predictor)
                return
            except Exception:
                pass

        raise RuntimeError(
            f"Could not load SAM3 model via transformers. "
            f"Model type '{model_type}' may require custom transformers with Sam3 support."
        )

    def _setup_predictor_interface(self, use_video_predictor):
        """Set up the predictor interface from the loaded model."""
        if self._sam_model is None:
            return

        # Set up predictor interface based on model capabilities
        if hasattr(self._sam_model, 'get_image_predictor'):
            self._sam_predictor = self._sam_model.get_image_predictor()
        elif hasattr(self._sam_model, 'image_predictor'):
            self._sam_predictor = self._sam_model.image_predictor

        if use_video_predictor:
            if hasattr(self._sam_model, 'get_video_predictor'):
                self._video_predictor = self._sam_model.get_video_predictor()
            elif hasattr(self._sam_model, 'video_predictor'):
                self._video_predictor = self._sam_model.video_predictor

    def _init_sam3_native(self, weights_path, config_path, use_video_predictor):
        """Load SAM3 using native sam3 module if available."""
        try:
            version = detect_sam3_version(weights_path, config_path)
            print(f"[SAM3] Detected version: {version}")

            if version == 'sam3.1':
                from sam3.model_builder import build_sam3_predictor
                _patch_sam3_sdpa_for_pre_ampere()

                # Build the multiplex predictor once and expose it through
                # both adapters.  We deliberately skip the separate 3.0-style
                # image model build to keep GPU memory manageable — one 3.1
                # model is enough.
                multiplex = build_sam3_predictor(
                    checkpoint_path=weights_path,
                    version='sam3.1',
                    compile=False,
                    use_fa3=False,
                    use_rope_real=False,
                )

                if use_video_predictor:
                    self._video_predictor = _Sam3p1VideoPredictorAdapter(
                        multiplex, self._device,
                    )

                # Always produce an image predictor too — the existing tracker
                # plugin calls init_models(use_video_predictor=True) but then
                # segment_with_sam uses self._sam_predictor.  Keep both wired.
                self._sam_model = multiplex
                self._sam_predictor = _Sam3p1ImagePredictorAdapter(
                    multiplex, self._device,
                )
                print(f"[SAM3] Successfully loaded SAM3.1 multiplex via native sam3 module")
                return

            # SAM 3.0 path (unchanged)
            from sam3.model_builder import build_sam3_video_model, build_sam3_image_model

            if use_video_predictor:
                self._video_predictor = build_sam3_video_model(
                    checkpoint_path=weights_path,
                    device=str(self._device),
                    load_from_HF=False,
                )
            else:
                model = build_sam3_image_model(
                    checkpoint_path=weights_path,
                    device=str(self._device),
                    eval_mode=True,
                    load_from_HF=False,
                    enable_segmentation=True,
                    enable_inst_interactivity=True,
                )
                if hasattr(model, 'inst_interactive_predictor'):
                    self._sam_predictor = model.inst_interactive_predictor
                    # The predictor's internal tracker model may not share
                    # the backbone with the parent model; fix this reference
                    if (hasattr(self._sam_predictor, 'model') and
                            hasattr(self._sam_predictor.model, 'backbone') and
                            self._sam_predictor.model.backbone is None and
                            hasattr(model, 'backbone') and
                            model.backbone is not None):
                        self._sam_predictor.model.backbone = model.backbone
                else:
                    self._sam_predictor = model
                self._sam_model = model

            print(f"[SAM3] Successfully loaded SAM3 via native sam3 module")
        except ImportError:
            raise ImportError("sam3 module not available for native loading")

    def _init_grounding_dino(self, model_id):
        """Initialize Grounding DINO for text-based detection.

        If model_id is a local directory, loads from there. Otherwise
        falls back to downloading from HuggingFace.
        """
        import os
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

            if os.path.isdir(str(model_id)):
                print(f"[SAM3] Loading Grounding DINO from local: {model_id}")

            self._grounding_processor = AutoProcessor.from_pretrained(model_id)
            self._grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_id
            ).to(self._device)
            self._grounding_model.eval()
            print(f"[SAM3] Grounding DINO loaded successfully")
        except Exception as e:
            print(f"[SAM3] Warning: Could not load Grounding DINO: {e}")
            self._grounding_model = None

    def _init_sam2_image(self, model_id):
        """Initialize SAM2 for single image prediction."""
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            sam_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            model = build_sam2(
                config_file=sam_cfg,
                ckpt_path=model_id,
                device=str(self._device),
                mode='eval',
            )
            self._sam_predictor = SAM2ImagePredictor(model)
        except ImportError:
            self._init_sam2_huggingface(model_id)

    def _init_sam2_video(self, model_id):
        """Initialize SAM2 for video prediction."""
        try:
            from sam2.build_sam import build_sam2_video_predictor

            sam_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            self._video_predictor = build_sam2_video_predictor(
                sam_cfg, model_id, device=self._device
            )
        except ImportError:
            self._init_sam2_huggingface(model_id)

    def _init_sam2_huggingface(self, model_id):
        """Fallback: Initialize SAM2 via HuggingFace transformers."""
        try:
            from transformers import Sam2Model, Sam2Processor

            self._sam_processor = Sam2Processor.from_pretrained(model_id)
            self._sam_model = Sam2Model.from_pretrained(model_id).to(self._device)
            self._sam_model.eval()
        except Exception as e:
            print(f"[SAM3] Warning: Could not load SAM2: {e}")

    def detect_with_text(self, image_np, text_query_list,
                         detection_threshold, text_threshold):
        """
        Detect objects in image using text query via Grounding DINO.

        Args:
            image_np: RGB image as numpy array
            text_query_list: List of text labels to detect
            detection_threshold: Confidence threshold
            text_threshold: Text matching threshold

        Returns:
            List of (box, score, class_name) tuples where box is [x1, y1, x2, y2]
        """
        if self._grounding_model is None:
            return []

        import torch
        from PIL import Image

        pil_img = Image.fromarray(image_np)
        # Grounding DINO expects a single string with labels separated
        # by periods, e.g. "object. animal. fish."
        text_str = '. '.join(text_query_list) + '.'

        inputs = self._grounding_processor(
            images=pil_img, text=text_str, return_tensors="pt"
        ).to(self._device)

        with torch.no_grad():
            outputs = self._grounding_model(**inputs)

        results = self._grounding_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=detection_threshold,
            text_threshold=text_threshold,
            target_sizes=[pil_img.size[::-1]]
        )

        detections = []
        if len(results) > 0:
            result = results[0]
            for box, score, label in zip(
                result["boxes"], result["scores"], result["text_labels"]
            ):
                box_np = box.cpu().numpy()
                score_val = float(score.cpu().numpy())
                detections.append((box_np, score_val, label))

        return detections

    def segment_with_sam(self, image_np, boxes):
        """
        Segment objects in image using SAM with box prompts.

        Args:
            image_np: RGB image as numpy array
            boxes: List of [x1, y1, x2, y2] bounding boxes

        Returns:
            List of binary masks (numpy arrays)
        """
        if len(boxes) == 0:
            return []

        import torch

        # Use SAM2 image predictor if available
        if self._sam_predictor is not None:
            with get_autocast_context(self._device):
                self._sam_predictor.set_image(image_np)

                prompts = {
                    'box': np.array(boxes),
                    'multimask_output': False
                }

                with torch.inference_mode():
                    masks, scores, _ = self._sam_predictor.predict(**prompts)

            # Handle shape - ensure we have [N, 1, H, W] or similar
            if len(masks.shape) == 3:
                masks = masks[None, :, :, :]

            return [masks[i, 0] for i in range(len(boxes))]

        # Fallback to HuggingFace SAM2
        if self._sam_model is not None:
            from PIL import Image

            pil_img = Image.fromarray(image_np)
            masks = []

            for box in boxes:
                inputs = self._sam_processor(
                    images=pil_img,
                    input_boxes=[[[box[0], box[1], box[2], box[3]]]],
                    return_tensors="pt"
                ).to(self._device)

                with torch.no_grad():
                    outputs = self._sam_model(**inputs)

                mask = outputs.pred_masks[0, 0, 0].cpu().numpy() > 0
                masks.append(mask)

            return masks

        # No SAM model - return full masks
        return [np.ones((image_np.shape[0], image_np.shape[1]), dtype=bool)] * len(boxes)

    def segment_single_with_mask(self, image_np, box, mask_input=None):
        """
        Segment a single object using a box prompt and optional mask prior.

        Passing the low-res mask logits from the previous frame as
        ``mask_input`` gives SAM temporal context and dramatically improves
        tracking consistency compared to a bare box prompt.

        Args:
            image_np: RGB image as numpy array (must call set_image first
                      or this will call it automatically).
            box: [x1, y1, x2, y2] bounding box.
            mask_input: Optional low-resolution mask logits from a previous
                        prediction (shape ``[1, 256, 256]``).  Returned as
                        the third element of the SAM predictor's output.

        Returns:
            (mask, low_res_mask) where *mask* is a binary HxW numpy array and
            *low_res_mask* is the low-resolution logits suitable for feeding
            back as ``mask_input`` on the next frame.
        """
        import torch

        if self._sam_predictor is None:
            h, w = image_np.shape[:2]
            return np.ones((h, w), dtype=bool), None

        with get_autocast_context(self._device):
            self._sam_predictor.set_image(image_np)

            prompts = {
                'box': np.array(box),
                'multimask_output': False,
            }
            if mask_input is not None:
                prompts['mask_input'] = mask_input

            with torch.inference_mode():
                masks, scores, low_res_masks = self._sam_predictor.predict(**prompts)

        # masks shape: [1, H, W] or [H, W]
        if len(masks.shape) == 3:
            mask = masks[0]
        else:
            mask = masks

        # low_res_masks shape: [1, 256, 256]
        low_res = low_res_masks if low_res_masks is not None else None

        return mask, low_res


def mask_to_points(mask, num_points):
    """
    Extract representative points from a mask.

    Args:
        mask: Binary mask as numpy array
        num_points: Number of points to extract

    Returns:
        List of (x, y) tuples
    """
    import cv2

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return []

    contour = max(contours, key=cv2.contourArea)

    M = cv2.moments(contour)
    if M["m00"] == 0:
        return []

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    points = [(cx, cy)]

    if num_points > 1 and len(contour) > 0:
        step = max(1, len(contour) // (num_points - 1))
        for i in range(0, len(contour), step):
            if len(points) >= num_points:
                break
            pt = contour[i].squeeze()
            points.append((int(pt[0]), int(pt[1])))

    return points[:num_points]


def compute_iou(box1, box2):
    """
    Compute IoU between two boxes [x1, y1, x2, y2].

    Args:
        box1: First box as [x1, y1, x2, y2]
        box2: Second box as [x1, y1, x2, y2]

    Returns:
        float: IoU value in [0, 1]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area
