# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Generic ONNX object-detector predictor.

Runs an arbitrary object-detection ONNX graph with onnxruntime -- no PyTorch.
It is a generalization of the ``kwcoco_detector_kit`` OnnxPredictor (formerly
vendored under plugins/pytorch, now superseded by this module): that one is
hard-wired to the DEIMv2/RT-DETR contract (inputs ``images`` +
``orig_target_sizes``; three outputs ``labels, boxes_xyxy, scores`` with NMS
baked into the graph). This predictor keeps that as one *decoder* among several
and drives everything else -- input size, preprocessing, thresholds, class
names, and the I/O contract -- from a ``.modelspec.json`` sidecar (falling back
to graph introspection and explicit overrides when the sidecar is absent).

Supported ``postprocess.decoder`` values:

* ``detr`` / ``baked`` (default): the DEIMv2/RT-DETR family. The graph takes an
  ``orig_target_sizes`` input and emits ``(labels, boxes_xyxy_pixels, scores)``
  with NMS already applied. Byte-for-byte compatible with the kwcoco predictor.
* ``yolo``: a single output of shape ``(1, 4+C, N)`` or ``(1, N, 4+C)`` in
  ``cxcywh`` (model-input pixels); this decodes, thresholds, rescales to the
  original frame, and runs NMS host-side.

Adding a new architecture is a new ``_decode_*`` method plus an enum value; the
preprocessing / spec / detection-assembly code is shared.

Returns, from :meth:`predict_image`, a list of
``{'label': int, 'bbox_xyxy': [x0,y0,x1,y1], 'score': float}`` in the ORIGINAL
image's pixel coordinates -- so an ``ocv_windowed`` wrapper can offset chip
detections back to the full frame exactly as it does for the other detectors.

Requires: onnxruntime, numpy, opencv (cv2).
"""
from __future__ import annotations

import json
import zipfile
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np


# ---------------------------------------------------------------------------
# device -> onnxruntime providers
# ---------------------------------------------------------------------------
def _providers_for_device(device: str) -> list:
    import onnxruntime as ort
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        import warnings
        warnings.warn(
            f"[OnnxPredictor] CUDAExecutionProvider not available for "
            f"device={device!r}; falling back to CPU")
        return ["CPUExecutionProvider"]
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def _cuda_device_id(device: str):
    if device == "cpu":
        return None
    parts = device.split(":")
    return int(parts[1]) if len(parts) == 2 else 0


def _safe_extract_zip(zf: zipfile.ZipFile, dst) -> None:
    """extractall with zip-slip protection: refuse member paths that would
    land outside the destination directory."""
    dst = Path(dst).resolve()
    for member in zf.infolist():
        target = (dst / member.filename).resolve()
        if not target.is_relative_to(dst):
            raise RuntimeError(f"unsafe zip member path: {member.filename}")
    zf.extractall(dst)


@contextmanager
def _open_onnx_package(package) -> Iterator[tuple]:
    """Yield (onnx_path, modelspec_dict) for a package dir, .onnx, or .zip.

    A package is either: a directory holding a ``*.onnx`` (+ optional
    ``*.modelspec.json`` sidecar), a bare ``*.onnx`` file (sidecar looked up
    next to it), or a ``.zip`` archive of a package dir.
    """
    package = Path(package).expanduser()

    if package.is_dir():
        onnx_files = sorted(package.rglob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"no .onnx file found under {package}")
        onnx_fpath = onnx_files[0]
        yield onnx_fpath, _load_spec(onnx_fpath)
        return

    if package.suffix == ".zip":
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(package) as zf:
                _safe_extract_zip(zf, tmp)
            onnx_files = sorted(Path(tmp).rglob("*.onnx"))
            if not onnx_files:
                raise FileNotFoundError(f"no .onnx file inside {package}")
            yield onnx_files[0], _load_spec(onnx_files[0])
        return

    # a bare .onnx path
    yield package, _load_spec(package)


def _load_spec(onnx_fpath: Path) -> dict:
    spec_fpath = onnx_fpath.with_suffix(".modelspec.json")
    if spec_fpath.exists():
        with open(spec_fpath) as f:
            return json.load(f)
    return {}


class OnnxPredictor:
    """Backend-agnostic ONNX detector inference.

    Args:
        package: package dir / ``.onnx`` / ``.zip`` (see :func:`_open_onnx_package`).
        device: ``"cpu"``, ``"cuda"``, or ``"cuda:N"``.
        score_thresh, nms_thresh: override the modelspec's postprocess values.
        decoder: override the modelspec's ``postprocess.decoder``.
        providers: explicit onnxruntime provider list (overrides ``device``).
    """

    def __init__(self, package, device="cpu", score_thresh=None,
                 nms_thresh=None, decoder=None, providers=None,
                 emit_masks=None):
        import onnxruntime as ort

        with _open_onnx_package(package) as (onnx_fpath, spec):
            inp = spec.get("input", {})
            shape_hw = inp.get("shape_hw", [640, 640])
            self._eval_h = int(shape_hw[0])
            self._eval_w = int(shape_hw[1])

            pre = spec.get("preprocess", {})
            self._scale = float(pre.get("scale", 1.0 / 255.0))
            self._mean = np.array(pre.get("normalize_mean", [0.0, 0.0, 0.0]),
                                  dtype=np.float32).reshape(1, 1, 3)
            self._std = np.array(pre.get("normalize_std", [1.0, 1.0, 1.0]),
                                 dtype=np.float32).reshape(1, 1, 3)
            # Resize interpolation: DEIM/RT-DETR export against INTER_AREA;
            # RF-DETR's graph was traced against torchvision BILINEAR, so its
            # spec sets "bilinear" to reproduce the torch detector's pixels.
            self._interp_name = str(pre.get("interpolation", "area")).lower()
            # Channel order the model expects. DEIM/kwcoco graphs are BGR (the
            # default, matching the historical detector conversion); RF-DETR is
            # RGB. The kwiver adapter flips only when this is "bgr", so an RGB
            # model receives the frame in the same order as its torch detector.
            self._channel_order = str(pre.get("channel_order", "bgr")).lower()

            post = spec.get("postprocess", {})
            self._score_thresh = float(
                score_thresh if score_thresh is not None
                else post.get("score_thresh", 0.30))
            self._nms_thresh = float(
                nms_thresh if nms_thresh is not None
                else post.get("nms_iou_thresh", 0.50))
            self._topk = int(post.get("topk", 300))
            self._decoder = str(
                decoder if decoder is not None
                else post.get("decoder", "detr")).lower()

            meta = spec.get("meta", {})
            self._category_names = list(meta.get("category_names", []))
            # Emit per-instance masks (rfdetr seg models expose a ``masks``
            # output). Defaults to the spec's segmentation flag; overridable.
            self._emit_masks = (bool(emit_masks) if emit_masks is not None
                                else bool(meta.get("segmentation", False)))

            if providers is None:
                providers = _providers_for_device(device)
                dev_id = _cuda_device_id(device)
                if dev_id is not None and "CUDAExecutionProvider" in providers:
                    providers = [("CUDAExecutionProvider", {"device_id": dev_id}),
                                 "CPUExecutionProvider"]

            self._session = ort.InferenceSession(str(onnx_fpath), providers=providers)

        self._input_names = [i.name for i in self._session.get_inputs()]
        # class names fall back to the label file, then to numeric ids
        if not self._category_names:
            self._category_names = self._labels_from_sidecar(package)

    @staticmethod
    def _labels_from_sidecar(package):
        package = Path(package).expanduser()
        root = package if package.is_dir() else package.parent
        for lbl in sorted(root.rglob("*.labels.txt")):
            return [ln.strip() for ln in open(lbl) if ln.strip()]
        return []

    # ------------------------------------------------------------------
    @property
    def category_names(self):
        return self._category_names

    @property
    def channel_order(self):
        return getattr(self, "_channel_order", "bgr")

    @property
    def eval_spatial_size(self):
        return (self._eval_h, self._eval_w)

    # ------------------------------------------------------------------
    def _preprocess(self, image_np: np.ndarray) -> np.ndarray:
        """Resize to eval size, normalise, NCHW float32: squash-resize with
        INTER_AREA, scale, then (x - mean) / std."""
        import cv2
        if image_np.ndim == 2:
            image_np = np.repeat(image_np[..., None], 3, axis=-1)
        elif image_np.shape[2] == 4:
            image_np = image_np[..., :3]
        interp = {"area": cv2.INTER_AREA, "bilinear": cv2.INTER_LINEAR,
                  "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC,
                  "nearest": cv2.INTER_NEAREST}.get(
                      getattr(self, "_interp_name", "area"), cv2.INTER_AREA)
        resized = cv2.resize(image_np, (self._eval_w, self._eval_h),
                             interpolation=interp)
        img_f32 = resized.astype(np.float32) * self._scale
        img_f32 = (img_f32 - self._mean) / self._std
        return img_f32.transpose(2, 0, 1)[None, ...]

    # ------------------------------------------------------------------
    def predict_image(self, image_np: np.ndarray, orig_size=None) -> list:
        if orig_size is None:
            h, w = image_np.shape[:2]
            orig_size = (w, h)
        W, H = int(orig_size[0]), int(orig_size[1])
        nchw = self._preprocess(image_np)

        if self._decoder in ("detr", "baked", "deimv2", "rtdetr"):
            return self._decode_detr(nchw, W, H)
        if self._decoder in ("rfdetr", "rf_detr"):
            return self._decode_rfdetr(nchw, W, H)
        if self._decoder == "yolo":
            return self._decode_yolo(nchw, W, H)
        raise ValueError(f"unknown decoder {self._decoder!r}")

    # ------------------------------------------------------------------
    def _rfdetr_raw(self, nchw):
        """Run the graph and return (boxes cxcywh (Q,4), logits (Q,C), and the
        per-query mask logits (Q,h,w) or None). Shared by the box and seg paths."""
        outputs = self._session.run(None, {self._input_names[0]: nchw})
        names = [o.name for o in self._session.get_outputs()]
        bi = next((i for i, n in enumerate(names) if "det" in n), None)
        li = next((i for i, n in enumerate(names) if "lab" in n), None)
        mi = next((i for i, n in enumerate(names) if "mask" in n), None)
        if bi is None or li is None:      # fall back on rank / last-dim
            b4 = [i for i, o in enumerate(outputs)
                  if o.ndim == 3 and o.shape[-1] == 4]
            lo = [i for i, o in enumerate(outputs)
                  if o.ndim == 3 and o.shape[-1] != 4]
            bi, li = (b4[0], lo[0]) if (len(b4) == 1 and len(lo) == 1) else (0, 1)
        masks = outputs[mi][0] if mi is not None else None
        return outputs[bi][0], outputs[li][0], masks

    def _rfdetr_topk(self, logits):
        """Replicate rfdetr PostProcess.forward: per-element sigmoid, then top
        ``num_select`` over the flattened (query x class) grid, threshold, and
        drop the trailing no-object class. Returns (query_idx, class_idx,
        scores) sorted by score. This -- not argmax-per-query -- is what makes
        the ONNX recall match the torch detector."""
        C = logits.shape[1]
        prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -88.0, 88.0)))
        flat = prob.reshape(-1)
        ns = min(self._topk or 300, flat.size)
        idx = np.argpartition(-flat, ns - 1)[:ns]
        idx = idx[np.argsort(-flat[idx])]
        scores = flat[idx]
        qi = idx // C
        ci = idx % C
        n_names = len(self._category_names) or (C - 1)
        keep = (scores > self._score_thresh) & (ci < n_names)
        return qi[keep], ci[keep], scores[keep]

    def _decode_rfdetr(self, nchw, W, H) -> list:
        """RF-DETR box decode: raw ``dets`` (cxcywh, normalised) + ``labels``
        (logits), no baked NMS / no ``orig_target_sizes``. NMS is intentionally
        omitted (RF-DETR is set-based); ocv_windowed dedups across tiles."""
        boxes, logits, masks = self._rfdetr_raw(nchw)
        qi, ci, scores = self._rfdetr_topk(logits)
        want_masks = self._emit_masks and masks is not None
        if want_masks:
            import cv2
        result = []
        for q, c, s in zip(qi, ci, scores):
            cx, cy, bw, bh = (float(v) for v in boxes[q])
            det = {"label": int(c),
                   "bbox_xyxy": [(cx - bw / 2) * W, (cy - bh / 2) * H,
                                 (cx + bw / 2) * W, (cy + bh / 2) * H],
                   "score": float(s)}
            if want_masks:
                # Per-query mask logits -> resize (bilinear) to the frame and
                # threshold at 0, exactly as rfdetr PostProcess does.
                m = cv2.resize(masks[q].astype(np.float32), (W, H),
                               interpolation=cv2.INTER_LINEAR)
                det["mask"] = (m > 0.0).astype(np.uint8)
            result.append(det)
        return result

    # ------------------------------------------------------------------
    def _decode_detr(self, nchw, W, H) -> list:
        """DEIMv2/RT-DETR: graph takes orig_target_sizes, returns pixel xyxy +
        baked NMS. Identical numerics to the kwcoco predictor."""
        feeds = {self._input_names[0]: nchw}
        if "orig_target_sizes" in self._input_names:
            feeds["orig_target_sizes"] = np.array([[W, H]], dtype=np.int64)
        outputs = self._session.run(None, feeds)
        labels, boxes, scores = (o[0] for o in outputs[:3])
        result = []
        for k in range(int(scores.shape[0])):
            s = float(scores[k])
            if s < self._score_thresh:
                continue
            x0, y0, x1, y1 = (float(v) for v in boxes[k])
            result.append({"label": int(labels[k]),
                           "bbox_xyxy": [x0, y0, x1, y1], "score": s})
        return result

    # ------------------------------------------------------------------
    def _decode_yolo(self, nchw, W, H) -> list:
        """YOLOv8-style single output (1,4+C,N) or (1,N,4+C), cxcywh in
        model-input pixels, no baked NMS."""
        out = self._session.run(None, {self._input_names[0]: nchw})[0][0]
        if out.shape[0] < out.shape[1]:   # (4+C, N) -> (N, 4+C)
            out = out.T
        boxes_cxcywh = out[:, :4]
        cls_scores = out[:, 4:]
        cls_ids = cls_scores.argmax(1)
        conf = cls_scores.max(1)
        keep = conf >= self._score_thresh
        boxes_cxcywh, cls_ids, conf = boxes_cxcywh[keep], cls_ids[keep], conf[keep]
        sx, sy = W / self._eval_w, H / self._eval_h
        xyxy = np.empty_like(boxes_cxcywh)
        xyxy[:, 0] = (boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2) * sx
        xyxy[:, 1] = (boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2) * sy
        xyxy[:, 2] = (boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2) * sx
        xyxy[:, 3] = (boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2) * sy
        keep_idx = self._nms(xyxy, conf, self._nms_thresh)
        return [{"label": int(cls_ids[i]),
                 "bbox_xyxy": [float(v) for v in xyxy[i]],
                 "score": float(conf[i])} for i in keep_idx]

    @staticmethod
    def _nms(boxes, scores, iou_thresh):
        if len(boxes) == 0:
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = (xx2 - xx1).clip(0)
            h = (yy2 - yy1).clip(0)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
            order = order[1:][iou <= iou_thresh]
        return keep

    def class_name(self, label: int) -> str:
        """Map an integer class index to its name (numeric fallback)."""
        if 0 <= label < len(self._category_names):
            return self._category_names[label]
        return str(label)
