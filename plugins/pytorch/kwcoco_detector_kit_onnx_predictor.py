# ============================================================================
# VENDORED FILE -- DO NOT EDIT HERE.
#
# This is a vendored copy of the canonical OnnxPredictor from the
# kwcoco_detector_kit repository. VIAME inference uses this copy so it does NOT
# depend on kwcoco_detector_kit being importable -- the kit evolves quickly and
# may intentionally break ONNX-package compatibility, so VIAME pins/resyncs on
# purpose rather than tracking the kit live.
#
# The CANONICAL source lives in the kit at:
#     kwcoco_detector_kit/predictors/onnx.py
# Edit it THERE, then resync this copy with the kit's vendoring tool:
#     # in the kwcoco_detector_kit repo:
#     python dev/vendor_onnx_to_viame.py --viame-root /path/to/VIAME
#
# Provenance is recorded in the importable ``__vendored_provenance__`` dict
# below. ``source_sha256`` is the hash of the canonical file's body; ``--check``
# recomputes it to detect drift.
# ============================================================================
"""
ONNX inference backend — wraps a kit-exported ONNX package for deployment
without PyTorch.

Requires:  onnxruntime, numpy, kwimage
No-import: torch, DEIMv2, kwcoco, kwconf, yaml
"""
from __future__ import annotations

__vendored_provenance__ = {
    "source_repo": "kwcoco_detector_kit",
    "source_path": "kwcoco_detector_kit/predictors/onnx.py",
    "kit_git_sha": "9d31086dad829260eaaf4c5f986695a25bbdb116",
    "kit_git_dirty": False,
    "source_sha256": "fedfbe3bfa3b8d6fac2b2aca0c11169c6f41de430e2d1b15da478eb7eaea5515",
    "vendored_at": "2026-06-30T20:24:02Z",
    "vendor_tool": "dev/vendor_onnx_to_viame.py",
}

import json
import os
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np


def _providers_for_device(device: str) -> list:
    """Map a device string to an onnxruntime provider list."""
    if device == "cpu":
        return ["CPUExecutionProvider"]
    import onnxruntime as ort
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        print(
            f"[OnnxPredictor] CUDAExecutionProvider not available "
            f"for device={device!r}; falling back to CPU"
        )
        return ["CPUExecutionProvider"]
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def _cuda_device_id(device: str) -> int | None:
    """Parse 'cuda' → 0, 'cuda:N' → N, 'cpu' → None."""
    if device == "cpu":
        return None
    parts = device.split(":")
    return int(parts[1]) if len(parts) == 2 else 0


def _safe_extract_zip(src: Path, dst: Path) -> None:
    dst = dst.resolve()
    with zipfile.ZipFile(src, "r") as zf:
        for member in zf.infolist():
            target = (dst / member.filename).resolve()
            if os.path.commonpath([str(dst), str(target)]) != str(dst):
                raise RuntimeError(f"unsafe zip member path: {member.filename}")
        zf.extractall(dst)


@contextmanager
def _open_onnx_package(package: Path) -> Iterator[tuple[Path, dict]]:
    """Yield (onnx_fpath, spec_dict) for a directory, .zip, or bare .onnx path."""
    tmp_ctx = None
    try:
        if package.is_dir():
            root = package
        elif package.suffix == ".zip":
            tmp_ctx = tempfile.TemporaryDirectory()
            root = Path(tmp_ctx.name)
            _safe_extract_zip(package, root)
        elif package.suffix == ".onnx":
            root = package.parent
        else:
            root = package.parent

        onnx_files = sorted(root.rglob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"no .onnx file found under {package}")
        onnx_fpath = onnx_files[0]

        spec_fpath = onnx_fpath.with_suffix(".modelspec.json")
        spec = json.loads(spec_fpath.read_text()) if spec_fpath.exists() else {}
        yield onnx_fpath, spec
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


class OnnxPredictor:
    """
    Inference adapter for a kit-exported ONNX package.

    Satisfies the :class:`~kwcoco_detector_kit.predictors._interface.DetectorPredictor`
    protocol. Requires only ``onnxruntime``, ``numpy``, and ``kwimage`` — no PyTorch.

    Args:
        package: Path to the exported package directory, ``.zip`` archive, or bare
            ``.onnx`` file. A directory or zip must contain a ``.onnx`` model and
            (optionally) a ``.modelspec.json`` sidecar with inference parameters.
        device: ``"cpu"`` (default), ``"cuda"``, or ``"cuda:N"``.
        score_thresh: Detection score threshold. When given, overrides the value
            from ``.modelspec.json``. Detections below this threshold are dropped.
        nms_thresh: NMS IoU threshold (stored for reference; DEIMv2 ONNX already
            runs NMS internally so this is not re-applied).
        providers: Explicit onnxruntime provider list. When given, overrides ``device``.
    """

    def __init__(
        self,
        package: str | Path,
        *,
        device: str = "cpu",
        score_thresh: float | None = None,
        nms_thresh: float | None = None,
        providers: list | None = None,
    ):
        import onnxruntime as ort

        package = Path(package).expanduser()

        with _open_onnx_package(package) as (onnx_fpath, spec):
            inp = spec.get("input", {})
            shape_hw = inp.get("shape_hw", [640, 640])
            self._eval_h = int(shape_hw[0])
            self._eval_w = int(shape_hw[1])

            pre = spec.get("preprocess", {})
            self._scale = float(pre.get("scale", 1.0 / 255.0))
            self._mean = np.array(
                pre.get("normalize_mean", [0.0, 0.0, 0.0]), dtype=np.float32
            ).reshape(1, 1, 3)
            self._std = np.array(
                pre.get("normalize_std", [1.0, 1.0, 1.0]), dtype=np.float32
            ).reshape(1, 1, 3)

            post = spec.get("postprocess", {})
            self._score_thresh = float(
                score_thresh if score_thresh is not None
                else post.get("score_thresh", 0.30)
            )
            self._nms_thresh = float(
                nms_thresh if nms_thresh is not None
                else post.get("nms_iou_thresh", 0.50)
            )

            meta = spec.get("meta", {})
            self._category_names: list[str] = list(meta.get("category_names", []))

            if providers is None:
                providers = _providers_for_device(device)
                device_id = _cuda_device_id(device)
                if device_id is not None and "CUDAExecutionProvider" in providers:
                    providers = [
                        ("CUDAExecutionProvider", {"device_id": device_id}),
                        "CPUExecutionProvider",
                    ]

            self._session = ort.InferenceSession(str(onnx_fpath), providers=providers)

    # ------------------------------------------------------------------
    # Protocol properties
    # ------------------------------------------------------------------

    @property
    def eval_spatial_size(self) -> tuple[int, int]:
        """(H, W) the model evaluates at."""
        return (self._eval_h, self._eval_w)

    @property
    def category_names(self) -> list[str]:
        return self._category_names

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _preprocess(self, image_np: np.ndarray) -> np.ndarray:
        """Resize to eval size, normalise, and convert to NCHW float32."""
        import kwimage

        if image_np.ndim == 2:
            image_np = np.repeat(image_np[..., None], 3, axis=-1)
        elif image_np.shape[2] == 4:
            image_np = image_np[..., :3]

        try:
            resized = kwimage.imresize(
                image_np, dsize=(self._eval_w, self._eval_h), interpolation="area"
            )
        except NotImplementedError:
            resized = kwimage.imresize(
                image_np, dsize=(self._eval_w, self._eval_h), interpolation="linear"
            )

        img_f32 = resized.astype(np.float32) * self._scale
        img_f32 = (img_f32 - self._mean) / self._std
        return img_f32.transpose(2, 0, 1)[None, ...]  # (1, 3, H, W)

    # ------------------------------------------------------------------
    # Public inference methods
    # ------------------------------------------------------------------

    def predict_image(
        self,
        image_np: np.ndarray,
        orig_size=None,
    ) -> list[dict]:
        """Score one image; return detections filtered by ``score_thresh``.

        Args:
            image_np: HxWx3 uint8 RGB array (grayscale 2-D and RGBA also accepted).
            orig_size: ``(W, H)`` of the original image. Inferred from ``image_np``
                when ``None``.

        Returns:
            List of ``{'label': int, 'bbox_xyxy': [x0, y0, x1, y1], 'score': float}``
            dicts, ordered by model output (score-descending after internal NMS).
        """
        if orig_size is None:
            h, w = image_np.shape[:2]
            orig_size = (w, h)

        nchw = self._preprocess(image_np)
        W, H = int(orig_size[0]), int(orig_size[1])
        orig_sizes = np.array([[W, H]], dtype=np.int64)

        outputs = self._session.run(
            None, {"images": nchw, "orig_target_sizes": orig_sizes}
        )
        labels_raw, boxes_raw, scores_raw = outputs[:3]
        # Batch dimension: pick item 0. shapes (K,), (K, 4), (K,)
        labels = labels_raw[0]
        boxes = boxes_raw[0]
        scores = scores_raw[0]

        result: list[dict] = []
        for k in range(int(scores.shape[0])):
            score = float(scores[k])
            if score < self._score_thresh:
                continue
            x0, y0, x1, y1 = (float(v) for v in boxes[k])
            result.append({
                "label": int(labels[k]),
                "bbox_xyxy": [x0, y0, x1, y1],
                "score": score,
            })
        return result

    def predict_image_kwimage(
        self,
        image_np: np.ndarray,
        orig_size=None,
    ):
        """Score one image; return a :class:`kwimage.Detections` object.

        Richer return type used by the VIAME plugin. ``.classes`` is populated
        from ``self.category_names``.

        Args:
            image_np: HxWx3 uint8 RGB array.
            orig_size: ``(W, H)``; inferred from ``image_np`` when ``None``.

        Returns:
            :class:`kwimage.Detections` with ``.boxes`` (ltrb),
            ``.scores``, ``.class_idxs``, and ``.classes``.
        """
        import kwimage

        if orig_size is None:
            h, w = image_np.shape[:2]
            orig_size = (w, h)

        nchw = self._preprocess(image_np)
        W, H = int(orig_size[0]), int(orig_size[1])
        orig_sizes = np.array([[W, H]], dtype=np.int64)

        outputs = self._session.run(
            None, {"images": nchw, "orig_target_sizes": orig_sizes}
        )
        labels_raw, boxes_raw, scores_raw = outputs[:3]
        labels = labels_raw[0].astype(np.int32)
        boxes = boxes_raw[0].astype(np.float32)
        scores = scores_raw[0].astype(np.float32)

        keep = scores >= self._score_thresh
        return kwimage.Detections(
            boxes=kwimage.Boxes(boxes[keep], "ltrb"),
            scores=scores[keep],
            class_idxs=labels[keep],
            classes=self._category_names,
        )
