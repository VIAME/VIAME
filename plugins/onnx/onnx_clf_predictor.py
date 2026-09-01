# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
ONNX image-classifier inference, via onnxruntime -- no PyTorch.

Companion to :mod:`viame.onnx.onnx_predictor` (detectors). Classification is a
different enough contract to keep separate: no boxes, no NMS, a dynamic batch
axis, and a letterbox resize rather than the detectors' squash.

Reads a ``.modelspec.json`` sidecar written by
:mod:`viame.pytorch.netharn_clf_to_onnx`, whose graph takes ``(N, 3, H, W)`` RGB
in ``[0, 1]`` and returns ``(N, C)`` softmax probabilities. The netharn
``InputNorm`` lives inside the graph, so by default this only rescales; a spec
carrying a non-identity ``normalize_mean``/``normalize_std`` is still honored.

Reproduces the preprocessing of netharn's ``ImageListDataset`` /
``ClfSamplerDataset``: ``kwimage.imresize(..., letterbox=True)`` -- aspect
preserved, centered, zero padded -- then ``/ 255``.

Requires: onnxruntime, numpy, opencv (cv2).
"""
from __future__ import annotations

import numpy as np

from viame.onnx.onnx_predictor import (
    _open_onnx_package, _providers_for_device, _cuda_device_id)


def letterbox_resize(image, height, width):
    """Aspect-preserving resize into a zero-padded (height, width) canvas.

    A deliberate, pixel-exact port of ``kwimage.imresize(..., letterbox=True)``,
    which is what netharn's classifier datasets use. Two details matter and are
    easy to get wrong:

    * **Interpolation is chosen from the scale.** kwimage defaults to
      ``INTER_AREA`` when shrinking and ``INTER_LANCZOS4`` when growing. Using
      ``INTER_LINEAR`` throughout visibly changes a whole-frame classification,
      where a 1360x1024 frame is shrunk ~6x into 224x224.
    * **The pad offset is rounded, not floored** -- ``round((target - embed)/2)``
      -- so an odd leftover puts the extra pixel on the top/left, not the
      bottom/right. Flooring shifts the content one pixel.

    Chips are rarely square, so this is not interchangeable with the squash
    resize the detectors use.
    """
    import cv2
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    elif image.shape[2] == 4:
        image = image[..., :3]
    src_h, src_w = image.shape[:2]
    if src_h <= 0 or src_w <= 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    orig_size = np.array([src_w, src_h])
    target_size = np.array([width, height])
    equal_sxy = (target_size / orig_size).min()
    embed_size = np.round(orig_size * equal_sxy).astype(int)
    embed_size = np.maximum(embed_size, 1)
    offset = np.round((target_size - embed_size) / 2).astype(int)
    left, top = offset
    right, bot = target_size - (embed_size + offset)

    interpolation = cv2.INTER_AREA if equal_sxy < 1 else cv2.INTER_LANCZOS4
    embedded = cv2.resize(image, (int(embed_size[0]), int(embed_size[1])),
                          interpolation=interpolation)
    return cv2.copyMakeBorder(embedded, int(top), int(bot), int(left),
                              int(right), borderType=cv2.BORDER_CONSTANT,
                              value=0)


class OnnxClassifierPredictor:
    """Batched ONNX image classification.

    Args:
        package: package dir / ``.onnx`` / ``.zip``.
        device: ``"cpu"``, ``"cuda"``, or ``"cuda:N"``.
        batch_size: chips per graph call. The exported graphs have a dynamic
            batch axis, so this only trades memory for per-call overhead.
        providers: explicit onnxruntime provider list (overrides ``device``).
    """

    def __init__(self, package, device="cpu", batch_size=4, providers=None):
        import onnxruntime as ort

        with _open_onnx_package(package) as (onnx_fpath, spec):
            inp = spec.get("input", {})
            shape_hw = inp.get("shape_hw", [224, 224])
            self._eval_h = int(shape_hw[0])
            self._eval_w = int(shape_hw[1])

            pre = spec.get("preprocess", {})
            self._scale = float(pre.get("scale", 1.0 / 255.0))
            self._mean = np.array(pre.get("normalize_mean", [0.0, 0.0, 0.0]),
                                  dtype=np.float32).reshape(1, 1, 3)
            self._std = np.array(pre.get("normalize_std", [1.0, 1.0, 1.0]),
                                 dtype=np.float32).reshape(1, 1, 3)
            self._resize_mode = str(pre.get("resize_mode", "letterbox")).lower()
            self._channel_order = str(pre.get("channel_order", "rgb")).lower()

            post = spec.get("postprocess", {})
            # The exporter bakes the softmax in; tolerate a graph that does not.
            self._softmax_applied = bool(post.get("softmax_applied", True))

            meta = spec.get("meta", {})
            self._category_names = list(meta.get("category_names", []))

            if providers is None:
                providers = _providers_for_device(device)
                dev_id = _cuda_device_id(device)
                if dev_id is not None and "CUDAExecutionProvider" in providers:
                    providers = [("CUDAExecutionProvider", {"device_id": dev_id}),
                                 "CPUExecutionProvider"]

            self._session = ort.InferenceSession(str(onnx_fpath),
                                                 providers=providers)

        self._input_name = self._session.get_inputs()[0].name
        self.batch_size = max(int(batch_size), 1)

    # ------------------------------------------------------------------
    @property
    def category_names(self):
        return self._category_names

    @property
    def channel_order(self):
        return self._channel_order

    @property
    def eval_spatial_size(self):
        return (self._eval_h, self._eval_w)

    # ------------------------------------------------------------------
    def _preprocess(self, image):
        import cv2
        if self._resize_mode == "letterbox":
            resized = letterbox_resize(image, self._eval_h, self._eval_w)
        else:
            if image.ndim == 2:
                image = np.repeat(image[..., None], 3, axis=-1)
            elif image.shape[2] == 4:
                image = image[..., :3]
            resized = cv2.resize(image, (self._eval_w, self._eval_h),
                                 interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) * self._scale
        arr = (arr - self._mean) / self._std
        return arr.transpose(2, 0, 1)

    @staticmethod
    def _softmax(logits):
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------
    def predict(self, images) -> np.ndarray:
        """Classify a list of HxWx3 uint8 images.

        Returns:
            ndarray: ``(len(images), num_classes)`` probabilities. An empty
            input yields a ``(0, num_classes)`` array rather than raising, so
            callers can treat "no chips this frame" as an ordinary case.
        """
        n_classes = len(self._category_names)
        if not len(images):
            return np.zeros((0, n_classes), dtype=np.float32)

        chips = np.stack([self._preprocess(im) for im in images])
        outputs = []
        for start in range(0, len(chips), self.batch_size):
            batch = np.ascontiguousarray(chips[start:start + self.batch_size])
            outputs.append(self._session.run(None, {self._input_name: batch})[0])
        probs = np.concatenate(outputs, axis=0)
        if not self._softmax_applied:
            probs = self._softmax(probs)
        return probs

    def class_name(self, label: int) -> str:
        if 0 <= label < len(self._category_names):
            return self._category_names[label]
        return str(label)
