# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from kwiver.vital.algo import ImageObjectDetector

from kwiver.vital.types import BoundingBoxD
from kwiver.vital.types import DetectedObject
from kwiver.vital.types import DetectedObjectSet
from kwiver.vital.types import DetectedObjectType

import scriptconfig as scfg
import ubelt as ub

from viame.pytorch.utilities import (
    report_cuda_errors,
    resolve_device_str,
    vital_config_update,
    register_vital_algorithm,
    parse_resolution,
    resolution_is_set,
    parse_temporal_offsets,
    temporal_offsets_to_frames,
)


class YOLOVDetectorConfig(scfg.DataConfig):
    """
    The configuration for :class:`YOLOVDetector`.
    """
    weight = scfg.Value(None, help=(
        'Path to a checkpoint written by the yolov trainer: a YOLOV++ video '
        'model, or a YOLOX base detector, which runs frame by frame.'))
    device = scfg.Value('auto', help='Device to run on: auto, cuda, or cuda:N')
    threshold = scfg.Value(0.1, help='Detection confidence threshold')
    nms_thresh = scfg.Value(-1.0, help=(
        'NMS IoU threshold. Negative = the value recorded in the checkpoint.'))
    resolution = scfg.Value(0, help=(
        'Network input: a square side ("640") or height x width ("960x1728"), '
        'multiples of 32. 0 = the resolution the checkpoint was trained at.'))
    temporal_offsets = scfg.Value('', help=(
        'Past frames each frame is aggregated with, as offsets in frames '
        '("-1,-2") or seconds ("-1s,-2s", converted with temporal_frame_rate). '
        'Empty = what the checkpoint was trained with; "none" forces the '
        'random draw from memory_frames. Only past frames are available: the '
        'detector keeps the features of the frames it is handed, in order. '
        'Run it on whole frames of one sequence, not on chips.'))
    temporal_frame_rate = scfg.Value(5.0, help=(
        'Rate (frames/s) of the images this detector is handed, used to turn '
        'second offsets into frame offsets. Match the pipeline\'s frame rate.'))
    reference_frames = scfg.Value(-1, help=(
        'Past frames drawn at random from the memory for each frame when no '
        'offsets are in use. Negative = the value recorded in the checkpoint.'))
    memory_frames = scfg.Value(-1, help=(
        'Past frames whose features are kept to draw from. Negative = the '
        'value recorded in the checkpoint.'))
    max_proposals = scfg.Value(-1, help=(
        'Cap on the per-frame proposals given to the aggregation head (0 = '
        'unbounded). Negative = the value recorded in the checkpoint.'))

    def __post_init__(self):
        super().__post_init__()


class YOLOVDetector(ImageObjectDetector):
    """
    Implementation of ImageObjectDetector using YOLOV++

    YOLOV++ re-scores the proposals of a YOLOX detector by attention over the
    proposals of other frames. Frames arrive one at a time, so the detector
    keeps the proposal features of past frames and aggregates against those.
    """

    def __init__(self):
        ImageObjectDetector.__init__(self)
        self._kwiver_config = YOLOVDetectorConfig()
        self._model = None
        self._online = None
        self._classes = []
        self._frame_shape = None

    def get_configuration(self):
        cfg = super(ImageObjectDetector, self).get_configuration()
        for key, value in self._kwiver_config.items():
            cfg.set_value(key, str(value))
        return cfg

    @report_cuda_errors("YOLOVDetector initialization")
    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        vital_config_update(cfg, cfg_in)

        for key in self._kwiver_config.keys():
            self._kwiver_config[key] = str(cfg.get_value(key))

        self._build_model()
        return True

    def _build_model(self):
        from viame.pytorch.yolov_common import (
            OnlineVideoDetector, build_from_checkpoint, quiet_upstream_logging,
            select_cuda_device)

        config = self._kwiver_config
        weight_fpath = config['weight']
        if not weight_fpath or not ub.Path(weight_fpath).exists():
            raise RuntimeError(f"YOLOV checkpoint not found: {weight_fpath}")

        quiet_upstream_logging()
        self._device = select_cuda_device(resolve_device_str(config['device']))
        model, meta, video = build_from_checkpoint(
            weight_fpath, max_proposals=int(config['max_proposals']))
        self._model = model.to(self._device).eval()
        self._video = video

        self._classes = list(meta.get('class_names') or [])
        if len(self._classes) != meta['num_classes']:
            self._classes = ['class_%d' % i for i in range(meta['num_classes'])]

        resolution = parse_resolution(config['resolution'])
        if resolution_is_set(resolution):
            height, width = (resolution if isinstance(resolution, (list, tuple))
                             else (resolution, resolution))
            self._input_size = (int(height), int(width))
        else:
            self._input_size = tuple(meta.get('input_size') or (640, 640))
        if any(dim % 32 for dim in self._input_size):
            raise ValueError(f"resolution {self._input_size} must be multiples of 32")

        nms_thresh = float(config['nms_thresh'])
        if nms_thresh < 0:
            nms_thresh = float(meta.get('nms_thresh' if video else 'base_nms_thresh',
                                        0.5 if video else 0.65))
        self._nms_thresh = nms_thresh

        print(f"[YOLOVDetector] Loaded {'YOLOV++' if video else 'YOLOX base'} "
              f"{meta['model_size']} model, {len(self._classes)} classes, input "
              f"{self._input_size[0]}x{self._input_size[1]} on {self._device}")
        if not video:
            return

        # Which past frames each frame is aggregated with.
        text = str(config['temporal_offsets']).strip()
        if text.lower() == 'none':
            offsets = []
        else:
            offsets = parse_temporal_offsets(text)
            if not offsets:
                offsets = parse_temporal_offsets(meta.get('temporal_offsets') or '')
        frame_offsets = temporal_offsets_to_frames(
            offsets, float(config['temporal_frame_rate']))
        if any(off >= 0 for off in frame_offsets):
            raise ValueError(
                f"temporal_offsets {offsets} needs future frames, which a "
                "per-frame detector never sees. Use negative offsets.")

        reference_frames = int(config['reference_frames'])
        if reference_frames < 0:
            reference_frames = int(meta.get('reference_frames', 15))
        memory_frames = int(config['memory_frames'])
        if memory_frames < 0:
            memory_frames = int(meta.get('memory_frames', 120))

        self._online = OnlineVideoDetector(
            self._model, self._input_size, nms_thresh=nms_thresh,
            reference_frames=reference_frames, memory_frames=memory_frames,
            frame_offsets=frame_offsets)
        if frame_offsets:
            print(f"[YOLOVDetector] Reference frames at offsets {offsets} -> "
                  f"frames {frame_offsets}")
        else:
            print(f"[YOLOVDetector] Reference frames: {reference_frames} drawn "
                  f"from the last {memory_frames}")

    def check_configuration(self, cfg):
        return True

    def reset(self):
        """Drop the frame memory; the next frame starts a new sequence."""
        if self._online is not None:
            self._online.reset()
        self._frame_shape = None

    @report_cuda_errors("YOLOVDetector detection")
    def detect(self, image_data):
        import numpy as np
        import torch
        from viame.pytorch.yolov_common import (
            base_detect, detections_to_numpy, preprocess_frame)

        threshold = float(self._kwiver_config['threshold'])

        # The networks were trained on OpenCV (BGR) frames.
        frame = image_data.asarray()
        if frame.ndim == 2:
            frame = frame[:, :, None]
        if frame.shape[2] == 1:
            frame = np.repeat(frame, 3, axis=2)
        bgr = np.ascontiguousarray(frame[:, :, 2::-1]).astype(np.uint8)

        # kwiver hands over no frame index; a new size means a new sequence.
        if self._frame_shape is not None and self._frame_shape != bgr.shape:
            self.reset()
        self._frame_shape = bgr.shape

        if self._online is not None:
            detections = self._online.step(bgr)
        else:
            img, ratio = preprocess_frame(bgr, self._input_size)
            img = torch.from_numpy(img).unsqueeze(0).to(self._device)
            result = base_detect(self._model, img, threshold, self._nms_thresh)[0]
            detections = detections_to_numpy(result, ratio, bgr.shape[:2])

        output = DetectedObjectSet()
        for x1, y1, x2, y2, score, cls in detections:
            if score < threshold or x2 <= x1 or y2 <= y1:
                continue
            name = self._classes[int(cls)]
            output.add(DetectedObject(
                BoundingBoxD(float(x1), float(y1), float(x2), float(y2)),
                float(score), DetectedObjectType(name, float(score))))
        return output


def __vital_algorithm_register__():
    register_vital_algorithm(
        YOLOVDetector, "yolov", "PyTorch YOLOV++ video detection routine"
    )
