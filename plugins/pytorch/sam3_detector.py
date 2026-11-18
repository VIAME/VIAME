# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
SAM 3.1 per-image detector.

Runs SAM 3.1's own open-vocabulary grounding head on each image: every
instance of the text query, with a box, score and mask. Unlike the
Grounding DINO path in sam3_refiner / sam3_text_query, this uses the SAM3
detector weights, so a checkpoint fine-tuned by SAM3Trainer (or by
SAM3TrackerTrainer with train_detector) is what runs. It is the inference
side of SAM3Trainer, and produces ordinary detections, so any tracker can
consume them.
"""

import scriptconfig as scfg

from kwiver.vital.algo import ImageObjectDetector

from viame.pytorch.utilities import (
    parse_bool,
    register_vital_algorithm,
    report_cuda_errors,
    resolve_device_str,
    supervision_to_kwiver_detections,
    vital_config_update,
)


class SAM3DetectorConfig(scfg.DataConfig):
    """
    The configuration for :class:`SAM3Detector`.
    """
    sam_model_id = scfg.Value(
        'models/sam3.1_weights.pt',
        help='SAM 3.1 checkpoint (sam3.1_weights.pt or a fine-tuned copy)')
    text_query = scfg.Value(
        'fish', help='Text query; comma-separate several to detect each, '
                     'labeled with its query')
    threshold = scfg.Value(0.3, help='Detection score threshold')
    max_detections = scfg.Value(100, help='Maximum detections per query')
    output_masks = scfg.Value(True, help='Attach instance masks')
    device = scfg.Value('auto', help='Device: auto, cpu, cuda, or cuda:N')
    amp = scfg.Value(True, help='Run the heads in fp16 on GPU')


class SAM3Detector(ImageObjectDetector):
    """
    Implementation of ImageObjectDetector using the SAM 3.1 detector.
    """

    def __init__(self):
        ImageObjectDetector.__init__(self)
        self._kwiver_config = SAM3DetectorConfig()
        self._predictor = None

    def get_configuration(self):
        cfg = super(ImageObjectDetector, self).get_configuration()
        for key, value in self._kwiver_config.items():
            cfg.set_value(key, str(value))
        return cfg

    @report_cuda_errors("SAM3Detector initialization")
    def set_configuration(self, cfg_in):
        from viame.pytorch.sam3_detector_training import DetectorPredictor
        from viame.pytorch.sam3_trainer import _resolve_model_path

        cfg = self.get_configuration()
        vital_config_update(cfg, cfg_in)
        for key in self._kwiver_config.keys():
            self._kwiver_config[key] = str(cfg.get_value(key))

        c = self._kwiver_config
        self._queries = [q.strip() for q in str(c['text_query']).split(',')
                         if q.strip()] or ['object']
        self._threshold = float(c['threshold'])
        self._max_detections = int(c['max_detections'])
        self._masks = parse_bool(c['output_masks'])
        device = resolve_device_str(str(c['device']))
        self._predictor = DetectorPredictor(
            _resolve_model_path(str(c['sam_model_id'])), device=device,
            amp=parse_bool(c['amp']))
        return True

    def check_configuration(self, cfg):
        return True

    @report_cuda_errors("SAM3Detector detection")
    def detect(self, image_data):
        from types import SimpleNamespace

        import numpy as np

        img = image_data.asarray()
        if img.ndim == 2:
            img = np.stack([img] * 3, -1)
        elif img.shape[2] > 3:
            img = img[:, :, :3]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        boxes, scores, qidx, masks = self._predictor.predict(
            np.ascontiguousarray(img), self._queries,
            threshold=self._threshold, max_detections=self._max_detections,
            masks=self._masks)
        dets = SimpleNamespace(xyxy=boxes, confidence=scores,
                               class_id=qidx.astype(int), mask=masks)
        return supervision_to_kwiver_detections(dets, self._queries)


def __vital_algorithm_register__():
    register_vital_algorithm(
        SAM3Detector, "sam3",
        "SAM 3.1 detector: every instance of a text query, per image")
