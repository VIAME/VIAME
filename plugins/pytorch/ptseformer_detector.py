# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from kwiver.vital.algo import ImageObjectDetector

import scriptconfig as scfg

from viame.pytorch.utilities import (
    report_cuda_errors,
    resolve_device_str,
    vital_config_update,
    register_vital_algorithm,
    parse_bool,
    parse_resolution,
    resolution_is_set,
    parse_temporal_offsets,
    temporal_offsets_to_frames,
)


class PTSEFormerDetectorConfig(scfg.DataConfig):
    """
    The configuration for :class:`PTSEFormerDetector`.
    """
    weight = scfg.Value(None, help='Path to a checkpoint written by the ptseformer trainer')
    device = scfg.Value('auto', help='Device to run on: auto, cpu, cuda, or cuda:N')
    threshold = scfg.Value(0.3, help='Detection confidence threshold')
    max_detections = scfg.Value(100, help=(
        'Most detections returned per frame, best first. The decoder scores '
        'num_queries boxes for the frame and for each reference frame.'))
    resolution = scfg.Value(0, help=(
        'Network input as HEIGHTxWIDTH (or one number for a square). 0 = the '
        'resolution the checkpoint was trained at.'))
    temporal_offsets = scfg.Value('', help=(
        'Reference frames as fixed offsets, in frames ("-1,-2") or seconds '
        '("-1s,-2s", converted with temporal_frame_rate). Empty = recover the '
        'training setup from the checkpoint: its fixed offsets, or '
        'num_reference_frames spread evenly over the past part of its '
        'reference window. Only past frames are available: the detector keeps '
        'the encoded features of the images it is handed, in order, and uses '
        'the oldest one it has until enough have arrived. Run it on whole '
        'frames of one sequence, not on chips.'))
    reference_window = scfg.Value('', help=(
        'Override of the checkpoint\'s reference window, "min,max" in frames '
        'or seconds. Unused when fixed offsets are in effect.'))
    num_reference_frames = scfg.Value(0, help=(
        'Override of the number of reference frames spread over the window. '
        '0 = the checkpoint\'s value.'))
    temporal_frame_rate = scfg.Value(5.0, help=(
        'Rate (frames/s) of the images this detector is handed, used to turn '
        'second offsets into frame offsets. Match the pipeline\'s frame rate.'))
    fp16 = scfg.Value(False, help='Run the network in half precision on CUDA')

    def __post_init__(self):
        super().__post_init__()


class PTSEFormerDetector(ImageObjectDetector):
    """
    Implementation of ImageObjectDetector using PTSEFormer

    Each frame goes through the backbone and encoder once; its features are
    kept and reused as the reference of later frames, so detection is causal.
    """

    def __init__(self):
        ImageObjectDetector.__init__(self)
        self._kwiver_config = PTSEFormerDetectorConfig()
        self._model = None
        self._classes = None
        self._offsets = []
        self._buffer = []

    def get_configuration(self):
        cfg = super(ImageObjectDetector, self).get_configuration()
        for key, value in self._kwiver_config.items():
            cfg.set_value(key, str(value))
        return cfg

    @report_cuda_errors("PTSEFormerDetector initialization")
    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        vital_config_update(cfg, cfg_in)

        for key in self._kwiver_config.keys():
            self._kwiver_config[key] = str(cfg.get_value(key))

        self._build_model()
        return True

    def _build_model(self):
        import torch
        from viame.pytorch import ptseformer_common as common

        config = self._kwiver_config
        checkpoint = common.read_checkpoint(config['weight'])
        args = checkpoint.get('args') if isinstance(checkpoint, dict) else None
        if not isinstance(args, dict) or 'arch' not in args:
            raise ValueError(
                f"{config['weight']} carries no PTSEFormer training arguments; "
                "use a checkpoint written by the ptseformer trainer")

        self._device = torch.device(resolve_device_str(config['device']))
        model = common.build_model(args['arch'])
        model.load_state_dict(checkpoint['model'])
        self._model = model.to(self._device).eval()
        self._common = common
        self._classes = list(args.get('class_names') or [])
        num_classes = int(args['arch']['num_classes'])
        self._classes += [str(i) for i in range(len(self._classes), num_classes)]

        resolution = parse_resolution(config['resolution'])
        if not resolution_is_set(resolution):
            resolution = args['resolution']
        if isinstance(resolution, int):
            resolution = (resolution, resolution)
        self._resolution = tuple(int(v) for v in resolution)
        self._fp16 = parse_bool(config['fp16']) and self._device.type == 'cuda'

        self._setup_references(args)
        print(f"[PTSEFormerDetector] Model loaded with {len(self._classes)} "
              f"classes at {self._resolution[0]}x{self._resolution[1]}; "
              f"reference frame offsets {self._offsets}")

    def _setup_references(self, args):
        """Resolve which past frames each frame is enhanced with."""
        common = self._common
        config = self._kwiver_config
        rate = float(config['temporal_frame_rate'])

        offsets = parse_temporal_offsets(config['temporal_offsets'])
        window_text = str(config['reference_window']).strip()
        if window_text.lower() == 'none':
            window_text = ''
        if not offsets and not window_text and args.get('temporal_offsets'):
            offsets = parse_temporal_offsets(args['temporal_offsets'])
        if offsets:
            offsets = temporal_offsets_to_frames(offsets, rate)
            if any(off >= 0 for off in offsets):
                raise ValueError(
                    f"temporal_offsets {offsets} needs future frames, which a "
                    "per-frame detector never sees. Train with negative offsets.")
        else:
            window = common.parse_reference_window(
                window_text or args.get('reference_window') or '-12,0')
            count = (int(config['num_reference_frames'])
                     or int(args.get('num_reference_frames') or 2))
            low, high = common.reference_window_to_frames(window, rate)
            # Upstream spreads test references over past and future; only the past exists here.
            offsets = common.causal_reference_offsets(low, high, count)
        self._offsets = offsets
        self._buffer = []

    def check_configuration(self, cfg):
        return True

    def _encode(self, rgb):
        """Encode one frame; returns its features and its reference queries."""
        import torch

        common = self._common
        tensor = common.preprocess_frame(rgb, self._resolution, self._device)
        with torch.autocast(self._device.type, enabled=self._fp16):
            frame = self._model.encode_frame(common.to_nested(tensor))
            queries = self._model.reference_queries(frame, frame['dec_utils'])
        return frame, queries

    @report_cuda_errors("PTSEFormerDetector detection")
    def detect(self, image_data):
        import numpy as np
        import torch

        from kwiver.vital.types import BoundingBoxD
        from kwiver.vital.types import DetectedObject
        from kwiver.vital.types import DetectedObjectSet
        from kwiver.vital.types import DetectedObjectType

        threshold = float(self._kwiver_config['threshold'])
        rgb = np.ascontiguousarray(image_data.asarray())
        if rgb.ndim == 2:
            rgb = rgb[:, :, None]
        if rgb.shape[2] == 1:
            rgb = np.repeat(rgb, 3, axis=2)
        rgb = rgb[:, :, :3]
        height, width = rgb.shape[:2]

        # The API carries no frame index; a new size is the only visible sequence change.
        if self._buffer and self._buffer[-1][0] != (height, width):
            self._buffer = []

        with torch.no_grad():
            frame, queries = self._encode(rgb)
            # Until enough frames have arrived the oldest one stands in, as
            # upstream clamps to the first frame of a sequence.
            history = self._buffer + [((height, width), frame, queries)]
            refs = [history[max(len(history) - 1 + off, 0)] for off in self._offsets]
            with torch.autocast(self._device.type, enabled=self._fp16):
                outputs = self._model.fuse(
                    frame, [ref[1] for ref in refs], [ref[2] for ref in refs])
            boxes, scores = self._common.decode_outputs(
                outputs, int(self._kwiver_config['max_detections']))

        self._buffer.append(((height, width), frame, queries))
        del self._buffer[:-max(-off for off in self._offsets)]

        scale = np.array([width, height, width, height], dtype=np.float64)
        boxes = boxes.cpu().numpy().astype(np.float64) * scale
        scores = scores.cpu().numpy().astype(np.float64)

        output = DetectedObjectSet()
        for box, class_scores in zip(boxes, scores):
            score = float(class_scores.max())
            if score < threshold or box[2] <= box[0] or box[3] <= box[1]:
                continue
            bbox = BoundingBoxD(float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            keep = [i for i, s in enumerate(class_scores) if s >= threshold]
            dot = DetectedObjectType(
                [self._classes[i] for i in keep], [float(class_scores[i]) for i in keep])
            output.add(DetectedObject(bbox, score, dot))
        return output


def __vital_algorithm_register__():
    register_vital_algorithm(
        PTSEFormerDetector, "ptseformer", "PyTorch PTSEFormer video detection routine"
    )
