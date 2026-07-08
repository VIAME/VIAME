# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from kwiver.vital.algo import ImageObjectDetector

import scriptconfig as scfg
import ubelt as ub

from viame.pytorch.utilities import (
    report_cuda_errors,
    resolve_device_str,
    vital_config_update,
    supervision_to_kwiver_detections,
    register_vital_algorithm,
    parse_bool,
    ensure_rfdetr_compatibility,
)


class RFDETRDetectorConfig(scfg.DataConfig):
    """
    The configuration for :class:`RFDETRDetector`.
    """
    weight = scfg.Value(None, help='Path to a trained RF-DETR checkpoint (.pt file)')
    model_size = scfg.Value('base', help='Model size: nano, small, medium, base, or large')
    num_channels = scfg.Value(3, help=(
        'Number of input channels. 3 = RGB; 4 = RGB + a motion/flow channel. '
        'Recovered from the checkpoint when present, otherwise this value.'))
    resolution = scfg.Value(0, help=(
        'Square input resolution. 0 = use the model-size default. Must match '
        'the resolution the checkpoint was trained at so the positional '
        'embeddings load correctly; recovered from the checkpoint when present, '
        'otherwise this value.'))
    device = scfg.Value('auto', help='Device to run on: auto, cpu, cuda, or cuda:N')
    threshold = scfg.Value(0.5, help='Detection confidence threshold')
    optimize_inference = scfg.Value(True, help='Whether to optimize model for inference')
    segmentation = scfg.Value(False, help=(
        'Load a segmentation (mask) RF-DETR variant (RFDETRSeg*) instead of the '
        'box-only variant. Must match how the checkpoint was trained, otherwise '
        'the segmentation-head weights will not load.'))

    def __post_init__(self):
        super().__post_init__()


class RFDETRDetector(ImageObjectDetector):
    """
    Implementation of ImageObjectDetector using RF-DETR

    RF-DETR is a real-time object detection model based on DETR architecture
    with DINOv2 backbone.
    """

    def __init__(self):
        ImageObjectDetector.__init__(self)
        self._kwiver_config = RFDETRDetectorConfig()
        self._model = None
        self._classes = None
        self._num_channels = 3

    def get_configuration(self):
        cfg = super(ImageObjectDetector, self).get_configuration()
        for key, value in self._kwiver_config.items():
            cfg.set_value(key, str(value))
        return cfg

    @report_cuda_errors("RFDETRDetector initialization")
    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()
        vital_config_update(cfg, cfg_in)

        for key in self._kwiver_config.keys():
            self._kwiver_config[key] = str(cfg.get_value(key))

        self._build_model()
        return True

    # RF-DETR size taxonomy as (patch_size, dec_layers) -> default resolution,
    # mirroring src/rfdetr/config.py. Used to recover the architecture from a
    # checkpoint's weight shapes when it carries no training args (e.g. a native
    # PyTorch Lightning checkpoint). Sizes sharing a (patch_size, dec_layers) key
    # differ only by resolution, so ties break on the nearest default.
    _DET_SIZES = {
        'nano': (16, 2, 384), 'small': (16, 3, 512), 'base': (14, 3, 560),
        'medium': (16, 4, 576), 'large': (16, 4, 704),
    }
    _SEG_SIZES = {
        'nano': (12, 4, 312), 'small': (12, 4, 384),
        'medium': (12, 5, 432), 'large': (12, 5, 504),
    }

    @staticmethod
    def _extract_state_dict(checkpoint):
        # Deployed checkpoints store the model weights under 'model'; a native
        # PyTorch Lightning checkpoint (last.ckpt / checkpoint_epoch=N.ckpt)
        # nests them under 'state_dict' with a 'model.' prefix.
        if not isinstance(checkpoint, dict):
            return checkpoint
        if 'model' in checkpoint:
            return checkpoint['model']
        if 'state_dict' in checkpoint:
            return {k[len('model.'):]: v
                    for k, v in checkpoint['state_dict'].items()
                    if k.startswith('model.')}
        return checkpoint

    @staticmethod
    def _checkpoint_args(checkpoint):
        args = checkpoint.get('args') if isinstance(checkpoint, dict) else None
        if isinstance(args, dict):
            return args
        try:
            return vars(args)
        except TypeError:
            return {}

    @classmethod
    def _infer_architecture(cls, state_dict):
        # Recover model_size, resolution, channel count and the segmentation
        # flag straight from the weight shapes so a checkpoint loads even when it
        # carries no training args. Leaves a field None when undeterminable.
        import math
        arch = {'model_size': None, 'resolution': None,
                'num_channels': None, 'segmentation': None}
        if not isinstance(state_dict, dict):
            return arch

        proj = state_dict.get(
            'backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight')
        proj_shape = getattr(proj, 'shape', None)
        if proj_shape is None or len(proj_shape) != 4:
            return arch  # unknown backbone; leave the config values in place
        patch_size = int(proj_shape[2])
        arch['num_channels'] = int(proj_shape[1])

        arch['segmentation'] = any(
            k.startswith('segmentation_head') for k in state_dict)

        prefix = 'transformer.decoder.layers.'
        layer_ids = [int(k[len(prefix):].split('.', 1)[0])
                     for k in state_dict if k.startswith(prefix)]
        dec_layers = max(layer_ids) + 1 if layer_ids else None

        # Resolution from the DINOv2 positional-embedding grid, allowing for a
        # cls token (and optional register tokens).
        pos = state_dict.get(
            'backbone.0.encoder.encoder.embeddings.position_embeddings')
        pos_shape = getattr(pos, 'shape', None)
        if pos_shape is not None and len(pos_shape) == 3:
            n_tokens = int(pos_shape[1])
            for n_special in (1, 5, 0):
                grid_sq = n_tokens - n_special
                grid = math.isqrt(grid_sq) if grid_sq > 0 else 0
                if grid and grid * grid == grid_sq:
                    arch['resolution'] = grid * patch_size
                    break

        table = cls._SEG_SIZES if arch['segmentation'] else cls._DET_SIZES
        if dec_layers is not None:
            candidates = [name for name, (ps, dl, _) in table.items()
                          if ps == patch_size and dl == dec_layers]
            if len(candidates) == 1:
                arch['model_size'] = candidates[0]
            elif candidates:
                target = arch['resolution'] or 0
                arch['model_size'] = min(
                    candidates, key=lambda n: abs(table[n][2] - target))
        return arch

    def _build_model(self):
        import torch

        weight_fpath = self._kwiver_config['weight']
        model_size = self._kwiver_config['model_size'].lower()
        device = resolve_device_str(self._kwiver_config['device'])
        optimize = parse_bool(self._kwiver_config['optimize_inference'])
        num_channels = int(self._kwiver_config['num_channels'])
        resolution = int(self._kwiver_config['resolution'])
        segmentation = parse_bool(self._kwiver_config['segmentation'])

        ensure_rfdetr_compatibility()

        # Load the checkpoint before building the network so its architecture can
        # be recovered from the weights. A deployed checkpoint embeds the training
        # args, but a native PyTorch Lightning checkpoint (last.ckpt /
        # checkpoint_epoch=N.ckpt) does not, so infer model_size, resolution,
        # channel count and the segmentation flag from the weight shapes and let
        # the embedded args, then the pipeline config, act as fallbacks. The
        # architecture must match the weights or load_state_dict fails.
        checkpoint = None
        state_dict = None
        ckpt_args = {}
        if weight_fpath and ub.Path(weight_fpath).exists():
            checkpoint = torch.load(weight_fpath, map_location=device, weights_only=False)
            state_dict = self._extract_state_dict(checkpoint)
            ckpt_args = self._checkpoint_args(checkpoint)
            arch = self._infer_architecture(state_dict)

            if arch['model_size'] is not None:
                model_size = arch['model_size']
            elif ckpt_args.get('model_size'):
                model_size = str(ckpt_args['model_size']).lower()

            if arch['segmentation'] is not None:
                segmentation = arch['segmentation']
            elif 'segmentation' in ckpt_args:
                segmentation = parse_bool(ckpt_args['segmentation'])

            if arch['num_channels'] is not None:
                num_channels = arch['num_channels']
            elif 'num_channels' in ckpt_args:
                num_channels = int(ckpt_args['num_channels'])

            if arch['resolution'] is not None:
                resolution = arch['resolution']
            elif 'resolution' in ckpt_args:
                resolution = int(ckpt_args['resolution'])

        # Import the appropriate RF-DETR model class based on size and whether a
        # segmentation (mask) head is present. Seg checkpoints carry extra
        # segmentation_head.* weights that only load into the RFDETRSeg* classes.
        import rfdetr
        det_models = {
            'nano': 'RFDETRNano', 'small': 'RFDETRSmall', 'medium': 'RFDETRMedium',
            'base': 'RFDETRBase', 'large': 'RFDETRLarge',
        }
        seg_models = {
            'nano': 'RFDETRSegNano', 'small': 'RFDETRSegSmall',
            'medium': 'RFDETRSegMedium', 'large': 'RFDETRSegLarge',
        }
        table = seg_models if segmentation else det_models
        if model_size not in table:
            kind = 'segmentation' if segmentation else 'detection'
            raise ValueError(f"Unknown {kind} model size: {model_size}. "
                           f"Expected one of: {', '.join(table)}")
        RFDETRModel = getattr(rfdetr, table[model_size])

        print(f"[RFDETRDetector] Loading {model_size} model on {device}")

        if checkpoint is not None:
            # Determine the actual class_embed dimension from weights.
            # RF-DETR's build_model adds +1 to num_classes for a background
            # class, but reinitialize_detection_head and the training loop
            # store weights with the raw dataset class count.  We therefore
            # read the true dimension from the checkpoint weights and call
            # reinitialize_detection_head to make the model match before
            # loading the state dict (mirroring RF-DETR's own loading path).
            if 'class_embed.weight' in state_dict:
                ckpt_num_classes = state_dict['class_embed.weight'].shape[0]
            elif ckpt_args.get('num_classes'):
                ckpt_num_classes = int(ckpt_args['num_classes'])
            else:
                ckpt_num_classes = 90  # default COCO classes

            # Get class names if available
            if ckpt_args.get('class_names'):
                self._classes = ckpt_args['class_names']

            model_kwargs = dict(
                pretrain_weights=None,
                num_classes=ckpt_num_classes,
                num_channels=num_channels,
                device=device
            )
            if resolution > 0:
                model_kwargs['resolution'] = resolution

            self._model = RFDETRModel(**model_kwargs)

            # The constructor adds +1 for background, so resize the
            # detection head to match the checkpoint's actual dimensions.
            self._model.model.reinitialize_detection_head(ckpt_num_classes)

            # Load the state dict
            self._model.model.model.load_state_dict(state_dict)

            if self._classes:
                self._model.model.class_names = self._classes
        else:
            # Use pretrained weights
            pre_kwargs = dict(num_channels=num_channels, device=device)
            if resolution > 0:
                pre_kwargs['resolution'] = resolution
            self._model = RFDETRModel(**pre_kwargs)

        self._num_channels = num_channels

        # Set up class names
        if self._classes is None:
            self._classes = list(self._model.class_names)

        # Optimize for inference if requested
        if optimize:
            print("[RFDETRDetector] Optimizing model for inference")
            self._model.optimize_for_inference(compile=False)

        print(f"[RFDETRDetector] Model loaded with {len(self._classes)} classes")

    def check_configuration(self, cfg):
        return True

    @report_cuda_errors("RFDETRDetector detection")
    def detect(self, image_data):
        import torch
        from PIL import Image

        threshold = float(self._kwiver_config['threshold'])

        # Convert kwiver image to numpy array
        full_rgb = image_data.asarray()

        # PIL cannot represent >4 channels (and only RGB/RGBA at all), so pass
        # multi-channel imagery (e.g. RGB + flow) to predict as a numpy array;
        # predict() converts it to a (C, H, W) tensor itself.
        if self._num_channels > 3 or full_rgb.ndim != 3 or full_rgb.shape[2] != 3:
            model_input = full_rgb
        else:
            model_input = Image.fromarray(full_rgb)

        # Run inference
        with torch.no_grad():
            detections = self._model.predict(model_input, threshold=threshold)

        # Convert supervision Detections to kwiver format
        output = supervision_to_kwiver_detections(detections, self._classes)
        return output


def __vital_algorithm_register__():
    register_vital_algorithm(
        RFDETRDetector, "rf_detr", "PyTorch RF-DETR detection routine"
    )
