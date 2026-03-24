# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from kwiver.vital.algo import ImageObjectDetector
import scriptconfig as scfg
import ubelt as ub
import torch

from viame.pytorch.utilities import kwimage_to_kwiver_detections, vital_config_update


class MITYoloConfig(scfg.DataConfig):
    """
    The configuration for :class:`MITYoloDetector`.
    """
    weight = scfg.Value(None, help='path to a checkpoint on disk')
    # accelerator = scfg.Value('auto', help='lightning accelerator. Can be cpu, gpu, or auto')
    device = scfg.Value('auto', help='a torch device string or number')

    def __post_init__(self):
        super().__post_init__()


def _patched_postprocess_call(self, predict, rev_tensor=None, image_size=None):
    """patch for PostProcess call to avoid doing nms.
    Originally from yolo/utils/model_utils"""
    if image_size is not None:
        self.converter.update(image_size)
    prediction = self.converter(predict["Main"])
    pred_class, _, pred_bbox = prediction[:3]
    pred_conf = prediction[3] if len(prediction) == 4 else None
    if rev_tensor is not None:
        pred_bbox = (pred_bbox - rev_tensor[:, None, 1:]) / rev_tensor[:, 0:1, None]
    # adapt raw yolo-mit bbox to [class_id, x1, y1, x2, y2, confidence]
    cls_dist = pred_class.sigmoid() * (1 if pred_conf is None else pred_conf)
    max_scores, class_ids = torch.max(cls_dist, dim=-1)
    predicts_all = []
    for b in range(cls_dist.size(0)):
        img_predicts = torch.cat([
            class_ids[b].unsqueeze(-1).float(),
            pred_bbox[b],
            max_scores[b].unsqueeze(-1)
        ], dim=-1)
        predicts_all.append(img_predicts)

    return predicts_all


class MITYoloDetector(ImageObjectDetector):
    """
    Implementation of ImageObjectDetector class

    Ignore:
        developer testing
        we pyenv3.8.19

    CommandLine:
        xdoctest -m /home/joncrall/code/VIAME/plugins/pytorch/mit_yolo_detector.py MITYoloDetector
        xdoctest -m pytorch.mit_yolo_detector MITYoloDetector

    Example:
        >>> import sys, ubelt
        >>> sys.path.append(ubelt.expandpath('~/code/VIAME/plugins'))
        >>> from pytorch.mit_yolo_detector import *  # NOQA
        >>> import kwimage
        >>> #
        >>> weight = ub.Path('~/code/YOLO-v9/viame-runs/train/viame-test/checkpoints/epoch=499-step=129000.ckpt').expand()
        >>> #
        >>> self = MITYoloDetector()
        >>> print(f'self = {ub.urepr(self, nl=1)}')
        >>> image_data = MITYoloDetector.demo_image()
        >>> cfg_in = dict(
        >>>     weight=weight,
        >>>     device='cuda:0',
        >>> )
        >>> self.set_configuration(cfg_in)
        >>> print(f'self = {ub.urepr(self, nl=1)}')
        >>> detected_objects = self.detect(image_data)
        >>> print(f'detected_objects = {ub.urepr(detected_objects, nl=1)}')
    """

    def __init__(self):
        ImageObjectDetector.__init__(self)

        # kwiver configuration variables
        self._kwiver_config = MITYoloConfig()
        self._yolo_objects = {
            'model': None,
            'transform': None,
            'converter': None,
            'post_process': None,
            'classes': None,
        }

        # setharn variables
        self._thresh = None
        self.predictor = None

    @classmethod
    def demo_weights(cls):
        """
        Returns a path to a mit_yolo deployed model

        Returns:
            str: file path to YOLO checkpoint
        """
        raise NotImplementedError

    @classmethod
    def demo_image(cls):
        """
        Returns an image which can be run through the detector

        Returns:
            ImageContainer: an image to test on
        """
        from PIL import Image as PILImage
        from kwiver.vital.util import VitalPIL
        from kwiver.vital.types import ImageContainer
        import kwimage
        image_fpath = kwimage.grab_test_image_fpath()
        pil_img = PILImage.open(image_fpath)
        image_data = ImageContainer(VitalPIL.from_pil(pil_img))
        return image_data

    def get_configuration(self):
        # Inherit from the base class
        cfg = super(ImageObjectDetector, self).get_configuration()
        for key, value in self._kwiver_config.items():
            cfg.set_value(key, str(value))
        return cfg

    def _build_model(self):
        import torch
        from hydra import compose, initialize_config_dir
        from yolo import (
            AugmentationComposer,
            create_converter,
            create_model,
            PostProcess
        )

        # TODO: need to be able to read metadata with weights
        device = self._kwiver_config.device
        if device == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
            else:
                device = torch.device('cpu')

        weight_fpath = ub.Path(self._kwiver_config.weight)
        print(f'weights_fpath={weight_fpath}')

        # Initialize pre-processing and inference with user train configuration
        train_config_dir = weight_fpath.parent
        train_config_name = "train_config.yaml"
        print(f'train_config_path={train_config_dir / train_config_name}')
        with initialize_config_dir(version_base=None, config_dir=str(train_config_dir), job_name="mit_yolo_detector"):
            train_cfg = compose(config_name=train_config_name)
            model = create_model(train_cfg.model, class_num=train_cfg.dataset.class_num, weight_path=weight_fpath).to(device)
            transform = AugmentationComposer([], train_cfg.image_size)
            converter = create_converter(train_cfg.model.name, model, train_cfg.model.anchor, train_cfg.image_size, device)
        # monkey-patch the PostProcess call to skip NMS from yolo-mit as NMS is already done in kwiver
        # this prevent running nms from yolo-mit which is not under user control in kwiver (e.g. if returning empty predictions)
        from yolo.config.config import NMSConfig
        # create a dummy nms for post process
        nms_config = NMSConfig(0.0, 0, 0)
        # Patches PostProcess to avoid NMS
        class CustomPostProcess(PostProcess):
            __call__ = _patched_postprocess_call
        post_process = CustomPostProcess(converter, nms_config)

        # Set the inference pipeline
        self._yolo_objects.update({
            'model': model,
            'transform': transform,
            'converter': converter,
            'post_process': post_process,
            'classes': list(train_cfg.dataset.class_list),
        })

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()

        # Imports used across this func
        import os
        # HACK: merge config doesn't support dictionary input
        vital_config_update(cfg, cfg_in)

        for key in self._kwiver_config.keys():
            self._kwiver_config[key] = str(cfg.get_value(key))

        if os.name == 'nt':
            os.environ["KWIMAGE_DISABLE_TORCHVISION_NMS"] = "1"

        self._build_model()
        return True

    def check_configuration(self, cfg):
        # if not cfg.has_value("deployed"):
        #     print("A network deploy file must be specified!")
        #     return False
        return True

    def detect(self, image_data):
        import torch
        from PIL import Image
        from yolo.utils.kwcoco_utils import tensor_to_kwimage
        full_rgb = image_data.asarray()
        pil_img = Image.fromarray(full_rgb)
        model = self._yolo_objects['model']
        transform = self._yolo_objects['transform']
        post_process = self._yolo_objects['post_process']
        classes = self._yolo_objects['classes']

        im_chw, bbox, rev_tensor = transform(pil_img)
        device = ub.peek(model.parameters()).device
        model.eval()
        with torch.no_grad():
            im_bchw = im_chw.to(device)[None, :, :, :]
            batched_rev_tensor = rev_tensor.to(device)[None]
            predict = model(im_bchw)
            pred_bbox = post_process(predict, batched_rev_tensor)
            yolo_boxes = pred_bbox[0]
            detections = tensor_to_kwimage(yolo_boxes, classes=classes)

        detections = detections.numpy()
        # flags = detections.scores >= self._thresh
        # detections = detections.compress(flags)

        # convert to kwiver format
        output = kwimage_to_kwiver_detections(detections)
        return output


def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    # Register Algorithm
    implementation_name = "mit_yolo"

    if algorithm_factory.has_algorithm_impl_name(
            MITYoloDetector.static_type_name(), implementation_name):
        return

    algorithm_factory.add_algorithm(
        implementation_name, "PyTorch MIT YOLO detection routine",
        MITYoloDetector)

    algorithm_factory.mark_algorithm_as_loaded(implementation_name)
