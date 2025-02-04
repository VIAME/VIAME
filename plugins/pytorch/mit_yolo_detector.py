# ckwg +29
# Copyright 2019 by Kitware, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  * Neither name of Kitware, Inc. nor the names of any contributors may be used
#  to endorse or promote products derived from this software without specific
#  prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import print_function

from kwiver.vital.algo import ImageObjectDetector

try:
    from kwiver.vital.types import BoundingBoxD
except ImportError:
    from kwiver.vital.types import BoundingBox as BoundingBoxD

from kwiver.vital.types import DetectedObjectSet
from kwiver.vital.types import DetectedObject
from kwiver.vital.types import DetectedObjectType

import scriptconfig as scfg
import numpy as np
import ubelt as ub


class MITYoloConfig(scfg.DataConfig):
    """
    The configuration for :class:`MITYoloDetector`.
    """
    weight = scfg.Value(None, help='path to a checkpoint on disk')
    # accelerator = scfg.Value('auto', help='lightning accelerator. Can be cpu, gpu, or auto')
    device = scfg.Value('auto', help='a torch device string or number')

    def __post_init__(self):
        super().__post_init__()


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
            'post_proccess': None,
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
        import pathlib
        import os
        import torch
        from hydra import compose, initialize
        import yolo
        from yolo import (
            AugmentationComposer,
            Config,
            create_converter,
            create_model,
            # custom_logger,
            # draw_bboxes,
        )
        from yolo import PostProcess
        import tempfile
        import kwutil

        weight_fpath = ub.Path(self._kwiver_config['weight'])

        train_config_fpath = weight_fpath.parent / 'train_config.yaml'
        if train_config_fpath.exists():
            train_config = kwutil.Yaml.load(train_config_fpath)
            # class_list = train_config['dataset']['class_list']
        else:
            raise Exception("Cannot introspect model config")

        # Create a dummy path to keep the API happy
        fake_image_path = tempfile.mktemp()

        # temp_config_dpath = ub.Path(tempfile.mkdtemp()).ensuredir()
        # inference_config_fpath = temp_config_dpath / 'config.yaml'
        # train_config_fpath.copy(inference_config_fpath)

        # This is annoying that we cant just specify an absolute path when it is
        # robustly built. Furthermore, the relative path seems like it isn't even
        # from the cwd, but the module that is currently being run.
        # Find the path that we need to be relative to in a somewhat portable
        # manner (i.e. will work in a Jupyter snippet).
        try:
            path_base = pathlib.Path(__file__).parent
        except NameError:
            path_base = pathlib.Path.cwd()
        yolo_path = pathlib.Path(yolo.__file__).parent
        rel_yolo_path = pathlib.Path(os.path.relpath(yolo_path, path_base))
        config_path = os.fspath(rel_yolo_path / 'config')

        # TODO: need to be able to read metadata with weights
        device = self._kwiver_config.device
        if device == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
            else:
                device = torch.device('cpu')

        weights_fpath = self._kwiver_config.weight
        print(f'weights_fpath={weights_fpath}')
        config_name = 'config'
        model = 'v9-c'

        train_config['dataset']

        # Write into the config dir a file that we can use for inference
        # TODO: it would be very nice if we could do this outside of the yolo
        # python module.
        dataset_config = train_config['dataset']
        cfgid = ub.hash_data(dataset_config, base='hex')[0:16]
        dataset_config_name = f'dataset_config_{cfgid}'
        dataset_config_dpath = ub.Path(config_path) / 'dataset'
        dataset_config_fpath = dataset_config_dpath / f'{dataset_config_name}.yaml'
        dataset_config_fpath.write_text(kwutil.Yaml.dumps(dataset_config))

        with initialize(config_path=config_path, version_base=None, job_name="mit_yolo_detector"):
            # Use the hydra system to populate the expected configuration,
            # but then use it to construct the model explicitly
            cfg: Config = compose(
                config_name=config_name,
                overrides=[
                    "task=inference",
                    f"task.data.source={fake_image_path}",
                    f"model={model}",
                    f"weight='{weights_fpath}'",
                    f'dataset={dataset_config_name}',
                    # f"dataset.class_list={class_list}",
                    # f"dataset.class_num={len(class_list)}",
                    "use_wandb=False",
                ]
            )
            model = create_model(cfg.model, class_num=cfg.dataset.class_num, weight_path=cfg.weight).to(device)
            transform = AugmentationComposer([], cfg.image_size)
            converter = create_converter(cfg.model.name, model, cfg.model.anchor, cfg.image_size, device)
            post_proccess = PostProcess(converter, cfg.task.nms)

        self._yolo_objects.update({
            'model': model,
            'transform': transform,
            'converter': converter,
            'post_proccess': post_proccess,
            'classes': list(cfg.dataset.class_list),
        })

    def set_configuration(self, cfg_in):
        cfg = self.get_configuration()

        # Imports used across this func
        import os
        # HACK: merge config doesn't support dictionary input
        _vital_config_update(cfg, cfg_in)

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
        post_proccess = self._yolo_objects['post_proccess']
        classes = self._yolo_objects['classes']

        im_chw, bbox, rev_tensor = transform(pil_img)
        device = ub.peek(model.parameters()).device
        with torch.no_grad():
            im_bchw = im_chw.to(device)[None, :, :, :]
            batched_rev_tensor = rev_tensor.to(device)[None]
            predict = model(im_bchw)
            pred_bbox = post_proccess(predict, batched_rev_tensor)
            yolo_boxes = pred_bbox[0]
            detections = tensor_to_kwimage(yolo_boxes, classes=classes)

        detections = detections.numpy()
        # flags = detections.scores >= self._thresh
        # detections = detections.compress(flags)

        # convert to kwiver format
        output = _kwimage_to_kwiver_detections(detections)
        return output


def _vital_config_update(cfg, cfg_in):
    """
    Treat a vital Config object like a python dictionary

    TODO: move to a utility module

    Args:
        cfg (kwiver.vital.config.config.Config): config to update
        cfg_in (dict | kwiver.vital.config.config.Config): new values
    """
    # vital cfg.merge_config doesnt support dictionary input
    if isinstance(cfg_in, dict):
        for key, value in cfg_in.items():
            if cfg.has_value(key):
                cfg.set_value(key, str(value))
            else:
                raise KeyError('cfg has no key={}'.format(key))
    else:
        cfg.merge_config(cfg_in)
    return cfg


def _kwiver_to_kwimage_detections(detected_objects):
    """
    Convert vital detected object sets to kwimage.Detections

    TODO: move to a utility module

    Args:
        detected_objects (kwiver.vital.types.DetectedObjectSet)

    Returns:
        kwimage.Detections
    """
    import ubelt as ub
    import kwimage
    boxes = []
    scores = []
    class_idxs = []

    classes = []
    if len(detected_objects) > 0:
        obj = ub.peek(detected_objects)
        classes = obj.type.all_class_names()

    for obj in detected_objects:
        box = obj.bounding_box
        tlbr = [box.min_x(), box.min_y(), box.max_x(), box.max_y()]
        score = obj.confidence
        cname = obj.type.get_most_likely_class()
        cidx = classes.index(cname)
        boxes.append(tlbr)
        scores.append(score)
        class_idxs.append(cidx)

    dets = kwimage.Detections(
        boxes=kwimage.Boxes(np.array(boxes), 'tlbr'),
        scores=np.array(scores),
        class_idxs=np.array(class_idxs),
        classes=classes,
    )
    return dets


def _kwimage_to_kwiver_detections(detections):
    """
    Convert kwimage detections to kwiver deteted object sets

    TODO: move to a utility module

    Args:
        detected_objects (kwimage.Detections)

    Returns:
        kwiver.vital.types.DetectedObjectSet
    """
    from kwiver.vital.types.types import ImageContainer, Image

    segmentations = None
    # convert segmentation masks
    if 'segmentations' in detections.data:
        segmentations = detections.data['segmentations']

    try:
        boxes = detections.boxes.to_ltrb()
    except Exception:
        boxes = detections.boxes.to_tlbr()

    scores = detections.scores
    class_idxs = detections.class_idxs

    if not segmentations:
        # Placeholders
        segmentations = (None,) * len(boxes)

    # convert to kwiver format, apply threshold
    detected_objects = DetectedObjectSet()

    for tlbr, score, cidx, seg in zip(boxes.data, scores, class_idxs, segmentations):
        class_name = detections.classes[cidx]

        bbox_int = np.round(tlbr).astype(np.int32)
        bounding_box = BoundingBoxD(
            bbox_int[0], bbox_int[1], bbox_int[2], bbox_int[3])

        detected_object_type = DetectedObjectType(class_name, score)
        detected_object = DetectedObject(
            bounding_box, score, detected_object_type)
        if seg:
            mask = seg.to_relative_mask().numpy().data
            detected_object.mask = ImageContainer(Image(mask))

        detected_objects.add(detected_object)
    return detected_objects


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
