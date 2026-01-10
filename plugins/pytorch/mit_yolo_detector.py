# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from __future__ import print_function

from kwiver.vital.algo import ImageObjectDetector
import scriptconfig as scfg
import ubelt as ub

from ._util_kwimage import kwimage_to_kwiver_detections
from ._utils import vital_config_update


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
        config_path = os.fspath(yolo_path / 'config')
        rel_config_path = pathlib.Path(os.path.relpath(config_path, path_base))
        print(f'path_base={path_base}')
        print(f'yolo_path={yolo_path}')
        print(f'rel_yolo_path={rel_yolo_path}')
        print(f'config_path={config_path}')

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

        with initialize(config_path=os.fspath(rel_config_path), version_base=None, job_name="mit_yolo_detector"):
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
