import scriptconfig as scfg
import numpy as np

from kwiver.vital.algo import ImageObjectDetector
from viame.pytorch.utilities import kwimage_to_kwiver_detections, vital_config_update


class HuggingFaceZeroShotDetectorConfig(scfg.DataConfig):
    """
    The configuration for :class:`HuggingFaceZeroShotDetector`.
    """
    model_id = scfg.Value("IDEA-Research/grounding-dino-tiny", help='huggingface model ID')
    device = scfg.Value('cuda', help='a torch device string or number')
    classes = scfg.Value('[foreground object]', help='A YAML list of text prompts')
    threshold = scfg.Value(0.25, help='Threshold to keep object detection predictions based on confidence score')
    text_threshold = scfg.Value(0.25, help='Score threshold to keep text detection predictions')

    def __post_init__(self):
        super().__post_init__()


class HuggingFaceZeroShotDetector(ImageObjectDetector):
    """
    References:
        https://github.com/IDEA-Research/GroundingDINO
        https://huggingface.co/docs/transformers/model_doc/grounding-dino

    Example:
        >>> from viame.pytorch.huggingface_zeroshot_detector import *
        >>> import kwimage
        >>> import ubelt as ub
        >>> self = HuggingFaceZeroShotDetector()
        >>> image_data = HuggingFaceZeroShotDetector.demo_image()
        >>> self.set_configuration({})
        >>> detected_objects = self.detect(image_data)
        >>> print(f'detected_objects = {ub.urepr(detected_objects, nl=1)}')
        >>> # Draw
        >>> from viame.pytorch import utilities
        >>> dets = utilities.kwiver_to_kwimage_detections(detected_objects)
        >>> canvas = dets.draw_on(image_data.asarray())
        >>> # xdoctest: +SKIP("only for interaction")
        >>> kwimage.imwrite('canvas.png', canvas)
    """

    def __init__(self):
        ImageObjectDetector.__init__(self)
        self._kwiver_config = HuggingFaceZeroShotDetectorConfig()
        self.predictor = None

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

    def set_configuration(self, cfg_in):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        cfg = self.get_configuration()
        vital_config_update(cfg, cfg_in)

        for key in self._kwiver_config.keys():
            self._kwiver_config[key] = str(cfg.get_value(key))

        model_id = self._kwiver_config['model_id']
        device = self._kwiver_config['device']
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        return True

    def check_configuration(self, cfg):
        return True

    def detect(self, image_data):
        from PIL import Image
        import torch
        import kwimage
        import kwcoco
        import kwarray
        import kwutil
        full_rgb = image_data.asarray()
        pil_img = Image.fromarray(full_rgb)

        model = self.model
        processor = self.processor

        device = self._kwiver_config.device
        classes: list[str] = kwutil.Yaml.coerce(self._kwiver_config.classes)
        text_labels : list[list[str]] = [classes]

        inputs = processor(images=pil_img, text=text_labels, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self._kwiver_config.threshold,
            text_threshold=self._kwiver_config.text_threshold,
            target_sizes=[pil_img.size[::-1]]
        )
        assert len(results) == 1
        result = results[0]

        boxes = kwimage.Boxes(result["boxes"], 'ltrb')
        classes = kwcoco.CategoryTree.coerce(sorted(set(result["text_labels"])))
        class_idxs = np.array([classes.node_to_idx[c] for c in result["text_labels"]])

        dets = kwimage.Detections(
            boxes=boxes.numpy(),
            scores=kwarray.ArrayAPI.numpy(result['scores']),
            class_idxs=class_idxs,
            classes=classes,
        )
        # convert to kwiver format
        output = kwimage_to_kwiver_detections(dets)
        return output


def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    # Register Algorithm
    implementation_name = "huggingface_zeroshot_detector"

    if algorithm_factory.has_algorithm_impl_name(
            HuggingFaceZeroShotDetector.static_type_name(), implementation_name):
        return

    algorithm_factory.add_algorithm(
        implementation_name, "HuggingFace ZeroShot Object Detection",
        HuggingFaceZeroShotDetector)

    algorithm_factory.mark_algorithm_as_loaded(implementation_name)
