#
# INSERT COPYRIGHT STATEMENT OR DELETE THIS
#

import PIL
import torch

import numpy as np
import PIL.Image as Image

from crf import densecrf
from scipy import ndimage
from torchvision import transforms
from collections import namedtuple

from kwiver.vital.algo import ImageObjectDetector

from kwiver.vital.types import Image
from kwiver.vital.types import ImageContainer
from kwiver.vital.types import DetectedObject
from kwiver.vital.types import DetectedObjectSet
from kwiver.vital.types import DetectedObjectType
from kwiver.vital.types import BoundingBox

# modfied by Xudong Wang based on third_party/TokenCut
from learn.algorithms.CutLER.CutLER_main.third_party.TokenCut.unsupervised_saliency_detection import utils, metric
from learn.algorithms.CutLER.CutLER_main.third_party.TokenCut.unsupervised_saliency_detection.object_discovery import detect_box

from learn.algorithms.CutLER.CutLER_main.maskcut import maskcut, dino


_Option = namedtuple('_Option', ['attr', 'config', 'default', 'parse', 'help'])

class MaskCutDetector( ImageObjectDetector ):
    """
    Implementation of ImageObjectDetector class
    """
    _options = [
        _Option('_cpu', 'cpu', False, bool, ''),
        # Dino ViT
        _Option('_pretrain_path', 'pretrain_path', '', str, 'path to pretrained model'),
        _Option('_vit_arch', 'vit_arch', 'small', str, 'which architecture'), # ['small', 'base']
        _Option('_vit_feat', 'vit_feat', 'k', str, 'which features'), # ['k', 'q', 'v', 'kqv']
        _Option('_patch_size', 'patch_size', 8, int, 'patch size'), # [16, 8]
        # maskcut
        _Option('_N', 'N', 3, int, 'the maximum number of pseudo-masks per image'),
        _Option('_tau', 'tau', 0.2, float, 'threshold used for producing binary graph'),
    ]

    def __init__( self ):
        print("initialized MaskCutDetector")
        ImageObjectDetector.__init__( self )

        for opt in self._options:
            setattr(self, opt.attr, opt.default)

        if self._vit_arch == 'base' and self._patch_size == 8:
            self._pretrain_path = "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
            self._feat_dim = 768
        elif self._vit_arch == 'small' and self._patch_size == 8:
            self._pretrain_path = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
            self._feat_dim = 384

    def get_configuration(self):
        # Inherit from the base class
        cfg = super( ImageObjectDetector, self ).get_configuration()

        for opt in self._options:
            cfg.set_value(opt.config, str(getattr(self, opt.attr)))
        return cfg

    def set_configuration( self, cfg_in ):
        cfg = self.get_configuration()
        cfg.merge_config( cfg_in )

        for opt in self._options:
            setattr(self, opt.attr, opt.parse(cfg.get_value(opt.config)))

        self.backbone = dino.ViTFeat(self._pretrain_path, self._feat_dim, 
                                     self._vit_arch, self._vit_feat, self._patch_size)
        print(msg = 'Load {self._vit_arch} pre-trained feature...')

        self.backbone.eval()
        if not self._cpu: self.backbone.cuda()

    def check_configuration( self, cfg ):
        if not cfg._pretrained_path:
            print("Pretrained path must be set")
            return False
        
        valid_vit_archs = ['small', 'base']
        if cfg._vit_arch not in valid_vit_archs:
            print(f"ViT arch {cfg._vit_arch} is not valid, must be one of {valid_vit_archs}")
            return False
        
        valid_vit_feats = ['k', 'q', 'v', 'kqv']
        if cfg._vit_feat not in valid_vit_feats:
            print(f"ViT feat {cfg._vit_feat} is not valid, must be one of {valid_vit_feats}")
            return False
        
        valid_patch_sizes = [16, 8]
        if cfg._patch_size not in valid_patch_sizes:
            print(f"Patch size {cfg._patch_size} is not valid, must be one of {valid_patch_sizes}")
            return False
        
        return True

    def detect( self, image_data ):
        output = DetectedObjectSet()

        # Convert image to 8-bit numpy
        input_image = image_data.asarray().astype( 'uint8' )
        width, height = input_image.size

        # preprocess image
        fixed_size = 480
        I_new = input_image.resize((int(fixed_size), int(fixed_size)), PIL.Image.LANCZOS)
        I_resize, w, h, feat_w, feat_h = utils.resize_pil(I_new, self._patch_size)

        ToTensor = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(
                                       (0.485, 0.456, 0.406),
                                       (0.229, 0.224, 0.225)),])

        tensor = ToTensor(I_resize).unsqueeze(0)
        if not self._cpu: tensor = tensor.cuda()

        # maskcut
        bipartitions, eigvecs = [], []
        feat = self.backbone(tensor)[0]

        _, bipartition, eigvec = maskcut.maskcut_forward(feat, [feat_h, feat_w], 
                                                 [self._patch_size, self._patch_size], 
                                                 [h,w], self._tau, N=self._N, cpu=self._cpu)
        bipartitions += bipartition
        eigvecs += eigvec

        for idx, bipartition in enumerate(bipartitions):
            # post-process pseudo-masks with CRF
            pseudo_mask = densecrf(np.array(I_new), bipartition)
            pseudo_mask = ndimage.binary_fill_holes(pseudo_mask>=0.5)

            # filter out the mask that have a very different pseudo-mask after the CRF
            mask1 = torch.from_numpy(bipartition)
            mask2 = torch.from_numpy(pseudo_mask)
            if not self._cpu: 
                mask1 = mask1.cuda()
                mask2 = mask2.cuda()
            if metric.IoU(mask1, mask2) < 0.5:
                pseudo_mask = pseudo_mask * -1

            # construct binary pseudo-masks
            pseudo_mask[pseudo_mask < 0] = 0
            pseudo_mask = Image.fromarray(np.uint8(pseudo_mask*255))
            pseudo_mask = np.asarray(pseudo_mask.resize((width, height)))

            # create coco-style annotation info
            annotation_info = maskcut.create_annotation_info(
                idx, 0, maskcut.category_info, 
                pseudo_mask.astype(np.uint8), None)

            # Convert detections to kwiver format
            bbox_int = annotation_info["bbox"].astype( np.int32 )
            bounding_box = BoundingBox( bbox_int[0], bbox_int[1],
                                        bbox_int[2], bbox_int[3] )
            
            label = annotation_info["category_id"]
            class_confidence = 1

            detected_object_type = DetectedObjectType( label, 1.0 )

            detected_object = DetectedObject( bounding_box,
                                              np.max( class_confidence ),
                                              detected_object_type )
            
            mask = annotation_info["segmentation"].to_relative_mask().numpy().data
            detected_object.mask = ImageContainer(Image(mask))

            output.add( detected_object )

        return output

def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    # Register Algorithm
    implementation_name = "maskcut"

    if algorithm_factory.has_algorithm_impl_name(
      MaskCutDetector.static_type_name(), implementation_name ):
        return

    algorithm_factory.add_algorithm( implementation_name,
      "maskcut dection inference routine", MaskCutDetector )

    algorithm_factory.mark_algorithm_as_loaded( implementation_name )
