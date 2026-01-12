# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

#
# INSERT COPYRIGHT STATEMENT OR DELETE THIS
#

import torch
import PIL
import kwimage

import numpy as np

from scipy import ndimage
from torchvision import transforms
from collections import namedtuple

from kwiver.vital.algo import ImageObjectDetector

from kwiver.vital.types import Image
from kwiver.vital.types import ImageContainer
from kwiver.vital.types import (
    BoundingBoxD, DetectedObject, DetectedObjectSet, DetectedObjectType
)

from viame.pytorch.learn.cutler.crf import densecrf
from viame.pytorch.learn.cutler.dino import ViTFeat
from viame.pytorch.learn.cutler.maskcut import maskcut_forward, create_annotation_info, category_info, resize_binary_mask

# modfied by Xudong Wang based on third_party/TokenCut
from viame.pytorch.learn.tokencut.unsupervised_saliency_detection import utils, metric
from viame.pytorch.learn.tokencut.unsupervised_saliency_detection.object_discovery import detect_box


_Option = namedtuple('_Option', ['attr', 'config', 'default', 'parse', 'help'])


class MaskCutDetector( ImageObjectDetector ):
    """
    Implementation of ImageObjectDetector class
    """
    _options = [
        _Option('_cpu', 'cpu', False, bool, ''),
        # Dino ViT
        _Option('_pretrained_path', 'pretrained_path', '', str, 'path to pretrained model'),
        _Option('_vit_arch', 'vit_arch', 'small', str, 'which architecture'), # ['small', 'base']
        _Option('_vit_feat', 'vit_feat', 'k', str, 'which features'), # ['k', 'q', 'v', 'kqv']
        _Option('_patch_size', 'patch_size', 8, int, 'patch size'), # [16, 8]
        # maskcut
        _Option('_N', 'N', 3, int, 'the maximum number of pseudo-masks per image'),
        _Option('_tau', 'tau', 0.2, float, 'threshold used for producing binary graph'),
    ]

    def __init__( self ):
        ImageObjectDetector.__init__( self )
        
        self.idx = 0

        for opt in self._options:
            setattr(self, opt.attr, opt.default)

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

        if self._vit_arch == 'base' and self._patch_size == 8:
            self._pretrained_path = "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
            self._feat_dim = 768
        else:#if self._vit_arch == 'small' and self._patch_size == 8:
            self._pretrained_path = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
            self._feat_dim = 384

        self.backbone = ViTFeat(self._pretrained_path, self._feat_dim, 
                                self._vit_arch, self._vit_feat, self._patch_size)
        print(f'Load {self._vit_arch} pre-trained feature...')

        self.backbone.eval()
        if not self._cpu: self.backbone.cuda()

    def check_configuration( self, cfg ):
        return True
        if not self._pretrained_path:
            print("Pretrained path must be set")
            return False
        
        valid_vit_archs = ['small', 'base']
        if self._vit_arch not in valid_vit_archs:
            print(f"ViT arch {self._vit_arch} is not valid, must be one of {valid_vit_archs}")
            return False
        
        valid_vit_feats = ['k', 'q', 'v', 'kqv']
        if self._vit_feat not in valid_vit_feats:
            print(f"ViT feat {self._vit_feat} is not valid, must be one of {valid_vit_feats}")
            return False
        
        valid_patch_sizes = [16, 8]
        if self._patch_size not in valid_patch_sizes:
            print(f"Patch size {self._patch_size} is not valid, must be one of {valid_patch_sizes}")
            return False
        
        return True

    def detect( self, image_data ):
        self.idx += 1
        print(f'detect image {self.idx}')
        output = DetectedObjectSet()

        # Convert image to PIL Image
        input_array = image_data.asarray().astype( 'uint8' )
        input_image = PIL.Image.fromarray(input_array)
        input_image_size = input_image.size

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

        _, bipartition, eigvec = maskcut_forward(feat, [feat_h, feat_w], 
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

            # create coco-style annotation info
            annotation_info = create_annotation_info(
                idx, 0, category_info, 
                pseudo_mask.astype(np.uint8), input_image_size)
            
            if not annotation_info:
            	return output

            # Convert detections to kwiver format
            [tl_x, tl_y, w, h] = np.array(annotation_info["bbox"]).astype( np.int32 ) # xywh
            bounding_box = BoundingBoxD( tl_x, tl_y, tl_x + w, tl_y + h ) # tlbr
            
            label = "maskcut"
            class_confidence = 1
            detected_object_type = DetectedObjectType( label, class_confidence )

            detected_object = DetectedObject( bounding_box,
                                              class_confidence,
                                              detected_object_type )

            poly = kwimage.Mask(annotation_info['segmentation'], 'bytes_rle').to_multi_polygon()
            mask = Image(poly.to_relative_mask().numpy().data)
            detected_object.mask = ImageContainer(mask)

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
      "maskcut detection inference routine", MaskCutDetector )

    algorithm_factory.mark_algorithm_as_loaded( implementation_name )
