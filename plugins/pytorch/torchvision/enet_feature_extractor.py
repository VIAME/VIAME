# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import torch
import torch.utils.data as data
import torch.nn as nn
from torchvision import models, transforms, datasets

from PIL import Image as pilImage

from viame.pytorch.utilities import get_gpu_device, init_cudnn

class SafeNormalize(object):
    """Per-channel normalize that avoids PyTorch vectorization bug on Windows.

    transforms.Normalize uses in-place broadcasting ops that produce garbage
    values on large tensors (>64x64 per channel) in PyTorch 2.10.0a0 Windows
    builds. This version normalizes one channel at a time to bypass the bug.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        result = torch.zeros_like(tensor)
        for i in range(tensor.shape[0]):
            result[i] = (tensor[i] - self.mean[i]) / self.std[i]
        return result

class EfficientNetDataLoader(data.Dataset):# This is the same as the siamese one it was based on
    def __init__(self, bbox_list, transform, frame_img, in_size):
        self._frame_img = pilImage.new( "RGB", frame_img.size )
        self._frame_img.paste( frame_img )
        self._transform = transform
        self._bbox_list = bbox_list
        self._in_size = in_size

    def __getitem__(self, index):
        bb = self._bbox_list[index].bounding_box
        # unwrap
        min_x = float( bb.min_x() )
        min_y = float( bb.min_y() )
        max_x = float( bb.max_x() )
        max_y = float( bb.max_y() )

        c_x = ( min_x + max_x ) / 2
        c_y = ( min_y + max_y ) / 2

        padding = 1.12 * max( max_x - min_x, max_y - min_y ) / 2

        # crop a square image
        im = self._frame_img.crop(
            (
                c_x - padding,
                c_y - padding,
                c_x + padding,
                c_y + padding
            )
        )

        im = im.resize((self._in_size, self._in_size), pilImage.BILINEAR)
        im = im.convert('RGB')

        if self._transform is not None:
            im = self._transform(im)

        return im

    def __len__(self):
        return self._bbox_list.size()


class EfficientNetFeatureExtractor(object):
    """
    Obtain the appearance features from a trained pytorch efficientnet50
    model
    """

    def __init__(self, model_path, img_size, batch_size, gpu_list=None):
        self._device, use_gpu_flag = get_gpu_device(gpu_list)

        # load the efficientnet50 model. Maybe this shouldn't be hardcoded?
        self._model = models.efficientnet_v2_s()
        weights = torch.load( model_path, weights_only=False )

        self._model.load_state_dict( weights )
        self._model.train( False )
        self._model.to( self._device )

        self._transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            SafeNormalize( [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] )
        ])
        self._img_size = img_size
        self._b_size = batch_size
        self.frame = None

        # Warmup pass to initialize cuDNN in this thread context
        init_cudnn(self._device)

    def __call__(self, bbox_list, MOT_flag):
        return self._obtain_feature(bbox_list, MOT_flag)

    def _obtain_feature(self, bbox_list, MOT_flag):
        kwargs = {'num_workers': 0, 'pin_memory': True}
        if self.frame is not None:
            bbox_loader_class = EfficientNetDataLoader(bbox_list, self._transform,
                                                       self.frame, self._img_size)
        else:
            raise ValueError("Trying to create ResenetDataLoader without a frame")

        bbox_loader = torch.utils.data.DataLoader(bbox_loader_class,
            batch_size=self._b_size, shuffle=False, **kwargs)

        torch.set_grad_enabled(False)

        def get_features( name ):
            def hook( model, input, output ):
                features[name] = output.detach()
            return hook

        features = {}
        self._model.avgpool.register_forward_hook(get_features('feats'))

        for idx, imgs in enumerate(bbox_loader):
            v_imgs = imgs.to(self._device)
            self._model(v_imgs)
            output = features['feats']
            if idx == 0:
                app_features = output
            else:
                app_features = torch.cat((app_features, output), dim=0)

        return app_features.cpu()
