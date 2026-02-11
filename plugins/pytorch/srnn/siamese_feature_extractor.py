# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import torch
import torch.utils.data as data

from torchvision import transforms

from PIL import Image as pilImage

from kwiver.vital.types import BoundingBoxD
from .models import Siamese
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
        import torch
        result = torch.zeros_like(tensor)
        for i in range(tensor.shape[0]):
            result[i] = (tensor[i] - self.mean[i]) / self.std[i]
        return result


class SiameseDataLoader(data.Dataset):
    def __init__(self, bbox_list, transform, frame_img, in_size):
        self._frame_img = frame_img
        self._transform = transform
        self._bbox_list = bbox_list
        self._in_size = in_size

    def __getitem__(self, index):
        bb = self._bbox_list[index]
        im = self._frame_img.crop((float(bb.min_x()), float(bb.min_y()),
                      float(bb.max_x()), float(bb.max_y())))
        im = im.resize((self._in_size, self._in_size), pilImage.BILINEAR)
        im = im.convert('RGB')
        if self._transform is not None:
            im = self._transform(im)

        return im

    def __len__(self):
        return len(self._bbox_list)


class SiameseFeatureExtractor(object):
    """
    Obtain the appearance features from a trained pytorch siamese
    model
    """

    def __init__(self, siamese_model_path, img_size, batch_size, gpu_list=None):
        self._device, use_gpu_flag = get_gpu_device(gpu_list)

        # load Siamese model (matching enet_feature_extractor pattern)
        self._siamese_model = Siamese()
        snapshot = torch.load(siamese_model_path)

        # Strip 'module.' prefix from state dict keys (from DataParallel training)
        state_dict = snapshot['state_dict']
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {self._strip_prefix(k, 'module.'): v
                          for k, v in state_dict.items()}

        self._siamese_model.load_state_dict(state_dict)
        self._siamese_model.train(False)
        self._siamese_model.to(self._device)

        print('Model loaded from {}'.format(siamese_model_path))

        self._transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            SafeNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self._img_size = img_size
        self._b_size = batch_size

        # Warmup pass to initialize cuDNN in this thread context
        init_cudnn(self._device)

    @classmethod
    def _strip_prefix(_cls, string, prefix):
        if not string.startswith(prefix):
            raise ValueError("{!r} was supposed to start with {!r} but does not"
                             .format(string, prefix))
        return string[len(prefix):]

    def __call__(self, frame, bbox_list):
        return self._obtain_features(frame, bbox_list)

    def _obtain_features(self, frame, bbox_list):
        kwargs = {'num_workers': 0, 'pin_memory': True}
        if frame is not None:
            bbox_loader_class = SiameseDataLoader(bbox_list, self._transform,
                                    frame, self._img_size)
        else:
            raise ValueError("Trying to create SiameseDataLoader without providing frame")

        bbox_loader = torch.utils.data.DataLoader(bbox_loader_class,
                            batch_size=self._b_size, shuffle=False, **kwargs)

        with torch.no_grad():
            app_features = [self._siamese_model(imgs.to(self._device))
                            for imgs in bbox_loader]
        return app_features and torch.cat(app_features).cpu()
