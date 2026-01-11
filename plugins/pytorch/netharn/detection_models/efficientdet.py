"""
Original model code "liberated" from https://github.com/toandaominh1997/EfficientDet.Pytorch

from liberator.closer import Closer

closer = Closer()
closer.add_static('EfficientDet', 'models/efficientdet.py')
closer.expand(['models'])
print(closer.current_sourcecode())
"""
import kwimage
import ndsampler
from viame.arrows.pytorch.netharn import core as nh
import ubelt as ub
from viame.arrows.pytorch.netharn.core.data.channel_spec import ChannelSpec

# from models.module import Anchors
# from models.module import BBoxTransform
# from models.bifpn import BIFPN
# from models.module import ClipBoxes
# from models.efficientnet import EfficientNet
# from models.losses import FocalLoss
# from models.retinahead import RetinaHead
# from torchvision.ops import nms
# from models.module import xavier_init
# from models.utils import MemoryEfficientSwish
# from models.utils import Swish
# from models.utils import efficientnet_params
# from models.utils import get_model_params
# from models.utils import get_same_padding_conv2d
# from models.utils import load_pretrained_weights
# from models.utils import round_filters
# from models.utils import round_repeats
# from models.utils import drop_connect
import collections
import re
from torch.utils import model_zoo
import math
import torch
# from models.module import ConvModule
# from models.module import bias_init_with_prob
# from models.module import normal_init
from functools import partial
import warnings
import torch.nn.functional as F
import numpy as np
import torch.nn as nn


def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)


def bias_init_with_prob(prior_prob):
    """ initialize conv/fc bias value according to giving probablity"""
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
    return bias_init


def conv_ws_2d(input,
               weight,
               bias=None,
               stride=1,
               padding=0,
               dilation=1,
               groups=1,
               eps=1e-5):
    c_in = weight.size(0)
    weight_flat = weight.view(c_in, -1)
    mean = weight_flat.mean(dim=1, keepdim=True).view(c_in, 1, 1, 1)
    std = weight_flat.std(dim=1, keepdim=True).view(c_in, 1, 1, 1)
    weight = (weight - mean) / (std + eps)
    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)


class ConvWS2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 eps=1e-5):
        super(ConvWS2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.eps = eps

    def forward(self, x):
        return conv_ws_2d(x, self.weight, self.bias, self.stride, self.padding,
                          self.dilation, self.groups, self.eps)


norm_cfg = {
    'BN': (
        'bn', nn.BatchNorm2d), 'SyncBN': (
            'bn', nn.SyncBatchNorm), 'GN': (
                'gn', nn.GroupNorm)}


conv_cfg = {'Conv': nn.Conv2d, 'ConvWS': ConvWS2d}


def build_norm_layer(cfg, num_features, postfix=''):
    """ Build normalization layer
    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            requires_grad (bool): [optional] whether stop gradient updates
        num_features (int): number of channels from input.
        postfix (int, str): appended into norm abbreviation to
            create named layer.
    Returns:
        name (str): abbreviation + postfix
        layer (nn.Module): created norm layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in norm_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN':
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer


def build_conv_layer(cfg, *args, **kwargs):
    """ Build convolution layer
    Args:
        cfg (None or dict): cfg should contain:
            type (str): identify conv layer type.
            layer args: args needed to instantiate a conv layer.
    Returns:
        layer (nn.Module): created conv layer
    """
    if cfg is None:
        cfg_ = dict(type='Conv')
    else:
        assert isinstance(cfg, dict) and 'type' in cfg
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in conv_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        conv_layer = conv_cfg[layer_type]

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer


class ConvModule(nn.Module):
    """A conv block that contains conv/norm/activation layers.
    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d.
        padding (int or tuple[int]): Same as nn.Conv2d.
        dilation (int or tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool or str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if norm_cfg is None, otherwise
            False.
        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
        activation (str or None): Activation type, "ReLU" by default.
        inplace (bool): Whether to use inplace mode for activation.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 activation='relu',
                 inplace=True,
                 order=('conv', 'norm', 'act')):
        super(ConvModule, self).__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation
        self.inplace = inplace
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == set(['conv', 'norm', 'act'])

        self.with_norm = norm_cfg is not None
        self.with_activatation = activation is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = False if self.with_norm else True
        self.with_bias = bias

        if self.with_norm and self.with_bias:
            warnings.warn('ConvModule has norm and bias at the same time')

        # build convolution layer
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm_name, norm = build_norm_layer(norm_cfg, norm_channels)
            self.add_module(self.norm_name, norm)

        # build activation layer
        if self.with_activatation:
            # TODO: introduce `act_cfg` and supports more activation layers
            if self.activation not in ['relu']:
                raise ValueError('{} is currently not supported.'.format(
                    self.activation))
            if self.activation == 'relu':
                self.activate = nn.ReLU(inplace=inplace)

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def forward(self, x, activate=True, norm=True):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activatation:
                x = self.activate(x)
        return x


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


class RetinaHead(nn.Module):
    """
    An anchor-based head used in [1]_.
    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.
    References:
        .. [1]  https://arxiv.org/pdf/1708.02002.pdf

    Example:
        >>> from .models.efficientdet import *  # NOQA
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[2]
        >>> box_per_anchor = bbox_pred.shape[2]
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 num_anchors=9,
                 conv_cfg=None,
                 norm_cfg=None,
                 **kwargs):
        super(RetinaHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.cls_out_channels = num_classes
        self.num_anchors = num_anchors
        self._init_layers()

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)
        self.output_act = nn.Sigmoid()

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        cls_energy = self.retina_cls(cls_feat)
        cls_score = self.output_act(cls_energy)
        # out is B x C x W x H, with C = n_classes + n_anchors
        cls_score = cls_score.permute(0, 2, 3, 1)
        batch_size, width, height, channels = cls_score.shape
        cls_score = cls_score.view(
            batch_size, width, height, self.num_anchors, self.num_classes)
        cls_score = cls_score.contiguous().view(x.size(0), -1, self.num_classes)

        bbox_pred = self.retina_reg(reg_feat)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1)
        bbox_pred = bbox_pred.contiguous().view(bbox_pred.size(0), -1, 4)
        return cls_score, bbox_pred

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)


def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(
        a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(
        a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) *
                         (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU


class FocalLoss(nn.Module):
    """
    TODO:
        validate loss formulation

        original derived from
        https://github.com/kuangliu/pytorch-retinanet/blob/master/loss.py

        Alternate impl
        https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/meta_arch/retinanet.py
    """
    # def __init__(self):

    def forward(criterion, classifications, regressions, anchors, annotations, ignore_idxs=None):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        if 0:
            anchor_boxes = kwimage.Boxes(anchors[0], 'tlbr').numpy()
            anchor_dsizes = list(zip(np.round(anchor_boxes.width.ravel()).tolist(), np.round(anchor_boxes.height.ravel()).tolist()))
            ub.dict_hist(anchor_dsizes)
            np.sqrt(anchor_boxes.area)

        # TODO :
        # [ ] - annotation weights
        # [ ] - ignore boxes
        # [ ] - option to ignore coarser pyramid levels in architecture

        if 0:
            # Equivalent impl but with 1 fewer lines and no magic numbers
            anchor_cxywh = kwimage.Boxes(anchor, 'tlbr').to_cxywh()
            (anchor_ctr_x, anchor_ctr_y,
             anchor_width, anchor_heights) = anchor_cxywh.data.T
        else:
            anchor_widths = anchor[:, 2] - anchor[:, 0]
            anchor_heights = anchor[:, 3] - anchor[:, 1]
            anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
            anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights

        device = annotations.device
        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            if ignore_idxs:
                ignore_flags = torch.zeros_like(bbox_annotation[:, 4], dtype=torch.bool)
                for idx in ignore_idxs:
                    ignore_flags |= (bbox_annotation[:, 4] == idx)

            if bbox_annotation.view(-1).shape[0] == 0:
                regression_losses.append(torch.tensor(0.0, requires_grad=True, device=device))
                classification_losses.append(torch.tensor(0.0, requires_grad=True, device=device))

                continue

            clf_eps = 1e-4
            classification = torch.clamp(classification, clf_eps, 1.0 - clf_eps)

            # num_anchors x num_annotations
            IoU = calc_iou(anchor, bbox_annotation[:, :4])

            # For each anchor, find its most similar true box
            # As each anchor is assigned to at most one object box
            IoU_max, IoU_argmax = torch.max(IoU, dim=1)  # num_anchors x 1

            #import pdb
            # pdb.set_trace()

            # compute the loss for classification
            targets = torch.full(classification.shape, fill_value=-1,
                                 device=device, dtype=torch.float)

            # anchor_fg_iou_thresh = 0.5  # original
            # anchor_bg_iou_thresh = 0.4  # original
            anchor_fg_iou_thresh = 0.1  # 0.1 did work
            anchor_bg_iou_thresh = 0.01

            # assign anchors that don't overlap well to background
            # assign anchors that overlap well to foreground
            # note: anchors between iou thresholds are ignored
            negative_flags = IoU_max <= anchor_bg_iou_thresh
            positive_flags = IoU_max > anchor_fg_iou_thresh

            targets[negative_flags, :] = 0
            # TODO: if there is a background class set that to 1.

            num_positive_anchors = positive_flags.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            positive_cidxs = assigned_annotations[positive_flags, 4].long()

            targets[positive_flags, :] = 0
            targets[positive_flags, positive_cidxs] = 1

            alpha_factor = torch.ones(targets.shape).to(device) * alpha

            alpha_factor = torch.where(targets == 1.,
                                       alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(targets == 1.,
                                       1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            # bce2 = -(targets * torch.log(classification) +
            #         (1.0 - targets) * torch.log(1.0 - classification))
            bce = nn.functional.binary_cross_entropy(
                classification, targets, reduction='none')
            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce
            # cls_loss = bce

            cls_loss = torch.where(
                targets != -1.0, cls_loss,
                torch.zeros(cls_loss.shape, device=device))

            clf_loss_norm = (
                cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0)
            )

            classification_losses.append(clf_loss_norm)

            # compute the loss for regression
            if positive_flags.sum() > 0:
                assigned_annotations = assigned_annotations[positive_flags, :]

                anchor_widths_pi = anchor_widths[positive_flags]
                anchor_heights_pi = anchor_heights[positive_flags]
                anchor_ctr_x_pi = anchor_ctr_x[positive_flags]
                anchor_ctr_y_pi = anchor_ctr_y[positive_flags]

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip minimum truth size to 1
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack(
                    (targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                # TODO: must be paramatarized via whatever the BBoxTransform.std / mean is
                targets = targets / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).to(device)

                if 0:
                    # This is the formulation for the actual box positions
                    # but it is unused here
                    tf = BBoxTransform()
                    tf(anchor[None, :], regression[None, :])

                # negative_indices = 1 + (~positive_flags)

                regression_pi = regression[positive_flags, :]
                regression_diff = torch.abs(targets - regression_pi)

                # This is a smooth L1 loss
                regression_loss = torch.where(
                    regression_diff < (1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                regression_losses.append(torch.tensor(0).float().to(device))

        clf_loss = torch.stack(classification_losses).mean(dim=0, keepdim=True)
        regression_loss = torch.stack(regression_losses).mean(dim=0, keepdim=True)

        return clf_loss, regression_loss


def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1],
                                dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


def round_repeats(repeats, global_params):
    """ Round number of filters based on depth multiplier. """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def round_filters(filters, global_params):
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(
        filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


url_map = {
    # Note: these urls no longer work
    'efficientnet-b0': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b0-355c32eb.pth',
    'efficientnet-b1': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b1-f1951068.pth',
    'efficientnet-b2': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b2-8bb594d6.pth',
    'efficientnet-b3': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b3-5fb5a3c3.pth',
    'efficientnet-b4': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b4-6ed6700e.pth',
    'efficientnet-b5': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b5-b6417697.pth',
    'efficientnet-b6': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b6-c76e70fd.pth',
    'efficientnet-b7': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b7-dcc49843.pth',
}


def load_pretrained_weights(model, model_name, load_fc=True):
    """ Loads pretrained weights, and downloads if loading for the first time. """
    state_dict = model_zoo.load_url(url_map[model_name])

    if load_fc:
        model.load_state_dict(state_dict)
    else:
        state_dict.pop('_fc.weight')
        state_dict.pop('_fc.bias')
        res = model.load_state_dict(state_dict, strict=False)
        assert set(res.missing_keys) == set(
            ['_fc.weight', '_fc.bias']), 'issue loading pretrained weights'
    print('Loaded pretrained weights for {}'.format(model_name))


class Identity(nn.Module):
    def __init__(self, ):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class Conv2dStaticSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a fixed image size"""

    def __init__(self, in_channels, out_channels,
                 kernel_size, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [
            self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = image_size if isinstance(image_size, list) else [
            image_size, image_size]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] +
                    (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] +
                    (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d(
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride,
                     self.padding, self.dilation, self.groups)
        return x


class Conv2dDynamicSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a dynamic image size """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels,
                         kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [
            self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] +
                    (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] +
                    (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w //
                          2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


def get_same_padding_conv2d(image_size=None):
    """ Chooses static padding if you have specified an image size, and dynamic padding otherwise.
        Static padding is necessary for ONNX exporting of models. """
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)


BlockArgs = collections.namedtuple('BlockArgs',
                                   ['kernel_size',
                                    'num_repeat',
                                    'input_filters',
                                    'output_filters',
                                    'expand_ratio',
                                    'id_skip',
                                    'stride',
                                    'se_ratio'])


GlobalParams = collections.namedtuple('GlobalParams',
                                      ['batch_norm_momentum',
                                       'batch_norm_epsilon',
                                       'dropout_rate',
                                       'num_classes',
                                       'width_coefficient',
                                       'depth_coefficient',
                                       'depth_divisor',
                                       'min_depth',
                                       'drop_connect_rate',
                                       'image_size'])


class BlockDecoder(object):
    """ Block Decoder for readability, straight from the official TensorFlow repository """

    @staticmethod
    def _decode_block_string(block_string):
        """ Gets a block through a string notation of arguments. """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=[int(options['s'][0])])

    @staticmethod
    def _encode_block_string(block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """
        Decodes a list of string notations to specify blocks inside the network.
        :param string_list: a list of strings, each string is a notation of block
        :return: a list of BlockArgs namedtuples of block args
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """
        Encodes a list of BlockArgs to a list of strings.
        :param blocks_args: a list of BlockArgs namedtuples of block args
        :return: a list of strings, each string is a notation of block
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


def efficientnet_params(model_name):
    """ Map EfficientNet model name to parameter coefficients. """
    params_dict = {
        # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    }
    return params_dict[model_name]


def efficientnet(width_coefficient=None, depth_coefficient=None, dropout_rate=0.2,
                 drop_connect_rate=0.2, image_size=None, num_classes=1000):
    """ Creates a efficientnet model. """

    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s22_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s22_e6_i192_o320_se0.25',
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        # data_format='channels_last',  # removed, this is always true in
        # PyTorch
        num_classes=num_classes,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None,
        image_size=image_size,
    )

    return blocks_args, global_params


def get_model_params(model_name, override_params):
    """ Get the block args and global params for a given model """
    if model_name.startswith('efficientnet'):
        w, d, s, p = efficientnet_params(model_name)
        # note: all models have drop connect rate = 0.2
        blocks_args, global_params = efficientnet(
            width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s)
    else:
        raise NotImplementedError(
            'model name is not pre-defined: %s' % model_name)
    if override_params:
        # ValueError will be raised here if override_params has fields not
        # included in global_params.
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block
    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above
    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (
            0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * \
            self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(
                in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(
                num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(
            num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(
                1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(
                in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(
                in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(
            in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(
            num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))

        x = self._swish(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(
                self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate,
                                 training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods
    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks
    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')
    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        # number of output channels
        out_channels = round_filters(32, self._global_params)
        self._conv_stem = Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(
            num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for i in range(len(self._blocks_args)):
            # Update block input and output filters based on depth multiplier.
            self._blocks_args[i] = self._blocks_args[i]._replace(
                input_filters=round_filters(
                    self._blocks_args[i].input_filters, self._global_params),
                output_filters=round_filters(
                    self._blocks_args[i].output_filters, self._global_params),
                num_repeat=round_repeats(
                    self._blocks_args[i].num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size
            # increase.
            self._blocks.append(MBConvBlock(
                self._blocks_args[i], self._global_params))
            if self._blocks_args[i].num_repeat > 1:
                self._blocks_args[i] = self._blocks_args[i]._replace(
                    input_filters=self._blocks_args[i].output_filters, stride=1)
            for _ in range(self._blocks_args[i].num_repeat - 1):
                self._blocks.append(MBConvBlock(
                    self._blocks_args[i], self._global_params))

        # Head'efficientdet-d0': 'efficientnet-b0',
        # output of final block
        in_channels = self._blocks_args[len(
            self._blocks_args) - 1].output_filters
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(
            num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """
        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        P = []
        index = 0
        num_repeat = 0
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            num_repeat = num_repeat + 1
            if(num_repeat == self._blocks_args[index].num_repeat):
                num_repeat = 0
                index = index + 1
                P.append(x)
        return P

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        # Convolution layers
        P = self.extract_features(inputs)
        return P

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(
            model_name, override_params)
        return cls(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000, in_channels=3):
        model = cls.from_name(model_name, override_params={
                              'num_classes': num_classes})
        try:
            load_pretrained_weights(
                model, model_name, load_fc=(num_classes == 1000))
        except Exception as ex:
            import warnings
            warnings.warn('Failed to get pretrained weights ex = {!r}'.format(ex))
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(
                image_size=model._global_params.image_size)
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = Conv2d(
                in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        return model

    # @classmethod
    # def from_pretrained(cls, model_name, num_classes=1000):
    #     model = cls.from_name(model_name, override_params={
    #                           'num_classes': num_classes})
    #     load_pretrained_weights(
    #         model, model_name, load_fc=(num_classes == 1000))

    #     return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(
            cls, model_name, also_need_pretrained_weights=False):
        """ Validates model name. None that pretrained weights are only available for
        the first four models (efficientnet-b{i} for i in 0,1,2,3) at the moment. """
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = ['efficientnet-b' + str(i) for i in range(num_models)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' +
                             ', '.join(valid_models))

    def get_list_features(self):
        list_feature = []
        for idx in range(len(self._blocks_args)):
            list_feature.append(self._blocks_args[idx].output_filters)

        return list_feature


class ClipBoxes(nn.Module):

    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):

        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

        return boxes


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias'):
        nn.init.constant_(module.bias, bias)


class BiFPNModule(nn.Module):
    def __init__(self,
                 channels,
                 levels,
                 init=0.5,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None,
                 eps=0.0001):
        super(BiFPNModule, self).__init__()
        self.activation = activation
        self.eps = eps
        self.levels = levels
        self.bifpn_convs = nn.ModuleList()
        # weighted
        self.w1 = nn.Parameter(torch.Tensor(2, levels).fill_(init))
        self.relu1 = nn.ReLU()
        self.w2 = nn.Parameter(torch.Tensor(3, levels - 2).fill_(init))
        self.relu2 = nn.ReLU()
        for jj in range(2):
            for i in range(self.levels - 1):  # 1,2,3
                fpn_conv = nn.Sequential(
                    ConvModule(
                        channels,
                        channels,
                        3,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        activation=self.activation,
                        inplace=False)
                )
                self.bifpn_convs.append(fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == self.levels
        # build top-down and down-top path with stack
        levels = self.levels
        # w relu
        w1 = self.relu1(self.w1)
        w1 /= torch.sum(w1, dim=0) + self.eps  # normalize
        w2 = self.relu2(self.w2)
        w2 /= torch.sum(w2, dim=0) + self.eps  # normalize
        # build top-down
        idx_bifpn = 0
        pathtd = inputs
        inputs_clone = []
        for in_tensor in inputs:
            inputs_clone.append(in_tensor.clone())

        for i in range(levels - 1, 0, -1):
            pathtd[i - 1] = (w1[0, i - 1] * pathtd[i - 1] + w1[1, i - 1] * F.interpolate(
                pathtd[i], scale_factor=2, mode='nearest')) / (w1[0, i - 1] + w1[1, i - 1] + self.eps)
            pathtd[i - 1] = self.bifpn_convs[idx_bifpn](pathtd[i - 1])
            idx_bifpn = idx_bifpn + 1
        # build down-top
        for i in range(0, levels - 2, 1):
            pathtd[i + 1] = (w2[0, i] * pathtd[i + 1] + w2[1, i] * F.max_pool2d(pathtd[i], kernel_size=2) +
                             w2[2, i] * inputs_clone[i + 1]) / (w2[0, i] + w2[1, i] + w2[2, i] + self.eps)
            pathtd[i + 1] = self.bifpn_convs[idx_bifpn](pathtd[i + 1])
            idx_bifpn = idx_bifpn + 1

        pathtd[levels - 1] = (w1[0, levels - 1] * pathtd[levels - 1] + w1[1, levels - 1] * F.max_pool2d(
            pathtd[levels - 2], kernel_size=2)) / (w1[0, levels - 1] + w1[1, levels - 1] + self.eps)
        pathtd[levels - 1] = self.bifpn_convs[idx_bifpn](pathtd[levels - 1])
        return pathtd


class BIFPN(nn.Module):
    """
    I think this means bidirectional feature pyramid network
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 stack=1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(BIFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.stack = stack

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.stack_bifpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                activation=self.activation,
                inplace=False)
            self.lateral_convs.append(l_conv)

        for ii in range(stack):
            self.stack_bifpn_convs.append(BiFPNModule(channels=out_channels,
                                                      levels=self.backbone_end_level - self.start_level,
                                                      conv_cfg=conv_cfg,
                                                      norm_cfg=norm_cfg,
                                                      activation=activation))
        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)
        self.init_weights()

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # part 1: build top-down and down-top path with stack
        used_backbone_levels = len(laterals)
        for bifpn_module in self.stack_bifpn_convs:
            laterals = bifpn_module(laterals)
        outs = laterals
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[0](orig))
                else:
                    outs.append(self.fpn_convs[0](outs[-1]))
                for i in range(1, self.num_outs - used_backbone_levels):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


class BBoxTransform(nn.Module):

    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            self.mean = torch.from_numpy(
                np.array([0, 0, 0, 0]).astype(np.float32))
        else:
            self.mean = mean
        if std is None:
            self.std = torch.from_numpy(
                np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
        else:
            self.std = std

    def forward(self, anchors, regressions):
        """
        boxes = anchors
        deltas = regressions
        """

        widths = anchors[:, :, 2] - anchors[:, :, 0]
        heights = anchors[:, :, 3] - anchors[:, :, 1]
        ctr_x = anchors[:, :, 0] + 0.5 * widths
        ctr_y = anchors[:, :, 1] + 0.5 * heights

        dx = regressions[:, :, 0] * self.std[0] + self.mean[0]
        dy = regressions[:, :, 1] * self.std[1] + self.mean[1]
        dw = regressions[:, :, 2] * self.std[2] + self.mean[2]
        dh = regressions[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack(
            [pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes


def shift(shape, stride, anchors):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors


def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    if ratios is None:
        ratios = np.array([0.5, 1, 2])

    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


class Anchors(nn.Module):
    """
    Example:
        >>> from .models.efficientdet import *  # NOQA
        >>> self = anchors = Anchors()
        >>> image_shape = (130, 130)
        >>> anchors = self.forward(image_shape)
        >>> anchors = kwimage.Boxes(anchors, 'tlbr')
        >>> print(anchors.to_cxywh())

    """
    def __init__(self, pyramid_levels=None, strides=None,
                 sizes=None, ratios=None, scales=None):
        super(Anchors, self).__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        if sizes is None:
            self.sizes = [2 ** (x + 2) for x in self.pyramid_levels]
        if ratios is None:
            self.ratios = np.array([0.5, 1, 2])
        if scales is None:
            self.scales = np.array(
                [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

        self.num_per_cell = len(self.scales) * len(self.ratios)

    def forward(self, image_shape, device=None):
        # image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        level_shapes = [(image_shape + 2 ** x - 1) // (2 ** x)
                        for x in self.pyramid_levels]

        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 4)).astype(np.float32)

        for idx, p in enumerate(self.pyramid_levels):
            anchors = generate_anchors(
                base_size=self.sizes[idx],
                ratios=self.ratios,
                scales=self.scales
            )
            level_shape = level_shapes[idx]
            # print('level_shape = {!r}'.format(level_shape))
            shifted_anchors = shift(level_shape, self.strides[idx], anchors)
            # print('shifted_anchors.shape = {!r}'.format(shifted_anchors.shape))
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

        all_anchors = np.expand_dims(all_anchors, axis=0)

        return torch.from_numpy(
            all_anchors.astype(np.float32)).to(device)


MODEL_MAP = {
    'efficientdet-d0': 'efficientnet-b0',
    'efficientdet-d1': 'efficientnet-b1',
    'efficientdet-d2': 'efficientnet-b2',
    'efficientdet-d3': 'efficientnet-b3',
    'efficientdet-d4': 'efficientnet-b4',
    'efficientdet-d5': 'efficientnet-b5',
    'efficientdet-d6': 'efficientnet-b6',
    'efficientdet-d7': 'efficientnet-b6',
}


class EfficientDetCoder(object):
    """
    Transforms output of EfficientDet into kwimage.Detections
    """
    def __init__(coder, classes, threshold, iou_threshold):
        coder.classes = classes
        coder.threshold = threshold
        coder.iou_threshold = iou_threshold

    def decode_batch(coder, outputs):
        classifications = outputs['classifications']
        transformed_anchors = outputs['transformed_anchors']

        batch_size = classifications.shape[0]

        n_top = 3  # always return at least 3 boxes
        batch_dets = []
        for bx in range(batch_size):
            pred_probs = classifications[bx]
            pred_score, pred_cidx = pred_probs.max(dim=1)
            pred_coords = transformed_anchors[bx]
            det = kwimage.Detections(
                boxes=kwimage.Boxes(pred_coords, 'tlbr'),
                scores=pred_score,
                probs=pred_probs,
                class_idxs=pred_cidx,
                classes=coder.classes
            )
            flags = det.scores > coder.threshold
            top_idxs = pred_score.argsort()[-n_top:]
            flags[top_idxs] = True
            det = det.compress(flags)
            det = det.non_max_supress(thresh=coder.iou_threshold)
            batch_dets.append(det)
        return batch_dets


class EfficientDet(nn.Module):
    """
    Ignore:
        >>> from .models.efficientdet import *  # NOQA
        >>> from .models.mm_models import _demo_batch
        >>> classes = ['class_{:0d}'.format(i) for i in range(81)]
        >>> channels = ChannelSpec.coerce('rgb')
        >>> self = EfficientDet(classes=classes, channels='rgb')
        >>> batch = _demo_batch(bsize=3, channels=channels, classes=classes, h=512, w=512)
        >>> outputs = self.forward(batch)
        >>> loss = sum(outputs['loss_parts'].values())
        >>> loss.backward()
        >>> # Test ignore class
        >>> classes = ['a', 'b', 'c', 'ignore']
        >>> channels = ChannelSpec.coerce('rgb')
        >>> self = EfficientDet(classes=classes, channels=channels, n_scales=5)
        >>> batch = _demo_batch(bsize=3, channels=channels, classes=classes, h=512, w=512)
        >>> outputs = self.forward(batch)
        >>> loss = sum(outputs['loss_parts'].values())
        >>> loss.backward()
        >>> # Test empty truth case
        >>> from .models.efficientdet import *  # NOQA
        >>> from .models.mm_models import _demo_batch
        >>> self = EfficientDet(classes=3, channels='rgb')
        >>> classes = ['class_{:0d}'.format(i) for i in range(81)]
        >>> channels = ChannelSpec.coerce('rgb')
        >>> batch = _demo_batch(bsize=[0, 0], channels=channels)
        >>> outputs = self.forward(batch)
        >>> loss = sum(outputs['loss_parts'].values())
        >>> loss.backward()

    Ignore:
        >>> from .detect_fit import *  # NOQA
        >>> harn = setup_harn(bsize=2, datasets='special:shapes256',
        >>>     arch='efficientdet', init='noop', xpu=0, channels='rgb',
        >>>     workers=0, normalize_inputs=False, sampler_backend=None)
        >>> harn.initialize()
        >>> self = harn.raw_model
        >>> batch = harn._demo_batch(tag='vali')
        >>> outputs = self.forward(batch)
        >>> batch_dets = self.coder.decode(outputs)
    """

    __BUILTIN_CRITERION__ = True

    def __init__(self, classes=None, input_stats=None, channels=None,
                 network='efficientdet-d0', D_bifpn=3, W_bifpn=88, D_class=3,
                 threshold=0.01, iou_threshold=0.5, n_scales=5):
        super(EfficientDet, self).__init__()

        classes = ndsampler.CategoryTree.coerce(classes)
        self.classes = classes
        num_classes = len(classes)

        if input_stats is None:
            input_stats = {}

        # TODO: use channels when input is not RGB
        self.channels = ChannelSpec.coerce(channels)
        chan_keys = list(self.channels.keys())
        if len(chan_keys) != 1:
            raise ValueError('this model can only do early fusion')
        if input_stats is None:
            input_stats = {}
        if len(input_stats):
            if chan_keys != list(input_stats.keys()):
                # Backwards compat for older pre-fusion input stats method
                assert 'mean' in input_stats or 'std' in input_stats
                input_stats = {
                    chan_keys[0]: input_stats,
                }
            if len(input_stats) != 1:
                print('GOT input_stats = {!r}'.format(input_stats))
                raise ValueError('this model can only do early fusion')
            main_input_stats = ub.peek(input_stats.values())
        else:
            main_input_stats = {}
        main_input_stats = ub.peek(input_stats.values())
        self.input_norm = nh.layers.InputNorm(**main_input_stats)

        self.backbone = EfficientNet.from_pretrained(MODEL_MAP[network])

        self.n_scales = n_scales

        backbone_level_channels = self.backbone.get_list_features()
        self.neck = BIFPN(in_channels=backbone_level_channels[-n_scales:],
                          out_channels=W_bifpn,
                          stack=D_bifpn,
                          num_outs=5)

        # Note which classes correspond to ignore categories
        self.ignore_class_idxs = []
        for idx, cname in enumerate(self.classes.idx_to_node):
            if cname.lower() == 'ignore':
                self.ignore_class_idxs.append(idx)

        max_level = len(backbone_level_channels)
        pyramid_levels = list(range(max_level - n_scales + 1, max_level + 1))
        self.anchors = Anchors(pyramid_levels=pyramid_levels)

        # TODO: need to parametarize based on anchor spec
        # or not, none of those params are used
        self.bbox_head = RetinaHead(num_classes=num_classes,
                                    in_channels=W_bifpn,
                                    num_anchors=self.anchors.num_per_cell)

        self.regressBoxes = BBoxTransform()  # TODO: move to coder
        # self.clipBoxes = ClipBoxes()  # TODO: move to coder

        if 0:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            self.freeze_bn()

        self.criterion = FocalLoss()

        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.coder = EfficientDetCoder(
            self.classes, self.threshold, self.iou_threshold)

    def _encode_batch(self, batch):
        """
        Transform bioharn inputs into efficientdet format
        """
        if not isinstance(batch, dict):
            raise Exception('expected bioharn style batch')

        if isinstance(batch['inputs'], dict):
            assert len(batch['inputs']) == 1, ('only early fusion for now')
            inputs = ub.peek(batch['inputs'].values())
        else:
            inputs = batch['inputs']
        assert len(inputs.data) == 1
        imgs = inputs.data[0]

        device = self.backbone._conv_stem.weight.device
        imgs = imgs.to(device)

        if 'label' in batch:
            label = batch['label']
            batch_size = imgs.shape[0]

            if 'tlbr' in label:
                assert len(label['tlbr'].data) == 1
                tlbr_tensor = nh.data.collate.padded_collate(label['tlbr'].data[0], -1)
                boxes_tensor = kwimage.Boxes(tlbr_tensor, 'tlbr')
            else:
                assert len(label['cxywh'].data) == 1
                cxywh_tensor = nh.data.collate.padded_collate(label['cxywh'].data[0], -1)
                boxes_tensor = kwimage.Boxes(cxywh_tensor, 'cxywh')

            tlbr_tensor = boxes_tensor.to_ltrb().data
            cidx_tensor = nh.data.collate.padded_collate(
                [c[:, None] for c in label['class_idxs'].data[0]], -1)
            annotations = torch.cat([tlbr_tensor, cidx_tensor.float()], axis=2)
            annotations = annotations.view(batch_size, -1, 5)
            annotations = annotations.to(device)
        else:
            annotations = None

        return imgs, annotations

    def forward(self, batch, return_result=None, return_loss=None):
        imgs, annotations = self._encode_batch(batch)

        imgs = self.input_norm(imgs)

        # x - a tuple of feature pyramid layers
        pyramid_feats = self.extract_feat(imgs)

        # get class scores / boxes for each pyramid layer
        outs = self.bbox_head(pyramid_feats)
        cls_score, bbox_pred = outs

        if 0:
            print('pyr = ' + ub.repr2([z.shape for z in pyramid_feats]))
            print('cls = ' + ub.repr2([z.shape for z in cls_score]))
            print('box = ' + ub.repr2([z.shape for z in bbox_pred]))

        classifications = torch.cat([out for out in cls_score], dim=1)
        regressions = torch.cat([out for out in bbox_pred], dim=1)

        # TODO: memoize this
        anchors = self.anchors(imgs.shape[-2:], imgs.device)

        # annotations should be B,5,N shaped tensor with cxywh + cidx label
        # cidx of -1 indicates unused
        # annotations = annotations.permute(0, 2, 1)

        if annotations is not None:
            ignore_idxs = self.ignore_class_idxs
            clf_loss, regression_loss = self.criterion(
                classifications, regressions, anchors, annotations,
                ignore_idxs)
            loss_parts = {
                'clf_loss': clf_loss,
                'regression_loss': regression_loss,
            }
        else:
            loss_parts = None

        transformed_anchors = self.regressBoxes(anchors, regressions)
        # transformed_anchors = self.clipBoxes(transformed_anchors, imgs)

        outputs = {
            'transformed_anchors': transformed_anchors,
            'classifications': classifications,
            'loss_parts': loss_parts,
        }
        return outputs

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def train(self, mode=True):
        r"""Sets the module in training mode.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        # Don't unfreeze BN layers when training
        self.training = mode
        for module in self.children():
            if not isinstance(module, nn.BatchNorm2d):
                module.train(mode)
        return self

    def extract_feat(self, imgs):
        """
            Directly extract features from the backbone+neck
        """
        bb_feats = self.backbone(imgs)
        pyramid_feats = self.neck(bb_feats[-self.n_scales:])
        if 0:
            print(ub.repr2([z.shape for z in bb_feats]))
            print(ub.repr2([z.shape for z in pyramid_feats]))
        return pyramid_feats
