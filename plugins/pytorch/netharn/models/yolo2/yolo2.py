#
#   Darknet YOLOv2 model
#   Copyright EAVISE
#   Modified for netharn
#
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import ubelt as ub
from viame.pytorch.netharn import layers
try:  # nocover
    from packaging.version import parse as LooseVersion
except ImportError:
    from distutils.version import LooseVersion


_TORCH_HAS_BOOL_COMP = LooseVersion(torch.__version__) >= LooseVersion('1.2.0')


class Conv2dBatchLeaky(layers.Module):
    """ This convenience layer groups a 2D convolution, a batchnorm and a leaky ReLU.
    They are executed in a sequential manner.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the kernel of the convolution
        stride (int or tuple): Stride of the convolution
        padding (int or tuple): padding of the convolution
        leaky_slope (number, optional): Controls the angle of the negative slope of the leaky ReLU; Default **0.1**

    Example:
        >>> self = Conv2dBatchLeaky(1, 1, 3, 1, 0)
        >>> self.output_shape_for((1, 1, 3, 3))
        >>> self.receptive_field_for()
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, leaky_slope=0.1):
        super(Conv2dBatchLeaky, self).__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.leaky_slope = leaky_slope

        # Layer
        self.layers = layers.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU(self.leaky_slope, inplace=True)
        )

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, negative_slope={leaky_slope})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        x = self.layers(x)
        return x

    def output_shape_for(self, input_shape):
        return self.layers.output_shape_for(input_shape)

    def receptive_field_for(self, input_field=None):
        return self.layers.receptive_field_for(input_field)


class Reorg(nn.Module):
    """
    This layer reorganizes a tensor according to a stride.

    The input must have 4 dimensions: [B, C, H, W]. The H and W dimesions will
    be sliced by the stride and then stacked in dimension C. There is some
    weirdness about how this works see [1] for details.

    References:
        [1] https://leimao.github.io/blog/Reorg-Layer-Explained/

    Args:
        stride (int): stride to divide the input tensor
        darknet (int): use original (slightly odd) darknet compatible behavior

    Example:
        >>> from .models.yolo2.yolo2 import *  # NOQA
        >>> self = Reorg(2)
        >>> x = torch.arange(64).view(1, 4, 4, 4)
        >>> y1 = self.forward(x)
        >>> self.output_shape_for((1, 8, 32, 32))
        >>> self.output_shape_for((1, 8, 16, 32))

    Ignore:
        make = lambda *a: torch.arange(int(np.prod(a))).view(*a)
        x = make(1, 4, 4, 4)
        self.darknet = True
        y1 = self.forward(x)
        self.darknet = False
        y2 = self.forward(x)
        print(ub.hzcat(['y1 = {!r}'.format(y1), ' | ', 'y2 = {!r}'.format(y2)]))
    """
    def __init__(self, stride=2, darknet=True):
        super(Reorg, self).__init__()
        if not isinstance(stride, int):
            raise TypeError('stride is not an int [{}]'.format(type(stride)))
        self.stride = stride
        self.darknet = darknet

    def __repr__(self):
        return '{} (stride={}, darknet_compatible_mode={})'.format(
            self.__class__.__name__, self.stride, self.darknet)

    def forward(self, x):
        assert(x.data.dim() == 4)
        B, C, H, W = x.shape

        if H % self.stride != 0:
            raise ValueError('Dimension mismatch: {} is not divisible by {}'.format(W, self.stride))
        if W % self.stride != 0:
            raise ValueError('Dimension mismatch: {} is not divisible by {}'.format(W, self.stride))

        # darknet compatible version from: https://github.com/thtrieu/darkflow/issues/173#issuecomment-296048648
        if self.darknet:
            x = x.view(B, C // (self.stride ** 2), H, self.stride, W, self.stride).contiguous()
            x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
            x = x.view(B, -1, H // self.stride, W // self.stride)
        else:
            ws, hs = self.stride, self.stride
            x = x.view(B, C, H // hs, hs, W // ws, ws).transpose(3, 4).contiguous()
            x = x.view(B, C, H // hs * W // ws, hs * ws).transpose(2, 3).contiguous()
            x = x.view(B, C, hs * ws, H // hs, W // ws).transpose(1, 2).contiguous()
            x = x.view(B, hs * ws * C, H // hs, W // ws)
        return x

    def output_shape_for(self, input_shape):
        B, C, H, W = input_shape

        C_new = C * int(self.stride ** 2)
        H_new = H // (self.stride)
        W_new = W // (self.stride)

        output_shape = (B, C_new, H_new, W_new)
        return output_shape


class Yolo2(layers.AnalyticModule):
    """
    `Yolov2`_ implementation with pytorch.

    Modified version original taken from lightnet

    Integrated netharn module mixins

    Example:
        >>> # xdoc: +REQUIRES(--download, module:ndsampler)
        >>> from .models.yolo2.yolo2 import *
        >>> torch.random.manual_seed(0)
        >>> B, C, Win, Hin = 2, 20, 96, 96
        >>> self = Yolo2(C, conf_thresh=4.9e-2)
        >>> im_data = torch.randn(B, 3, Hin, Win)
        >>> # the _forward function produces raw YOLO output
        >>> network_output = self.forward(im_data)
        >>> A = len(self.anchors)
        >>> Wout, Hout = Win // 32, Hin // 32
        >>> # The default coder function will construct the bounding boxes
        >>> # Each item in `batch_dets` is a list of Detections objects.
        >>> batch_dets = self.coder.decode_batch(network_output)
        >>> boxes = batch_dets[0].numpy().take([0, 1])
        >>> print(ub.repr2(boxes.data, nl=1))  # xdoc: +IGNORE_WANT
    """

    def __init__(self, classes, conf_thresh=.25, nms_thresh=.4,
                 in_channels=3, anchors=None, input_stats=None):
        """ Network initialisation """
        super(Yolo2, self).__init__()

        if anchors is None:
            anchors = np.array([(1.3221, 1.73145), (3.19275, 4.00944),
                                (5.05587, 8.09892), (9.47112, 4.84053),
                                (11.2364, 10.0071)], dtype=float)

        import ndsampler
        classes = ndsampler.CategoryTree.coerce(classes)

        if input_stats is None:
            input_stats = {}
        self.input_norm = layers.InputNorm(**input_stats)

        self.classes = classes
        self.num_classes = len(classes)
        self.anchors = anchors

        # Network
        layer_list = [
            # Sequence 0 : input = image tensor
            OrderedDict([
                ('1_convbatch',     Conv2dBatchLeaky(in_channels, 32, 3, 1, 1)),
                ('2_max',           nn.MaxPool2d(2, 2)),
                ('3_convbatch',     Conv2dBatchLeaky(32, 64, 3, 1, 1)),
                ('4_max',           nn.MaxPool2d(2, 2)),
                ('5_convbatch',     Conv2dBatchLeaky(64, 128, 3, 1, 1)),
                ('6_convbatch',     Conv2dBatchLeaky(128, 64, 1, 1, 0)),
                ('7_convbatch',     Conv2dBatchLeaky(64, 128, 3, 1, 1)),
                ('8_max',           nn.MaxPool2d(2, 2)),
                ('9_convbatch',     Conv2dBatchLeaky(128, 256, 3, 1, 1)),
                ('10_convbatch',    Conv2dBatchLeaky(256, 128, 1, 1, 0)),
                ('11_convbatch',    Conv2dBatchLeaky(128, 256, 3, 1, 1)),
                ('12_max',          nn.MaxPool2d(2, 2)),
                ('13_convbatch',    Conv2dBatchLeaky(256, 512, 3, 1, 1)),
                ('14_convbatch',    Conv2dBatchLeaky(512, 256, 1, 1, 0)),
                ('15_convbatch',    Conv2dBatchLeaky(256, 512, 3, 1, 1)),
                ('16_convbatch',    Conv2dBatchLeaky(512, 256, 1, 1, 0)),
                ('17_convbatch',    Conv2dBatchLeaky(256, 512, 3, 1, 1)),
            ]),

            # Sequence 1 : input = sequence0
            OrderedDict([
                ('18_max',          nn.MaxPool2d(2, 2)),
                ('19_convbatch',    Conv2dBatchLeaky(512, 1024, 3, 1, 1)),
                ('20_convbatch',    Conv2dBatchLeaky(1024, 512, 1, 1, 0)),
                ('21_convbatch',    Conv2dBatchLeaky(512, 1024, 3, 1, 1)),
                ('22_convbatch',    Conv2dBatchLeaky(1024, 512, 1, 1, 0)),
                ('23_convbatch',    Conv2dBatchLeaky(512, 1024, 3, 1, 1)),
                ('24_convbatch',    Conv2dBatchLeaky(1024, 1024, 3, 1, 1)),
                ('25_convbatch',    Conv2dBatchLeaky(1024, 1024, 3, 1, 1)),
            ]),

            # Sequence 2 : input = sequence0
            OrderedDict([
                ('26_convbatch',    Conv2dBatchLeaky(512, 64, 1, 1, 0)),
                ('27_reorg',        Reorg(2)),
            ]),

            # Sequence 3 : input = sequence2 + sequence1
            OrderedDict([
                ('28_convbatch',    Conv2dBatchLeaky((4 * 64) + 1024, 1024, 3, 1, 1)),
                ('29_conv',         nn.Conv2d(1024, len(self.anchors) * (5 + self.num_classes), 1, 1, 0)),
            ])
        ]
        self.layers = nn.ModuleList(
            [nn.Sequential(layer_dict) for layer_dict in layer_list])

        self.reduction = 32   # = 2 ** 5 =  2 ** (num_downsample_ops)
        self.coder = YoloCoder(
            self.classes, self.anchors, conf_thresh, nms_thresh,
            reduction=self.reduction,
        )

    def forward(self, inputs):
        """
        CommandLine:
            xdoctest -m netharn.models.yolo2.yolo2 Yolo2.forward

        Example:
            >>> # DISABLE_DOCTEST
            >>> # xdoc: +REQUIRES(--download, module:ndsampler)
            >>> from .models.yolo2.yolo2 import *
            >>> self = Yolo2.demo()
            >>> inputs = self.demo_inputs()
            >>> output = self(inputs)
            >>> batch_dets = self.coder.decode_batch(output)
            >>> dets = batch_dets[0]
            >>> print('dets.boxes = {!r}'.format(dets.boxes))
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()  # xdoc: +SKIP
            >>> kwplot.imshow(inputs[0], colorspace='rgb', fnum=1, doclf=True)
            >>> dets.scale(inputs.shape[-2:][::-1]).draw()
            >>> kwplot.show_if_requested()
        """
        normed = self.input_norm(inputs)
        out0 = self.layers[0](normed)
        out1 = self.layers[1](out0)
        # Route : layers=-9
        out2 = self.layers[2](out0)
        # Route : layers=-1,-4
        combo = torch.cat((out2, out1), 1)
        raw0 = self.layers[3](combo)

        # Reshape output to separate the anchor dimension from the box and class dimension
        nB, nO, nH, nW = raw0.shape
        nA = len(self.anchors)
        nC = self.num_classes
        raw = raw0.view(nB, nA, 5 + nC, nH, nW)
        output = {
            'cxywh_energy': raw[:, :, 0:4, :, :],
            'score_energy': raw[:, :, 4:5, :, :],
            'class_energy': raw[:, :, 5:,  :, :],
        }
        return output

    def _analytic_forward(self, inputs, _OutputFor, _Output, _Hidden,
                          **kwargs):
        """
        Define forward in a symbolic way such that output shape and receptive
        field can be analytically computed.

        Ignore:
            >>> from viame.pytorch import netharn as nh
            >>> globals().update(nh.layers.AnalyticModule._analytic_shape_kw())
            >>> inputs = [2, 3, 288, 288]
            >>> self = Yolo2.demo()
            >>> out = self._analytic_forward(inputs, _OutputFor, _Output, _Hidden)
            >>> print(ub.repr2(out.hidden.shallow(2), nl=-1))
        """
        hidden = _Hidden()
        normed = hidden['input_norm'] = _OutputFor(self.input_norm)(inputs)
        # Preproc
        out0 = hidden['out0'] = _OutputFor(self.layers[0])(normed)

        # High level and reorg branches
        out1 = hidden['out1'] = _OutputFor(self.layers[1])(out0)
        out2 = hidden['out2'] = _OutputFor(self.layers[2])(out0)

        combo = hidden['combo'] = _OutputFor(torch.cat)((out2, out1), 1)

        raw0 = hidden['raw0'] = _OutputFor(self.layers[3])(combo)

        nA = len(self.anchors)
        nC = self.num_classes
        nB, nO, nH, nW = _OutputFor.shape(raw0)

        raw = hidden['raw'] = _OutputFor.view(raw0, nB, nA, 5 + nC, nH, nW)

        output = hidden['output'] = {
            'cxywh_energy': _OutputFor.getitem(raw)[:, :, 0:4, :, :],
            'score_energy': _OutputFor.getitem(raw)[:, :, 4:5, :, :],
            'class_energy': _OutputFor.getitem(raw)[:, :, 5:,  :, :],
        }
        return _Output.coerce(output, hidden)

    # def output_shape_for(self, input_shape):
    #     """
    #     Shape of the output produced by this network
    #     """
    #     nB, c, inH, inW = input_shape
    #     outH = inH / self.factor
    #     outW = inW / self.factor
    #     nA = len(self.anchors)
    #     nC = self.num_classes
    #     return (nB, nA, 5 + nC, outH, outW)

    @classmethod
    def demo(cls, key='lightnet'):

        classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                   'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                   'train', 'tvmonitor']
        self = Yolo2(classes=classes, conf_thresh=0.01, nms_thresh=0.4)

        from viame.pytorch import netharn as nh
        weights_fpath = demo_voc_weights(key)
        initializer = nh.initializers.Pretrained(weights_fpath)
        init_info = initializer(self, verbose=0)  # NOQA

        self.eval()
        return self

    def demo_inputs(self):
        inp_size = (288, 288)
        im_data, rgb255 = demo_image(inp_size)
        inputs = torch.cat([im_data, im_data])  # make a batch size of 2
        return inputs


class YoloCoder(object):
    """
    Translate yolo inputs and outputs between input and output space.

    Example:
        >>> # xdoctest: +REQUIRES(--download)
        >>> from .models.yolo2.yolo2 import *  # NOQA
        >>> self = YoloCoder.demo()
        >>> print(self.__json__())
    """
    _coder_version = 3

    def __init__(self, classes, anchors, conf_thresh=0.001, nms_thresh=0.4,
                 reduction=32):
        import ndsampler
        self.classes = ndsampler.CategoryTree.coerce(classes)
        self.num_anchors = len(anchors)
        self.anchor_step = len(anchors[0])
        self.anchors = torch.Tensor(anchors)
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        self.reduction = reduction  # set based on model downsampling

    @classmethod
    def coerce(YoloCoder, arg):
        if isinstance(arg, dict):
            return YoloCoder(**arg)
        elif isinstance(arg, YoloCoder):
            return arg
        else:
            raise TypeError(type(arg))

    def __json__(self):
        return self.__getstate__()

    def __getstate__(self):
        data = self.__dict__.copy()
        data['classes'] = self.classes.__json__()
        data['anchors'] = self.anchors.data.cpu().numpy().tolist()
        data.pop('num_anchors')
        data.pop('anchor_step')
        return data

    def __setstate__(self, state):
        import ndsampler
        state['classes'] = ndsampler.CategoryTree.coerce(state['classes'])
        state['anchors'] = torch.Tensor(state['anchors'])
        state['num_anchors'] = len(state['anchors'])
        state['anchor_step'] = len(state['anchors'][0])
        self.__dict__.update(state)

    @classmethod
    def demo(cls):
        anchors = np.array([
            (1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892),
            (9.47112, 4.84053), (11.2364, 10.0071)])
        self = cls(classes=20, anchors=anchors, conf_thresh=.14, nms_thresh=0.5)
        return self

    def demo_output(self, bsize=3):
        nA = len(self.anchors)
        nC = len(self.classes)
        raw = torch.randn(bsize, nA, 5 + nC, 9, 9)
        output = {
            'cxywh_energy': raw[:, :, 0:4, :, :],
            'score_energy': raw[:, :, 4:5, :, :],
            'class_energy': raw[:, :, 5:,  :, :],
        }
        return output

    def decode_batch(self, output, forloss=False):
        """
        Returns array of detections for every image in batch

        Example:
            >>> # xdoc: +REQUIRES(--download, module:ndsampler)
            >>> from .models.yolo2.yolo2 import *  # NOQA
            >>> self = YoloCoder.demo()
            >>> output = self.demo_output()
            >>> batch_dets = self.decode_batch(output)
            >>> batch_dets = self.decode_batch(output, forloss=True)

        Example:
            >>> # xdoc: +REQUIRES(--download, module:ndsampler)
            >>> info = dev_demodata()
            >>> self, output = ub.take(info, ['coder', 'outputs'])
            >>> batch_dets = self.decode_batch(output)
            >>> dets = batch_dets[0].sort().scale(info['orig_sizes'][0])
            >>> print('dets.boxes = {!r}'.format(dets.boxes))
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.figure(fnum=1, doclf=True)
            >>> kwplot.imshow(info['rgb255'], colorspace='rgb')
            >>> dets.draw()
            >>> kwplot.show_if_requested()
        """
        import kwimage
        # dont modify inplace
        # output = output.clone()

        class_energy = output['class_energy']
        score_energy = output['score_energy']
        cxywh_energy = output['cxywh_energy']

        # Variables
        nB = class_energy.shape[0]
        nH, nW = class_energy.shape[-2:]
        nA = self.num_anchors

        device = class_energy.device

        if self.anchors.device != device:
            self.anchors = self.anchors.to(device)

        # Compute xc,yc, nW,nH, box_score on Tensor
        lin_x = torch.linspace(0, nW - 1, nW, device=device).repeat(nH, 1)
        lin_y = torch.linspace(0, nH - 1, nH, device=device).repeat(nW, 1).t().contiguous()
        anchor_w = self.anchors[:, 0].contiguous().view(1, nA, 1, 1)
        anchor_h = self.anchors[:, 1].contiguous().view(1, nA, 1, 1)

        if forloss:
            # TODO : rectify
            coord = torch.empty_like(cxywh_energy)
            coord[:, :, 0:2, :, :] = cxywh_energy[:, :, 0:2, :, :].sigmoid()    # cx,cy
            coord[:, :, 2:4, :, :] = cxywh_energy[:, :, 2:4, :, :]              # w,h

            with torch.no_grad():
                pred_boxes = torch.empty_like(cxywh_energy, device=device).view(-1, 4)
                pred_boxes[:, 0] = (coord[:, :, 0, :, :] + lin_x).view(-1)
                pred_boxes[:, 1] = (coord[:, :, 1, :, :] + lin_y).view(-1)
                pred_boxes[:, 2] = (coord[:, :, 2, :, :].exp() * anchor_w).view(-1)
                pred_boxes[:, 3] = (coord[:, :, 3, :, :].exp() * anchor_h).view(-1)

            info = {
                'coord': coord,
                'pred_boxes': pred_boxes,
            }
            return info
        else:
            cxywh = cxywh_energy.clone()
            # cxywh_ = cxywh.view(nB, self.num_anchors, -1, nH,  nW)
            cxywh[:, :, 0, :].sigmoid_().add_(lin_x).div_(nW)          # X center
            cxywh[:, :, 1, :].sigmoid_().add_(lin_y).div_(nH)          # Y center
            cxywh[:, :, 2, :].exp_().mul_(anchor_w).div_(nW)           # Width
            cxywh[:, :, 3, :].exp_().mul_(anchor_h).div_(nH)           # Height

        score = score_energy.sigmoid()                             # Box score

        # Compute class_score
        if len(self.classes) > 1:
            cls_scores = torch.nn.functional.softmax(class_energy, dim=2)
            cls_max, cls_max_idx = torch.max(cls_scores, 2, keepdim=True)
            cls_max.mul_(score)
        else:
            cls_max = score
            cls_max_idx = torch.zeros_like(cls_max)

        # Save detection if conf*class_conf is higher than threshold
        flags = cls_max >= self.conf_thresh
        flags_flat = flags.view(-1)

        if flags.sum() == 0:
            batch_dets = []
            for i in range(nB):
                batch_dets.append(kwimage.Detections(
                    boxes=kwimage.Boxes(torch.empty((0, 4), dtype=torch.float32, device=device), 'cxywh'),
                    scores=torch.empty(0, dtype=torch.float32, device=device),
                    class_idxs=torch.empty(0, dtype=torch.int64, device=device),
                    classes=self.classes
                ))
        else:
            # Permute so the bbox dim (i.e. xywh) is trailing
            coords = cxywh.permute(0, 1, 3, 4, 2).contiguous().view(-1, 4)
            coords = coords[flags.view(-1)]

            scores = cls_max[flags]
            class_idxs = cls_max_idx[flags]

            stacked_dets = kwimage.Detections(
                boxes=kwimage.Boxes(coords, 'cxywh'),
                scores=scores,
                class_idxs=class_idxs,
                classes=self.classes
            )

            # Get indexes of splits between images of batch
            max_det_per_batch = len(self.anchors) * nH * nW
            m = max_det_per_batch
            flags_flat.int = flags_flat.int()
            slices = [slice(m * i, m * (i + 1)) for i in range(nB)]
            det_per_batch = torch.IntTensor([flags_flat[s].sum()
                                             for s in slices])
            split_idx = torch.cumsum(det_per_batch, dim=0)

            batch_dets = []
            start = 0
            for end in split_idx:
                dets = stacked_dets[start: end]
                dets = dets.non_max_supress(thresh=self.nms_thresh)
                batch_dets.append(dets)
                start = end
        return batch_dets


class YoloLoss(layers.common.Loss):
    """ Computes region loss from darknet network output and target annotation.

    Args:
        coder (YoloCoder): coder for the network

        # classes (int): number of categories
        # anchors (list): 2D list representing anchor boxes (see :class:`lightnet.network.Darknet`)

        coord_scale (float): weight of bounding box coordinates
        noobject_scale (float): weight of regions without target boxes
        object_scale (float): weight of regions with target boxes
        class_scale (float): weight of categorical predictions
        thresh (float): minimum iou between a predicted box and ground truth for them to be considered matching
        seen (torch.Tensor): How many images the network has already been trained on.
    """

    def __init__(self, coder, seen=0, coord_scale=1.0,
                 noobject_scale=1.0, object_scale=5.0, class_scale=1.0,
                 thresh=0.6, seen_thresh=12800):
        super(YoloLoss, self).__init__()
        coder = YoloCoder.coerce(coder)
        self.coder = coder
        self.classes = coder.classes
        self.num_classes = len(coder.classes)
        self.num_anchors = len(coder.anchors)
        self.anchor_step = len(coder.anchors[0])
        self.anchors = torch.Tensor(coder.anchors)
        self.register_buffer('seen', torch.tensor(seen))
        self.seen_thresh = seen_thresh

        self.coord_scale = coord_scale
        self.noobject_scale = noobject_scale
        self.object_scale = object_scale
        self.class_scale = class_scale
        self.thresh = thresh

        self.mse = nn.MSELoss(reduction='sum')
        self.clf = nn.CrossEntropyLoss(reduction='sum')

    @classmethod
    def demo(YoloLoss):
        coder = YoloCoder.demo()
        self = YoloLoss(coder)
        return self

    def demo_truth(self, bsize=2):
        """
        Example of truth variable
        """
        # true boxes for each item in the batch
        # each box encodes class, x_center, y_center, width, and height
        # coordinates are normalized in the range 0 to 1
        # items in each batch are padded with dummy boxes with class_id=-1
        cxywh = torch.FloatTensor([
            # boxes for batch item 1
            [[0.50, 0.50, 1.00, 1.00],
             [0.32, 0.42, 0.22, 0.12]],
            # boxes for batch item 2 (it has no objects, note the pad!)
            [[0, 0, 0, 0],
             [0, 0, 0, 0]],
        ])

        class_idxs = torch.FloatTensor([
            [2, 4],
            [-1, -1],
        ])

        weight = torch.FloatTensor([
            [0, 1],
            [-1, -1],
        ])

        if bsize == 0:
            cxywh = cxywh[0:0]
            weight = weight[0:0]
            class_idxs = class_idxs[0:0]

        if bsize != 2:
            cxywh = cxywh[0:1].repeat(bsize, 1, 1)
            weight = weight[0:1].repeat(bsize, 1)
            class_idxs = class_idxs[0:1].repeat(bsize, 1)

        target = {
            'cxywh': cxywh,
            'class_idxs': class_idxs,
            'weight': weight,
        }
        return target

    def forward(self, output, target, seen=None):
        """ Compute Region loss.

        CommandLine:
            xdoctest -m netharn.models.yolo2.yolo2 YoloLoss.forward

        Example:
            >>> # xdoc: +REQUIRES(--download, module:ndsampler)
            >>> from .models.yolo2.yolo2 import *  # NOQA
            >>> model = Yolo2.demo()
            >>> self = YoloLoss(model.coder)
            >>> target = self.demo_truth()
            >>> output = model.coder.demo_output(bsize=target['cxywh'].shape[0])
            >>> print('target = ' + ub.repr2(ub.map_vals(lambda x: x.shape, target), nl=1))
            >>> print('output = ' + ub.repr2(ub.map_vals(lambda x: x.shape, output), nl=1))
            >>> seen = None
            >>> loss_parts = self(output, target)
            >>> print('loss_parts = ' + ub.repr2(loss_parts))

        Example:
            >>> # xdoc: +REQUIRES(--download, module:ndsampler)
            >>> # Test cases with no truth
            >>> from .models.yolo2.yolo2 import *  # NOQA
            >>> self = YoloLoss.demo()
            >>> target = self.demo_truth(bsize=2)
            >>> output = self.coder.demo_output(bsize=2)
            >>> # Case 1: weights are -1
            >>> target['weight'][:] = -1
            >>> loss_parts = self(output, target)
            >>> # Case 2: empty with shape
            >>> target = {
            >>>     'cxywh': target['cxywh'][:, 0:0],
            >>>     'class_idxs': target['class_idxs'][:, 0:0],
            >>>     'weight': target['weight'][:, 0:0],
            >>> }
            >>> loss_parts = self(output, target)
            >>> # Case 3: empty without shape
            >>> target = {
            >>>     'cxywh': torch.Tensor(),
            >>>     'class_idxs': torch.Tensor(),
            >>>     'weight': torch.Tensor(),
            >>> }
            >>> loss_parts = self(output, target)

        Example:
            >>> # xdoc: +REQUIRES(--download, module:ndsampler)
            >>> from .models.yolo2.yolo2 import *  # NOQA
            >>> info = dev_demodata()
            >>> self, output, target = ub.take(info, ['criterion', 'outputs', 'target'])
            >>> model = info['model']
            >>> self = YoloLoss(model.coder)
            >>> loss_parts = self.forward(output, target)
            >>> print('loss_parts = {!r}'.format(loss_parts))
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.figure(fnum=1, doclf=True)
            >>> sf = info['orig_sizes'][0]
            >>> dets.boxes.scale(sf, inplace=True)
            >>> kwplot.imshow(info['rgb255'], colorspace='rgb')
            >>> dets.draw()
            >>> kwplot.show_if_requested()
        """
        class_energy = output['class_energy']
        score_energy = output['score_energy']
        cxywh_energy = output['cxywh_energy']

        # Get x,y,w,h,conf,cls
        nB, nA, nC, nH, nW = class_energy.data.shape
        nC = self.num_classes
        assert nA == self.num_anchors
        assert nC == self.num_classes

        device = class_energy.device
        if seen is not None:
            self.seen = torch.tensor(seen)
        elif self.training:
            self.seen += nB

        if self.anchors.device != device:
            self.anchors = self.anchors.to(device)

        conf = score_energy.sigmoid()
        if nC > 1:
            # Swaps the dimensions from [B, A, C, H, W] to be [B, A, H, W, C]
            cls = class_energy.permute(0, 1, 3, 4, 2).contiguous()

        # Note: can't do inplace ops here.
        coord = torch.empty_like(cxywh_energy)
        coord[:, :, 0:2, :, :] = cxywh_energy[:, :, 0:2, :, :].sigmoid()    # cx,cy
        coord[:, :, 2:4, :, :] = cxywh_energy[:, :, 2:4, :, :]              # w,h

        info = self.coder.decode_batch(output, forloss=True)
        coord = info['coord']

        with torch.no_grad():
            # We only use the "decoded" coords (which are pred_boxes) to
            # determine the ground truth mask. We don't need to backprop
            # through here.

            # Get target values
            # CREATE ENCODED TRUTH VALUES *BASED ON THE OUTPUT*
            # TODO: refactor, minimize, verify correctness
            pred_boxes = info['pred_boxes']
            masks, truth = self.build_targets(pred_boxes, target, nB, nA, nC, nH, nW, device)

            if nC > 1:
                pcls_mask = masks['cls'].view(-1, 1)
                if _TORCH_HAS_BOOL_COMP:
                    tcls = truth['cls'][masks['cls'].bool()].view(-1).long()
                else:
                    tcls = truth['cls'][masks['cls']].view(-1).long()

        if nC > 1:
            if _TORCH_HAS_BOOL_COMP:
                pcls = cls.view(-1, nC)[pcls_mask.view(-1).bool()]
            else:
                pcls = cls.view(-1, nC)[pcls_mask.view(-1)]

        coord_mask = masks['coord'].repeat(1, 1, 4, 1)
        coord_ = coord.view(coord_mask.shape)
        conf_ = conf.view(masks['conf'].shape)

        # Compute losses
        coord_part = self.mse(coord_ * coord_mask, truth['coord'] * coord_mask)
        conf_part = self.mse(conf_ * masks['conf'], truth['conf'] * masks['conf'])

        loss_parts = {}
        loss_parts['coord'] = self.coord_scale * (coord_part / nB)
        loss_parts['conf'] = conf_part / nB
        if nC > 1 and pcls.numel() > 0:
            clf_part = (self.clf(pcls, tcls) / nB)
            loss_parts['cls'] = self.class_scale * 2 * clf_part

        return loss_parts

    def build_targets(self, pred_boxes, target, nB, nA, nC, nH, nW, device=None):
        """
        For each ground truth, assign it to a predicted box, and construct
        appropriate truth tensors and masks.

        Args:
            pred_boxes : OUTPUT_SPACE cywh boxes
            target : contains 01-NORMALIZED cxywh boxes, cidxs, and weights

        TODO:
            standardize nonrmalization of inputs (best case is probably to
            simply allow raw targets and normalize insidet this func)

        Example:
            >>> # xdoc: +REQUIRES(--download, module:ndsampler)
            >>> from .models.yolo2.yolo2 import *  # NOQA
            >>> info = dev_demodata()
            >>> self, output, target = ub.take(info, ['criterion', 'outputs', 'target'])
            >>> decoded_info = self.coder.decode_batch(output, forloss=True)
            >>> pred_boxes = decoded_info['pred_boxes']
            >>> nB, nA, nC, nH, nW = output['class_energy'].shape
            >>> device = pred_boxes.device
            >>> self.seen += 100000
            >>> masks, truth = self.build_targets(pred_boxes, target, nB, nA, nC, nH, nW, device)
            >>> print('masks sum = {}'.format(ub.map_vals(lambda x: x.sum(), masks)))
            >>> print('truth sum = {}'.format(ub.map_vals(lambda x: x.sum(), truth)))
            >>> print('masks shape = {}'.format(ub.map_vals(lambda x: x.shape, masks)))
            >>> print('truth shape = {}'.format(ub.map_vals(lambda x: x.shape, truth)))

        Example:
            >>> # xdoc: +REQUIRES(--download, module:ndsampler)
            >>> # Test empty case
            >>> from .models.yolo2.yolo2 import *  # NOQA
            >>> self = YoloLoss.demo()
            >>> target = self.demo_truth(bsize=2)
            >>> for k in target:
            >>>     target[k] = torch.Tensor()
            >>> output = self.coder.demo_output(bsize=2)
            >>> decoded_info = self.coder.decode_batch(output, forloss=True)
            >>> pred_boxes = decoded_info['pred_boxes']
            >>> nB, nA, nC, nH, nW = output['class_energy'].shape
            >>> device = pred_boxes.device
            >>> self.seen += 100000
            >>> masks, truth = self.build_targets(pred_boxes, target, nB, nA, nC, nH, nW, device)
            >>> print('masks sum = {}'.format(ub.map_vals(lambda x: x.sum(), masks)))
            >>> print('truth sum = {}'.format(ub.map_vals(lambda x: x.sum(), truth)))
            >>> print('masks shape = {}'.format(ub.map_vals(lambda x: x.shape, masks)))
            >>> print('truth shape = {}'.format(ub.map_vals(lambda x: x.shape, truth)))
            >>> print('masks sum = {}'.format(ub.map_vals(lambda x: x.sum(), masks)))
            >>> print('truth sum = {}'.format(ub.map_vals(lambda x: x.sum(), truth)))
            >>> print('masks shape = {}'.format(ub.map_vals(lambda x: x.shape, masks)))
            >>> print('truth shape = {}'.format(ub.map_vals(lambda x: x.shape, truth)))
        """
        import math
        # Parameters

        target_cxwh = target['cxywh']  # normalized
        target_cidx = target['class_idxs']
        target_weight = target['weight']

        # TODO: ensure normalized

        # nB = target_cxwh.size(0)
        # nA = self.num_anchors
        nPixels = nH * nW
        item_stride = nA * nH * nW

        # Tensors
        conf_mask = torch.full((nB, nA, nPixels), self.noobject_scale, requires_grad=False)
        coord_mask = torch.zeros(nB, nA, 1, nPixels, requires_grad=False)
        cls_mask = torch.zeros(nB, nA, nPixels, requires_grad=False).byte()

        tcoord = torch.zeros(nB, nA, 4, nPixels, requires_grad=False)
        tconf = torch.zeros(nB, nA, nPixels, requires_grad=False)
        tcls = torch.zeros(nB, nA, nPixels, requires_grad=False)

        # 12800
        if self.seen < self.seen_thresh:
            coord_mask.fill_(1)
            # coord_mask.fill_(.01 / self.coord_scale)

            if self.anchor_step == 4:
                # TODO: use permute
                tcoord[:, :, 0] = self.anchors[:, 2].contiguous().view(1, nA, 1, 1).repeat(nB, 1, 1, nPixels)
                tcoord[:, :, 1] = self.anchors[:, 3].contiguous().view(
                    1, nA, 1, 1).repeat(nB, 1, 1, nPixels)
            else:
                tcoord[:, :, 0].fill_(0.5)
                tcoord[:, :, 1].fill_(0.5)

        self.anchors = self.anchors.to(pred_boxes.device)
        if self.anchor_step == 4:
            anchors_wh = self.anchors[:, 2:4]
        else:
            anchors_wh = self.anchors

        if target_weight.numel() > 0:

            # Determine the number of true boxes in each batch item by checking
            # if the weight has a padding value (which should be -1)
            target_numgt = (target_weight >= 0).sum(dim=1).data.cpu().numpy()

            # For each batch item, ...
            for b, num_gt in enumerate(target_numgt):
                if num_gt > 0:
                    # Remove dummy batch padding (which should be trailing)
                    gtb = target_cxwh[b, 0:num_gt].view(-1, 4)
                    gtc = target_cidx[b, 0:num_gt].view(-1)
                    gtw = target_weight[b, 0:num_gt].view(-1)

                    # Build up tensors
                    cur_pred_boxes = pred_boxes[b * item_stride: (b + 1) * item_stride]

                    # convert to output space
                    cur_true_boxes = gtb.clone()
                    cur_true_boxes[:, ::2] *= nW
                    cur_true_boxes[:, 1::2] *= nH

                    # Set confidence mask of matching detections to 0, we
                    # will selectively reenable a subset of these values later
                    iou_gt_pred = bbox_ious(cur_true_boxes, cur_pred_boxes)
                    mask = (iou_gt_pred > self.thresh).sum(0) >= 1
                    conf_mask_b = conf_mask[b]
                    conf_mask_b[mask.view_as(conf_mask_b)] = 0

                    # Find best anchor for each gt
                    gt_wh = cur_true_boxes[:, 2:4]
                    iou_gt_anchors = wh_ious(gt_wh, anchors_wh)
                    _, best_anchors = iou_gt_anchors.max(1)

                    # Set masks and target values for each gt
                    for i in range(num_gt):
                        cx, cy, w, h = cur_true_boxes[i, 0:4]

                        gi = min(nW - 1, max(0, int(cx)))
                        gj = min(nH - 1, max(0, int(cy)))
                        anch_idx = best_anchors[i]
                        grid_idx = gj * nW + gi

                        weight = gtw[i]

                        coord_mask[b, anch_idx, 0, grid_idx] = (2 - (w * h) / nPixels) * weight
                        cls_mask[b, anch_idx, grid_idx] = weight
                        conf_mask[b, anch_idx, grid_idx] = self.object_scale * weight

                        tcoord[b, anch_idx, 0, grid_idx] = cx - gi
                        tcoord[b, anch_idx, 1, grid_idx] = cy - gj
                        tcoord[b, anch_idx, 2, grid_idx] = math.log(w / self.anchors[anch_idx, 0])
                        tcoord[b, anch_idx, 3, grid_idx] = math.log(h / self.anchors[anch_idx, 1])

                        iou = iou_gt_pred[i, anch_idx * nPixels + grid_idx]
                        tconf[b, anch_idx, grid_idx] = iou

                        tcls[b, anch_idx, grid_idx] = gtc[i]

        masks = {
            'coord': coord_mask.to(device).sqrt_(),
            'conf': conf_mask.to(device).sqrt_(),
            'cls': cls_mask.to(device),
        }
        truth = {
            'coord': tcoord.to(device),
            'conf': tconf.to(device),
            'cls': tcls.to(device),
        }
        return masks, truth


def wh_ious(wh1, wh2):
    """
    Compute IOU between centered boxes with given wh.
    Slightly faster than zeroing the center coords.
    """
    half_wh1 = (wh1 / 2)
    half_wh2 = (wh2 / 2)

    b1x2, b1y2 = (half_wh1).split(1, 1)
    b2x2, b2y2 = (half_wh2).split(1, 1)
    b1x1, b1y1 = -b1x2, -b1y2
    b2x1, b2y1 = -b2x2, -b2y2

    dx = (b1x2.min(b2x2.t()) - b1x1.max(b2x1.t())).clamp_(min=0)
    dy = (b1y2.min(b2y2.t()) - b1y1.max(b2y1.t())).clamp_(min=0)
    intersections = dx * dy

    areas1 = wh1.prod(dim=1, keepdim=True)
    areas2 = wh2.prod(dim=1, keepdim=True)

    unions = (areas1 + areas2.t()) - intersections
    return intersections / unions


def bbox_ious(boxes1, boxes2):
    """ Compute IOU between all boxes from ``boxes1`` with all boxes from ``boxes2``.

    Args:
        boxes1 (torch.Tensor): List of bounding boxes
        boxes2 (torch.Tensor): List of bounding boxes

    Note:
        List format: [[xc, yc, w, h],...]
    """
    b1x1, b1y1 = (boxes1[:, :2] - (boxes1[:, 2:4] / 2)).split(1, 1)
    b1x2, b1y2 = (boxes1[:, :2] + (boxes1[:, 2:4] / 2)).split(1, 1)
    b2x1, b2y1 = (boxes2[:, :2] - (boxes2[:, 2:4] / 2)).split(1, 1)
    b2x2, b2y2 = (boxes2[:, :2] + (boxes2[:, 2:4] / 2)).split(1, 1)

    dx = (b1x2.min(b2x2.t()) - b1x1.max(b2x1.t())).clamp(min=0)
    dy = (b1y2.min(b2y2.t()) - b1y1.max(b2y1.t())).clamp(min=0)
    intersections = dx * dy

    areas1 = (b1x2 - b1x1) * (b1y2 - b1y1)
    areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)
    unions = (areas1 + areas2.t()) - intersections

    return intersections / unions


def find_anchors2(sampler, reduction=32, num_anchors=5):
    """
    Ignore:
        import ndsampler
        sampler = ndsampler.CocoSampler.demo('shapes')
        import sys, ubelt
        from .models.yolo2.yolo2 import *  # NOQA
        sys.path.append(ubelt.expandpath('~/code/netharn/examples'))
        from object_detection import setup_harn
        cmdline = False
        kw = {
            'train_dataset': '~/data/VOC/voc-trainval.mscoco.json',
            'vali_dataset': '~/data/VOC/voc-test-2007.mscoco.json',
        }
        harn = setup_harn(**kw)
        torch_dset = harn.hyper.datasets['train']
        sampler = torch_dset.sampler
        anchors = find_anchors2(sampler, reduction)
        print('anchors = {!r}'.format(anchors))
    """
    from sklearn import cluster
    # all_cxywh = sampler.regions.targets._getcols(['cx', 'cy', 'width', 'height'])
    all_wh = sampler.regions.targets._getcols(['width', 'height'])

    all_imgwh = sampler.regions.targets._getcols(['img_width', 'img_height'])

    # foo = all_imgwh / reduction
    # foo * input_dims

    all_norm_wh = all_wh * reduction / all_imgwh

    ogrid_wh = 18
    all_ogrid_wh = all_norm_wh * ogrid_wh

    algo = cluster.KMeans(
        n_clusters=num_anchors, n_init=20, max_iter=10000, tol=1e-6,
        algorithm='elkan', verbose=0)
    algo.fit(all_ogrid_wh)
    anchors = algo.cluster_centers_
    return anchors


def find_anchors(dset):
    """
    OLD CODE, dont use

    Ignore:
        >>> # xdoctest: +SKIP
        >>> # xdoc: +REQUIRES(--download, module:ndsampler)
        >>> self = YoloVOCDataset(split='train', years=[2007])
        >>> anchors = self._find_anchors()
        >>> print('anchors = {}'.format(ub.repr2(anchors, precision=2)))
        >>> # xdoctest: +REQUIRES(--show)
        >>> xy = -anchors / 2
        >>> wh = anchors
        >>> show_boxes = np.hstack([xy, wh])
        >>> kwplot.autompl()  # xdoc: +SKIP
        >>> kwplot.figure(doclf=True, fnum=1)
        >>> kwplot.draw_boxes(show_boxes, box_format='tlwh')
        >>> from matplotlib import pyplot as plt
        >>> plt.gca().set_xlim(xy.min() - 1, wh.max() / 2 + 1)
        >>> plt.gca().set_ylim(xy.min() - 1, wh.max() / 2 + 1)
        >>> plt.gca().set_aspect('equal')
    """
    import numpy as np
    from PIL import Image
    from sklearn import cluster
    all_norm_wh = []
    for i in ub.ProgIter(range(len(dset)), desc='find anchors'):
        annots = dset._load_annotation(i)
        img_wh = np.array(Image.open(dset.gpaths[i]).size)
        boxes = np.array(annots['boxes'])
        box_wh = boxes[:, 2:4] - boxes[:, 0:2]
        # normalize to 0-1
        norm_wh = box_wh / img_wh
        all_norm_wh.extend(norm_wh.tolist())
    # Re-normalize to the size of the grid
    all_wh = np.array(all_norm_wh) * dset.base_wh[0] / dset.factor
    algo = cluster.KMeans(
        n_clusters=5, n_init=20, max_iter=10000, tol=1e-6,
        algorithm='elkan', verbose=0)
    algo.fit(all_wh)
    anchors = algo.cluster_centers_
    return anchors


def demo_voc_weights(key='lightnet'):
    """
    Demo weights for Pascal VOC dataset
    """

    if key == 'lightnet':
        # url = 'https://gitlab.com/EAVISE/lightnet/raw/master/examples/yolo-voc/lightnet_weights.pt'
        # hash_prefix = 'c4597fed8eb1b01da3495'
        fpath = ub.grabdata('https://data.kitware.com/api/v1/file/5c2e6e1a8d777f072bf2dc65/download',
                            fname='lightnet_weights.pt',
                            appname='netharn',
                            hasher='sha512',
                            hash_prefix='c4597fed8eb1b01')
        return fpath
    elif key == 'darknet':
        url = 'https://pjreddie.com/media/files/yolo-voc.weights'
        hash_prefix = '3033f5f510c25ab3ff6b9'
        fpath = ub.grabdata(url, appname='netharn', hash_prefix=hash_prefix)
        return fpath
    else:
        raise KeyError(key)
    # import lightnet
    # from os.path import dirname, join
    # dpath = dirname(dirname(lightnet.__file__))
    # fpath = join(dpath, 'examples', 'yolo-voc', 'lightnet_weights.pt')
    return fpath


def initial_imagenet_weights():
    # import os
    try:
        darknet_weight_fpath = ub.grabdata(
            'https://pjreddie.com/media/files/darknet19_448.conv.23',
            appname='netharn', hash_prefix='8016f5b7ddc15c5d7dad2315')
        torch_fpath = darknet_weight_fpath + '_lntf.pt'
        import os
        if not os.path.exists(torch_fpath):
            import lightnet.models
            # hack to transform initial state
            model = lightnet.models.Yolo2(classes=20)
            model.load_weights(darknet_weight_fpath)
            torch.save(model.state_dict(), torch_fpath)
    except (Exception, ImportError):
        # Maybe this had a weird bad init state?
        torch_fpath = ub.grabdata('https://data.kitware.com/api/v1/file/5b16b81c8d777f15ebe1ffce/download',
                                  fname='darknet19_448.conv.23.pt',
                                  appname='netharn',
                                  hasher='sha512',
                                  hash_prefix='f38968224a81a')
    return torch_fpath


def demo_image(inp_size):
    import kwimage
    import numpy as np
    import cv2
    rgb255 = kwimage.grab_test_image('astro', 'rgb')
    rgb01 = cv2.resize(rgb255, inp_size).astype(np.float32) / 255
    im_data = torch.FloatTensor([rgb01.transpose(2, 0, 1)])
    return im_data, rgb255


def dev_demodata():
    """
    Create demodata for each part of the system
    """
    inp_size = (288, 288)

    classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor']

    model = Yolo2(classes=classes, conf_thresh=0.01, nms_thresh=0.4).eval()
    state_dict = torch.load(demo_voc_weights())['weights']
    model.load_state_dict(state_dict)

    criterion = YoloLoss(model.coder)

    im_data, rgb255 = demo_image(inp_size)
    inputs = torch.cat([im_data, im_data])  # make a batch size of 2
    outputs = model(inputs)

    label = {
        'cxywh': torch.FloatTensor([
            [205, 255, 350, 500],
        ]) / 512,
        'class_idxs': torch.LongTensor([14]),
        'weight': torch.FloatTensor([1.0]),
    }
    from viame.pytorch import netharn as nh
    target = nh.data.collate.padded_collate([label, label])

    orig_sizes = torch.LongTensor([rgb255.shape[0:2][::-1]] * len(inputs))

    raw = torch.cat([
        outputs['cxywh_energy'],
        outputs['score_energy'],
        outputs['class_energy']], dim=2)

    demodata = {
        'model': model,
        'coder': model.coder,
        'inputs': inputs,
        'outputs': outputs,
        'criterion': criterion,
        'target': target,
        'rgb255': rgb255,
        'raw': raw,
        'orig_sizes': orig_sizes,
    }
    return demodata


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m netharn.models.yolo2.yolo2
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
