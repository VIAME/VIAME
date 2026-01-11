#
#   Darknet YOLOv2 model
#   Copyright EAVISE
#
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import ubelt as ub
from viame.pytorch.netharn.models.yolo2 import light_postproc

__all__ = ['Yolo']


class Conv2dBatchLeaky(nn.Module):
    """ This convenience layer groups a 2D convolution, a batchnorm and a leaky ReLU.
    They are executed in a sequential manner.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the kernel of the convolution
        stride (int or tuple): Stride of the convolution
        padding (int or tuple): padding of the convolution
        leaky_slope (number, optional): Controls the angle of the negative slope of the leaky ReLU; Default **0.1**
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
        self.layers = nn.Sequential(
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


class Reorg(nn.Module):
    """ This layer reorganizes a tensor according to a stride.
    The dimensions 2,3 will be sliced by the stride and then stacked in dimension 1. (input must have 4 dimensions)

    Args:
        stride (int): stride to divide the input tensor
    """
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        if not isinstance(stride, int):
            raise TypeError('stride is not an int [{}]'.format(type(stride)))
        self.stride = stride
        self.darknet = True

    def __repr__(self):
        return '{} (stride={}, darknet_compatible_mode={})'.format(
            self.__class__.__name__, self.stride, self.darknet)

    def forward(self, x):
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)

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


class Yolo(nn.Module):
    """ `Yolo v2`_ implementation with pytorch.

    Modified version original taken from lightnet

    This network uses :class:`~lightnet.network.RegionLoss` as its loss function
    and :class:`~lightnet.data.GetBoundingBoxes` as its default postprocessing function.

    Args:
        num_classes (Number, optional): Number of classes; Default **20**
        weights_file (str, optional): Path to the saved weights; Default **None**
        conf_thresh (Number, optional): Confidence threshold for postprocessing of the boxes; Default **0.25**
        nms_thresh (Number, optional): Non-maxima suppression threshold for postprocessing; Default **0.4**
        input_channels (Number, optional): Number of input channels; Default **3**
        anchors (list): 2D list representing anchor boxes. These width and
            height values should be in network output coordinates.

    Attributes:
        self.loss (fn): loss function. Usually this is :class:`~lightnet.network.RegionLoss`
        self.postprocess (fn): Postprocessing function. By default this is :class:`~lightnet.data.GetBoundingBoxes`

    .. _Yolo v2: https://github.com/pjreddie/darknet/blob/777b0982322142991e1861161e68e1a01063d76f/cfg/yolo-voc.cfg

    CommandLine:
        python ~/code/netharn/netharn/models/yolo2/light_yolo.py Yolo

    Example:
        >>> # xdoctest: +REQUIRES(module:kwimage)
        >>> from .models.yolo2.light_yolo import *
        >>> torch.random.manual_seed(0)
        >>> B, C, Win, Hin = 2, 20, 96, 96
        >>> self = Yolo(num_classes=C, conf_thresh=4.9e-2)
        >>> im_data = torch.randn(B, 3, Hin, Win)
        >>> # the _forward function produces raw YOLO output
        >>> network_output = self.forward(im_data)
        >>> A = len(self.anchors)
        >>> Wout, Hout = Win // 32, Hin // 32
        >>> assert list(network_output.shape) == [2, 5, 25, 3, 3]
        >>> assert list(network_output.shape) == [B, A, (C + 5), Wout, Hout]
        >>> # The default postprocess function will construct the bounding boxes
        >>> # Each item in `batch_dets` is a list of Detections objects.
        >>> batch_dets = self.postprocess(network_output)
        >>> boxes = batch_dets[0].numpy()
        >>> print(ub.repr2(boxes.data, nl=1))  # xdoc: +IGNORE_WANT
        {
            'boxes': <Boxes(cxywh,
                         array([[0.834205  , 0.49842083, 5.505774  , 3.8463774 ],
                                [0.8421171 , 0.16787954, 1.5673765 , 1.9569172 ],
                                [0.4886469 , 0.5097498 , 0.84630543, 0.79987913]], dtype=float32))>,
            'class_idxs': np.array([ 5, 15, 10], dtype=np.int64),
            'scores': np.array([0.05158454, 0.051546  , 0.04938344], dtype=np.float32),
        }
    """

    def __init__(self, num_classes=20, conf_thresh=.25,
                 nms_thresh=.4, input_channels=3, anchors=None):
        """ Network initialisation """
        super(Yolo, self).__init__()

        if anchors is None:
            anchors = np.array([(1.3221, 1.73145), (3.19275, 4.00944),
                                (5.05587, 8.09892), (9.47112, 4.84053),
                                (11.2364, 10.0071)], dtype=float)
            # np.asarray([(1.08, 1.19), (3.42, 4.41), (6.63, 11.38),
            #                       (9.42, 5.11), (16.62, 10.52)],
            #                      dtype=float)

        # Parameters
        self.num_classes = num_classes
        self.anchors = anchors
        self.reduction = 32             # input_dim/output_dim

        # Network
        layer_list = [
            # Sequence 0 : input = image tensor
            OrderedDict([
                ('1_convbatch',     Conv2dBatchLeaky(
                    input_channels, 32, 3, 1, 1)),
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

        self.postprocess = light_postproc.GetBoundingBoxes(
            self.num_classes, self.anchors, conf_thresh, nms_thresh,
        )

    def output_shape_for(self, input_shape):
        """
        Shape of the output produced by this network
        """
        nB, c, inH, inW = input_shape
        outH = inH / self.factor
        outW = inW / self.factor
        nA = len(self.anchors)
        nC = self.num_classes
        return (nB, nA, 5 + nC, outH, outW)

    def forward(self, x):
        """
        Example:
            >>> # xdoc: +REQUIRES(--download)
            >>> # xdoctest: +REQUIRES(module:kwimage)
            >>> from .models.yolo2.light_yolo import *
            >>> inp_size = (288, 288)
            >>> self = Yolo(num_classes=20, conf_thresh=0.01, nms_thresh=0.4)
            >>> state_dict = torch.load(demo_voc_weights())['weights']
            >>> self.load_state_dict(state_dict)
            >>> im_data, rgb255 = demo_image(inp_size)
            >>> inputs = torch.cat([im_data, im_data])  # make a batch size of 2
            >>> output = self(inputs)
            >>> # Define remaining params
            >>> orig_sizes = torch.LongTensor([rgb255.shape[0:2][::-1]] * len(inputs))
            >>> batch_dets = self.postprocess(output)
            >>> dets = batch_dets[0]
            >>> # xdoc: +REQUIRES(--show)
            >>> from viame.pytorch import netharn as nh
            >>> import kwplot
            >>> kwplot.autompl()  # xdoc: +SKIP
            >>> dets.meta['classes'] = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            >>>  'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
            >>>  'dog', 'horse', 'motorbike', 'person',
            >>>  'pottedplant', 'sheep', 'sofa', 'train',
            >>>  'tvmonitor')
            >>> kwplot.figure(fnum=1, doclf=True)
            >>> sf = orig_sizes[0]
            >>> dets.boxes.scale(sf, inplace=True)
            >>> kwplot.imshow(rgb255, colorspace='rgb')
            >>> dets.draw()
            >>> kwplot.show_if_requested()
        """
        outputs = []

        outputs.append(self.layers[0](x))
        outputs.append(self.layers[1](outputs[0]))
        # Route : layers=-9
        outputs.append(self.layers[2](outputs[0]))
        # Route : layers=-1,-4
        out = self.layers[3](torch.cat((outputs[2], outputs[1]), 1))

        # Reshape output to separate the anchor dimension from the box and class dimension
        nB, nO, nH, nW = out.shape
        nA = len(self.anchors)
        nC = self.num_classes
        out = out.view(nB, nA, 5 + nC, nH, nW)
        return out


def find_anchors(dset):
    """
    Example:
        >>> # xdoc: +SKIP
        >>> self = YoloVOCDataset(split='train', years=[2007])
        >>> anchors = self._find_anchors()
        >>> print('anchors = {}'.format(ub.repr2(anchors, precision=2)))
        >>> # xdoctest: +REQUIRES(--show)
        >>> xy = -anchors / 2
        >>> wh = anchors
        >>> show_boxes = np.hstack([xy, wh])
        >>> import kwplot
        >>> kwplot.figure(doclf=True, fnum=1)
        >>> kwplot.autompl()  # xdoc: +SKIP
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
            model = lightnet.models.Yolo(num_classes=20)
            model.load_weights(darknet_weight_fpath)
            torch.save(model.state_dict(), torch_fpath)
    except ImportError:
        # Maybe this had a weird bad init state?
        torch_fpath = ub.grabdata('https://data.kitware.com/api/v1/file/5b16b81c8d777f15ebe1ffce/download',
                                  fname='darknet19_448.conv.23.pt',
                                  appname='netharn',
                                  hasher='sha512',
                                  hash_prefix='f38968224a81a')
    return torch_fpath


def demo_image(inp_size):
    from ... import util
    import numpy as np
    import cv2
    rgb255 = util.grab_test_image('astro', 'rgb')
    rgb01 = cv2.resize(rgb255, inp_size).astype(np.float32) / 255
    im_data = torch.FloatTensor([rgb01.transpose(2, 0, 1)])
    return im_data, rgb255


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m lightnet.models.network_yolo all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
