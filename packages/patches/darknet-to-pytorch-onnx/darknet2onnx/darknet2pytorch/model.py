import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from darknet2onnx.darknet2pytorch.darknet_config import parse_cfg, load_conv, load_conv_bn, load_fc
from darknet2onnx.darknet2pytorch.yolo_ops import (
    YoloLayer, Mish, MaxPoolDark, Upsample_expand, Reorg, GlobalAvgPool2d, EmptyModule)


def get_region_boxes(boxes_and_confs):
    """Stack all yolo outputs (boxes, probs and confs) into a single output."""

    # print('Getting boxes from boxes and confs ...')

    boxes_list = []
    probs_list = []
    confs_list = []

    for item in boxes_and_confs:
        boxes_list.append(item[0])
        probs_list.append(item[1])
        confs_list.append(item[2])

    # boxes: [batch, num1 + num2 + num3, 1, 4]
    # confs: [batch, num1 + num2 + num3, num_classes]
    boxes = torch.cat(boxes_list, dim=1)
    probs = torch.cat(probs_list, dim=1)
    confs = torch.cat(confs_list, dim=1)

    return [boxes, probs, confs]


class RegionLayer(nn.Module):
    """The YOLOv2 ``[region]`` head.

    Upstream only implements the YOLOv3+ ``[yolo]`` head. YOLOv2 differs in
    three ways that matter:

    * anchors are expressed in **grid cells**, not network pixels, so widths
      scale by ``anchor_w * exp(tw) / grid_w`` rather than by ``/ net_width``;
    * class scores go through a **softmax** over classes (``softmax=1``), not
      an independent sigmoid per class;
    * there is a single detection scale, so no anchor mask.

    Emits the same ``(boxes, probs, confs)`` triple as :class:`YoloLayer` --
    boxes as normalized ``cxcywh`` -- so ``get_region_boxes`` can stack either.
    """

    def __init__(self):
        super(RegionLayer, self).__init__()
        self.anchors = []
        self.num_anchors = 5
        self.anchor_step = 2
        self.num_classes = 0
        self.use_softmax = True

    def forward(self, output, target=None):
        if self.training:
            return output

        n_batch = output.size(0)
        grid_h = output.size(2)
        grid_w = output.size(3)
        n_anchor = self.num_anchors
        n_cls = self.num_classes

        # [B, A*(5+C), H, W] -> [B, A, 5+C, H, W]
        out = output.view(n_batch, n_anchor, 5 + n_cls, grid_h, grid_w)

        # The preceding conv is linear, so the activations are applied here.
        tx = torch.sigmoid(out[:, :, 0])
        ty = torch.sigmoid(out[:, :, 1])
        tw = out[:, :, 2]
        th = out[:, :, 3]
        conf = torch.sigmoid(out[:, :, 4])
        cls = out[:, :, 5:]
        cls = (F.softmax(cls, dim=2) if self.use_softmax
               else torch.sigmoid(cls))

        device = output.device
        dtype = output.dtype
        grid_x = torch.arange(grid_w, device=device, dtype=dtype).view(1, 1, 1, grid_w)
        grid_y = torch.arange(grid_h, device=device, dtype=dtype).view(1, 1, grid_h, 1)

        anchors = torch.tensor(self.anchors, device=device, dtype=dtype)
        anchors = anchors.view(n_anchor, self.anchor_step)[:, 0:2]
        anchor_w = anchors[:, 0].view(1, n_anchor, 1, 1)
        anchor_h = anchors[:, 1].view(1, n_anchor, 1, 1)

        # Normalized centre/extent, matching the YoloLayer convention.
        bx = (tx + grid_x) / grid_w
        by = (ty + grid_y) / grid_h
        bw = (anchor_w * torch.exp(tw)) / grid_w
        bh = (anchor_h * torch.exp(th)) / grid_h

        # [B, A, H, W] -> [B, A*H*W, 1], anchor-major like YoloLayer
        def _flat(t):
            return t.reshape(n_batch, n_anchor * grid_h * grid_w, 1)

        boxes = torch.cat([_flat(bx), _flat(by), _flat(bw), _flat(bh)], dim=-1)
        confs = _flat(conf)
        probs = cls.permute(0, 1, 3, 4, 2).reshape(
            n_batch, n_anchor * grid_h * grid_w, n_cls)
        return boxes, probs, confs


class Darknet(nn.Module):
    """Darknet pytorch model with support for route shortcut, reorg and forward."""
    def __init__(self, cfgfile, inference=False):
        super(Darknet, self).__init__()
        self.inference = inference
        self.training = not self.inference

        self.blocks = parse_cfg(cfgfile)
        self.width = int(self.blocks[0]['width'])
        self.height = int(self.blocks[0]['height'])

        self.models = self.create_network(self.blocks)  # merge conv, bn,leaky
        self.loss = self.models[len(self.models) - 1]

        if self.blocks[(len(self.blocks) - 1)]['type'] == 'region':
            self.anchors = self.loss.anchors
            self.num_anchors = self.loss.num_anchors
            self.anchor_step = self.loss.anchor_step
            self.num_classes = self.loss.num_classes

        self.header = torch.IntTensor([0, 0, 0, 0])
        self.seen = 0

    def forward(self, x):
        ind = -2
        self.loss = None
        outputs = dict()
        out_boxes = []

        for block in self.blocks:
            ind = ind + 1

            if block['type'] == 'net':
                continue
            elif block['type'] in ['convolutional', 'maxpool', 'reorg', 'upsample', 'avgpool', 'softmax', 'connected']:
                x = self.models[ind](x)
                outputs[ind] = x
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
                if len(layers) == 1:
                    if 'groups' not in block.keys() or int(block['groups']) == 1:
                        x = outputs[layers[0]]
                        outputs[ind] = x
                    else:
                        groups = int(block['groups'])
                        group_id = int(block['group_id'])
                        _, b, _, _ = outputs[layers[0]].shape
                        x = outputs[layers[0]][:, b // groups * group_id:b // groups * (group_id + 1)]
                        outputs[ind] = x
                elif len(layers) == 2:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x = torch.cat((x1, x2), 1)
                    outputs[ind] = x
                elif len(layers) == 4:
                    x1 = outputs[layers[0]]
                    x2 = outputs[layers[1]]
                    x3 = outputs[layers[2]]
                    x4 = outputs[layers[3]]
                    x = torch.cat((x1, x2, x3, x4), 1)
                    outputs[ind] = x
                else:
                    warnings.warn("rounte number > 2 ,is {}".format(len(layers)))

            elif block['type'] == 'shortcut':
                from_layer = int(block['from'])
                activation = block['activation']
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                x1 = outputs[from_layer]
                x2 = outputs[ind - 1]
                x = x1 + x2
                if activation == 'leaky':
                    x = F.leaky_relu(x, 0.1, inplace=True)
                elif activation == 'relu':
                    x = F.relu(x, inplace=True)
                outputs[ind] = x
            elif block['type'] == 'sam':
                from_layer = int(block['from'])
                from_layer = from_layer if from_layer > 0 else from_layer + ind
                x1 = outputs[from_layer]
                x2 = outputs[ind - 1]
                x = x1 * x2
                outputs[ind] = x
            elif block['type'] == 'region':
                boxes = self.models[ind](x)
                out_boxes.append(boxes)
                outputs[ind] = boxes
            elif block['type'] == 'yolo':
                boxes = self.models[ind](x)
                out_boxes.append(boxes)
                outputs[ind] = boxes
            elif block['type'] == 'cost':
                continue
            else:
                warnings.warn('unknown type %s' % (block['type']))

        if self.training:
            return out_boxes
        else:
            return get_region_boxes(out_boxes)

    def create_network(self, blocks):
        models = nn.ModuleList()

        prev_filters = 3
        out_filters = []
        prev_stride = 1
        out_strides = []
        conv_id = 0
        for block in blocks:
            if block['type'] == 'net':
                prev_filters = int(block['channels'])
                if 'letter_box' in block.keys():
                    letter_box = bool(block['letter_box'])
                    if not letter_box:
                        warnings.warn(f"YOLO parameter letter_box=1 not supported yet!")
                else:
                    letter_box = False
                continue
            elif block['type'] == 'convolutional':
                conv_id = conv_id + 1
                batch_normalize = int(block['batch_normalize'])
                filters = int(block['filters'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])
                is_pad = int(block['pad'])
                pad = (kernel_size - 1) // 2 if is_pad else 0
                activation = block['activation']
                model = nn.Sequential()
                if batch_normalize:
                    model.add_module('conv{0}'.format(conv_id),
                                     nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=False))
                    model.add_module('bn{0}'.format(conv_id), nn.BatchNorm2d(filters))
                    # model.add_module('bn{0}'.format(conv_id), BN2d(filters))
                else:
                    model.add_module('conv{0}'.format(conv_id),
                                     nn.Conv2d(prev_filters, filters, kernel_size, stride, pad))
                if activation == 'leaky':
                    model.add_module('leaky{0}'.format(conv_id), nn.LeakyReLU(0.1, inplace=True))
                elif activation == 'relu':
                    model.add_module('relu{0}'.format(conv_id), nn.ReLU(inplace=True))
                elif activation == 'mish':
                    model.add_module('mish{0}'.format(conv_id), Mish())
                elif activation == 'linear':
                    model.add_module('linear{0}'.format(conv_id), nn.Identity())
                elif activation == 'logistic':
                    model.add_module('sigmoid{0}'.format(conv_id), nn.Sigmoid())
                else:
                    print("No convolutional activation named {}".format(activation))

                prev_filters = filters
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'maxpool':
                pool_size = int(block['size'])
                stride = int(block['stride'])
                if stride == 1 and pool_size % 2:
                    # You can use Maxpooldark instead, here is convenient to convert onnx.
                    # Example: [maxpool] size=3 stride=1
                    model = nn.MaxPool2d(kernel_size=pool_size, stride=stride, padding=pool_size // 2)
                elif stride == pool_size:
                    # You can use Maxpooldark instead, here is convenient to convert onnx.
                    # Example: [maxpool] size=2 stride=2
                    model = nn.MaxPool2d(kernel_size=pool_size, stride=stride, padding=0)
                else:
                    model = MaxPoolDark(pool_size, stride)
                out_filters.append(prev_filters)
                prev_stride = stride * prev_stride
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'avgpool':
                model = GlobalAvgPool2d()
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'softmax':
                model = nn.Softmax()
                out_strides.append(prev_stride)
                out_filters.append(prev_filters)
                models.append(model)
            elif block['type'] == 'cost':
                if block['_type'] == 'sse':
                    model = nn.MSELoss(reduction='mean')
                elif block['_type'] == 'L1':
                    model = nn.L1Loss(reduction='mean')
                elif block['_type'] == 'smooth':
                    model = nn.SmoothL1Loss(reduction='mean')
                out_filters.append(1)
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'reorg':
                stride = int(block['stride'])
                prev_filters = stride * stride * prev_filters
                out_filters.append(prev_filters)
                prev_stride = prev_stride * stride
                out_strides.append(prev_stride)
                models.append(Reorg(stride))
            elif block['type'] == 'upsample':
                stride = int(block['stride'])
                out_filters.append(prev_filters)
                prev_stride = prev_stride // stride
                out_strides.append(prev_stride)

                models.append(Upsample_expand(stride))

            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                ind = len(models)
                layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
                if len(layers) == 1:
                    if 'groups' not in block.keys() or int(block['groups']) == 1:
                        prev_filters = out_filters[layers[0]]
                        prev_stride = out_strides[layers[0]]
                    else:
                        prev_filters = out_filters[layers[0]] // int(block['groups'])
                        prev_stride = out_strides[layers[0]] // int(block['groups'])
                elif len(layers) == 2:
                    # No assert on ordering: YOLOv7-tiny routes do not
                    # list the preceding layer first, and the filter
                    # sum below does not depend on the order.
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]]
                    prev_stride = out_strides[layers[0]]
                elif len(layers) == 4:
                    prev_filters = out_filters[layers[0]] + out_filters[layers[1]] + out_filters[layers[2]] + \
                                   out_filters[layers[3]]
                    prev_stride = out_strides[layers[0]]
                else:
                    print("route error!!!")

                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(EmptyModule())
            elif block['type'] == 'shortcut':
                ind = len(models)
                prev_filters = out_filters[ind - 1]
                out_filters.append(prev_filters)
                prev_stride = out_strides[ind - 1]
                out_strides.append(prev_stride)
                models.append(EmptyModule())
            elif block['type'] == 'sam':
                ind = len(models)
                prev_filters = out_filters[ind - 1]
                out_filters.append(prev_filters)
                prev_stride = out_strides[ind - 1]
                out_strides.append(prev_stride)
                models.append(EmptyModule())
            elif block['type'] == 'connected':
                filters = int(block['output'])
                if block['activation'] == 'linear':
                    model = nn.Linear(prev_filters, filters)
                elif block['activation'] == 'leaky':
                    model = nn.Sequential(
                        nn.Linear(prev_filters, filters),
                        nn.LeakyReLU(0.1, inplace=True))
                elif block['activation'] == 'relu':
                    model = nn.Sequential(
                        nn.Linear(prev_filters, filters),
                        nn.ReLU(inplace=True))
                prev_filters = filters
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(model)
            elif block['type'] == 'region':
                region_layer = RegionLayer()
                region_layer.anchors = [float(i) for i in block['anchors'].split(',')]
                region_layer.num_classes = int(block['classes'])
                self.num_classes = region_layer.num_classes
                region_layer.num_anchors = int(block['num'])
                region_layer.anchor_step = len(region_layer.anchors) // region_layer.num_anchors
                region_layer.use_softmax = bool(int(block.get('softmax', 1)))
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(region_layer)
            elif block['type'] == 'yolo':
                yolo_layer = YoloLayer()
                anchors = block['anchors'].split(',')
                anchor_mask = block['mask'].split(',')
                yolo_layer.anchor_mask = [int(i) for i in anchor_mask]
                yolo_layer.anchors = [float(i) for i in anchors]
                yolo_layer.num_classes = int(block['classes'])
                self.num_classes = yolo_layer.num_classes
                yolo_layer.num_anchors = int(block['num'])
                yolo_layer.anchor_step = len(yolo_layer.anchors) // yolo_layer.num_anchors
                yolo_layer.stride = prev_stride
                yolo_layer.scale_x_y = float(block['scale_x_y'])
                if 'new_coords' in block.keys():
                    yolo_layer.new_coords = bool(block['new_coords'])
                else:
                    yolo_layer.new_coords = False
                yolo_layer.net_height = self.height
                yolo_layer.net_width = self.width
                # yolo_layer.object_scale = float(block['object_scale'])
                # yolo_layer.noobject_scale = float(block['noobject_scale'])
                # yolo_layer.class_scale = float(block['class_scale'])
                # yolo_layer.coord_scale = float(block['coord_scale'])
                out_filters.append(prev_filters)
                out_strides.append(prev_stride)
                models.append(yolo_layer)
            else:
                print('unknown type %s' % (block['type']))

        return models

    def load_weights(self, weightfile):
        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, count=5, dtype=np.int32)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        buf = np.fromfile(fp, dtype=np.float32)
        fp.close()

        start = 0
        ind = -2
        for block in self.blocks:
            if start >= buf.size:
                break
            ind = ind + 1
            if block['type'] == 'net':
                continue
            elif block['type'] == 'convolutional':
                model = self.models[ind]
                batch_normalize = int(block['batch_normalize'])
                if batch_normalize:
                    start = load_conv_bn(buf, start, model[0], model[1])
                else:
                    start = load_conv(buf, start, model[0])
            elif block['type'] == 'connected':
                model = self.models[ind]
                if block['activation'] != 'linear':
                    start = load_fc(buf, start, model[0])
                else:
                    start = load_fc(buf, start, model)
            elif block['type'] == 'maxpool':
                pass
            elif block['type'] == 'reorg':
                pass
            elif block['type'] == 'upsample':
                pass
            elif block['type'] == 'route':
                pass
            elif block['type'] == 'shortcut':
                pass
            elif block['type'] == 'sam':
                pass
            elif block['type'] == 'region':
                pass
            elif block['type'] == 'yolo':
                pass
            elif block['type'] == 'avgpool':
                pass
            elif block['type'] == 'softmax':
                pass
            elif block['type'] == 'cost':
                pass
            else:
                print('unknown type %s' % (block['type']))
