"""
DEPRICATE USE THE ONE IN YOLO2

Originally by EAVISE

Note: my alternative implementation of region_loss is in netharn/dev/ somewhere.
I probably should test if that is faster / more correct and them perhaps use
that instead.

"""
import math
import torch
import torch.nn as nn


class BaseLossWithCudaState(torch.nn.modules.loss._Loss):
    """
    Helper to keep track of if a loss module is in cpu or gpu mod
    """

    def __init__(self):
        super(BaseLossWithCudaState, self).__init__()
        self._iscuda = False
        self._device_num = None

    def cuda(self, device_num=None, **kwargs):
        self._iscuda = True
        self._device_num = device_num
        return super(BaseLossWithCudaState, self).cuda(device_num, **kwargs)

    def cpu(self):
        self._iscuda = False
        self._device_num = None
        return super(BaseLossWithCudaState, self).cpu()

    @property
    def is_cuda(self):
        return self._iscuda

    def get_device(self):
        if self._device_num is None:
            return torch.device('cpu')
        return self._device_num


class RegionLoss(BaseLossWithCudaState):
    """ Computes region loss from darknet network output and target annotation.

    Args:
        num_classes (int): number of categories
        anchors (list): 2D list representing anchor boxes (see :class:`lightnet.network.Darknet`)
        coord_scale (float): weight of bounding box coordinates
        noobject_scale (float): weight of regions without target boxes
        object_scale (float): weight of regions with target boxes
        class_scale (float): weight of categorical predictions
        thresh (float): minimum iou between a predicted box and ground truth for them to be considered matching
        seen (torch.Tensor): How many images the network has already been trained on.
    """

    def __init__(self, num_classes, anchors, reduction=32, seen=0, coord_scale=1.0, noobject_scale=1.0, object_scale=5.0, class_scale=1.0, thresh=0.6):
        super(RegionLoss, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.anchor_step = len(anchors[0])
        self.anchors = torch.Tensor(anchors)
        self.reduction = reduction
        self.register_buffer('seen', torch.tensor(seen))

        self.coord_scale = coord_scale
        self.noobject_scale = noobject_scale
        self.object_scale = object_scale
        self.class_scale = class_scale
        self.thresh = thresh

        self.mse = nn.MSELoss(reduction='sum')
        self.clf = nn.CrossEntropyLoss(reduction='sum')

    # def extra_repr(self):
    #     repr_str = 'classes={self.num_classes}, reduction={self.reduction}, threshold={self.thresh}, seen={self.seen.item()}\n'
    #     repr_str += 'coord_scale={self.coord_scale}, object_scale={self.object_scale}, noobject_scale={self.noobject_scale}, class_scale={self.class_scale}\n'
    #     repr_str += 'anchors='
    #     for a in self.anchors:
    #         repr_str += '[{a[0]:.5g}, {a[1]:.5g}] '
    #     return repr_str

    def forward(self, output, target, seen=None):
        """ Compute Region loss.

        Args:
            output (torch.autograd.Variable): Output from the network
            target (brambox.boxes.annotations.Annotation or torch.Tensor): Brambox annotations or tensor containing the annotation targets (see :class:`lightnet.data.BramboxToTensor`)
            seen (int, optional): How many images the network has already been trained on; Default **Add batch_size to previous seen value**

        Note:
            The example below only shows this function working with a target tensor. |br|
            This loss function also works with a list of brambox annotations as target and will work the same.
            The added benefit of using brambox annotations is that this function will then also look at the ``ignore`` flag of the annotations
            and ignore detections that match with it. This allows you to have annotations that will not influence the loss in any way,
            as opposed to having them removed and counting them as false detections.

        Example:
            >>> # DISABLE_DOCTEST
            >>> from .models.yolo2.region_loss2 import *
            >>> from .models.yolo2.light_yolo import Yolo
            >>> torch.random.manual_seed(0)
            >>> network = Yolo(num_classes=2, conf_thresh=4e-2)
            >>> #_ = torch.random.manual_seed(0)
            >>> #network = ln.models.Yolo(num_classes=2, conf_thresh=4e-2)
            >>> #region_loss = ln.network.loss.RegionLoss(network.num_classes, network.anchors)
            >>> self = RegionLoss(network.num_classes, network.anchors)
            >>> Win, Hin = 96, 96
            >>> Wout, Hout = 1, 1
            >>> # true boxes for each item in the batch
            >>> # each box encodes class, x_center, y_center, width, and height
            >>> # coordinates are normalized in the range 0 to 1
            >>> # items in each batch are padded with dummy boxes with class_id=-1
            >>> target = torch.FloatTensor([
            ...     # boxes for batch item 1
            ...     [[0, 0.50, 0.50, 1.00, 1.00],
            ...      [1, 0.32, 0.42, 0.22, 0.12]],
            ...     # boxes for batch item 2 (it has no objects, note the pad!)
            ...     [[-1, 0, 0, 0, 0],
            ...      [-1, 0, 0, 0, 0]],
            ... ])
            >>> im_data = torch.randn(len(target), 3, Hin, Win, requires_grad=True)
            >>> output = network.forward(im_data)
            >>> loss = float(self(output, target))
            >>> print('loss = {loss:.2f}'.format(loss=loss))
            loss = 20.43
        """
        if isinstance(target, dict):
            target = target['target']

        # Get x,y,w,h,conf,cls
        if False:
            nB = output.data.size(0)
            nA = self.num_anchors
            nC = self.num_classes
            nH = output.data.size(2)
            nW = output.data.size(3)
            nPixels = nH * nW  # number of ouput grid cells

            device = output.device
            if seen is not None:
                self.seen = torch.tensor(seen)
            elif self.training:
                self.seen += nB

            output = output.view(nB, nA, -1, nPixels)
            coord = torch.zeros_like(output[:, :, :4])
            coord[:, :, :2] = output[:, :, :2].sigmoid()    # tx,ty
            coord[:, :, 2:4] = output[:, :, 2:4]            # tw,th
            conf = output[:, :, 4].sigmoid()
            if nC > 1:
                cls = output[:, :, 5:].contiguous().view(
                    nB * nA, nC, nPixels).transpose(1, 2).contiguous().view(-1, nC)

            # Create prediction boxes
            pred_boxes = torch.FloatTensor(nB * nA * nPixels, 4)
            lin_x = torch.linspace(0, nW - 1, nW).repeat(nH, 1).view(nPixels).to(device)
            lin_y = torch.linspace(0, nH - 1, nH).view(nH, 1).repeat(1, nW).view(nPixels).to(device)
            anchor_w = self.anchors[:, 0].contiguous().view(nA, 1).to(device)
            anchor_h = self.anchors[:, 1].contiguous().view(nA, 1).to(device)

            pred_boxes[:, 0] = (coord[:, :, 0].detach() + lin_x).view(-1)
            pred_boxes[:, 1] = (coord[:, :, 1].detach() + lin_y).view(-1)
            pred_boxes[:, 2] = (coord[:, :, 2].detach().exp() * anchor_w).view(-1)
            pred_boxes[:, 3] = (coord[:, :, 3].detach().exp() * anchor_h).view(-1)
            pred_boxes = pred_boxes.cpu()

        else:
            nB, nA, nC5, nH, nW = output.data.shape
            nC = self.num_classes
            assert nA == self.num_anchors
            assert nC5 == self.num_classes + 5
            output = output.view(nB, nA, 5 + nC, nH, nW)

            coord = torch.zeros_like(output[:, :, 0:4, :, :])
            coord[:, :, 0:2, :, :] = output[:, :, 0:2, :, :].sigmoid()  # tx,ty
            coord[:, :, 2:4, :, :] = output[:, :, 2:4, :, :]            # tw,th

            nPixels = nH * nW  # number of ouput grid cells

            device = output.device
            if seen is not None:
                self.seen = torch.tensor(seen)
            elif self.training:
                self.seen += nB

            conf = output[:, :, 4:5, :, :].sigmoid()
            if nC > 1:
                # Swaps the dimensions from [B, A, C, H, W] to be [B, A, H, W, C]
                cls = output[:, :, 5:, :, :].contiguous().view(
                    nB * nA, nC, nH * nW).transpose(1, 2).contiguous().view(
                        nB, nA, nH, nW, nC)

            pred_boxes = torch.empty(nB * nA * nH * nW, 4,
                                     dtype=torch.float32, device=device)
            lin_x = torch.linspace(0, nW - 1, nW).repeat(nH, 1).view(nPixels).to(device)
            lin_y = torch.linspace(0, nH - 1, nH).view(nH, 1).repeat(1, nW).view(nPixels).to(device)

            lin_x = torch.linspace(0, nW - 1, nW).repeat(nH, 1).to(device)
            lin_y = torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().contiguous().to(device)

            anchor_w = self.anchors[:, 0].contiguous().view(nA, 1).view(1, nA, 1, 1, 1).to(device)
            anchor_h = self.anchors[:, 1].contiguous().view(nA, 1).view(1, nA, 1, 1, 1).to(device)

            pred_boxes[:, 0] = (coord[:, :, 0:1, :, :].detach() + lin_x).view(-1)
            pred_boxes[:, 1] = (coord[:, :, 1:2, :, :].detach() + lin_y).view(-1)
            pred_boxes[:, 2] = (coord[:, :, 2:3, :, :].detach().exp() * anchor_w).view(-1)
            pred_boxes[:, 3] = (coord[:, :, 3:4, :, :].detach().exp() * anchor_h).view(-1)

        # Get target values
        coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls = self.build_targets(
            pred_boxes, target, nH, nW)
        coord_mask = coord_mask.expand_as(tcoord).to(device).sqrt()
        conf_mask = conf_mask.to(device).sqrt()
        tcoord = tcoord.to(device)
        tconf = tconf.to(device)

        if nC > 1:
            if False:
                tcls = tcls[cls_mask].view(-1).long().to(device)
                cls_mask = cls_mask.view(-1, 1).repeat(1, nC).to(device)
                cls = cls[cls_mask].view(-1, nC)
                cls_ = cls
            else:
                tcls = tcls[cls_mask].view(-1).long().to(device)
                cls_mask = cls_mask.view(-1, 1).repeat(1, nC).to(device)
                cls_ = cls.view(-1, nC)[cls_mask.view(-1, nC)].view(-1, nC)

            cls_ = cls_.view(-1, nC)
        coord_ = coord.view(coord_mask.shape)
        conf_ = conf.view(conf_mask.shape)

        # Compute losses
        self.loss_coord = self.coord_scale * (
            self.mse(coord_ * coord_mask, tcoord * coord_mask) / nB)
        self.loss_conf = self.mse(conf_ * conf_mask, tconf * conf_mask) / nB
        if nC > 1:
            self.loss_cls = self.class_scale * 2 * (self.clf(cls_, tcls) / nB)
            self.loss_tot = self.loss_coord + self.loss_conf + self.loss_cls
        else:
            self.loss_cls = None
            self.loss_tot = self.loss_coord + self.loss_conf

        return self.loss_tot

    def build_targets(self, pred_boxes, ground_truth, nH, nW):
        """ Compare prediction boxes and ground truths, convert ground truths to network output tensors """
        # Parameters
        nB = ground_truth.size(0)
        nA = self.num_anchors
        nAnchors = nA * nH * nW
        nPixels = nH * nW

        # Tensors
        conf_mask = torch.ones(
            nB, nA, nPixels, requires_grad=False) * self.noobject_scale
        coord_mask = torch.zeros(nB, nA, 1, nPixels, requires_grad=False)
        cls_mask = torch.zeros(nB, nA, nPixels, requires_grad=False).byte()
        tcoord = torch.zeros(nB, nA, 4, nPixels, requires_grad=False)
        tconf = torch.zeros(nB, nA, nPixels, requires_grad=False)
        tcls = torch.zeros(nB, nA, nPixels, requires_grad=False)

        if self.seen < 12800:
            coord_mask.fill_(1)
            # coord_mask.fill_(.01 / self.coord_scale)

            if self.anchor_step == 4:
                tcoord[:, :, 0] = self.anchors[:, 2].contiguous().view(
                    1, nA, 1, 1).repeat(nB, 1, 1, nPixels)
                tcoord[:, :, 1] = self.anchors[:, 3].contiguous().view(
                    1, nA, 1, 1).repeat(nB, 1, 1, nPixels)
            else:
                tcoord[:, :, 0].fill_(0.5)
                tcoord[:, :, 1].fill_(0.5)

        self.anchors = self.anchors.to(pred_boxes.device)
        if self.anchor_step == 4:
            anchors = self.anchors.clone()
            anchors[:, :2] = 0
        else:
            anchors = torch.cat(
                [torch.zeros_like(self.anchors), self.anchors], 1)

        for b in range(nB):
            gt = ground_truth[b][(ground_truth[b, :, 0] >= 0)[:, None].expand_as(ground_truth[b])].view(-1, 5)

            if gt.numel() > 0:     # No gt for this image
                # Build up tensors
                cur_pred_boxes = pred_boxes[b * nAnchors: (b + 1) * nAnchors]

                gt = gt[:, 1:]
                gt[:, ::2] *= nW
                gt[:, 1::2] *= nH

                # Set confidence mask of matching detections to 0
                iou_gt_pred = bbox_ious(gt, cur_pred_boxes)
                mask = (iou_gt_pred > self.thresh).sum(0) >= 1
                conf_mask[b][mask.view_as(conf_mask[b])] = 0

                # Find best anchor for each gt
                gt_wh = gt.clone()
                gt_wh[:, :2] = 0
                iou_gt_anchors = bbox_ious(gt_wh, anchors)
                _, best_anchors = iou_gt_anchors.max(1)

                # Set masks and target values for each gt
                gt_size = gt.size(0)
                for i in range(gt_size):
                    gi = min(nW - 1, max(0, int(gt[i, 0])))
                    gj = min(nH - 1, max(0, int(gt[i, 1])))
                    best_n = best_anchors[i]
                    iou = iou_gt_pred[i][best_n * nPixels + gj * nW + gi]

                    coord_mask[b][best_n][0][gj * nW + gi] = 2 - (gt[i, 2]  *  gt[i, 3]) / nPixels
                    cls_mask[b][best_n][gj * nW + gi] = 1
                    conf_mask[b][best_n][gj * nW + gi] = self.object_scale
                    tcoord[b][best_n][0][gj * nW + gi] = gt[i, 0] - gi
                    tcoord[b][best_n][1][gj * nW + gi] = gt[i, 1] - gj
                    tcoord[b][best_n][2][gj * nW + gi] = math.log(gt[i, 2] / self.anchors[best_n, 0])
                    tcoord[b][best_n][3][gj * nW  + gi] = math.log(gt[i, 3] / self.anchors[best_n, 1])
                    tconf[b][best_n][gj * nW + gi] = iou
                    tcls[b][best_n][gj * nW + gi] = ground_truth[b, i, 0]

        return coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls


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
