"""
Reproduces RegionLoss from Darknet:
    https://github.com/pjreddie/darknet/blob/master/src/region_layer.c

Based off RegionLoss from Lightnet:
    https://gitlab.com/EAVISE/lightnet/blob/master/lightnet/network/loss/_regionloss.py

Speedups
    [ ] - Preinitialize anchor tensors
"""

import torch
import torch.nn as nn
import numpy as np  # NOQA
try:  # nocover
    from packaging.version import parse as LooseVersion
except ImportError:
    from distutils.version import LooseVersion


_TORCH_HAS_BOOL_COMP = LooseVersion(torch.__version__) >= LooseVersion('1.2.0')

__all__ = ['RegionLoss']


class BaseLossWithCudaState(torch.nn.modules.loss._Loss):
    """
    Helper to keep track of if a loss module is in cpu or gpu mod
    """
    def __init__(self):
        super(BaseLossWithCudaState, self).__init__()
        self._iscuda = False
        self._device_num = None
        self._device = None

    def cuda(self, device_num=None, **kwargs):
        self._iscuda = True
        self._device_num = device_num
        return super(BaseLossWithCudaState, self).cuda(device_num, **kwargs)

    def cpu(self):
        self._iscuda = False
        self._device_num = None
        return super(BaseLossWithCudaState, self).cpu()

    def to(self, device):
        if isinstance(device, int):
            device = torch.device('cuda', device)
        elif device == 'cuda':
            device = torch.device('cuda')
        elif device == 'cpu':
            device = torch.device('cpu')
        elif device is None:
            device = torch.device('cpu')
        self._iscuda = device.type != 'cpu'
        self._device_num = device.index
        self._device = device
        return super(BaseLossWithCudaState, self).to(device)

    @property
    def device(self):
        return self._device

    @property
    def is_cuda(self):
        return self._iscuda

    def get_device(self):
        if self._device is not None:
            return self._device

        if self._device_num is None:
            return torch.device('cpu')
        return self._device_num


class RegionLoss(BaseLossWithCudaState):
    """ Computes region loss from darknet network output and target annotation.

    Args:
        num_classes (int): number of categories
        anchors (list): 2D list representing anchor boxes (see :class:`lightnet.network.Darknet`)
            These width and height values should be in network output coordinates.
        coord_scale (float): weight of bounding box coordinates
        noobject_scale (float): weight of regions without target boxes
        object_scale (float): weight of regions with target boxes
        class_scale (float): weight of categorical predictions
        thresh (float): minimum iou for a predicted box to be assigned to a target

    CommandLine:
        python ~/code/netharn/netharn/models/yolo2/light_region_loss.py RegionLoss:0

    Example:
        >>> # DISABLE_DOCTEST
        >>> # xdoctest: +REQUIRES(module:kwimage)
        >>> from .models.yolo2.light_yolo import Yolo
        >>> torch.random.manual_seed(0)
        >>> network = Yolo(num_classes=2, conf_thresh=4e-2)
        >>> self = RegionLoss(num_classes=network.num_classes, anchors=network.anchors)
        >>> Win, Hin = 96, 96
        >>> Wout, Hout = 1, 1
        >>> # true boxes for each item in the batch
        >>> # each box encodes class, center, width, and height
        >>> # coordinates are normalized in the range 0 to 1
        >>> # items in each batch are padded with dummy boxes with class_id=-1
        >>> target = torch.FloatTensor([
        >>>     # boxes for batch item 1
        >>>     [[0, 0.50, 0.50, 1.00, 1.00],
        >>>      [1, 0.32, 0.42, 0.22, 0.12]],
        >>>     # boxes for batch item 2 (it has no objects, note the pad!)
        >>>     [[-1, 0, 0, 0, 0],
        >>>      [-1, 0, 0, 0, 0]],
        >>> ])
        >>> im_data = torch.randn(len(target), 3, Hin, Win)
        >>> output = network.forward(im_data)
        >>> loss = float(self.forward(output, target, seen=0))
        >>> #print('self.loss_cls = {!r}'.format(self.loss_cls))
        >>> #print('self.loss_coord = {!r}'.format(self.loss_coord))
        >>> #print('self.loss_conf = {!r}'.format(self.loss_conf))
        >>> print('loss = {:.2f}'.format(loss))
        >>> print('output.sum() = {:.2f}'.format(output.sum()))
        loss = 8.79
        output.sum() = 2.15

        loss = 20.18
        output.sum() = 2.15

    Example:
        >>> # DISABLE_DOCTEST
        >>> from .models.yolo2.light_yolo import Yolo
        >>> torch.random.manual_seed(0)
        >>> network = Yolo(num_classes=2, conf_thresh=4e-2)
        >>> self = RegionLoss(num_classes=network.num_classes, anchors=network.anchors)
        >>> Win, Hin = 96, 96
        >>> Wout, Hout = 1, 1
        >>> target = torch.FloatTensor([])
        >>> im_data = torch.randn(2, 3, Hin, Win)
        >>> output = network.forward(im_data)
        >>> loss = float(self.forward(output, target))
        >>> print('loss = {:.2f}'.format(loss))
        >>> print('output.sum() = {:.2f}'.format(output.sum()))
        loss = 5.86
        output.sum() = 2.15

        loss = 5.96
        output.sum() = 2.15

        loss = 16.47
    """

    def __init__(self, num_classes, anchors, coord_scale=1.0,
                 noobject_scale=1.0, object_scale=5.0, class_scale=1.0,
                 thresh=0.6, seen_thresh=12800,
                 small_boxes=False,
                 mse_factor=0.5):
        import kwimage
        super(RegionLoss, self).__init__()

        self.num_classes = num_classes

        self.seen_thresh = seen_thresh

        self.anchors = torch.Tensor(anchors)
        self.num_anchors = len(anchors)

        self.coord_scale = coord_scale
        self.noobject_scale = noobject_scale
        self.object_scale = object_scale
        self.class_scale = class_scale
        self.thresh = thresh

        self.loss_coord = None
        self.loss_conf = None
        self.loss_cls = None
        self.loss_tot = None

        self.coord_mse = nn.MSELoss(reduction='sum')
        self.conf_mse = nn.MSELoss(reduction='sum')
        self.cls_critrion = nn.CrossEntropyLoss(reduction='sum')

        # Precompute relative anchors in tlbr format for iou computation
        rel_anchors_cxywh = torch.cat([torch.zeros_like(self.anchors), self.anchors], 1)
        self.rel_anchors_boxes = kwimage.Boxes(rel_anchors_cxywh, 'cxywh')

        self.small_boxes = small_boxes
        self.mse_factor = mse_factor

    def forward(self, output, target, seen=0, gt_weights=None):
        """ Compute Region loss.

        Args:
            output (torch.Tensor): Output from the network
                should have shape [B, A, 5 + C, H, W]

            target (torch.Tensor): the shape should be [B, T, 5], where B is
                the batch size, T is the maximum number of boxes in an item,
                and the final dimension should correspond to [class_idx,
                center_x, center_y, width, height]. Items with fewer than T
                boxes should be padded with dummy boxes with class_idx=-1.

            seen (int): number of training batches the networks has "seen"

        Example:
            >>> # DISABLE_DOCTEST
            >>> from viame.arrows.pytorch.netharn import core as nh
            >>> nC = 2
            >>> self = RegionLoss(num_classes=nC, anchors=np.array([[1, 1]]))
            >>> nA = len(self.anchors)
            >>> # one batch, with one anchor, with 2 classes and 3x3 grid cells
            >>> import kwarray
            >>> rng = kwarray.ensure_rng(0)
            >>> output = torch.Tensor(rng.rand(1, nA, 5 + nC, 3, 3))
            >>> # one batch, with one true box
            >>> target = torch.Tensor(rng.rand(1, 1, 5))
            >>> target[..., 0] = 0
            >>> seen = 0
            >>> gt_weights = None
            >>> self.forward(output, target, seen).item()
            7.491...


            2.374...

            4.528...
        """
        if isinstance(target, dict):
            gt_weights = target.get('gt_weights', gt_weights)
            target = target['target']

        # Parameters
        nB, nA, nC5, nH, nW = output.data.shape
        nC = self.num_classes
        assert nA == self.num_anchors
        assert nC5 == self.num_classes + 5

        device = self.get_device()
        if self.rel_anchors_boxes.device != device:
            self.rel_anchors_boxes.data = self.rel_anchors_boxes.data.to(device)
            self.anchors = self.anchors.to(device)

        # Get x,y,w,h,conf,*cls_probs from the third dimension
        # output_ = output.view(nB, nA, 5 + nC, nH, nW)

        coord = torch.zeros_like(output[:, :, 0:4, :, :], device=device)
        coord[:, :, 0:2, :, :] = output[:, :, 0:2, :, :].sigmoid()  # tx,ty
        coord[:, :, 2:4, :, :] = output[:, :, 2:4, :, :]            # tw,th

        conf = output[:, :, 4:5, :, :].sigmoid()
        if nC > 1:
            # Swaps the dimensions from [B, A, C, H, W] to be [B, A, H, W, C]
            cls_probs = output[:, :, 5:, :, :].contiguous().view(
                nB * nA, nC, nH * nW).transpose(1, 2).contiguous().view(
                    nB, nA, nH, nW, nC)

        with torch.no_grad():
            # Create prediction boxes
            pred_cxywh = torch.empty(nB * nA * nH * nW, 4,
                                     dtype=torch.float32, device=device)

            # Grid cell center offsets
            lin_x = torch.linspace(0, nW - 1, nW, device=device).repeat(nH, 1)
            lin_y = torch.linspace(0, nH - 1, nH, device=device).repeat(nW, 1).t().contiguous()
            anchor_w = self.anchors[:, 0].contiguous().view(nA, 1).view(1, nA, 1, 1, 1)
            anchor_h = self.anchors[:, 1].contiguous().view(nA, 1).view(1, nA, 1, 1, 1)

            # Convert raw network output to bounding boxes in network output coordinates
            pred_cxywh[:, 0] = (coord[:, :, 0:1, :, :].data + lin_x).view(-1)
            pred_cxywh[:, 1] = (coord[:, :, 1:2, :, :].data + lin_y).view(-1)
            pred_cxywh[:, 2] = (coord[:, :, 2:3, :, :].data.exp() * anchor_w).view(-1)
            pred_cxywh[:, 3] = (coord[:, :, 3:4, :, :].data.exp() * anchor_h).view(-1)

            # Get target values
            _tup = self.build_targets(
                pred_cxywh, target, nH, nW, seen=seen, gt_weights=gt_weights)
            coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls = _tup

            if nC > 1:
                if _TORCH_HAS_BOOL_COMP:
                    masked_tcls = tcls[cls_mask.bool()].view(-1).long()
                else:
                    masked_tcls = tcls[cls_mask].view(-1).long()

        if nC > 1:
            # Swaps the dimensions to be [B, A, H, W, C]
            # (Allowed because 3rd dimension is guarneteed to be 1 here)
            cls_probs_mask = cls_mask.reshape(nB, nA, nH, nW, 1).repeat(1, 1, 1, 1, nC)
            cls_probs_mask.requires_grad = False
            if _TORCH_HAS_BOOL_COMP:
                masked_cls_probs = cls_probs[cls_probs_mask.bool()].view(-1, nC)
            else:
                masked_cls_probs = cls_probs[cls_probs_mask].view(-1, nC)

        # Compute losses

        # Bounding Box Loss
        # To be compatible with the original YOLO code we add a seemingly
        # random multiply by .5 in our MSE computation so the torch autodiff
        # algorithm produces the same result as darknet. (but maybe its not a
        # good idea?)
        loss_coord = self.mse_factor * self.coord_scale * self.coord_mse(coord_mask * coord, coord_mask * tcoord) / nB

        # Objectness Loss
        # object_scale and noobject_scale are incorporated in conf_mask.
        loss_conf = self.mse_factor * self.conf_mse(conf_mask * conf, conf_mask * tconf) / nB

        # Class Loss
        if nC > 1 and masked_cls_probs.numel():
            loss_cls = self.class_scale * self.cls_critrion(masked_cls_probs, masked_tcls) / nB
            self.loss_cls = float(loss_cls.data.cpu().item())
        else:
            self.loss_cls = loss_cls = 0

        loss_tot = loss_coord + loss_conf + loss_cls

        # Record loss components as module members
        self.loss_tot = float(loss_tot.data.cpu().item())
        self.loss_coord = float(loss_coord.data.cpu().item())
        self.loss_conf = float(loss_conf.data.cpu().item())

        return loss_tot

    def build_targets(self, pred_cxywh, target, nH, nW, seen=0, gt_weights=None):
        """
        Compare prediction boxes and targets, convert targets to network output tensors

        Args:
            pred_cxywh (Tensor):   shape [B * A * W * H, 4] in normalized cxywh format
            target (Tensor): shape [B, max(gtannots), 4]

        CommandLine:
            python ~/code/netharn/netharn/models/yolo2/light_region_loss.py RegionLoss.build_targets:1

        Example:
            >>> # xdoctest: +REQUIRES(module:kwimage)
            >>> from .models.yolo2.light_yolo import Yolo
            >>> torch.random.manual_seed(0)
            >>> network = Yolo(num_classes=2, conf_thresh=4e-2)
            >>> self = RegionLoss(num_classes=network.num_classes, anchors=network.anchors)
            >>> Win, Hin = 96, 96
            >>> nW, nH = 3, 3
            >>> target = torch.FloatTensor([])
            >>> gt_weights = torch.FloatTensor([[-1, -1, -1], [1, 1, 0]])
            >>> #pred_cxywh = torch.rand(90, 4)
            >>> nB = len(gt_weights)
            >>> pred_cxywh = torch.rand(nB, len(self.anchors), nH, nW, 4).view(-1, 4)
            >>> seen = 0
            >>> self.build_targets(pred_cxywh, target, nH, nW, seen, gt_weights)

        Example:
            >>> # xdoctest: +REQUIRES(module:kwimage)
            >>> torch.random.manual_seed(0)
            >>> anchors = np.array([[.75, .75], [1.0, .3], [.3, 1.0]])
            >>> self = RegionLoss(num_classes=2, anchors=anchors)
            >>> nW, nH = 2, 2
            >>> # true boxes for each item in the batch
            >>> # each box encodes class, center, width, and height
            >>> # coordinates are normalized in the range 0 to 1
            >>> # items in each batch are padded with dummy boxes with class_id=-1
            >>> target = torch.FloatTensor([
            >>>     # boxes for batch item 0 (it has no objects, note the pad!)
            >>>     [[-1, 0, 0, 0, 0],
            >>>      [-1, 0, 0, 0, 0],
            >>>      [-1, 0, 0, 0, 0]],
            >>>     # boxes for batch item 1
            >>>     [[0, 0.50, 0.50, 1.00, 1.00],
            >>>      [1, 0.34, 0.32, 0.12, 0.32],
            >>>      [1, 0.32, 0.42, 0.22, 0.12]],
            >>> ])
            >>> gt_weights = torch.FloatTensor([[-1, -1, -1], [1, 1, 0]])
            >>> nB = len(gt_weights)
            >>> pred_cxywh = torch.rand(nB, len(anchors), nH, nW, 4).view(-1, 4)
            >>> seen = 0
            >>> coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls = self.build_targets(pred_cxywh, target, nH, nW, seen, gt_weights)
        """
        import kwimage
        from .util import torch_ravel_multi_index
        gtempty = (target.numel() == 0)

        # Parameters
        nB = target.shape[0] if not gtempty else 0
        # nT = target.shape[1] if not gtempty else 0
        nA = self.num_anchors

        nPixels = nW * nH

        if nB == 0:
            # torch does not preserve shapes when any dimension goes to 0
            # fix nB if there is no groundtruth
            nB = int(len(pred_cxywh) / (nA * nH * nW))
        else:
            assert nB == int(len(pred_cxywh) / (nA * nH * nW)), 'bad assumption'

        seen = seen + nB

        # Tensors
        device = target.device

        # Put the groundtruth in a format comparable to output
        tcoord = torch.zeros(nB, nA, 4, nH, nW, device=device)
        tconf = torch.zeros(nB, nA, 1, nH, nW, device=device)
        tcls = torch.zeros(nB, nA, 1, nH, nW, device=device)

        # Create weights to determine which outputs are punished
        # By default we punish all outputs for not having correct iou
        # objectness prediction. The other masks default to zero meaning that
        # by default we will not punish a prediction for having a different
        # coordinate or class label (later the groundtruths will override these
        # defaults for select grid cells and anchors)
        coord_mask = torch.zeros(nB, nA, 1, nH, nW, device=device)
        conf_mask = torch.ones(nB, nA, 1, nH, nW, device=device)
        # TODO: this could be a weight instead
        cls_mask = torch.zeros(nB, nA, 1, nH, nW, device=device, dtype=torch.uint8)

        # Default conf_mask to the noobject_scale
        conf_mask.fill_(self.noobject_scale)

        # encourage the network to predict boxes centered on the grid cells by
        # setting the default target xs and ys to be (.5, .5) (i.e. the
        # relative center of a grid cell) fill the mask with ones so all
        # outputs are punished for not predicting center anchor locations ---
        # unless tcoord is overriden by a real groundtruth target later on.
        if seen < self.seen_thresh:
            # PJreddies version
            # https://github.com/pjreddie/darknet/blob/master/src/region_layer.c#L254

            # By default encourage the network to predict no shift
            tcoord[:, :, 0:2, :, :].fill_(0.5)
            # By default encourage the network to predict no scale (in logspace)
            tcoord[:, :, 2:4, :, :].fill_(0.0)

            if False:
                # In the warmup phase we care about changing the coords to be
                # exactly the anchors if they don't predict anything, but the
                # weight is only 0.01, set it to 0.01 / self.coord_scale.
                # Note we will apply the required sqrt later
                coord_mask.fill_((0.01 / self.coord_scale))
                # This hurts even thought it seems like its what darknet does
            else:
                coord_mask.fill_(1)

        if gtempty:
            coord_mask = coord_mask.sqrt()
            conf_mask = conf_mask.sqrt()
            coord_mask = coord_mask.expand_as(tcoord)
            return coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls

        # Put this back into a non-flat view
        pred_cxywh = pred_cxywh.view(nB, nA, nH, nW, 4)
        pred_boxes = kwimage.Boxes(pred_cxywh, 'cxywh')

        gt_class = target[..., 0].data
        gt_boxes_norm = kwimage.Boxes(target[..., 1:5], 'cxywh')

        # Put GT boxes into output coordinates
        gt_boxes = gt_boxes_norm.scale([nW, nH])
        # Construct "relative" versions of the true boxes, centered at 0
        # This will allow them to be compared to the anchor boxes.
        rel_gt_boxes = gt_boxes.copy()
        rel_gt_boxes.data[..., 0:2] = 0

        # true boxes with a class of -1 are fillers, ignore them
        gt_isvalid = (gt_class >= 0)
        batch_nT = gt_isvalid.sum(dim=1).cpu().numpy()

        # Compute the grid cell for each groundtruth box
        true_xs = gt_boxes.data[..., 0]
        true_ys = gt_boxes.data[..., 1]
        true_is = true_xs.long().clamp_(0, nW - 1)
        true_js = true_ys.long().clamp_(0, nH - 1)

        if gt_weights is None:
            # If unspecified give each groundtruth a default weight of 1
            gt_weights = torch.ones_like(target[..., 0], device=device)

        # Undocumented darknet detail: multiply coord weight by two minus the
        # area of the true box in normalized coordinates.  the square root is
        # because the weight.
        if self.small_boxes:
            gt_coord_weights = (gt_weights * (2.0 - gt_boxes_norm.area[..., 0]))
        else:
            gt_coord_weights = gt_weights
        # Pre multiply weights with object scales
        gt_conf_weights = gt_weights * self.object_scale
        # Pre threshold classification weights
        gt_cls_weights = (gt_weights > .5).byte()

        # Loop over ground_truths and construct tensors
        for bx in range(nB):
            # Get the actual groundtruth boxes for this batch item
            nT = batch_nT[bx]
            if nT == 0:
                continue

            # Batch ground truth
            cur_rel_gt_boxes = rel_gt_boxes[bx, 0:nT]
            cur_gt_boxes = gt_boxes[bx, 0:nT]
            cur_gt_cls = target[bx, 0:nT, 0]
            # scalars, one for each true object
            cur_true_is = true_is[bx, 0:nT]
            cur_true_js = true_js[bx, 0:nT]
            cur_true_coord_weights = gt_coord_weights[bx, 0:nT]
            cur_true_conf_weights = gt_conf_weights[bx, 0:nT]
            cur_true_cls_weights = gt_cls_weights[bx, 0:nT]

            cur_gx, cur_gy, cur_gw, cur_gh = cur_gt_boxes.data.t()

            # Batch predictions
            cur_pred_boxes = pred_boxes[bx]

            # NOTE: IOU computation is the bottleneck in this function

            # Assign groundtruth boxes to anchor boxes
            cur_anchor_gt_ious = self.rel_anchors_boxes.ious(
                cur_rel_gt_boxes, bias=0)
            _, cur_true_anchor_axs = cur_anchor_gt_ious.max(dim=0)  # best_ns in YOLO

            # Get the anchor (w,h) assigned to each true object
            cur_true_anchor_w, cur_true_anchor_h = self.anchors[cur_true_anchor_axs].t()

            # Find the IOU of each predicted box with the groundtruth
            cur_pred_true_ious = cur_pred_boxes.ious(cur_gt_boxes, bias=0)
            # Assign groundtruth boxes to predicted boxes
            cur_ious, _ = cur_pred_true_ious.max(dim=-1)

            # Set loss to zero for any predicted boxes that had a high iou with
            # a groundtruth target (we wont punish them for not being
            # background), One of these will be selected as the best and be
            # punished for not predicting the groundtruth value.
            conf_mask[bx].view(-1)[cur_ious.view(-1) > self.thresh] = 0

            ####
            # Broadcast the loop over true boxes
            ####
            # Convert the true box coordinates to be comparable with pred output
            # * translate each gtbox to be relative to its assignd gridcell
            # * make w/h relative to anchor box w / h and convert to logspace
            cur_tcoord_x = cur_gx - cur_true_is.float()
            cur_tcoord_y = cur_gy - cur_true_js.float()
            cur_tcoord_w = (cur_gw / cur_true_anchor_w).log()
            cur_tcoord_h = (cur_gh / cur_true_anchor_h).log()

            if 0:
                cur_true_anchor_axs_ = cur_true_anchor_axs.cpu().numpy()
                cur_true_js_ = cur_true_js.cpu().numpy()
                cur_true_is_ = cur_true_is.cpu().numpy()

                iou_raveled_idxs = np.ravel_multi_index([
                    cur_true_anchor_axs_, cur_true_js_, cur_true_is_, np.arange(nT)
                ], cur_pred_true_ious.shape)
                # Get the ious with the assigned boxes for each truth
                cur_true_ious = cur_pred_true_ious.view(-1)[iou_raveled_idxs]

                raveled_idxs = np.ravel_multi_index([
                    [bx], cur_true_anchor_axs_, [0], cur_true_js_, cur_true_is_
                ], coord_mask.shape)

                # --------------------------------------------
                multi_index = ([bx], cur_true_anchor_axs_, [0], cur_true_js_, cur_true_is_)
                # multi_index_ = multi_index
                raveled_idxs_b0 = np.ravel_multi_index(multi_index, tcoord.shape)
                # A bit faster than ravel_multi_indexes with [1], [2], and [3]
                raveled_idxs_b1 = raveled_idxs_b0 + nPixels
                raveled_idxs_b2 = raveled_idxs_b0 + nPixels * 2
                raveled_idxs_b3 = raveled_idxs_b0 + nPixels * 3
            else:
                iou_raveled_idxs = torch_ravel_multi_index([
                    cur_true_anchor_axs, cur_true_js, cur_true_is,
                    torch.arange(nT, device=device, dtype=torch.long)
                ], cur_pred_true_ious.shape, device)
                # Get the ious with the assigned boxes for each truth
                cur_true_ious = cur_pred_true_ious.view(-1)[iou_raveled_idxs]

                Bxs = torch.full_like(cur_true_anchor_axs, bx)
                Zxs = torch.full_like(cur_true_anchor_axs, 0)

                multi_index = [Bxs, cur_true_anchor_axs, Zxs, cur_true_js, cur_true_is]
                multi_index = torch.cat([x.view(-1, 1) for x in multi_index], dim=1)
                raveled_idxs = torch_ravel_multi_index(multi_index, coord_mask.shape, device)

                # --------------------------------------------
                # We reuse the previous multi-index because the dims are
                # broadcastable at [:, :, [0], :, :]
                raveled_idxs_b0 = torch_ravel_multi_index(multi_index, tcoord.shape, device)
                # A bit faster than ravel_multi_indexes with [1], [2], and [3]
                raveled_idxs_b1 = raveled_idxs_b0 + nPixels
                raveled_idxs_b2 = raveled_idxs_b0 + nPixels * 2
                raveled_idxs_b3 = raveled_idxs_b0 + nPixels * 3
                # --------------------------------------------

            coord_mask.view(-1)[raveled_idxs] = cur_true_coord_weights
            cls_mask.view(-1)[raveled_idxs]   = cur_true_cls_weights
            conf_mask.view(-1)[raveled_idxs]  = cur_true_conf_weights

            tcoord.view(-1)[raveled_idxs_b0] = cur_tcoord_x
            tcoord.view(-1)[raveled_idxs_b1] = cur_tcoord_y
            tcoord.view(-1)[raveled_idxs_b2] = cur_tcoord_w
            tcoord.view(-1)[raveled_idxs_b3] = cur_tcoord_h

            tcls.view(-1)[raveled_idxs]  = cur_gt_cls
            tconf.view(-1)[raveled_idxs] = cur_true_ious

        # because coord and conf masks are witin this MSE we need to sqrt them
        coord_mask = coord_mask.sqrt()
        conf_mask = conf_mask.sqrt()
        coord_mask = coord_mask.expand_as(tcoord)
        return coord_mask, conf_mask, cls_mask, tcoord, tconf, tcls


if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.models.yolo2.light_region_loss all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
