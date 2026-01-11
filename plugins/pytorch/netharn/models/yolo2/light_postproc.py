"""
Decode YOLO network outputs into netharn-style Detections

Adapted from code by EAVISE
"""
import torch
import numpy as np  # NOQA
import ubelt as ub  # NOQA


class GetBoundingBoxes(object):
    """ Convert output from darknet networks to bounding box tensor.

    Args:
        network (lightnet.network.Darknet): Network the converter will be used with
        conf_thresh (Number [0-1]): Confidence threshold to filter detections
        nms_thresh(Number [0-1]): Overlapping threshold to filter detections with non-maxima suppresion

    Returns:
        List[kwimage.Detections]:
            detection object for each image in the batch.

    Note:
        The boxes in the detections use relative values for its coordinates.

    Examples:
        >>> # xdoctest: +REQUIRES(module:kwimage)
        >>> import torch
        >>> torch.random.manual_seed(0)
        >>> anchors = np.array([(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)])
        >>> self = GetBoundingBoxes(anchors=anchors, num_classes=20, conf_thresh=.14, nms_thresh=0.5)
        >>> output = torch.randn(8, 5, 5 + 20, 9, 9)
        >>> batch_dets = self(output)
        >>> assert len(batch_dets) == 8
    """

    def __init__(self, num_classes, anchors, conf_thresh=0.001, nms_thresh=0.4):
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.anchor_step = len(anchors[0])
        self.anchors = torch.Tensor(anchors)
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

    # @util.profile
    def __call__(self, output):
        """ Compute bounding boxes after thresholding and nms """
        batch_dets = self._decode(output.data)
        return batch_dets

    # @util.profile
    def _decode(self, output):
        """
        Returns array of detections for every image in batch

        CommandLine:
            python ~/code/netharn/netharn/box_models/yolo2/light_postproc.py GetBoundingBoxes._decode

        Examples:
            >>> # xdoctest: +REQUIRES(module:kwimage)
            >>> import torch
            >>> torch.random.manual_seed(0)
            >>> anchors = np.array([(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)])
            >>> self = GetBoundingBoxes(anchors=anchors, num_classes=20, conf_thresh=.14, nms_thresh=0.5)
            >>> output = torch.randn(16, 5, 5 + 20, 9, 9)
            >>> from ... import XPU
            >>> output = XPU.coerce('auto').move(output)
            >>> batch_dets = self._decode(output.data)
            >>> assert len(batch_dets) == 16

        Ignore:
            >>> from .models.yolo2.yolo2 import *  # NOQA
            >>> info = dev_demodata()
            >>> outputs = info['outputs']
            >>> cxywh_energy = output['cxywh_energy']
            >>> raw = info['raw']
            >>> raw_ = raw.clone()

            >>> self = GetBoundingBoxes(anchors=info['model'].anchors, num_classes=20, conf_thresh=.14, nms_thresh=0.5)
            >>> dets = self._decode(raw)[0]
            >>> dets.scores

            >>> self, output = ub.take(info, ['coder', 'outputs'])
            >>> batch_dets = self.decode_batch(output)
            >>> dets = batch_dets[0]
            >>> dets.scores

        """
        import kwimage
        # dont modify inplace
        raw_ = output.clone()

        # Variables
        bsize = raw_.shape[0]
        h, w = raw_.shape[-2:]

        device = raw_.device

        if self.anchors.device != device:
            self.anchors = self.anchors.to(device)

        # Compute xc,yc, w,h, box_score on Tensor
        lin_x = torch.linspace(0, w - 1, w, device=device).repeat(h, 1).view(h * w)
        lin_y = torch.linspace(0, h - 1, h, device=device).repeat(w, 1).t().contiguous().view(h * w)

        anchor_w = self.anchors[:, 0].contiguous().view(1, self.num_anchors, 1)
        anchor_h = self.anchors[:, 1].contiguous().view(1, self.num_anchors, 1)

        # -1 == 5+num_classes (we can drop feature maps if 1 class)
        output_ = raw_.view(bsize, self.num_anchors, -1, h * w)

        output_[:, :, 0, :].sigmoid_().add_(lin_x).div_(w)          # X center
        output_[:, :, 1, :].sigmoid_().add_(lin_y).div_(h)          # Y center
        output_[:, :, 2, :].exp_().mul_(anchor_w).div_(w)           # Width
        output_[:, :, 3, :].exp_().mul_(anchor_h).div_(h)           # Height
        output_[:, :, 4, :].sigmoid_()                              # Box score

        # output_[:, :, 0:4].sum()
        # torch.all(cxywh.view(-1) == output_[:, :, 0:4].contiguous().view(-1))

        # Compute class_score
        if self.num_classes > 1:
            cls_scores = torch.nn.functional.softmax(output_[:, :, 5:, :], 2)

            cls_max, cls_max_idx = torch.max(cls_scores, 2)
            cls_max.mul_(output_[:, :, 4, :])
        else:
            cls_max = output_[:, :, 4, :]
            cls_max_idx = torch.zeros_like(cls_max)

        # Save detection if conf*class_conf is higher than threshold

        # Newst lightnet code, which is based on my mode1 code
        score_thresh = cls_max > self.conf_thresh
        score_thresh_flat = score_thresh.view(-1)

        if score_thresh.sum() == 0:
            batch_dets = []
            for i in range(bsize):
                batch_dets.append(kwimage.Detections(
                    boxes=kwimage.Boxes(torch.empty((0, 4), dtype=torch.float32, device=device), 'cxywh'),
                    scores=torch.empty(0, dtype=torch.float32, device=device),
                    class_idxs=torch.empty(0, dtype=torch.int64, device=device),
                ))
        else:
            # Mask select boxes > conf_thresh
            coords = output_.transpose(2, 3)[..., 0:4]
            coords = coords[score_thresh[..., None].expand_as(coords)].view(-1, 4)

            scores = cls_max[score_thresh]

            class_idxs = cls_max_idx[score_thresh]

            stacked_dets = kwimage.Detections(
                boxes=kwimage.Boxes(coords, 'cxywh'),
                scores=scores,
                class_idxs=class_idxs,
            )

            # Get indexes of splits between images of batch
            max_det_per_batch = len(self.anchors) * h * w
            slices = [slice(max_det_per_batch * i, max_det_per_batch * (i + 1)) for i in range(bsize)]
            det_per_batch = torch.IntTensor([score_thresh_flat[s].int().sum() for s in slices])
            split_idx = torch.cumsum(det_per_batch, dim=0)

            batch_dets = []
            start = 0
            for end in split_idx:
                dets = stacked_dets[start: end]
                dets = dets.non_max_supress(thresh=self.nms_thresh)
                batch_dets.append(dets)
                start = end
        return batch_dets


if __name__ == '__main__':
    """
    CommandLine:
        python -m netharn.models.yolo2.light_postproc all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
