"""

Tested against mmdet 1.0 on sha 4c94f10d0ebb566701fb5319f5da6808df0ebf6a

Notes:
    https://github.com/open-mmlab/mmdetection/blob/master/docs/MODEL_ZOO.md

SeeAlso:
    ~/code/bioharn/dev/misc/autogen_mm_models.py
"""
import numpy as np
import ubelt as ub
from viame.pytorch import netharn as nh
import torch
import kwimage
import kwarray
from collections import OrderedDict
import warnings  # NOQA
from viame.pytorch.netharn.data.channel_spec import ChannelSpec  # TODO: kwcoco.ChannelSpec
from viame.pytorch.netharn.data import data_containers
from packaging.version import parse as Version


def _hack_mm_backbone_in_channels(backbone_cfg):
    """
    Verify the backbone supports input channels
    """
    if 'in_channels' not in backbone_cfg:
        return
    import mmdet
    _NEEDS_CHECK = True
    if _NEEDS_CHECK:
        import inspect
        from mmdet import models
        backbone_key = backbone_cfg['type']
        if backbone_key == 'ResNeXt':
            backbone_key = 'ResNet'

        backbone_cls = models.builder.BACKBONES.get(backbone_key)

        cls_kw = inspect.signature(backbone_cls).parameters
        if 'in_channels' not in cls_kw:
            if backbone_cfg['in_channels'] == 3:
                backbone_cfg.pop('in_channels')
            else:
                raise ValueError((
                    'mmdet.__version__={!r} does not support in_channels'
                ).format(mmdet.__version__))


def _ensure_unwrapped_and_mounted(mm_inputs, xpu):
    from kwcoco.util.util_json import IndexableWalker
    walker = IndexableWalker(mm_inputs)
    for path, val in walker:
        if isinstance(val, dict):
            walker[path] = val.copy()

        if isinstance(val, data_containers.BatchContainer):
            if len(val.data) != 1:
                raise ValueError('data not scattered correctly')
            if val.cpu_only:
                val = val.data[0]
            else:
                val = xpu.move(val.data[0])
            walker[path] = val
    return mm_inputs


def _hack_mmdet_masks(gt_masks):
    r"""
    Convert masks into a mmdet compliant format

    Example:
        >>> # xdoctest: +REQUIRES(module:mmdet)
        >>> channels = 3
        >>> batch = _demo_batch(bsize=4, with_mask='bitmap', channels=channels)
        >>> gt_masks = batch['label']['class_masks']
        >>> bitmask_container = _hack_mmdet_masks(gt_masks)
        >>> gt_masks = batch['label']['class_masks'].data[0]
        >>> bitmask_single = _hack_mmdet_masks(gt_masks)
        >>> #
        >>> channels = 3
        >>> batch = _demo_batch(bsize=4, with_mask='polygon', channels=channels)
        >>> gt_masks = batch['label']['class_masks']
        >>> poly_container = _hack_mmdet_masks(gt_masks)
        >>> gt_masks = batch['label']['class_masks'].data[0]
        >>> poly_single = _hack_mmdet_masks(gt_masks)
        >>> #
        >>> print('\nbitmask_single = {!r}'.format(bitmask_single))
        >>> print('\npoly_single = {!r}'.format(poly_single))
        >>> print('\nbitmask_container = {!r}'.format(bitmask_container))
        >>> print('\npoly_container = {!r}'.format(poly_container))

    """
    if isinstance(gt_masks, data_containers.BatchContainer):
        # handle data containers as well
        DC = type(gt_masks)
        kw = gt_masks.meta
        gt_masks_ = DC([_hack_mmdet_masks(d) for d in gt_masks.data], **kw)
        return gt_masks_
    else:
        def _hack_hw(m):
            w, h = 1, 1
            for obj_poly in m:
                flat_coords = list(ub.flatten(obj_poly))
                xs = flat_coords[0::2]
                ys = flat_coords[1::2]
                w = max(w, max(xs))
                h = max(h, max(ys))
            w = int(np.ceil(w))
            h = int(np.ceil(h))
            return {'height': h, 'width': w}

        if isinstance(gt_masks, list) and len(gt_masks) > 0:
            if isinstance(gt_masks[0], list):
                # Input is determined to be mmdet polygons
                from mmdet.core.mask import PolygonMasks
                poly_masks = [PolygonMasks(m, **_hack_hw(m)) for m in gt_masks]
                return poly_masks
            elif isinstance(gt_masks[0], (torch.Tensor, np.array)):
                from mmdet.core.mask import BitmapMasks
                # Input is determined to be mmdet bitmasks

                numpy_masks = [kwarray.ArrayAPI.numpy(mask) for mask in gt_masks]
                bitmap_masks = [
                    BitmapMasks(m, height=m.shape[1], width=m.shape[2])
                    for m in numpy_masks]
                return bitmap_masks
            else:
                raise Exception('unknown case 1')
        else:
            raise Exception('unknown case 2')


def _hack_numpy_gt_masks(gt_masks):
    # mmdet only allows numpy inputs
    from mmdet.core.mask import BitmapMasks
    numpy_masks = [kwarray.ArrayAPI.numpy(mask) for mask in gt_masks]

    bitmap_masks = [
        BitmapMasks(m, height=m.shape[1], width=m.shape[2])
        for m in numpy_masks]
    return bitmap_masks


def _demo_batch(bsize=1, channels='rgb', h=256, w=256, classes=3,
                with_mask=True, packed=False):
    """
    Input data for testing this detector

    Args:
        packed (bool, default=False): if True return dict of data containers
            otherwise unwrap the datacontainers by packing their contents.

    Example:
        >>> # xdoctest: +REQUIRES(module:mmdet)
        >>> from .models.mm_models import *  # NOQA
        >>> from .models.mm_models import _demo_batch, _batch_to_mm_inputs
        >>> #globals().update(**xdev.get_func_kwargs(_demo_batch))
        >>> classes = ['class_{:0d}'.format(i) for i in range(81)]
        >>> channels = ChannelSpec.coerce('rgb,mx|my')
        >>> #
        >>> batch = _demo_batch(bsize=4, with_mask=True, channels=channels)
        >>> mm_inputs = _batch_to_mm_inputs(batch)
        >>> #
        >>> #batch = _demo_batch(bsize=4, with_mask=True, channels=channels, packed=True)
        >>> #mm_inputs = _batch_to_mm_inputs(batch)


        >>> batch = _demo_batch(bsize=4, with_mask='bitmap', channels=channels)

        gt_masks = batch['label']['class_masks'].data[0]

        >>> batch = _demo_batch(bsize=4, with_mask='polygon', channels=channels)

        gt_masks = batch['label']['class_masks'].data[0]

        >>> mm_inputs = _batch_to_mm_inputs(batch)

        from mmdet.core.mask import PolygonMasks
        for ms in batch['label']['class_masks'].data:
            for m in ms:
                PolygonMasks(m, **_hack_hw(m))

        DC
        data_containers.BatchContainer(
            [[PolygonMasks(m, **_hack_hw(m)) for m in ms] for ms in batch['label']['class_masks'].data],
            stack=False
        )

        [0][0][0])

    Ignore:
        import xdev
        globals().update(**xdev.get_func_kwargs(_demo_batch))
        with_mask = 'polygon'
    """
    rng = kwarray.ensure_rng(0)
    if isinstance(bsize, list):
        item_sizes = bsize
        bsize = len(item_sizes)
    else:
        item_sizes = [rng.randint(0, 10) for bx in range(bsize)]

    channels = ChannelSpec.coerce(channels)
    B, H, W = bsize, h, w

    input_shapes = {
        key: (B, c, H, W)
        for key, c in channels.sizes().items()
    }
    inputs = {
        key: torch.rand(*shape)
        for key, shape in input_shapes.items()
    }

    batch_items = []
    for bx in range(B):

        item_sizes[bx]

        dets = kwimage.Detections.random(num=item_sizes[bx],
                                         classes=classes,
                                         segmentations=True)
        dets = dets.scale((W, H))

        # Extract segmentations if they exist
        if with_mask:

            if isinstance(with_mask, str):
                if with_mask not in {'bitmap', 'polygon'}:
                    raise KeyError(with_mask)

            # Not supported by mmdet as of 2.9.0
            USE_RELATIVE_MASKS = False
            USE_POLYGON_MASKS = with_mask == 'polygon'

            has_mask_list = []
            class_mask_list = []
            for sseg in dets.data['segmentations']:
                if sseg is not None:

                    if USE_POLYGON_MASKS:
                        # Coerce into polygons for mmdet
                        pts = sseg.to_coco(style='orig')
                        pts = [np.array(p) for p in pts]
                        class_mask_list.append(pts)
                        has_mask_list.append(1)
                        continue
                    elif USE_RELATIVE_MASKS:
                        mask = sseg.to_relative_mask()
                        c_mask = mask.to_c_mask().data
                        mask_tensor = torch.tensor(c_mask, dtype=torch.uint8)
                    else:
                        mask = sseg.to_mask(dims=(H, W))
                        c_mask = mask.to_c_mask().data
                        mask_tensor = torch.tensor(c_mask, dtype=torch.uint8)

                    class_mask_list.append(mask_tensor.unsqueeze(0))
                    has_mask_list.append(1)
                else:
                    class_mask_list.append(None)
                    has_mask_list.append(-1)

            has_mask = torch.tensor(has_mask_list, dtype=torch.int8)
            if USE_POLYGON_MASKS:
                class_masks = class_mask_list
            elif USE_RELATIVE_MASKS:
                class_masks = class_mask_list
            else:
                if len(class_mask_list) == 0:
                    class_masks = torch.empty((0, H, W), dtype=torch.uint8)
                else:
                    class_masks = torch.cat(class_mask_list, dim=0)
        else:
            class_masks = None

        dets = dets.tensor()
        label = {
            'tlbr': data_containers.ItemContainer(dets.boxes.to_ltrb().data.float(), stack=False),
            'class_idxs': data_containers.ItemContainer(dets.class_idxs, stack=False),
            'weight': data_containers.ItemContainer(torch.FloatTensor(rng.rand(len(dets))), stack=False)
        }

        if with_mask:
            label['class_masks'] = data_containers.ItemContainer(class_masks, stack=False, cpu_only=True)
            label['has_mask'] = data_containers.ItemContainer(has_mask, stack=False, cpu_only=True)

        item = {
            'inputs': {
                key: data_containers.ItemContainer(vals[bx], stack=True)
                for key, vals in inputs.items()
            },
            'label': label,
        }
        batch_items.append(item)

    # from viame.pytorch import netharn as nh
    # from .data_containers import container_collate
    batch = data_containers.container_collate(batch_items, num_devices=1)
    # batch = nh.data.collate.padded_collate(batch_items)

    if packed:
        from kwcoco.util.util_json import IndexableWalker
        walker = IndexableWalker(batch)
        for path, val in walker:
            if isinstance(val, data_containers.BatchContainer):
                walker[path] = val.pack()
                walker.send(False)

    return batch


def _dummy_img_metas(B, H, W, C):
    import mmdet
    MMDET_GE_2_12 = Version(mmdet.__version__) >= Version('2.12.0')
    if MMDET_GE_2_12:
        scale_factor = np.array([1., 1.0])
    else:
        scale_factor = 1.0

    img_metas = [{
        'img_shape': (H, W, C),
        'ori_shape': (H, W, C),
        'pad_shape': (H, W, C),
        'filename': '<memory>.png',
        'scale_factor': scale_factor,
        'flip': False,
    } for _ in range(B)]
    return img_metas


def _batch_to_mm_inputs(batch, ignore_thresh=0.1):
    """
    Convert our netharn style batch to mmdet style

    Example:
        >>> # xdoctest: +REQUIRES(module:mmdet)
        >>> # Test batch with empty item
        >>> bsize = [2, 0, 1, 1]
        >>> batch = _demo_batch(bsize)
        >>> mm_inputs = _batch_to_mm_inputs(batch)

        >>> # Test empty batch
        >>> from .models.mm_models import *  # NOQA
        >>> bsize = [0, 0, 0, 0]
        >>> batch = _demo_batch(bsize)
        >>> mm_inputs = _batch_to_mm_inputs(batch)

        >>> channels = ChannelSpec.coerce('rgb,mx|my')
        >>> batch = _demo_batch(bsize=4, with_mask=True, channels=channels, packed=False)
        >>> mm_inputs = _batch_to_mm_inputs(batch)

        >>> # batch = _demo_batch(bsize=4, with_mask=True, channels=channels, packed=True)
        >>> # mm_inputs = _batch_to_mm_inputs(batch)

        parent_batch = _demo_batch(bsize=4, with_mask=True, channels=channels, packed=False)
        batch = parent_batch['inputs']['rgb']
        mm_inputs = _batch_to_mm_inputs(batch)
        mm_inputs = _batch_to_mm_inputs(batch.pack())

        batch = _demo_batch(bsize=4, with_mask='polygon', channels=channels, packed=False)
        mm_inputs = _batch_to_mm_inputs(batch)
        z = mm_inputs['gt_masks']
        z = z.to(0)

        # Note: padded collate seems to not work in this case
        from kwcoco.util.util_json import IndexableWalker
        self = batch['label']['class_masks']
        inbatch = list(ub.flatten(self.data))
        walker = IndexableWalker(inbatch)
        for path, val in walker:
            if isinstance(val, np.ndarray):
                walker[path] = torch.from_numpy(val)
        from viame.pytorch.netharn.data.collate import padded_collate
        packed = padded_collate(inbatch, fill_value=self.padding_value)
    """
    if isinstance(batch, torch.Tensor):
        # Absolute simplest case where the input is given as a
        # (presumably RGB or grayscale) Tensor
        B, C, H, W = batch.shape

        # hack in img meta
        img_metas = _dummy_img_metas(B, H, W, C)

        mm_inputs = {
            'imgs': batch,
            'img_metas': img_metas,
        }
        return mm_inputs
    if isinstance(batch, data_containers.BatchContainer):
        # Second simplest case, no labels are given and just an one main stream
        # is given as a batch container.
        main_stream = batch
        stream_type = type(batch)
        # Get the number of batch items for each GPU / group
        groupsizes = [item.shape[0] for item in main_stream.data]
        B = len(main_stream.data)
        C, H, W = main_stream.data[0].shape[1:]
        DC = stream_type

        # hack in img meta
        img_metas = DC([
            _dummy_img_metas(num, H, W, C)
            for num in groupsizes
        ], stack=False, cpu_only=True)

        mm_inputs = {
            'imgs': main_stream,
            'img_metas': img_metas,
        }
        return mm_inputs

    if not isinstance(batch, dict):
        raise TypeError('We expected input batch to be a dictionary')

    if isinstance(batch['inputs'], dict):
        inputs = batch['inputs']
    else:
        # Input streams were not specified
        inputs = {'???': batch['inputs']}

    # Determine what the main input stream is.
    # Typically this will be RGB.
    if len(inputs) == 0:
        raise ValueError('No inputs are given')
    elif len(inputs) == 1 and False:
        # NOTE: the len(inputs) > 1 case should work for this as well hack this
        # to be false for testing for now. Can remove the and False after we
        # verify the fusion case works well.
        #
        # Simple case, only one channel
        main_key, main_stream = ub.peek(inputs.items())
        stream_type = type(main_stream)
    else:
        # Early fusion case
        input_types = ub.map_vals(type, inputs)
        if not ub.allsame(input_types.values()):
            raise Exception('Input streams should all be the same type')
        stream_type = ub.peek(input_types.values())
        key_priority_lut = {
            'rgb': 2,
            'gray': 1,
        }
        stream_priority = {
            key: (key_priority_lut.get(key, 0), value.numel())
            for key, value in inputs.items()
        }
        main_key = ub.argmax(stream_priority)
        main_stream = inputs[main_key]

    if isinstance(main_stream, data_containers.BatchContainer):
        # Things are already in data containers

        # Get the number of batch items for each GPU / group
        groupsizes = [item.shape[0] for item in main_stream.data]

        B = len(main_stream.data)
        C, H, W = main_stream.data[0].shape[1:]

        DC = stream_type

        # hack in img meta
        img_metas = DC([
            _dummy_img_metas(num, H, W, C)
            for num in groupsizes
        ], stack=False, cpu_only=True)

        mm_inputs = {
            'imgs': main_stream,
            'main_key': main_key,
            'inputs': inputs,
            'img_metas': img_metas,
        }

        # Handled pad collated batches. Ensure shapes are correct.
        if 'label' in batch:
            label = batch['label']
            mm_inputs['gt_labels'] = DC(
                [
                    list(cidxs) for cidxs in label['class_idxs'].data
                ], label['class_idxs'].stack,
                label['class_idxs'].padding_value)

            if 'cxywh' in label:
                mm_inputs['gt_bboxes'] = DC(
                    [[kwimage.Boxes(b, 'cxywh').to_ltrb().data for b in bbs]
                     for bbs in label['cxywh'].data],
                    label['cxywh'].stack,
                    label['cxywh'].padding_value)

            if 'tlbr' in label:
                assert 'gt_bboxes' not in mm_inputs, 'already have boxes'
                mm_inputs['gt_bboxes'] = DC(
                    [[kwimage.Boxes(b, 'tlbr').to_ltrb().data for b in bbs]
                     for bbs in label['tlbr'].data],
                    label['tlbr'].stack,
                    label['tlbr'].padding_value)

            if 'class_masks' in label:
                mm_inputs['gt_masks'] = _hack_mmdet_masks(label['class_masks'])
                # .data
                # [mask for mask in label['class_masks'].data]

            if 'weight' in label:
                ignore_flags = DC(
                    [[w < ignore_thresh for w in ws]
                     for ws in label['weight'].data], label['weight'].stack)

                # filter ignore boxes
                outer_bboxes_ignore = []
                for outer_bx in range(len(ignore_flags.data)):
                    inner_bboxes_ignore = []
                    for inner_bx in range(len(ignore_flags.data[outer_bx])):
                        flags = ignore_flags.data[outer_bx][inner_bx]
                        ignore_bboxes = mm_inputs['gt_bboxes'].data[outer_bx][inner_bx][flags]
                        mm_inputs['gt_labels'].data[outer_bx][inner_bx] = (
                            mm_inputs['gt_labels'].data[outer_bx][inner_bx][~flags])
                        mm_inputs['gt_bboxes'].data[outer_bx][inner_bx] = (
                            mm_inputs['gt_bboxes'].data[outer_bx][inner_bx][~flags])
                        inner_bboxes_ignore.append(ignore_bboxes)
                    outer_bboxes_ignore.append(inner_bboxes_ignore)

                mm_inputs['gt_bboxes_ignore'] = DC(outer_bboxes_ignore,
                                                   label['weight'].stack)

    else:
        B, C, H, W = main_stream.shape

        # hack in img meta
        img_metas = _dummy_img_metas(B, H, W, C)

        mm_inputs = {
            'imgs': main_stream,
            'main_key': main_key,
            'inputs': inputs,
            'img_metas': img_metas,
        }

        # Handled pad collated batches. Ensure shapes are correct.
        if 'label' in batch:

            label = batch['label']

            if isinstance(label['class_idxs'], list):
                # Data was already collated as a list
                mm_inputs['gt_labels'] = label['class_idxs']
                if 'cxywh' in label:
                    mm_inputs['gt_bboxes'] = [
                        kwimage.Boxes(b, 'cxywh').to_ltrb().data
                        for b in label['cxywh']
                    ]
                elif 'tlbr' in label:
                    assert 'gt_bboxes' not in mm_inputs, 'already have boxes'
                    mm_inputs['gt_bboxes'] = label['tlbr']

                if 'class_masks' in label:
                    mm_inputs['gt_masks'] = _hack_mmdet_masks(label['class_masks'])

                if 0:
                    # TODO:
                    if 'weight' in label:
                        gt_bboxes_ignore = [[w < ignore_thresh for w in ws]
                                            for ws in label['weight']]
                        mm_inputs['gt_bboxes_ignore'] = gt_bboxes_ignore
            else:
                raise NotImplementedError('use batch containers')

    return mm_inputs


class MM_Coder(object):
    """
    Standardize network inputs and outputs
    """

    def __init__(self, classes):
        self.classes = classes

    def decode_batch(self, outputs):
        """
        Transform mmdet outputs into a list of detections objects

        Args:
            outputs (Dict): dict containing loss_parts and batch_results

                b = 0  # indexes over batches
                k = 0  # indexes over the different classes

                # batch_results are an mmdet based list format
                batch_results = outputs['batch_results']
                result = batch_results[b]

                # result - can be a list with
                result[k] is an (N, 5) tensor encoding bboxes for class k

                # result - can be a 2-tuple with
                result[0][k] is a (N, 5) tensor encoding bboxes for class k
                result[1][k] is a N-len list of coco sseg dicts for class k

        Example:
            >>> # xdoctest: +REQUIRES(module:mmdet)
            >>> from .models.mm_models import *  # NOQA
            >>> import torch
            >>> classes = ['a', 'b', 'c']
            >>> xpu = data_containers.ContainerXPU('cpu')
            >>> model = MM_CascadeRCNN(classes).to(xpu.main_device)
            >>> batch = model.demo_batch(bsize=1, h=256, w=256)
            >>> outputs = model.forward(batch)
            >>> self = model.coder
            >>> batch_dets = model.coder.decode_batch(outputs)

        Example:
            >>> # xdoctest: +REQUIRES(module:mmdet)
            >>> from .models.mm_models import *  # NOQA
            >>> classes = ['a', 'b', 'c']
            >>> xpu = data_containers.ContainerXPU(0)
            >>> model = MM_MaskRCNN(classes, channels='rgb|d').to(xpu.main_device)
            >>> batch = model.demo_batch(bsize=1, h=256, w=256)
            >>> mm_inputs = _batch_to_mm_inputs(batch)
            >>> outputs = model.forward(batch)
            >>> self = model.coder
            >>> batch_dets = model.coder.decode_batch(outputs)
        """
        batch_results = outputs['batch_results']
        batch_dets = []

        if isinstance(batch_results, data_containers.BatchContainer):
            batch_result_data = batch_results.data
        else:
            batch_result_data = batch_results

        class_offset = 0
        start = 0
        mm_sseg_results = None

        for result in batch_result_data:
            if isinstance(result, tuple) and len(result) == 2:
                # bbox and segmentation result
                mm_bbox_results = result[0]
                mm_sseg_results = result[1]
            elif isinstance(result, list):
                # Hack for mmdet 2.4+ (not sure exactly what format change is
                # so this may need to be reworked)
                import mmdet
                if Version(mmdet.__version__) >= Version('2.4.0'):

                    mm_sseg_results = None
                    if len(result) == 1:
                        if isinstance(result[0], tuple):
                            mm_bbox_results = result[0][0]
                            mm_sseg_results = result[0][1]

                    if mm_sseg_results is None:
                        # They seem to have added another level of nesting
                        # Not sure what the semantics of it are yet.
                        import itertools as it
                        mm_bbox_results = list(it.chain.from_iterable(result))
                else:
                    # bbox only result
                    mm_bbox_results = result
            else:
                # TODO: when using data parallel, we have
                # Note: this actually only happened when we failed to use
                # netharn.data.data_containers.ContainerXPU
                # type(result) = <class 'netharn.data.data_containers.BatchContainer'>
                raise NotImplementedError(
                    'unknown mmdet result format. '
                    'type(result) = {}'.format(type(result))
                )

            if mm_bbox_results is not None:
                # Stack the results into a detections object
                # Note: avoid [:, x] syntax due to torch_liberator AST unparsing bug
                pred_cidxs = []
                pred_tlbr_parts = []
                pred_score_parts = []
                for cidx, cls_results in enumerate(mm_bbox_results, start=start):
                    # assert cls_results.shape == (None, 5)
                    pred_cidxs.extend([cidx + class_offset] * len(cls_results))
                    # Use transpose to avoid multi-dim slice syntax
                    pred_tlbr_parts.append(cls_results.T[0:4].T)
                    pred_score_parts.append(cls_results.T[4])
                pred_tlbr = np.vstack(pred_tlbr_parts)
                pred_score = np.hstack(pred_score_parts)
            else:
                raise AssertionError('should always have bboxes')

            if mm_sseg_results is not None:
                pred_ssegs = []
                for cidx, cls_ssegs in enumerate(mm_sseg_results, start=start):
                    pred_ssegs.extend([
                        kwimage.Mask.coerce(sseg) for sseg in cls_ssegs])
                    # kwimage.Mask(sseg, 'bytes_rle') for sseg in cls_ssegs])
                pred_ssegs = kwimage.MaskList(pred_ssegs)
            else:
                pred_ssegs = kwimage.MaskList([None] * len(pred_cidxs))

            det = kwimage.Detections(
                boxes=kwimage.Boxes(pred_tlbr, 'tlbr'),
                scores=pred_score,
                class_idxs=np.array(pred_cidxs, dtype=int),
                segmentations=pred_ssegs,
                classes=self.classes
            )
            batch_dets.append(det)
        return batch_dets


class MM_Detector(nh.layers.Module):
    """
    """
    _mmdet_is_version_1x = False  # needed to prevent autoconvert
    __bioharn_model_vesion__ = 4  # needed to prevent autoconvert
    __BUILTIN_CRITERION__ = True

    def __init__(self, mm_model, train_cfg=None, test_cfg=None,
                 classes=None, input_stats=None, channels=None):
        super(MM_Detector, self).__init__()
        import mmcv
        import mmdet
        from mmdet.models import build_detector
        import kwcoco
        from packaging.version import parse as Version

        if input_stats is None:
            input_stats = {}

        self.classes = kwcoco.CategoryTree.coerce(classes)
        self.channels = ChannelSpec.coerce(channels)

        chan_keys = list(self.channels.keys())
        if len(chan_keys) != 1:
            print('GOT chan_keys = {!r}'.format(chan_keys))
            raise ValueError('this model can only do early fusion')
        if len(input_stats):
            if chan_keys != list(input_stats.keys()):
                # Backwards compat for older pre-fusion input stats method
                if 'mean' not in input_stats and 'std' not in input_stats:
                    raise AssertionError(
                        'input_stats = {!r}, self.channels={!r}'.format(
                            input_stats, self.channels)
                    )
                input_stats = {
                    chan_keys[0]: input_stats,
                }
            if len(input_stats) != 1:
                print('GOT input_stats = {!r}'.format(input_stats))
                raise ValueError('this model can only do early fusion')
            main_input_stats = ub.peek(input_stats.values())
        else:
            main_input_stats = {}
        self.input_norm = nh.layers.InputNorm(**main_input_stats)

        MMDET_GE_2_20 = Version(mmdet.__version__) >= Version('2.20.0')
        if MMDET_GE_2_20:
            # Not sure what the exact version break is here
            mm_model['backbone']['pretrained'] = mm_model.pop('pretrained')
            if test_cfg is not None:
                mm_model['test_cfg'] = test_cfg
            if train_cfg is not None:
                mm_model['train_cfg'] = train_cfg
            test_cfg = None
            train_cfg = None

        if train_cfg is not None:
            train_cfg = mmcv.utils.config.ConfigDict(train_cfg)

        if test_cfg is not None:
            test_cfg = mmcv.utils.config.ConfigDict(test_cfg)

        MMDET_GE_2_12 = Version(mmdet.__version__) >= Version('2.12.0')

        if MMDET_GE_2_12:
            # mmdet v2.12.0 introduced new registry stuff that forces use of
            # config dictionaries
            mm_model = mmcv.ConfigDict(mm_model)

            if MMDET_GE_2_20:
                if train_cfg is not None:
                    train_cfg = mmcv.ConfigDict(train_cfg)
                if test_cfg is not None:
                    test_cfg = mmcv.ConfigDict(test_cfg)
            else:
                train_cfg = mmcv.ConfigDict(train_cfg)
                test_cfg = mmcv.ConfigDict(test_cfg)

        self.detector = build_detector(
            mm_model, train_cfg=train_cfg, test_cfg=test_cfg)

        if MMDET_GE_2_12:
            self.detector.init_weights()

        self.coder = MM_Coder(self.classes)

    def demo_batch(self, bsize=3, h=256, w=256, with_mask=None):
        """
        Input data for testing this detector

        Example:
            >>> # xdoctest: +REQUIRES(module:mmdet)
            >>> classes = ['class_{:0d}'.format(i) for i in range(81)]
            >>> self = MM_RetinaNet(classes)
            >>> #globals().update(**xdev.get_func_kwargs(MM_Detector.demo_batch))
            >>> self.demo_batch()
        """
        if with_mask is None:
            with_mask = getattr(self.detector, 'with_mask', False)
        channels = self.channels
        batch = _demo_batch(bsize, channels, h, w, with_mask=with_mask)
        return batch

    def forward(self, batch, return_loss=True, return_result=True):
        """
        Wraps the mm-detection interface with something that plays nicer with
        netharn.

        Args:
            batch (Dict): containing:
                - inputs (Dict[str, Tensor]):
                    mapping of input streams (e.g. rgb or motion) to
                    corresponding tensors.
                - label (None | Dict): optional if loss is needed. Contains:
                    tlbr: bounding boxes in tlbr space
                    class_idxs: bounding box class indices
                    weight: bounding box class weights (only used to set ignore
                        flags)

                OR an mmdet style batch containing:
                    imgs
                    img_metas
                    gt_bboxes
                    gt_labels
                    etc...

                    # OR new auxillary information
                    auxs
                    main_key
                    <subject to change>

            return_loss (bool): compute the loss
            return_result (bool): compute the result
                TODO: make this more efficient if loss was computed as well

        Returns:
            Dict: containing results and losses depending on if return_loss and
                return_result were specified.
        """
        if 'img_metas' in batch and 'imgs' in batch:
            # already in mm_inputs format
            orig_mm_inputs = batch
        else:
            orig_mm_inputs = _batch_to_mm_inputs(batch)

        mm_inputs = orig_mm_inputs.copy()

        # from .data_containers import _report_data_shape
        # print('--------------')
        # print('--------------')
        # _report_data_shape(mm_inputs)
        # print('--------------')
        # print('--------------')

        # Hack: remove data containers if it hasn't been done already
        from viame.pytorch import netharn as nh
        xpu = nh.XPU.from_data(self)
        mm_inputs = _ensure_unwrapped_and_mounted(mm_inputs, xpu)

        imgs = mm_inputs.pop('imgs')
        img_metas = mm_inputs.pop('img_metas')

        # with warnings.catch_warnings():
        #     warnings.filterwarnings('ignore', 'indexing with dtype')

        outputs = {}
        if return_loss:
            gt_bboxes = mm_inputs['gt_bboxes']
            gt_labels = mm_inputs['gt_labels']

            # _report_data_shape(mm_inputs)
            gt_bboxes_ignore = mm_inputs.get('gt_bboxes_ignore', None)

            trainkw = {}
            try:
                with_mask = self.detector.with_mask
            except AttributeError:
                with_mask = False
            if with_mask:
                if 'gt_masks' in mm_inputs:
                    # mmdet only allows numpy inputs
                    trainkw['gt_masks'] = mm_inputs['gt_masks']

            # Compute input normalization
            imgs_norm = self.input_norm(imgs)

            losses = self.detector.forward(imgs_norm, img_metas,
                                           gt_bboxes=gt_bboxes,
                                           gt_labels=gt_labels,
                                           gt_bboxes_ignore=gt_bboxes_ignore,
                                           return_loss=True, **trainkw)
            loss_parts = OrderedDict()
            for loss_name, loss_value in losses.items():
                if 'loss' in loss_name:
                    # Ensure these are tensors and not scalars for
                    # DataParallel
                    if isinstance(loss_value, torch.Tensor):
                        loss_parts[loss_name] = loss_value.mean().unsqueeze(0)
                    elif isinstance(loss_value, list):
                        loss_parts[loss_name] = sum(_loss.mean().unsqueeze(0) for _loss in loss_value)
                    else:
                        raise TypeError(
                            '{} is not a tensor or list of tensors'.format(loss_name))

            if hasattr(self, '_fix_loss_parts'):
                self._fix_loss_parts(loss_parts)

            outputs['loss_parts'] = loss_parts

        if return_result:
            with torch.no_grad():
                imgs_norm = self.input_norm(imgs)
                hack_imgs = [g.unsqueeze(0) for g in imgs_norm]
                # For whaver reason we cant run more than one test image at the
                # same time.
                batch_results = []
                for one_img, one_meta in zip(hack_imgs, img_metas):
                    result = self.detector.forward([one_img], [[one_meta]],
                                                   return_loss=False)
                    batch_results.append(result)
                outputs['batch_results'] = data_containers.BatchContainer(
                    batch_results, stack=False, cpu_only=True)
        return outputs

    def _init_backbone_from_pretrained(self, filename):
        """
        Loads pretrained backbone weights
        """
        model_state = _load_mmcv_weights(filename)
        info = nh.initializers.functional.load_partial_state(
            self.detector.backbone, model_state, verbose=1,
            mangle=True, leftover='kaiming_normal',
        )
        return info


class MM_RetinaNet(MM_Detector):
    """
    CommandLine:
        xdoctest -m ~/code/bioharn/bioharn/models/mm_models.py MM_RetinaNet


    SeeAlso:
        ~/code/mmdetection/mmdet/models/detectors/cascade_rcnn.py
        ~/code/mmdetection/mmdet/models/detectors/retinanet.py

    Example:
        >>> # xdoctest: +REQUIRES(module:mmdet)
        >>> from .models.mm_models import *  # NOQA
        >>> import torch
        >>> import mmcv
        >>> classes = ['class_{:0d}'.format(i) for i in range(80)]
        >>> self = MM_RetinaNet(classes)
        >>> head = self.detector.bbox_head
        >>> batch = self.demo_batch()

        >>> xpu = nh.XPU(0)
        >>> self = self.to(xpu.main_device)
        >>> filename = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_r50_fpn_1x_20181125-7b0c2548.pth'
        >>> _ = mmcv.runner.checkpoint.load_checkpoint(self.detector, filename)
        >>> batch = xpu.move(batch)
        >>> outputs = self.forward(batch)

        import kwplot
        kwplot.autompl()

        kwplot.imshow(batch['inputs']['rgb'][0])
        det = outputs['batch_dets'][0]
        det.draw()

        # filename = '/home/joncrall/Downloads/retinanet_r50_fpn_1x_20181125-7b0c2548.pth'
        checkpoint = torch.load(filename, weights_only=False)
        state_dict = checkpoint['state_dict']
        ours = self.detector.state_dict()

        print(ub.urepr(list(state_dict.keys()),nl=1))
        print(ub.urepr(list(ours.keys()),nl=1))
    """

    def __init__(self, classes, channels='rgb', input_stats=None):

        # from mmcv.runner.checkpoint import load_from_http
        # url =
        # checkpoint = load_from_http(url)
        # pretrained = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_r50_fpn_1x_20181125-7b0c2548.pth'
        # pretrained = '/home/joncrall/Downloads/retinanet_r50_fpn_1x_20181125-7b0c2548.pth'
        # pretrained = 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/retinanet_r50_fpn_2x_20190616-75574209.pth'

        # NOTE: mmdetect is weird how it determines which category is
        # background. When use_sigmoid_cls is True, there is physically one
        # less class that is evaluated. When softmax is True the first output
        # is the background, but this is obfuscated by slicing, which makes it
        # seem as if your foreground class idxs do start at zero (even though
        # they don't in this case).
        #
        # Either way I think we can effectively treat these detectors as if the
        # bacground class is at the end of the list.
        import kwcoco
        classes = kwcoco.CategoryTree.coerce(classes)
        self.classes = classes
        # if 'background' in classes:
        #     assert classes.node_to_idx['background'] == 0
        #     num_classes = len(classes)
        # else:
        #     num_classes = len(classes) + 1
        num_classes = len(classes)

        self.channels = ChannelSpec.coerce(channels)
        chann_norm = self.channels.normalize()
        assert len(chann_norm) == 1
        in_channels = len(ub.peek(chann_norm.values()))

        compat_params = {}
        compat_params['bbox_head'] = dict(
            type='RetinaHead',
            num_classes=num_classes,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                octave_base_scale=4,
                scales_per_octave=3,
                ratios=[0.5, 1.0, 2.0],
                strides=[8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))

        # model settings
        mm_config = dict(
            type='RetinaNet',
            pretrained=None,
            backbone=dict(
                type='ResNet',
                depth=50,
                in_channels=in_channels,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                frozen_stages=-1,
                style='pytorch'),
            neck=dict(
                type='FPN',
                in_channels=[256, 512, 1024, 2048],
                out_channels=256,
                start_level=1,
                add_extra_convs=True,
                num_outs=5),
            **compat_params)
        # training and testing settings
        train_cfg = dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=0.5),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)
        test_cfg = dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)

        backbone_cfg = mm_config['backbone']
        _hack_mm_backbone_in_channels(backbone_cfg)
        super().__init__(mm_config, train_cfg=train_cfg, test_cfg=test_cfg,
                         classes=classes, input_stats=input_stats,
                         channels=channels)


class MM_MaskRCNN(MM_Detector):
    """
    Example:
        >>> # xdoctest: +REQUIRES(module:mmdet)
        >>> # Test multiple channels
        >>> from .models.mm_models import *  # NOQA
        >>> import torch
        >>> classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        >>> xpu = nh.XPU(0)
        >>> self = MM_MaskRCNN(classes, channels='rgb|d').to(xpu.main_device)
        >>> batch = self.demo_batch(bsize=1, h=256, w=256, with_mask='bitmap')
        >>> self.detector.test_cfg['rcnn']['score_thr'] = 1e-9
        >>> self.detector.test_cfg['rcnn']['mask_thr_binary'] = 1e-9
        >>> outputs = self.forward(batch)
        >>> batch_dets = self.coder.decode_batch(outputs)
        >>> sseg = batch_dets[0].data['segmentations'][0]
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> mask = sseg.data
        >>> mask.draw()

    Example:
        >>> # xdoctest: +REQUIRES(module:mmdet)
        >>> from .models.mm_models import *  # NOQA
        >>> import torch
        >>> classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        >>> xpu = nh.XPU(0)
        >>> self = MM_MaskRCNN(classes, channels='rgb|d').to(xpu.main_device)
        >>> batch = self.demo_batch(bsize=1, h=256, w=256, with_mask='polygon')
        >>> self.detector.test_cfg['rcnn']['score_thr'] = 1e-9
        >>> self.detector.test_cfg['rcnn']['mask_thr_binary'] = 1e-9
        >>> outputs = self.forward(batch)
    """
    def __init__(self, classes, channels='rgb', input_stats=None):
        import kwcoco
        classes = kwcoco.CategoryTree.coerce(classes)
        # if 'background' in classes:
        #     assert classes.node_to_idx['background'] == 0
        #     num_classes = len(classes)
        # else:
        #     num_classes = len(classes) + 1
        num_classes = len(classes)

        # pretrained = 'torchvision://resnet50'
        self.channels = ChannelSpec.coerce(channels)
        chann_norm = self.channels.normalize()
        assert len(chann_norm) == 1
        in_channels = len(ub.peek(chann_norm.values()))

        compat_params = {}
        compat_params['rpn_head'] = dict(
            type='RPNHead',
            in_channels=256,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[8],
                ratios=[0.5, 1.0, 2.0],
                strides=[4, 8, 16, 32, 64]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))
        compat_params['roi_head'] = dict(
            type='StandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
            mask_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            mask_head=dict(
                type='FCNMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=num_classes,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)))

        # model settings
        mm_config = dict(
            type='MaskRCNN',
            pretrained=None,
            backbone=dict(
                type='ResNet',
                depth=50,
                num_stages=4,
                in_channels=in_channels,
                out_indices=(0, 1, 2, 3),
                frozen_stages=-1,
                style='pytorch'),
            neck=dict(
                type='FPN',
                in_channels=[256, 512, 1024, 2048],
                out_channels=256,
                num_outs=5),
            **compat_params)
        # model training and testing settings
        train_cfg = dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    ignore_iof_thr=0.5),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=0,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms=dict(type='nms', iou_threshold=0.7),  # hack for mmdet=2.12.0
                max_per_img=2000,  # hack for mmdet=2.12.0
                nms_across_levels=False,
                nms_pre=2000,
                nms_post=2000,
                max_num=2000,
                nms_thr=0.7,
                min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    ignore_iof_thr=0.5),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False))
        test_cfg = dict(
            rpn=dict(
                nms=dict(type='nms', iou_threshold=0.7),  # hack for mmdet=2.12.0
                max_per_img=1000,  # hack for mmdet=2.12.0
                nms_across_levels=False,
                nms_pre=1000,
                nms_post=1000,
                max_num=1000,
                nms_thr=0.7,
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100,
                mask_thr_binary=0.5))

        backbone_cfg = mm_config['backbone']
        _hack_mm_backbone_in_channels(backbone_cfg)
        super().__init__(mm_config, train_cfg=train_cfg, test_cfg=test_cfg,
                         classes=classes, input_stats=input_stats,
                         channels=channels)


def _load_mmcv_weights(filename, map_location=None):
    import os
    from mmcv.runner.checkpoint import (get_torchvision_models, load_from_http)

    # load checkpoint from modelzoo or file or url
    if filename.startswith('modelzoo://'):
        warnings.warn('The URL scheme of "modelzoo://" is deprecated, please '
                      'use "torchvision://" instead')
        model_urls = get_torchvision_models()
        model_name = filename[11:]
        checkpoint = load_from_http(model_urls[model_name])
    elif filename.startswith('torchvision://'):
        model_urls = get_torchvision_models()
        model_name = filename[14:]
        checkpoint = load_from_http(model_urls[model_name])
    elif filename.startswith('open-mmlab://'):
        model_name = filename[13:]
        try:
            from mmcv.runner.checkpoint import open_mmlab_model_urls
            checkpoint = load_from_http(open_mmlab_model_urls[model_name])
        except ImportError:
            from mmcv.runner.checkpoint import get_external_models
            mmlab_urls = get_external_models()
            checkpoint = load_from_http(mmlab_urls[model_name])
    elif filename.startswith(('http://', 'https://')):
        checkpoint = load_from_http(filename)
    else:
        if not os.path.isfile(filename):
            raise IOError('{} is not a checkpoint file'.format(filename))
        checkpoint = torch.load(filename, map_location=map_location, weights_only=False)
    # get state_dict from checkpoint
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError(
            'No state_dict found in checkpoint file {}'.format(filename))
    return state_dict


class MM_CascadeRCNN(MM_Detector):
    """
    CommandLine:
        xdoctest -m ~/code/bioharn/bioharn/models/mm_models.py MM_CascadeRCNN

    Example:
        >>> # xdoctest: +REQUIRES(module:mmdet)
        >>> from .models.mm_models import *  # NOQA
        >>> import torch
        >>> classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        >>> xpu = data_containers.ContainerXPU(0)
        >>> self = MM_CascadeRCNN(classes).to(xpu.main_device)
        >>> batch = self.demo_batch(bsize=1, h=256, w=256)
        >>> outputs = self.forward(batch)
        >>> batch_dets = self.coder.decode_batch(outputs)

    Example:
        >>> # xdoctest: +REQUIRES(module:mmdet)
        >>> # Test multiple channels
        >>> from .models.mm_models import *  # NOQA
        >>> import torch
        >>> classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        >>> xpu = data_containers.ContainerXPU('cpu')
        >>> self = MM_CascadeRCNN(classes, channels='rgb|d').to(xpu.main_device)
        >>> batch = self.demo_batch(bsize=1, h=256, w=256)
        >>> batch = xpu.move(batch)
        >>> outputs = self.forward(batch)
        >>> batch_dets = self.coder.decode_batch(outputs)
    """
    def __init__(self, classes, channels='rgb', input_stats=None):
        import kwcoco
        classes = kwcoco.CategoryTree.coerce(classes)

        if 'background' in classes:
            # Mmdet changed its "background category" conventions
            # https://mmdetection.readthedocs.io/en/latest/compatibility.html#codebase-conventions
            if classes.node_to_idx['background'] != len(classes) - 1:
                raise AssertionError('mmdet 2.x needs background to be the last class')
            num_classes = len(classes) - 1
        else:
            num_classes = len(classes)

        # else:
        #     num_classes = len(classes)
        # if 'background' in classes:
        #     assert classes.node_to_idx['background'] == 0
        #     num_classes = len(classes)
        # else:
        #     num_classes = len(classes) + 1

        self.channels = ChannelSpec.coerce(channels)

        chann_norm = self.channels.normalize()
        assert len(chann_norm) == 1
        in_channels = len(ub.peek(chann_norm.values()))
        # pretrained = 'open-mmlab://resnext101_32x4d'
        # pretrained = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_x101_64x4d_fpn_1x_20181218-e2dc376a.pth'
        # pretrained = 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_x101_32x4d_fpn_1x_20190501-af628be5.pth'

        self.in_channels = in_channels

        compat_params = {}
        # Compatibility for mmdet 2.x
        compat_params['rpn_head'] = dict(
            type='RPNHead',
            in_channels=256,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[8],
                ratios=[0.5, 1.0, 2.0],
                strides=[4, 8, 16, 32, 64]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0))

        compat_params['roi_head'] = dict(
            type='CascadeRoIHead',
            num_stages=3,
            stage_loss_weights=[1, 0.5, 0.25],
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            bbox_head=[
                dict(
                    type='Shared2FCBBoxHead',
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=num_classes,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.1, 0.1, 0.2, 0.2]),
                    reg_class_agnostic=True,
                    loss_cls=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=False,
                        loss_weight=1.0),
                    loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                                   loss_weight=1.0)),
                dict(
                    type='Shared2FCBBoxHead',
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=num_classes,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.05, 0.05, 0.1, 0.1]),
                    reg_class_agnostic=True,
                    loss_cls=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=False,
                        loss_weight=1.0),
                    loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                                   loss_weight=1.0)),
                dict(
                    type='Shared2FCBBoxHead',
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=num_classes,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.033, 0.033, 0.067, 0.067]),
                    reg_class_agnostic=True,
                    loss_cls=dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=False,
                        loss_weight=1.0),
                    loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
            ])

        mm_config =  dict(
            type='CascadeRCNN',
            # pretrained='open-mmlab://resnext101_32x4d',
            pretrained=None,
            backbone=dict(
                type='ResNeXt',
                # pretrained=None,
                depth=101,
                groups=32,
                base_width=4,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                frozen_stages=1,
                style='pytorch',
                in_channels=in_channels
            ),
            neck=dict(
                type='FPN',
                in_channels=[256, 512, 1024, 2048],
                out_channels=256,
                num_outs=5),
            **compat_params,
        )

        train_cfg = dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    ignore_iof_thr=0.5),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=0,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms=dict(type='nms', iou_threshold=0.7),  # hack for mmdet=2.12.0
                max_per_img=1000,  # hack for mmdet=2.12.0
                nms_across_levels=False,
                nms_pre=2000,
                nms_post=2000,
                max_num=2000,
                nms_thr=0.7,
                min_bbox_size=0),
            rcnn=[
                dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.5,
                        neg_iou_thr=0.5,
                        min_pos_iou=0.5,
                        ignore_iof_thr=0.5),
                    sampler=dict(
                        type='RandomSampler',
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True),
                    mask_size=28,
                    pos_weight=-1,
                    debug=False),
                dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.6,
                        neg_iou_thr=0.6,
                        min_pos_iou=0.6,
                        ignore_iof_thr=0.5),
                    sampler=dict(
                        type='RandomSampler',
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True),
                    mask_size=28,
                    pos_weight=-1,
                    debug=False),
                dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.7,
                        neg_iou_thr=0.7,
                        min_pos_iou=0.7,
                        ignore_iof_thr=0.5),
                    sampler=dict(
                        type='RandomSampler',
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True),
                    mask_size=28,
                    pos_weight=-1,
                    debug=False)
            ],
            stage_loss_weights=[1, 0.5, 0.25])

        test_cfg = dict(
            rpn=dict(
                nms=dict(type='nms', iou_threshold=0.7),  # hack for mmdet=2.12.0
                max_per_img=1000,  # hack for mmdet=2.12.0
                nms_across_levels=False,
                nms_pre=1000,
                nms_post=1000,
                max_num=1000,
                nms_thr=0.7,
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100,
                mask_thr_binary=0.5),
            keep_all_stages=False)

        backbone_cfg = mm_config['backbone']
        _hack_mm_backbone_in_channels(backbone_cfg)
        super(MM_CascadeRCNN, self).__init__(mm_config, train_cfg=train_cfg,
                                             test_cfg=test_cfg,
                                             classes=classes,
                                             channels=channels,
                                             input_stats=input_stats)

    def _fix_loss_parts(self, loss_parts):
        """
        Hack for data parallel runs where the loss dicts need to have the same
        exact keys.
        """
        num_stages = self.detector.roi_head.num_stages
        for i in range(num_stages):
            for name in ['loss_cls', 'loss_bbox']:
                key = 's{}.{}'.format(i, name)
                if key not in loss_parts:
                    loss_parts[key] = torch.zeros(1, device=self.main_device)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/bioharn/bioharn/models/mm_models.py all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
