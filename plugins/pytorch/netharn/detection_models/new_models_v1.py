"""
Ignore:
    >>> # xdoctest: +SKIP
    >>> from .detect_fit import *  # NOQA
    >>> harn = setup_harn(bsize=2, datasets='special:shapes256',
    >>>     arch='MM_HRNetV2_w18_MaskRCNN', xpu='auto',
    >>>     workers=0, normalize_inputs='imagenet', sampler_backend=None)
    >>> harn.initialize()
    >>> batch = harn._demo_batch(1, 'vali')
    >>> #del batch['label']['has_mask']
    >>> #del batch['label']['class_masks']
    >>> from viame.pytorch.netharn.models.mm_models import _batch_to_mm_inputs
    >>> mm_batch = _batch_to_mm_inputs(batch)
    >>> outputs, loss = harn.run_batch(batch)

Ignore:
    import mmdet
    import liberator
    closer = liberator.closer.Closer()
    # closer.add_dynamic(mmdet.models.roi_heads.mask_heads.fcn_mask_head.FCNMaskHead_V2)
    # closer.add_dynamic(mmdet.models.detectors.MaskRCNN)
    # closer.add_dynamic(mmdet.models.detectors.TwoStageDetector)
    # closer.add_dynamic(mmdet.models.roi_heads.StandardRoIHead)
    #closer.add_dynamic(mmdet.models.roi_heads.StandardRoIHead)
    # closer.add_dynamic(mmdet.models.necks.HRFPN)
    # closer.add_dynamic(mmdet.models.roi_heads.test_mixins.BBoxTestMixin)
    # closer.add_dynamic(mmdet.models.roi_heads.test_mixins.MaskTestMixin)
    # closer.add_dynamic(mmdet.models.backbones.HRNet)
    # closer.add_dynamic(mmdet.models.roi_heads_V2.Shared2FCBBoxHead)
    closer.add_dynamic(mmdet.models.roi_heads.BBoxHead)

    # closer.expand(['mmdet'])
    print(closer.current_sourcecode())
"""

import ubelt as ub
import warnings  # NOQA
from viame.pytorch.netharn.data.channel_spec import ChannelSpec

try:
    import mmdet
    import mmcv
except Exception:
    mmdet = None
    mmcv = None
    build_backbone = None
else:
    # from mmdet.models.detectors.base import BaseDetector
    from mmdet.models.builder import build_backbone
    # from mmdet.models.builder import build_head
    # from mmdet.models.builder import build_neck

import torch.nn as nn
import torch

from .new_backbone import HRNet_V2
from .new_neck import HRFPN_V2
from .new_head import Shared2FCBBoxHead_V2
from .new_head import StandardRoIHead_V2
from .new_head import FCNMaskHead_V2
from .new_detector import MaskRCNN_V2

import kwcoco
from viame.pytorch import netharn as nh
from collections import OrderedDict
import warnings  # NOQA
from viame.pytorch.netharn.data import data_containers

# from .mm_models import MM_Detector
from .mm_models import MM_Coder
from .mm_models import _demo_batch
from .mm_models import _batch_to_mm_inputs
from .mm_models import _load_mmcv_weights
from .mm_models import _hack_numpy_gt_masks
from .mm_models import _ensure_unwrapped_and_mounted


def monkeypatch_build_norm_layer():
    """
    NOTE: This is structured in a very particular way so torch-liberator
    correctly carries the monkey patch with it.
    """
    # FIXME: need to inject into deploy files

    def build_norm_layer_hack(cfg, num_features, postfix=''):
        """Build normalization layer.

        Args:
            cfg (dict): The norm layer config, which should contain:

                - type (str): Layer type.
                - layer args: Args needed to instantiate a norm layer.
                - requires_grad (bool, optional): Whether stop gradient updates.
            num_features (int): Number of input channels.
            postfix (int | str): The postfix to be appended into norm abbreviation
                to create named layer.

        Returns:
            (str, nn.Module): The first element is the layer name consisting of
                abbreviation and postfix, e.g., bn1, gn. The second element is the
                created norm layer.
        """
        from mmcv.cnn.bricks.registry import NORM_LAYERS
        from mmcv.cnn.bricks.norm import infer_abbr
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

        layer_type = cfg_.pop('type')
        if layer_type not in NORM_LAYERS:
            raise KeyError(f'Unrecognized norm type {layer_type}')

        norm_layer = NORM_LAYERS.get(layer_type)
        abbr = infer_abbr(norm_layer)

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
            if cfg_['num_groups'] == 'auto':
                valid_num_groups = [
                    factor for factor in range(1, num_features)
                    if num_features % factor == 0
                ]
                infos = [
                    {'ng': ng, 'nc': num_features / ng}
                    for ng in valid_num_groups
                ]
                ideal = num_features ** (0.5)
                for item in infos:
                    item['heuristic'] = abs(ideal - item['ng']) * abs(ideal - item['nc'])
                chosen = sorted(infos, key=lambda x: (x['heuristic'], 1 - x['ng']))[0]
                cfg_['num_groups'] = chosen['ng']

            layer = norm_layer(num_channels=num_features, **cfg_)

        for param in layer.parameters():
            param.requires_grad = requires_grad

        return name, layer

    from mmcv import cnn as mm_cnn  # NOQA
    from mmcv.cnn.bricks import norm  # NOQA
    from mmdet.models.backbones import hrnet  # NOQA
    from mmdet.models.backbones import resnet  # NOQA
    from mmdet.models.utils import res_layer  # NOQA

    # norm.build_norm_layer = build_norm_layer_hack
    # hrnet.build_norm_layer = build_norm_layer_hack
    # resnet.build_norm_layer = build_norm_layer_hack
    # res_layer.build_norm_layer = build_norm_layer_hack
    # mm_cnn.build_norm_layer = build_norm_layer_hack

    def find_modules_with_function(func):
        import gc
        dependants = gc.get_referrers(func)
        for dependant in dependants:
            if isinstance(dependant, dict) and '__name__' in dependant:
                yield dependant

    func = mm_cnn.build_norm_layer
    for mod_dict in find_modules_with_function(func):
        if func.__name__ in mod_dict:
            mod_dict[func.__name__] = build_norm_layer_hack

    # from .models import new_backbone
    # new_backbone.build_norm_layer = build_norm_layer_hack

MMCV_MONKEY_PATCH = 1
if MMCV_MONKEY_PATCH and mmcv is not None and mmdet is not None:
    monkeypatch_build_norm_layer()


BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
# determine it based on available resources.
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit


class MM_Detector_V3(nh.layers.Module):
    """
    Wraps mm detectors. Attempt to include logic for late fusion.
    """
    __BUILTIN_CRITERION__ = True
    _mmdet_is_version_1x = False  # needed to prevent autoconvert

    def __init__(self, detector=None, classes=None, channels=None):
        super().__init__()
        self.detector = detector
        self.channels = ChannelSpec.coerce(channels)
        self.classes = kwcoco.CategoryTree.coerce(classes)
        self.coder = MM_Coder(self.classes)

    def demo_batch(self, bsize=3, h=256, w=256, with_mask=None):
        """
        Input data for testing this detector
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
        # print(type(batch['inputs']['rgb']))
        if 'img_metas' in batch and ('inputs' in batch or 'imgs' in batch):
            # already in mm_inputs format
            orig_mm_inputs = batch
        else:
            orig_mm_inputs = _batch_to_mm_inputs(batch)

        mm_inputs = orig_mm_inputs.copy()

        # Hack: remove data containers if it hasn't been done already
        from viame.pytorch import netharn as nh
        xpu = nh.XPU.from_data(self)
        mm_inputs = _ensure_unwrapped_and_mounted(mm_inputs, xpu)

        if 'inputs' not in mm_inputs:
            raise Exception('Experimental MMDet stuff requires an inputs dict')

        inputs = mm_inputs.pop('inputs')
        img_metas = mm_inputs.pop('img_metas')

        if not isinstance(inputs, dict):
            raise ValueError('expected dict mapping channel names to tensors')

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
                    trainkw['gt_masks'] = _hack_numpy_gt_masks(mm_inputs['gt_masks'])

            # Compute input normalization
            losses = self.detector(inputs, img_metas, gt_bboxes=gt_bboxes,
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
                an_input = ub.peek(inputs.values())
                bsize = an_input.shape[0]

                hack_inputs = [
                    {k: v[b:b + 1] for k, v in inputs.items()}
                    for b in range(bsize)
                ]
                # For whaver reason we cant run more than one test image at the
                # same time.
                batch_results = []
                for one_input, one_meta in zip(hack_inputs, img_metas):
                    result = self.detector([one_input], [[one_meta]],
                                                   return_loss=False)
                    batch_results.append(result)
                outputs['batch_results'] = data_containers.BatchContainer(
                    batch_results, stack=False, cpu_only=True)

        return outputs

    def _init_backbone_from_pretrained(self, filename):
        """
        Loads pretrained backbone weights
        """
        from viame.pytorch import netharn as nh
        model_state = _load_mmcv_weights(filename)

        # HACK TO ONLY INIT THE RGB PART
        if 1:
            print('hacked off init backbone from pretrained')
        else:
            print('init backbone from pretrained')
            info = nh.initializers.functional.load_partial_state(
                self.detector.backbone.chan_backbones.rgb,
                model_state, verbose=1,
                mangle=False,
                association='embedding',
                leftover='kaiming_normal',
            )
            # print('info = {}'.format(ub.repr2(info, nl=True)))
            return info


class LateFusionPyramidBackbone(nn.Module):
    """
    Wraps another backbone to perform late fusion

    Ignore:
        >>> # xdoctest: +REQUIRES(module:mmcv)
        >>> from .models.new_models_v1 import *  # NOQA
        >>> monkeypatch_build_norm_layer()
        >>> from .models.new_models_v1 import *  # NOQA
        >>> from viame.pytorch.netharn.models.mm_models import _demo_batch  # NOQA
        >>> channels = ChannelSpec.coerce('rgb,mx|my,disparity')
        >>> out_channels = [18, 36, 72, 144]
        >>> self = LateFusionPyramidBackbone(
        >>>     channels=channels, out_channels=out_channels)
        >>> # self2 = LateFusionPyramidBackbone(
        >>> #    channels=channels, out_channels=None)
        >>> batch = _demo_batch(3, channels, 256, 256, packed=True)
        >>> inputs = batch['inputs']
        >>> fused_outputs = self(inputs)
        >>> print(nh.data.data_containers.nestshape(fused_outputs))
        [torch.Size([4, 18, 64, 64]), torch.Size([4, 36, 32, 32]),
         torch.Size([4, 72, 16, 16]), torch.Size([4, 144, 8, 8])]
        >>> nh.util.number_of_parameters(self)
    """
    def __init__(self, channels='rgb', input_stats=None, fuse_method='cat',
                 out_channels=None, hack_shrink=False):
        super().__init__()
        channels = ChannelSpec.coerce(channels)
        chann_norm = channels.normalize()
        if input_stats is not None:
            assert set(input_stats.keys()) == set(chann_norm.keys())

        # norm_cfg = {'type': 'BN'}

        hack_shrink_channels = {
            # Hack to shrink model size for this type of data
            'disparity',
        }

        chan_backbones = {}
        for chan_key, chan_labels in chann_norm.items():
            if input_stats is None:
                chan_input_stats = None
            else:
                chan_input_stats = input_stats[chan_key]

            # TODO: generalize so different channels can use different
            # backbones
            if chan_key in hack_shrink_channels and hack_shrink:
                hrnet_backbone_config = {
                    'extra': {
                        'stage1': {
                            'block': 'BOTTLENECK',
                            'num_blocks': (4,),
                            'num_branches': 1,
                            'num_channels': (64,),
                            'num_modules': 1,
                        },
                        'stage2': {
                            'block': 'BASIC',
                            'num_blocks': (4, 4),
                            'num_branches': 2,
                            'num_channels': (6, 12),
                            'num_modules': 1,
                        },
                        'stage3': {
                            'block': 'BASIC',
                            'num_blocks': (4, 4, 4),
                            'num_branches': 3,
                            'num_channels':  (6, 12, 24),
                            'num_modules': 4,
                        },
                        'stage4': {
                            'block': 'BASIC',
                            'num_blocks': (4, 4, 4, 4),
                            'num_branches': 4,
                            'num_channels': (6, 12, 24, 48),  # hack, we need this to be the same for now
                            'num_modules': 3,
                        }
                    },
                    'in_channels': len(chan_labels),
                    'input_stats': chan_input_stats,
                    'norm_cfg': {'type': 'GN', 'num_groups': 'auto'},
                    'type': HRNet_V2
                }
            else:
                hrnet_backbone_config = {
                    'extra': {
                        'stage1': {
                            'block': 'BOTTLENECK',
                            'num_blocks': (4,),
                            'num_branches': 1,
                            'num_channels': (64,),
                            'num_modules': 1,
                        },
                        'stage2': {
                            'block': 'BASIC',
                            'num_blocks': (4, 4),
                            'num_branches': 2,
                            'num_channels': (18, 36),
                            'num_modules': 1,
                        },
                        'stage3': {
                            'block': 'BASIC',
                            'num_blocks': (4, 4, 4),
                            'num_branches': 3,
                            'num_channels': (18, 36, 72),
                            'num_modules': 4,
                        },
                        'stage4': {
                            'block': 'BASIC',
                            'num_blocks': (4, 4, 4, 4),
                            'num_branches': 4,
                            'num_channels': (18, 36, 72, 144),
                            'num_modules': 3,
                        }
                    },
                    'in_channels': len(chan_labels),
                    'input_stats': chan_input_stats,
                    'norm_cfg': {'type': 'GN', 'num_groups': 'auto'},
                    'type': HRNet_V2
                }
            if build_backbone is None:
                raise Exception('mmcv is not installed')
            chan_backbone = build_backbone(hrnet_backbone_config)
            chan_backbones[chan_key] = chan_backbone

        self.channels = channels
        self.chan_backbones = torch.nn.ModuleDict(chan_backbones)

        # self.chan_backbones['disparity'].out_channels
        self.fuse_method = fuse_method

        # Once channels are combined we will smooth them at each level using
        # the same 1x1 convolution
        prefused_level_shapes = ub.ddict(dict)
        for chan_key, bb in self.chan_backbones.items():
            for level, num in enumerate(bb.stage4_cfg['num_channels']):
                prefused_level_shapes[level][chan_key] = num

        level_smoothers_ = {}
        # Use given out channels, or compute them
        out_channels_ = []
        for level, chan_to_num in prefused_level_shapes.items():
            if self.fuse_method == 'cat':
                # hack, we need this to be a specific value for now
                # TODO FIXME
                t_in = sum(n for n in chan_to_num.values())
                if out_channels is None:
                    t_out = t_in
                else:
                    t_out = out_channels[level]
            elif self.fuse_method == 'sum':
                t_in = max(n for n in chan_to_num.values())
                if out_channels is None:
                    t_out = t_in
                else:
                    t_out = out_channels[level]
            else:
                raise KeyError(self.fuse_method)
            out_channels_.append(t_out)
            level_smoothers_[str(level)] = torch.nn.Conv2d(t_in, t_out, 1, 1, 0, bias=True)

        self.out_channels = out_channels_
        self.level_smoothers = torch.nn.ModuleDict(level_smoothers_)

    def forward(self, inputs):
        prefused_outputs = ub.ddict(dict)
        for chan_key in inputs.keys():
            chan_imgs = inputs[chan_key]
            chan_backbone = self.chan_backbones[chan_key]
            chan_outputs = chan_backbone(chan_imgs)
            # chan_outputs is a list for each pyramid level
            for level, lvl_out in enumerate(chan_outputs):
                prefused_outputs[level][chan_key] = lvl_out

        # prefused_shapes = ub.map_vals(lambda x: ub.map_vals( lambda y: y.shape,  x), prefused_outputs)
        # print('prefused_shapes = {}'.format(ub.repr2(prefused_shapes, nl=1)))

        fused_outputs = []
        for level, prefused in prefused_outputs.items():
            # Fuse by summing.
            # TODO: if the input streams are not aligned we should do that
            # here.
            # TODO: allow alternate late fusion schemes other than sum?
            smoother = self.level_smoothers[str(level)]
            if self.fuse_method == 'sum':
                fused = sum(prefused.values())
            if self.fuse_method == 'cat':
                # list(map(lambda x: x.shape, prefused.values()))
                fused = torch.cat(list(prefused.values()), dim=1)
            else:
                raise KeyError(self.fuse_method)

            smoothed = smoother(fused)
            fused_outputs.append(smoothed)

        return fused_outputs

    def init_weights(self, pretrained=None):
        for chan_key, chan_backbone in self.chan_backbones.items():
            chan_backbone.init_weights(pretrained=pretrained)


class MM_HRNetV2_w18_MaskRCNN(MM_Detector_V3):
    """
    SeeAlso:
        ~/.local/conda/envs/py38/lib/python3.8/site-packages/mmdet/models/detectors/base.py
        ~/.local/conda/envs/py38/lib/python3.8/site-packages/mmdet/models/detectors/two_stage.py
        ~/.local/conda/envs/py38/lib/python3.8/site-packages/mmdet/models/roi_heads/mask_heads/fcn_mask_head.py

    CommandLine:
        xdoctest -m /home/joncrall/code/bioharn/bioharn/models/new_models_v1.py MM_HRNetV2_w18_MaskRCNN

    Example:
        >>> # xdoctest: +REQUIRES(module:mmdet)
        >>> # xdoctest: +REQUIRES(module:mmcv)
        >>> # xdoctest: +REQUIRES(--cuda)
        >>> from .models.new_models_v1 import *  # NOQA
        >>> channels = ChannelSpec.coerce('rgb,mx|my')
        >>> input_stats = None
        >>> self = MM_HRNetV2_w18_MaskRCNN(classes=3, channels=channels)
        >>> batch = self.demo_batch()
        >>> import xdev
        >>> xdev.make_warnings_print_tracebacks()
        >>> from viame.pytorch import netharn as nh
        >>> print(nh.util.number_of_parameters(self))
        >>> self.to(0)
        >>> batch = self.demo_batch()
        >>> print('batch = {!r}'.format(batch))
        >>> outputs = self(batch)
        >>> batch_dets = self.coder.decode_batch(outputs)
        >>> print('batch_dets = {!r}'.format(batch_dets))

    Example:
        >>> # xdoctest: +REQUIRES(module:mmdet)
        >>> # xdoctest: +REQUIRES(--cuda)
        >>> from .models.new_models_v1 import *  # NOQA
        >>> channels = ChannelSpec.coerce('rgb')
        >>> input_stats = None
        >>> self = MM_HRNetV2_w18_MaskRCNN(classes=3, channels=channels)
        >>> batch = self.demo_batch()
        >>> import xdev
        >>> xdev.make_warnings_print_tracebacks()
        >>> from viame.pytorch import netharn as nh
        >>> print(nh.util.number_of_parameters(self))
        >>> self.to(0)
        >>> batch = self.demo_batch()
        >>> print('batch = {!r}'.format(batch))
        >>> outputs = self(batch)
        >>> batch_dets = self.coder.decode_batch(outputs)
        >>> print('batch_dets = {!r}'.format(batch_dets))

    Example:
        >>> # xdoctest: +REQUIRES(module:mmdet)
        >>> # xdoctest: +REQUIRES(--cuda)
        >>> from .models.new_models_v1 import *  # NOQA
        >>> channels = ChannelSpec.coerce('rgb')
        >>> input_stats = None
        >>> self = MM_HRNetV2_w18_MaskRCNN(classes=3, channels=channels, with_mask=False)
        >>> print('self.detector.with_mask = {!r}'.format(self.detector.with_mask))
    """
    pretrained_url = 'open-mmlab://msra/hrnetv2_w18'

    def __init__(self, classes=None, input_stats=None, channels='rgb',
                 with_mask=True, fuse_method='sum', hack_shrink=False):
        classes = kwcoco.CategoryTree.coerce(classes)
        channels = ChannelSpec.coerce(channels)

        # ensure torch-liberator takes the monkey patch
        monkeypatch_build_norm_layer()

        rpn_head_v1 = {
            'anchor_generator': {
                'ratios': [0.5, 1.0, 2.0],
                'scales': [8],
                'strides': [4, 8, 16, 32, 64],
                'type': 'AnchorGenerator'
            },
            'bbox_coder': {
                'target_means': [0.0, 0.0, 0.0, 0.0],
                'target_stds': [1.0, 1.0, 1.0, 1.0],
                'type': 'DeltaXYWHBBoxCoder'
            },
            'feat_channels': 256,
            'in_channels': 256,
            'loss_bbox': {'loss_weight': 1.0, 'type': 'L1Loss'},
            'loss_cls': {'loss_weight': 1.0, 'type': 'CrossEntropyLoss', 'use_sigmoid': True},
            'type': 'RPNHead'
        }

        rpn_train_cfg_v1 = {
            'allowed_border': -1,
            'assigner': {
                'ignore_iof_thr': -1,
                'match_low_quality': True,
                'min_pos_iou': 0.3,
                'neg_iou_thr': 0.3,
                'pos_iou_thr': 0.7,
                'type': 'MaxIoUAssigner'},
            'debug': False,
            'pos_weight': -1,
            'sampler': {
                'add_gt_as_proposals': False,
                'neg_pos_ub': -1,
                'num': 256,
                'pos_fraction': 0.5,
                'type': 'RandomSampler'}
        }

        # rpn_head_v2 = dict(
        #     # _delete_=True,
        #     type='GARPNHead',
        #     in_channels=256,
        #     feat_channels=256,
        #     approx_anchor_generator=dict(
        #         type='AnchorGenerator',
        #         octave_base_scale=8,
        #         scales_per_octave=3,
        #         ratios=[0.5, 1.0, 2.0],
        #         strides=[4, 8, 16, 32, 64]),
        #     square_anchor_generator=dict(
        #         type='AnchorGenerator',
        #         ratios=[1.0],
        #         scales=[8],
        #         strides=[4, 8, 16, 32, 64]),
        #     anchor_coder=dict(
        #         type='DeltaXYWHBBoxCoder',
        #         target_means=[.0, .0, .0, .0],
        #         target_stds=[0.07, 0.07, 0.14, 0.14]),
        #     bbox_coder=dict(
        #         type='DeltaXYWHBBoxCoder',
        #         target_means=[.0, .0, .0, .0],
        #         target_stds=[0.07, 0.07, 0.11, 0.11]),
        #     loc_filter_thr=0.01,
        #     loss_loc=dict(
        #         type='FocalLoss',
        #         use_sigmoid=True,
        #         gamma=2.0,
        #         alpha=0.25,
        #         loss_weight=1.0),
        #     loss_shape=dict(type='BoundedIoULoss', beta=0.2, loss_weight=1.0),
        #     loss_cls=dict(
        #         type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        #     loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))

        # rpn_train_cfg_v2 = dict(
        #     ga_assigner=dict(
        #         type='ApproxMaxIoUAssigner',
        #         pos_iou_thr=0.7,
        #         neg_iou_thr=0.3,
        #         min_pos_iou=0.3,
        #         ignore_iof_thr=-1),
        #     assigner=dict(
        #         type='ApproxMaxIoUAssigner',
        #         pos_iou_thr=0.7,
        #         neg_iou_thr=0.3,
        #         min_pos_iou=0.3,
        #         ignore_iof_thr=-1),
        #     ga_sampler=dict(
        #         type='RandomSampler',
        #         num=256,
        #         pos_fraction=0.5,
        #         neg_pos_ub=-1,
        #         add_gt_as_proposals=False),
        #     allowed_border=-1,
        #     center_ratio=0.2,
        #     ignore_ratio=0.5)

        rpn_train_cfg = rpn_train_cfg_v1
        rpn_head = rpn_head_v1

        if with_mask:
            mask_head = {
                'conv_out_channels': 256,
                'in_channels': 256,
                'loss_mask': {'loss_weight': 1.0, 'type': 'CrossEntropyLoss', 'use_mask': True},
                'classes': classes,
                'num_convs': 4,
                'type': FCNMaskHead_V2,
                'norm_cfg': {'type': 'GN', 'num_groups': 32},
            }
        else:
            mask_head = None
        from mmdet.models.builder import build_backbone

        default_args = mmcv.Config({
            'test_cfg': {
                'rcnn': {
                    'mask_thr_binary': 0.5,
                    'max_per_img': 100,
                    'nms': {'iou_threshold': 0.5, 'type': 'nms'},
                    'score_thr': 0.05
                },
                'rpn': {'max_num': 1000, 'min_bbox_size': 0,
                        'nms_across_levels': False, 'nms_post': 1000,
                        'nms_pre': 1000, 'nms_thr': 0.7}
            },
            'train_cfg': {
                'rpn': rpn_train_cfg,
                'rpn_proposal': {
                    'max_num': 1000,
                    'min_bbox_size': 0,
                    'nms_across_levels': False,
                    'nms_post': 1000,
                    'nms_pre': 2000,
                    'nms_thr': 0.7
                },
                'rcnn': {
                    'assigner': {
                        'ignore_iof_thr': -1,
                        'match_low_quality': True,
                        'min_pos_iou': 0.5,
                        'neg_iou_thr': 0.5,
                        'pos_iou_thr': 0.5,
                        'type': 'MaxIoUAssigner'},
                    'debug': False,
                    'mask_size': 28,
                    'pos_weight': -1,
                    'sampler': {
                        'add_gt_as_proposals': True,
                        'neg_pos_ub': -1,
                        'num': 512,
                        'pos_fraction': 0.25,
                        'type': 'RandomSampler'
                    }
                },
            }
        })

        backbone_cfg = {
            'channels': channels,
            'input_stats': input_stats,
            'fuse_method': fuse_method,
            'out_channels': [18, 36, 72, 144],
            'hack_shrink': hack_shrink,
            'type': LateFusionPyramidBackbone
        }
        backbone = build_backbone(backbone_cfg)

        mm_cfg = mmcv.Config({
            'model': {
                'backbone': {
                    'instance': backbone,
                },
                'neck': {
                    'in_channels': backbone_cfg['out_channels'],
                    'out_channels': 256,
                    'type': HRFPN_V2,
                    'norm_cfg': {'type': 'GN', 'num_groups': 32},
                },
                'rpn_head': rpn_head,
                'roi_head': {
                    'bbox_roi_extractor': {
                        'featmap_strides': [4, 8, 16, 32],
                        'out_channels': 256,
                        'roi_layer': {'output_size': 7, 'sampling_ratio': 0, 'type': 'RoIAlign'},
                        'type': 'SingleRoIExtractor'
                    },
                    'mask_roi_extractor': {
                        'featmap_strides': [4, 8, 16, 32],
                        'out_channels': 256,
                        'roi_layer': {'output_size': 14, 'sampling_ratio': 0, 'type': 'RoIAlign'},
                        'type': 'SingleRoIExtractor'
                    },
                    'bbox_head': {
                        'bbox_coder': {
                            'target_means': [0.0, 0.0, 0.0, 0.0],
                            'target_stds': [0.1, 0.1, 0.2, 0.2],
                            'type': 'DeltaXYWHBBoxCoder'
                        },
                        'fc_out_channels': 1024,
                        'in_channels': 256,
                        'loss_bbox': {'loss_weight': 1.0, 'type': 'L1Loss'},
                        'loss_cls': {'loss_weight': 1.0, 'type': 'CrossEntropyLoss', 'use_sigmoid': False},
                        'classes': classes,
                        'reg_class_agnostic': False,
                        'roi_feat_size': 7,
                        'norm_cfg': {'type': 'GN', 'num_groups': 32},
                        'type': Shared2FCBBoxHead_V2,
                    },
                    'mask_head': mask_head,
                    'type': StandardRoIHead_V2,
                },
                'pretrained': None,
                'type': MaskRCNN_V2,
            },
        })

        from packaging.version import parse as Version
        import mmdet

        MMDET_GT_2_12 = Version(mmdet.__version__) >= Version('2.12.0')

        if MMDET_GT_2_12:
            # mmdet v2.12.0 introduced new registry stuff that forces use of
            # config dictionaries
            mm_cfg = mmcv.ConfigDict(mm_cfg)

        from mmdet.models import build_detector
        detector = build_detector(
            mm_cfg['model'], train_cfg=default_args['train_cfg'],
            test_cfg=default_args['test_cfg'])

        if MMDET_GT_2_12:
            detector.init_weights()

        super().__init__(detector, classes=classes, channels=channels)
