from os.path import exists
from os.path import join
import scriptconfig as scfg
import re
import tempfile
import torch
from collections import OrderedDict
import ubelt as ub
import copy
import json
import inspect
import six
import textwrap


class UpgradeMMDetConfig(scfg.Config):
    default = {
        'deployed': scfg.Path(None, help='path to torch_liberator zipfile to convert'),
        'use_cache': scfg.Path(False, help='do nothing if we already converted'),
        'out_dpath': scfg.Path(None, help='place to write the new model to. (uses a cache directory if unspecified)'),
    }


def get_func_sourcecode(func, strip_def=False):
    """
    wrapper around inspect.getsource but takes into account utool decorators
    strip flags are very hacky as of now

    Example:
        >>> # build test data
        >>> func = get_func_sourcecode
        >>> strip_def = True
        >>> sourcecode = get_func_sourcecode(func, strip_def)
        >>> print('sourcecode = {}'.format(sourcecode))
    """
    inspect.linecache.clearcache()  # HACK: fix inspect bug
    sourcefile = inspect.getsourcefile(func)
    if sourcefile is not None and (sourcefile != '<string>'):
        try_limit = 2
        for num_tries in range(try_limit):
            try:
                sourcecode = inspect.getsource(func)
                if not isinstance(sourcecode, six.text_type):
                    sourcecode = sourcecode.decode('utf-8')
            except (IndexError, OSError, SyntaxError):
                print('WARNING: Error getting source')
                inspect.linecache.clearcache()
                if num_tries + 1 != try_limit:
                    tries_left = try_limit - num_tries - 1
                    print('Attempting %d more time(s)' % (tries_left))
                else:
                    raise
    else:
        sourcecode = None
    if strip_def:
        # hacky
        # TODO: use redbaron or something like that for a more robust appraoch
        REGEX_NONGREEDY = '*?'
        sourcecode = textwrap.dedent(sourcecode)
        regex_decor = '^@.' + REGEX_NONGREEDY
        regex_defline = '^def [^:]*\\):\n'
        patern = '(' + regex_decor + ')?' + regex_defline
        RE_FLAGS = re.MULTILINE | re.DOTALL
        RE_KWARGS = {'flags': RE_FLAGS}
        nodef_source = re.sub(patern, '', sourcecode, **RE_KWARGS)
        sourcecode = textwrap.dedent(nodef_source)
        pass
    return sourcecode


def upgrade_deployed_mmdet_model(config):
    """

    CLI:
        python -m bioharn.compat.upgrade_mmdet_model \
            --deployed=/home/joncrall/.cache/bioharn/deploy_MM_CascadeRCNN_myovdqvi_035_MVKVVR_fix3.zip

    Example:
        >>> # xdoctest: +REQUIRES(--slow)
        >>> # xdoctest: +REQUIRES(module:mmdet)
        >>> from .compat.upgrade_mmdet_model import *  # NOQA
        >>> from .util.util_girder import grabdata_girder
        >>> api_url = 'https://data.kitware.com/api/v1'
        >>> file_id = '5dd3eb8eaf2e2eed3508d604'
        >>> old_fpath = grabdata_girder(api_url, file_id)
        >>> config = {'deployed': old_fpath}
        >>> new_fpath = upgrade_deployed_mmdet_model(config)
        >>> print('new_fpath = {!r}'.format(new_fpath))

    Ignore:
        import xinspect
        # Close xinspect so we dont need to depend on it
        import liberator
        closer = liberator.Closer()
        closer.add_dynamic(xinspect.dynamic_kwargs.get_func_sourcecode)
        print(closer.current_sourcecode())

    Ignore:
        5dd3181eaf2e2eed3505827c
        girder-client --api-url https://data.kitware.com/api/v1 list 5dd3eb8eaf2e2eed3508d604
        girder-client --api-url https://data.kitware.com/api/v1 list 5dd3181eaf2e2eed3505827c

        girder-client --api-url https://data.kitware.com/api/v1 download 5eb9c21f9014a6d84e638b49 $HOME/tmp/deploy_MM_CascadeRCNN_rgb-fine-coi-v40_ntjzrxlb_007_FVMWBU.zip

        girder-client --api-url https://data.kitware.com/api/v1 download 5dd3eb8eaf2e2eed3508d604 $HOME/tmp/deploy_MM_CascadeRCNN_myovdqvi_035_MVKVVR_fix3.zip
        girder-client --api-url https://data.kitware.com/api/v1 download 5dd3eb8eaf2e2eed3508d604 $HOME/tmp/deploy_MM_CascadeRCNN_myovdqvi_035_MVKVVR_fix3.zip
    """
    from torch_liberator import deployer
    import ndsampler
    from viame.pytorch import netharn as nh
    from ..detect_predict import setup_module_aliases
    from ..detection_models import mm_models

    # Set up module aliases for backwards compatibility with old models
    setup_module_aliases()

    config = UpgradeMMDetConfig(config)

    deploy_fpath = config['deployed']

    if config['out_dpath'] is None:
        extract_dpath = ub.ensure_app_cache_dir('torch_liberator/extracted')
    else:
        extract_dpath = config['out_dpath']

    new_name = ub.augpath(deploy_fpath, dpath='', suffix='_mm2x')
    new_fpath = join(extract_dpath, new_name)

    print('Upgrade deployed model config: config = {!r}'.format(config))

    if config['use_cache']:
        if exists(new_fpath):
            print('Returning cached new_fpath = {!r}'.format(new_fpath))
            return new_fpath

    deployed = deployer.DeployedModel(deploy_fpath)

    print('Extracting old snapshot to: {}'.format(extract_dpath))
    temp_fpath = deployed.extract_snapshot(extract_dpath)

    model_cls, model_initkw = deployed.model_definition()
    old_classes = ndsampler.CategoryTree.coerce(model_initkw['classes'])
    num_classes = len(old_classes)

    # old mmdet has background as class 0, new has it as class K
    # https://mmdetection.readthedocs.io/en/latest/compatibility.html#codebase-conventions
    if 'background' in old_classes:
        num_classes_old = num_classes - 1
        new_classes = ndsampler.CategoryTree.from_mutex(list((ub.oset(list(old_classes)) - {'background'}) | {'background'}), bg_hack=False)
    else:
        num_classes_old = num_classes
        new_classes = old_classes

    # model_src = print(inspect.getsource(model_cls.__init__))
    # import xinspect
    # model_src = xinspect.dynamic_kwargs.get_func_sourcecode(model_cls.__init__, strip_def=True)
    model_src = get_func_sourcecode(model_cls.__init__, strip_def=True)

    xpu = nh.XPU.coerce('cpu')
    old_snapshot = xpu.load(temp_fpath)
    # Extract just the model state
    model_state = old_snapshot['model_state_dict']

    model_state_2 = {k.replace('module.detector.', ''): v for k, v in model_state.items()}

    # These are handled by the initkw
    model_state_2.pop('module.input_norm.mean', None)
    model_state_2.pop('module.input_norm.std', None)

    # Add major hacks to the config string to attempt to re-create what mmdet can handle
    config_strings = model_src.replace('mm_config', 'model')
    config_strings = 'from viame.pytorch.netharn.data.channel_spec import ChannelSpec\n' + config_strings
    config_strings = 'import ubelt as ub\n' + config_strings
    config_strings = 'classes = {!r}\n'.format(list(old_classes)) + config_strings
    if 'in_channels' in model_initkw:
        config_strings = 'in_channels = {!r}\n'.format(model_initkw['in_channels']) + config_strings
    if 'channels' in model_initkw:
        config_strings = 'channels = {!r}\n'.format(model_initkw['channels']) + config_strings
    if 'input_stats' in model_initkw:
        config_strings = 'input_stats = {!r}\n'.format(model_initkw['input_stats']) + config_strings
    config_strings = config_strings[:config_strings.find('_hack_mm_backbone_in_channels')]
    config_strings = config_strings.replace('self.', '')

    import re
    # UserWarning: "out_size" is deprecated in `RoIAlign.__init__`, please use "output_size" instead
    # UserWarning: "sample_num" is deprecated in `RoIAlign.__init__`, please use "sampling_ratio" instead
    #UserWarning: "iou_thr" is deprecated in `nms`, please use "iou_threshold" instead

    from functools import partial
    def careful_replace(match, repl, requires=None):
        # Only replace the match if the context line contains the required text
        start, stop = match.span()
        # Look back to find the context line
        prev_nl = match.string[:start][::-1].find('\n')
        prev_text = match.string[start - prev_nl:start]
        if requires is None or requires in prev_text:
            return repl
        else:
            # Return original text
            return match.string[start:stop]

    def whole_word(regex):
        return r'\b{}\b'.format(regex)

    # This actually doesnt matter because we will extract the new model from
    # the bioharn.models.mm_models source dir
    new = re.sub(whole_word('iou_thr'), partial(careful_replace, repl='iou_threshold', requires='nms'), config_strings)
    new = re.sub(whole_word('sample_num'), partial(careful_replace, repl='sampling_ratio', requires='RoIAlign'), new)
    new = re.sub(whole_word('out_size'), partial(careful_replace, repl='output_size', requires='RoIAlign'), new)
    config_strings = new

    print(config_strings)

    checkpoint = {
        'state_dict': model_state_2,
        'meta': {
            # hack in mmdet metadata
            'mmdet_version': '1.0.0',
            'config': config_strings,
        },
    }

    config_strings = checkpoint['meta']['config']
    in_file = ub.augpath(temp_fpath, suffix='_prepared')
    torch.save(checkpoint, in_file)

    # checkpoint = torch.load(in_file, weights_only=False)
    out_file = ub.augpath(temp_fpath, suffix='_upgrade2x')

    print('num_classes_old = {!r}'.format(num_classes_old))
    print('new_classes = {!r}'.format(new_classes))
    convert(in_file, out_file, num_classes_old + 1)

    input_stats = model_initkw['input_stats']
    new_initkw = dict(classes=new_classes.__json__(), channels='rgb', input_stats=input_stats)
    new_model = mm_models.MM_CascadeRCNN(**new_initkw)
    new_model._initkw = new_initkw

    new_model_state = torch.load(out_file, weights_only=False)

    # print(new_model.detector.roi_head.bbox_head[0].fc_cls.weight.shape)
    # print(model_state_2['bbox_head.0.fc_cls.weight'].shape)
    # print(new_model_state['state_dict']['roi_head.bbox_head.0.fc_cls.weight'].shape)
    _ = new_model.detector.load_state_dict(new_model_state['state_dict'])

    TEST_FORWARD = 0
    if TEST_FORWARD:
        batch = {
            'inputs': {
                'rgb': torch.rand(1, 3, 256, 256),
            }
        }
        outputs = new_model.forward(batch, return_loss=False)
        batch_dets = new_model.coder.decode_batch(outputs)
        dets = batch_dets[0]
        print('dets = {!r}'.format(dets))

    new_train_info = copy.deepcopy(deployed.train_info())
    new_train_info['__mmdet_conversion__'] = '1x_to_2x'

    new_train_info_fpath = join(extract_dpath, 'train_info.json')
    new_snap_fpath = join(extract_dpath, 'converted_deploy_snapshot.pt')
    with open(new_train_info_fpath, 'w') as file:
        json.dump(new_train_info, file, indent='    ')

    new_snapshot = {
        'model_state_dict': new_model.state_dict(),
        # {'detector.' + k: v for k, v in new_model_state['state_dict'].items()},
        'epoch': old_snapshot['epoch'],
        '__mmdet_conversion__': '1x_to_2x',
    }
    torch.save(new_snapshot, new_snap_fpath)

    new_deployed = deployer.DeployedModel.custom(
        model=new_model, snap_fpath=new_snap_fpath,
        train_info_fpath=new_train_info_fpath, initkw=new_initkw)

    new_name = ub.augpath(deploy_fpath, dpath='', suffix='_mm2x')
    fpath = new_deployed.package(dpath=extract_dpath, name=new_name)
    print('fpath = {!r}'.format(fpath))
    return fpath


def main():
    config = UpgradeMMDetConfig(cmdline=True)
    upgrade_deployed_mmdet_model(config)


##################################################################
# The following code is copied from mmdetection because it is not
# generically importable as a python module.
# See:
# https://github.com/open-mmlab/mmdetection/blob/master/tools/upgrade_model_version.py


def is_head(key):
    valid_head_list = [
        'bbox_head', 'mask_head', 'semantic_head', 'grid_head', 'mask_iou_head'
    ]

    return any(key.startswith(h) for h in valid_head_list)


def parse_config(config_strings):
    from mmcv import Config  # NOQA
    temp_file = tempfile.NamedTemporaryFile()
    config_path = f'{temp_file.name}.py'
    with open(config_path, 'w') as f:
        f.write(config_strings)

    config = Config.fromfile(config_path)
    is_two_stage = True
    is_ssd = False
    is_retina = False
    reg_cls_agnostic = False
    if 'rpn_head' not in config.model:
        is_two_stage = False
        # check whether it is SSD
        if config.model.bbox_head.type == 'SSDHead':
            is_ssd = True
        elif config.model.bbox_head.type == 'RetinaHead':
            is_retina = True
    elif isinstance(config.model['bbox_head'], list):
        reg_cls_agnostic = True
    elif 'reg_class_agnostic' in config.model.bbox_head:
        reg_cls_agnostic = config.model.bbox_head \
            .reg_class_agnostic
    temp_file.close()
    return is_two_stage, is_ssd, is_retina, reg_cls_agnostic


def reorder_cls_channel(val, num_classes=81):
    # bias
    if val.dim() == 1:
        new_val = torch.cat((val[1:], val[:1]), dim=0)
    # weight
    else:
        out_channels, in_channels = val.shape[:2]
        # conv_cls for softmax output
        if out_channels != num_classes and out_channels % num_classes == 0:
            new_val = val.reshape(-1, num_classes, in_channels, *val.shape[2:])
            new_val = torch.cat((new_val[:, 1:], new_val[:, :1]), dim=1)
            new_val = new_val.reshape(val.size())
        # fc_cls
        elif out_channels == num_classes:
            new_val = torch.cat((val[1:], val[:1]), dim=0)
        # agnostic | retina_cls | rpn_cls
        else:
            new_val = val

    return new_val


def truncate_cls_channel(val, num_classes=81):

    # bias
    if val.dim() == 1:
        if val.size(0) % num_classes == 0:
            new_val = val[:num_classes - 1]
        else:
            new_val = val
    # weight
    else:
        out_channels, in_channels = val.shape[:2]
        # conv_logits
        if out_channels % num_classes == 0:
            new_val = val.reshape(num_classes, in_channels, *val.shape[2:])[1:]
            new_val = new_val.reshape(-1, *val.shape[1:])
        # agnostic
        else:
            new_val = val

    return new_val


def truncate_reg_channel(val, num_classes=81):
    # bias
    if val.dim() == 1:
        # fc_reg|rpn_reg
        if val.size(0) % num_classes == 0:
            new_val = val.reshape(num_classes, -1)[:num_classes - 1]
            new_val = new_val.reshape(-1)
        # agnostic
        else:
            new_val = val
    # weight
    else:
        out_channels, in_channels = val.shape[:2]
        # fc_reg|rpn_reg
        if out_channels % num_classes == 0:
            new_val = val.reshape(num_classes, -1, in_channels,
                                  *val.shape[2:])[1:]
            new_val = new_val.reshape(-1, *val.shape[1:])
        # agnostic
        else:
            new_val = val

    return new_val


def convert(in_file, out_file, num_classes):
    """Convert keys in checkpoints.

    There can be some breaking changes during the development of mmdetection,
    and this tool is used for upgrading checkpoints trained with old versions
    to the latest one.
    """
    checkpoint = torch.load(in_file, weights_only=False)
    in_state_dict = checkpoint.pop('state_dict')
    out_state_dict = OrderedDict()
    meta_info = checkpoint['meta']
    is_two_stage, is_ssd, is_retina, reg_cls_agnostic = parse_config(
        meta_info['config'])
    if meta_info['mmdet_version'] <= '0.5.3' and is_retina:
        upgrade_retina = True
    else:
        upgrade_retina = False

    for key, val in in_state_dict.items():
        new_key = key
        new_val = val
        if is_two_stage and is_head(key):
            new_key = 'roi_head.{}'.format(key)

        # classification
        m = re.search(
            r'(conv_cls|retina_cls|rpn_cls|fc_cls|fcos_cls|'
            r'fovea_cls).(weight|bias)', new_key)
        if m is not None:
            print(f'reorder cls channels of {new_key}')
            new_val = reorder_cls_channel(val, num_classes)

        # regression
        m = re.search(r'(fc_reg|rpn_reg).(weight|bias)', new_key)
        if m is not None and not reg_cls_agnostic:
            print(f'truncate regression channels of {new_key}')
            new_val = truncate_reg_channel(val, num_classes)

        # mask head
        m = re.search(r'(conv_logits).(weight|bias)', new_key)
        if m is not None:
            print(f'truncate mask prediction channels of {new_key}')
            new_val = truncate_cls_channel(val, num_classes)

        m = re.search(r'(cls_convs|reg_convs).\d.(weight|bias)', key)
        # Legacy issues in RetinaNet since V1.x
        # Use ConvModule instead of nn.Conv2d in RetinaNet
        # cls_convs.0.weight -> cls_convs.0.conv.weight
        if m is not None and upgrade_retina:
            param = m.groups()[1]
            new_key = key.replace(param, f'conv.{param}')
            out_state_dict[new_key] = val
            print(f'rename the name of {key} to {new_key}')
            continue

        m = re.search(r'(cls_convs).\d.(weight|bias)', key)
        if m is not None and is_ssd:
            print(f'reorder cls channels of {new_key}')
            new_val = reorder_cls_channel(val, num_classes)

        out_state_dict[new_key] = new_val
    checkpoint['state_dict'] = out_state_dict
    torch.save(checkpoint, out_file)

#################################################################


if __name__ == '__main__':
    """
    CommandLine:
        python -m bioharn.compat.upgrade_mmdet_model
    """
    main()
