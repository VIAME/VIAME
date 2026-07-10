"""
Logic for predicting on a new dataset using a trained detector.

Notes:

    https://data.kitware.com/#collection/58b747ec8d777f0aef5d0f6a

    source $HOME/internal/secrets

    girder-client --api-url https://data.kitware.com/api/v1 list 58b747ec8d777f0aef5d0f6a

    girder-client --api-url https://data.kitware.com/api/v1 list 58c49f668d777f0aef5d7960

    girder-client --api-url https://data.kitware.com/api/v1 list 5c423a5f8d777f072b0ba58f

    girder-client --api-url https://data.kitware.com/api/v1 list 5dd3181eaf2e2eed3505827c

    girder-client --api-url https://data.kitware.com/api/v1 list 5aac22638d777f068578d53c --columns=id,type,name

    girder-client --api-url https://data.kitware.com/api/v1 list 5dd3eb8eaf2e2eed3508d604

    girder-client --api-url https://data.kitware.com/api/v1 download 5dd3eb8eaf2e2eed3508d604

"""
from os.path import join, dirname, basename, isfile, exists
import ubelt as ub
import torch.utils.data as torch_data
from viame.pytorch import netharn as nh
import numpy as np
import torch
import scriptconfig as scfg
import kwimage
import kwarray
import warnings
import kwcoco
import torch_liberator
from viame.pytorch.netharn.data.channel_spec import ChannelSpec
from viame.pytorch.netharn.data.data_containers import ContainerXPU
import os


try:
    import xdev
    profile = xdev.profile
except Exception:
    profile = ub.identity


ENABLE_BIOHARN_WARNINGS = os.environ.get('ENABLE_BIOHARN_WARNINGS', '')


class DetectPredictConfig(scfg.Config):
    default = {

        'deployed': None,
        'batch_size': 4,
        'xpu': 'auto',

        'window_dims': scfg.Value('native', help='size of a sliding window'),  # (512, 512),
        'input_dims': scfg.Value('native', help='The size of the inputs to the network'),

        'workers': 0,

        'window_overlap': scfg.Value(0.0, help='overlap of the sliding window'),

        'channels': scfg.Value(
            'native', type=str,
            help='list of channels needed by the model. '
            'Typically this can be inferred from the model'),

        # Note: these dont work exactly correct due to mmdetection model
        # differences
        'nms_thresh': 0.4,
        'conf_thresh': 0.1,

        'skip_upgrade': scfg.Value(False, help='if true skips upgrade model checks'),

        'verbose': 1,
    }


def patch_numpy():
    import numpy as np
    np.bool = bool
    np.int = int
    np.float = float
    np.str = str
    if np.lib.NumpyVersion(np.__version__) >= '2.0.0b1':
        np.string_ = np.bytes_
        np.unicode_ = np.str_
        np.Inf = np.inf


def setup_module_aliases():
    """
    Set up module aliases for backwards compatibility with old models.

    Old models may have been trained with the standalone 'netharn' or 'bioharn'
    packages, which have now been merged into 'viame.pytorch.netharn'. This
    function creates aliases in sys.modules so that imports like:
        - import netharn
        - from netharn.data import data_containers
        - import bioharn
        - from bioharn.models import mm_models

    Will resolve to the corresponding modules under viame.pytorch.netharn.

    This function is idempotent and safe to call multiple times.
    """
    import sys
    from viame.pytorch import netharn as nh

    # Define the mapping of old module names to new modules
    # netharn -> viame.pytorch.netharn
    # bioharn -> viame.pytorch.netharn (bioharn was merged into netharn)
    alias_mappings = {
        # Top-level aliases
        'netharn': nh,
        'bioharn': nh,
        # netharn submodules
        'netharn.data': nh.data,
        'netharn.data.channel_spec': nh.data.channel_spec,
        'netharn.data.data_containers': nh.data.data_containers,
        'netharn.util': nh.util,
        'netharn.models': nh.models,
        'netharn.layers': nh.layers,
        'netharn.initializers': nh.initializers,
        'netharn.criterions': nh.criterions,
        'netharn.schedulers': nh.schedulers,
        'netharn.optimizers': nh.optimizers,
        # bioharn submodules (map to netharn equivalents)
        'bioharn.models': nh.detection_models,
        'bioharn.models.mm_models': nh.detection_models.mm_models,
        'bioharn.detect_predict': nh.detect_predict,
        'bioharn.detect_fit': nh.detect_fit,
        'bioharn.detect_dataset': nh.detect_dataset,
        'bioharn.clf_predict': nh.clf_predict,
        'bioharn.clf_fit': nh.clf_fit,
        'bioharn.clf_dataset': nh.clf_dataset,
    }

    for old_name, new_module in alias_mappings.items():
        if old_name not in sys.modules:
            sys.modules[old_name] = new_module


def _ensure_upgraded_model(deployed_fpath):
    """
    Example:
        >>> # xdoctest: +REQUIRES(module:mmdet)
        >>> from .detect_predict import _ensure_upgraded_model
        >>> deployed_fpath = deployed_fpath1 = ub.grabdata(
        >>>     'https://data.kitware.com/api/v1/file/5dd3eb8eaf2e2eed3508d604/download',
        >>>     fname='deploy_MM_CascadeRCNN_myovdqvi_035_MVKVVR_fix3.zip',
        >>>     appname='viame', hasher='sha512',
        >>>     hash_prefix='22a1eeb18c9e5706f6578e66abda1e97a88eee5', verbose=0)
        >>> ensured_fpath1 = _ensure_upgraded_model(deployed_fpath1)
        >>> #
        >>> deployed_fpath = deployed_fpath2 = ub.grabdata(
        >>>     'https://data.kitware.com/api/v1/file/5ee93f639014a6d84ec52b7f/download',
        >>>     fname='deploy_MM_CascadeRCNN_myovdqvi_035_MVKVVR_fix3_mm2x.zip',
        >>>     appname='viame', hasher='sha512',
        >>>     hash_prefix='63b7c3981b3446b079c1d83541a5666c496f6148', verbose=0)
        >>> ensured_fpath2 = _ensure_upgraded_model(deployed_fpath2)

    """
    # Set up module aliases for backwards compatibility with old models
    # that use 'import netharn' or 'import bioharn' instead of 'viame.pytorch.netharn'
    setup_module_aliases()

    if deployed_fpath:
        print('Ensure upgraded model: deployed_fpath = {!r}'.format(deployed_fpath))

    if not exists(deployed_fpath):
        # Case where old model was upgraded and may have been deleted
        # This logic is a bit hacky, could be refactored to be more robust.
        upgrade_candidates = [
            ub.augpath(deployed_fpath, suffix='_bio3x')
        ]
        candidates = [c for c in upgrade_candidates if exists(c)]
        if candidates:
            deployed_fpath = candidates[0]

    if isinstance(deployed_fpath, str) and exists(deployed_fpath) and deployed_fpath.endswith('.pt'):
        # hack
        return deployed_fpath

    deployed = torch_liberator.DeployedModel.coerce(deployed_fpath)

    # Hueristic to determine if the model needs update or not
    needs_update = False
    if 'model_fpath' in deployed.info:
        with ub.zopen(deployed.info['model_fpath'], 'r') as file:
            topology_text = file.read()
            if 'MM_Detector' in topology_text and not '_force_no_upgrade_' in topology_text:
                if '_mmdet_is_version_1x' not in topology_text:
                    needs_update = 'to_2x'
                else:
                    # Super hack to "parse" out the bioharn model version
                    import re
                    match = re.search(r'^\W*__bioharn_model_vesion__\W*=\W*(\d+)',
                                      topology_text, flags=re.MULTILINE)
                    need_version = 3
                    if match:
                        found = match.groups()[0]
                        found_version = int(found)
                    else:
                        found_version = 2

                    if found_version < need_version:
                        needs_update = 'to_latest'

    print('needs_update = {!r}'.format(needs_update))
    if needs_update == 'to_2x':
        from .compat.upgrade_mmdet_model import upgrade_deployed_mmdet_model
        ensured_fpath = upgrade_deployed_mmdet_model({
            'deployed': deployed_fpath, 'use_cache': True,
            'out_dpath': dirname(deployed_fpath),
        })
    elif needs_update == 'to_latest':
        from .compat.update_bioharn_model import update_deployed_bioharn_model
        ensured_fpath = update_deployed_bioharn_model({
            'deployed': deployed_fpath, 'use_cache': True,
            'out_dpath': dirname(deployed_fpath),
        })
    else:
        ensured_fpath = deployed_fpath
    return ensured_fpath


class DetectPredictor(object):
    """
    A detector API for bioharn trained models


    Example:
        >>> # xdoctest: +REQUIRES(--network)
        >>> # xdoctest: +REQUIRES(module:mmdet)
        >>> import ubelt as ub
        >>> from .detect_predict import *  # NOQA
        >>> deployed_fpath = ub.grabdata(
        >>>     'https://data.kitware.com/api/v1/file/5dd3eb8eaf2e2eed3508d604/download',
        >>>     fname='deploy_MM_CascadeRCNN_myovdqvi_035_MVKVVR_fix3.zip',
        >>>     appname='viame', hasher='sha512',
        >>>     hash_prefix='22a1eeb18c9e5706f6578e66abda1e97a88eee5')
        >>> image_fpath = ub.grabdata(
        >>>     'https://data.kitware.com/api/v1/file/5dcf0d1faf2e2eed35fad5d1/download',
        >>>     fname='scallop.jpg', appname='viame', hasher='sha512',
        >>>     hash_prefix='3bd290526c76453bec7')
        >>> rgb = kwimage.imread(image_fpath)
        >>> inputs = {'rgb': rgb}
        >>> config = dict(
        >>>     deployed=deployed_fpath,
        >>> )
        >>> predictor = DetectPredictor(config)
        >>> final = predictor.predict(inputs)
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(inputs['rgb'], doclf=True)
        >>> final2 = final.compress(final.scores > .0)
        >>> final2.draw()

    Ignore:
        >>> deployed_fpath = ub.grabdata(
        >>>     'https://data.kitware.com/api/v1/file/5ee93f639014a6d84ec52b7f/download',
        >>>     fname='deploy_MM_CascadeRCNN_myovdqvi_035_MVKVVR_fix3_mm2x.zip',
        >>>     appname='viame', hasher='sha512',
        >>>     hash_prefix='63b7c3981b3446b079c1d83541a5666c496f6148', verbose=3)

        >>> from .detect_predict import *  # NOQA
        >>> deployed_fpath = ub.grabdata(
        >>>     'https://data.kitware.com/api/v1/file/5f99c37c50a41e3d1918fdbe/download',
        >>>     fname='trained_detector_for_jc.zip',
        >>>     appname='viame', hasher='sha512',
        >>>     hash_prefix='73250a9f2bdc4b746f1edf1baec1593403229c4d7972863', verbose=3)
        >>> config = dict(
        >>>     deployed=deployed_fpath,
        >>> )
        >>> predictor = DetectPredictor(config)
        >>> predictor._ensure_mounted_model()
        >>> inputs = {'rgb': (np.random.rand(512, 512, 3) * 255).astype(np.uint8),
        >>>           'disparity': (np.random.rand(512, 512, 1) * 255).astype(np.uint8)}
        >>> #inputs = {'rgb|disparity': np.random.rand(512, 512, 4)}
        >>> final = predictor.predict(inputs)
    """
    def __init__(predictor, config):

        patch_numpy()  # HACK

        predictor.config = DetectPredictConfig(config)
        predictor.model = None
        predictor.xpu = None
        predictor.coder = None

        # This is populated if we need to modify behavior for backwards
        # compatibility.
        predictor._compat_hack = None

    def info(predictor, text):
        if predictor.config['verbose']:
            print(text)

    @classmethod
    def _infer_native(cls, config):
        """
        Preforms whatever hacks are necessary to introspect the correct
        values of special "native" config options depending on the model.
        """
        # Set default fallback values
        native_defaults = {
            'input_dims': (512, 512),
            'window_dims': 'full',
            'channels': 'hack_old_method'
        }
        @ub.memoize
        def _native_config():
            deployed = torch_liberator.DeployedModel.coerce(config['deployed'])
            # New models should have relevant params here, which is slightly
            # less hacky than using the eval.
            native_config = deployed.train_info()['other']
            common = set(native_defaults) & set(native_config)
            if len(common) != len(native_defaults):
                # Fallback on the hacky string encoding of the configs
                cfgstr = deployed.train_info()['extra']['config']
                # import ast
                # parsed = ast.literal_eval(cfgstr)
                parsed = eval(cfgstr, {'inf': float('inf')})
                native_config.update(parsed)
            return native_config

        native = {}
        native_config = _native_config()
        for key in list(native_defaults.keys()):
            if config[key] == 'native':
                try:
                    native[key] = native_config[key]
                except Exception:
                    if ENABLE_BIOHARN_WARNINGS:
                        warnings.warn((
                            'WARNING: Unable to determine native {} from model. '
                            'Defaulting to {}! Please ensure this is OK.').format(
                                key, native_defaults[key]
                        ))
                    native[key] = native_defaults[key]
            else:
                native[key] = config[key]

        if native['channels'] == 'hack_old_method':
            # Hueristic to determine what channels an older model wants.  This
            # should not be necessary for newer models which directly encode
            # this.
            native['channels'] = 'rgb'
            if native_config.get('use_disparity', False):
                native['channels'] += '|disparity'

        return native

    def _ensure_model(predictor):
        # Just make sure the model is in memory (it might not be on the XPU yet)
        if predictor.model is None:
            # Set up module aliases for backwards compatibility with old models
            # that use 'import netharn' or 'import bioharn' instead of 'viame.pytorch.netharn'
            setup_module_aliases()

            # TODO: we want to use ContainerXPU when dealing with an mmdet
            # model but we probably want regular XPU otherwise. Not sure what
            # the best way to do this is yet.
            # NOTE: ContainerXPU might actually work with non-container returns
            # need to test this.
            xpu = ContainerXPU.coerce(predictor.config['xpu'])
            deployed = predictor.config['deployed']
            if isinstance(predictor.config['deployed'], str):
                if not predictor.config['skip_upgrade']:
                    deployed = _ensure_upgraded_model(deployed)
            deployed = torch_liberator.DeployedModel.coerce(deployed)
            model = deployed.load_model()
            model.train(False)
            predictor.xpu = xpu
            predictor.model = model
            # The model must have a coder
            predictor.raw_model = predictor.xpu.raw(predictor.model)
            predictor.coder = predictor.raw_model.coder

    def _ensure_mounted_model(predictor):
        predictor._ensure_model()
        model = predictor.model
        _ensured_mount = getattr(model, '_ensured_mount', False)
        if not _ensured_mount:
            xpu = predictor.xpu
            if xpu != ContainerXPU.from_data(model):
                predictor.info('Mount model on {}'.format(xpu))
                model = xpu.mount(model)
                predictor.model = model
                # The model must have a coder
                predictor.raw_model = predictor.xpu.raw(predictor.model)
                predictor.coder = predictor.raw_model.coder
            # hack to prevent multiple XPU data checks
            predictor.model._ensured_mount = True

    @profile
    def predict(predictor, inputs):
        """
        Predict on a single large image using a sliding window_dims

        Args:
            inputs (Dict[str, PathLike | ndarray] | PathLike | ndarray):
                An 8-bit RGB numpy image or a path to the image for single-rgb
                detectors. For multi-channel detectors this should be a
                dictionary where the channel name maps to a path or ndarray.

        Returns:
            kwimage.Detections: a wrapper around predicted boxes, scores,
                and class indices. See the `.data` attribute for more info.

        SeeAlso:
            :method:`predict_sampler` - this can be a faster alternative to
            predict, but it requires that your dataset is formatted as a
            sampler.
        """
        predictor.info('Begin detection prediction (via predict)')

        # Ensure model is in prediction mode and disable gradients for speed
        predictor._ensure_mounted_model()

        inputs = predictor._ensure_inputs_dict(inputs)
        predictor.info('Detect objects in image (shape={})'.format(
            ub.map_vals(lambda x: x.shape, inputs)))

        full_inputs, pad_offset_rc, window_dims = predictor._prepare_single_inputs(inputs)

        pad_offset_xy = torch.FloatTensor(
            np.ascontiguousarray(pad_offset_rc[::-1], dtype=np.float32))

        slider_dataset = predictor._make_dataset(full_inputs, window_dims)

        # Its typically faster to use num_workers=0 here because the full image
        # is already in memory. We only need to slice and cast to float32.
        slider_loader = torch.utils.data.DataLoader(
            slider_dataset, shuffle=False, num_workers=predictor.config['workers'],
            batch_size=predictor.config['batch_size'])

        # TODO:
        # mmdetection models need to modify predictor._raw_model.detector.test_cfg
        prog = ub.ProgIter(slider_loader, total=len(slider_loader),
                           desc='predict', enabled=predictor.config['verbose'] > 1)
        accum_dets = []
        with torch.set_grad_enabled(False):
            for raw_batch in prog:
                batch = {
                    'inputs': predictor.xpu.move(raw_batch['inputs']),
                    'tf_chip_to_full': raw_batch['tf_chip_to_full'],
                    'pad_offset_xy': pad_offset_xy,
                }
                results = predictor._predict_batch(batch)
                for dets in results:
                    accum_dets.append(dets)

        # Stitch predicted detections together
        predictor.info('Accumulate detections')
        all_dets = kwimage.Detections.concatenate(accum_dets)

        # Perform final round of NMS on the stiched boxes
        predictor.info('Finalize detections')

        if len(all_dets) > 0:
            keep = all_dets.non_max_supression(
                thresh=predictor.config['nms_thresh'],
                daq={'diameter': all_dets.boxes.width.max()},
            )
            final_dets = all_dets.take(keep)
        else:
            final_dets = all_dets

        predictor.info('Finished prediction')
        return final_dets

    @profile
    def predict_sampler(predictor, sampler, gids=None):
        """
        Predict on all images in a dataset wrapped in a ndsampler.CocoSampler

        Args:
            sampler (ndsampler.CocoSampler): dset wrapped in a sampler
            gids (List[int], default=None): if specified, then only predict
                on these image ids.

        Yields:
            Tuple[int, Detections] : image_id, detection pairs

        SeeAlso:
            :method:`predict` - this is a simpler alternative to
            predict_sampler. It only requires that you pass your data in as an
            image.
        """
        predictor.info('Begin detection prediction (via predict_sampler)')

        predictor._ensure_mounted_model()

        native = predictor._infer_native(predictor.config)
        predictor.info('native = {}'.format(ub.urepr(native, nl=1)))
        input_dims = native['input_dims']
        window_dims = native['window_dims']
        channels = native['channels']

        torch_dset = WindowedSamplerDataset(sampler, window_dims=window_dims,
                                            input_dims=input_dims,
                                            channels=channels, gids=gids)
        if len(torch_dset) == 0:
            return
        slider_loader = torch.utils.data.DataLoader(
            torch_dset, shuffle=False, num_workers=predictor.config['workers'],
            batch_size=predictor.config['batch_size'])

        prog = ub.ProgIter(slider_loader, total=len(slider_loader),
                           chunksize=predictor.config['batch_size'],
                           desc='predict', enabled=predictor.config['verbose'] > 1)

        xpu = predictor.xpu

        # raw_batch = ub.peek(prog)
        with torch.set_grad_enabled(False):

            # ----
            buffer_gids = []
            buffer_dets = []

            for raw_batch in prog:
                batch = {
                    'tf_chip_to_full': raw_batch['tf_chip_to_full'],
                }
                if 'inputs' in raw_batch:
                    batch['inputs'] = xpu.move(raw_batch['inputs'])
                else:
                    raise NotImplementedError

                batch_gids = raw_batch['gid'].view(-1).numpy()
                batch_dets = list(predictor._predict_batch(batch))

                # Determine if we have finished an image (assuming images are
                # passed in sequentially in order)
                can_yield = (
                    np.any(np.diff(batch_gids)) or
                    (len(buffer_gids) and buffer_gids[-1] != batch_gids[0])
                )

                buffer_gids.extend(batch_gids)
                buffer_dets.extend(batch_dets)

                if can_yield:
                    ready_idx = max(np.where(np.diff(buffer_gids))[0]) + 1
                    ready_gids = buffer_gids[:ready_idx]
                    ready_dets = buffer_dets[:ready_idx]

                    #
                    buffer_gids = buffer_gids[ready_idx:]
                    buffer_dets = buffer_dets[ready_idx:]
                    for gid, dets in predictor._finalize_dets(ready_dets, ready_gids):
                        yield gid, dets
            # ----

            # Finalize anything that remains
            ready_gids = buffer_gids
            ready_dets = buffer_dets
            for gid, dets in predictor._finalize_dets(ready_dets, ready_gids):
                yield gid, dets

    @profile
    def _finalize_dets(predictor, ready_dets, ready_gids):
        """ Helper for predict_sampler """
        gid_to_ready_dets = ub.group_items(ready_dets, ready_gids)
        for gid, dets_list in gid_to_ready_dets.items():
            if len(dets_list) == 0:
                dets = kwimage.Detections.concatenate([])
            elif len(dets_list) == 1:
                dets = dets_list[0]
            elif len(dets_list) > 1:
                dets = kwimage.Detections.concatenate(dets_list)
                keep = dets.non_max_supression(
                    thresh=predictor.config['nms_thresh'],
                )
                dets = dets.take(keep)
            yield (gid, dets)

    def _ensure_inputs_dict(predictor, inputs):
        if isinstance(inputs, str):
            # Assume inputs has been given as a single file path
            fpath = inputs
            predictor.info('Reading {!r}'.format(fpath))
            inputs = kwimage.imread(fpath, space='rgb')
        if isinstance(inputs, np.ndarray):
            # Assume inputs has been given as a single rgb image
            inputs = {'rgb': inputs}

        if not isinstance(inputs, dict):
            raise TypeError(type(inputs))

        return inputs

    def _prepare_single_inputs(predictor, inputs):

        if predictor.config['window_dims'] == 'native':
            native = predictor._infer_native(predictor.config)
            window_dims = native['window_dims']
        else:
            window_dims = predictor.config['window_dims']

        inputs_dims = ub.map_vals(lambda x: tuple(x.shape[0:2]), inputs)
        if not ub.allsame(inputs_dims.values()):
            # TODO: handle non-aligned inputs
            raise ValueError('inputs must have same spatial dims for now')

        full_dims = ub.peek(inputs_dims.values())

        if window_dims == 'full':
            window_dims = full_dims

        # Pad small images to be at least the minimum window_dims size
        dims_delta = np.array(full_dims) - np.array(window_dims)
        needs_pad = np.any(dims_delta < 0)
        if needs_pad:
            padding = np.maximum(-dims_delta, 0)
            lower_pad = padding // 2
            upper_pad = padding - lower_pad
            pad_width = list(zip(lower_pad, upper_pad))
            ndims_spti = len(padding)
            pad_offset_rc = lower_pad[0:2]
        else:
            pad_offset_rc = np.array([0, 0])

        if 0:
            if getattr(predictor.raw_model, 'input_norm', None) is not None:
                # todo: use known mean if possible
                # input_norm = predictor.raw_model.input_norm
                pass
        else:
            pad_value = 127

        full_inputs = {}
        for chan_key, imdata in inputs.items():
            if needs_pad:
                ndims_all = len(imdata.shape)
                if ndims_all > ndims_spti:
                    # Handle channels
                    extra = [(0, 0)] * (ndims_all - ndims_spti)
                    pad_width_ = pad_width + extra
                else:
                    pad_width_ = pad_width
                full_imdata = np.pad(
                    imdata, pad_width_, mode='constant',
                    constant_values=pad_value)
            else:
                full_imdata = imdata
            full_inputs[chan_key] = full_imdata

        return full_inputs, pad_offset_rc, window_dims

    def _make_dataset(predictor, full_inputs, window_dims):
        """ helper for predict """

        full_dims = tuple(ub.peek(full_inputs.values()).shape[0:2])

        native = predictor._infer_native(predictor.config)
        predictor.info('native = {}'.format(ub.urepr(native, nl=1)))

        # Break large images into chunks to fit on the GPU
        slider = nh.util.SlidingWindow(full_dims, window=window_dims,
                                       overlap=predictor.config['window_overlap'],
                                       keepbound=True, allow_overshoot=True)

        input_dims = native['input_dims']
        if input_dims == 'full' or input_dims == window_dims:
            input_dims = None

        slider_dataset = SingleImageDataset(full_inputs, slider, input_dims,
                                            channels=native['channels'])
        return slider_dataset

    @profile
    def _predict_batch(predictor, batch):
        """
        Runs the torch network on a single batch and postprocesses the outputs

        Yields:
            kwimage.Detections
        """
        tf_chip_to_full = batch['tf_chip_to_full']

        scale_xy = tf_chip_to_full['scale_xy']
        shift_xy = tf_chip_to_full['shift_xy']

        if 'pad_offset_xy' in batch:
            pad_offset_xy = batch['pad_offset_xy']
            shift_xy_ = shift_xy - pad_offset_xy[None, :]
        else:
            shift_xy_ = shift_xy

        outputs = None

        if predictor._compat_hack is None:
            # All GPU work happens in this line
            if hasattr(predictor.model.module, 'detector'):
                # HACK FOR MMDET MODELS
                # TODO: hack for old detectors that require "im" inputs
                try:
                    outputs = predictor.model.forward(
                        batch, return_loss=False, return_result=True)
                except KeyError:
                    predictor._compat_hack = 'old_mmdet_im_model'
                except NotImplementedError:
                    predictor._compat_hack = 'fixup_mm_inputs'
                if predictor._compat_hack:
                    if ENABLE_BIOHARN_WARNINGS:
                        warnings.warn(
                            'Normal mm-detection input failed. '
                            'Attempting to find backwards compatible solution')
            else:
                # assert len(batch['inputs']) == 1
                try:
                    # im = ub.peek(batch['inputs'].values())
                    outputs = predictor.model.forward(batch['inputs'])
                except Exception:
                    try:
                        # Hack for old efficientdet models with bad input checking
                        from viame.pytorch.netharn.data.data_containers import BatchContainer
                        if isinstance(batch['inputs']['rgb'], torch.Tensor):
                            batch['inputs']['rgb'] = BatchContainer([batch['inputs']['rgb']])
                        outputs = predictor.model.forward(batch)
                        predictor._compat_hack = 'efficientdet_hack'
                    except Exception:
                        raise Exception('Unsure about expected model inputs')
                # raise NotImplementedError('only works on mmdet models')

        if outputs is None:
            # HACKS FOR BACKWARDS COMPATIBILITY
            if predictor._compat_hack == 'old_mmdet_im_model':
                batch['im'] = batch.pop('inputs')['rgb']
                outputs = predictor.model.forward(batch, return_loss=False)
            if predictor._compat_hack == 'fixup_mm_inputs':
                from .detection_models.mm_models import _batch_to_mm_inputs
                mm_inputs = _batch_to_mm_inputs(batch)
                outputs = predictor.model.forward(mm_inputs, return_loss=False)
            if predictor._compat_hack == 'efficientdet_hack':
                from viame.pytorch.netharn.data.data_containers import BatchContainer
                batch['inputs']['rgb'] = BatchContainer([batch['inputs']['rgb']])
                outputs = predictor.model.forward(batch)

        # Postprocess GPU outputs
        if 'Container' in str(type(outputs)):
            # HACK
            outputs = outputs.data

        batch_dets = predictor.coder.decode_batch(outputs)

        for idx, det in enumerate(batch_dets):
            item_scale_xy = scale_xy[idx].numpy()
            item_shift_xy = shift_xy_[idx].numpy()
            det = det.numpy()
            det = det.compress(det.scores > predictor.config['conf_thresh'])

            # Ensure that masks are transformed into polygons for transformation efficiency
            # import xdev
            # xdev.embed()
            if det.data.get('segmentations', None) is not None:
                if 0:
                    import kwplot
                    kwplot.imshow(batch['inputs']['rgb'].data[-1].cpu().numpy())

                    for sseg in det.data['segmentations']:
                        s = sseg.data.sum()
                        if s:
                            mp = sseg.to_multi_polygon()
                            print(s)
                            print(mp)
                            break

                # Patch to fix issue with kwimage / opencv
                import kwimage
                if isinstance(det.data['segmentations'].data, list):
                    for seg in det.data['segmentations'].data:
                        if isinstance(seg, kwimage.Mask):
                            if seg.data.dtype.kind == 'b':
                                seg.data = seg.data.astype(np.uint8)

                det.data['segmentations'] = det.data['segmentations'].to_polygon_list()

            if True and len(det) and np.all(det.boxes.width <= 1) and len(batch['inputs']) == 1:
                # HACK FOR YOLO
                # TODO: decode should return detections in batch input space
                # assert len(batch['inputs']) == 1
                im = ub.peek(batch['inputs'].values())
                inp_size = np.array(im.shape[-2:][::-1])
                det = det.scale(inp_size, inplace=True)

            det = det.scale(item_scale_xy, inplace=True)
            det = det.translate(item_shift_xy, inplace=True)
            # Fix type issue
            if 'class_idxs' in det.data:
                det.data['class_idxs'] = det.data['class_idxs'].astype(int)
            yield det


class SingleImageDataset(torch_data.Dataset):
    """
    Wraps a SlidingWindow in a torch dataset for fast data loading

    This maps image slices into an indexable set for the torch dataloader.

    Calling __getitem__ will result in a dictionary containing a chip for a
    particular window and that chip's offset in the original image.

    Args:
        full_inputs (Dict[str, ndarray]):
            mapping from channel names (e.g. rgb) to aligned ndarrays.
            These arrays must be smaller than the slider window dimensions.

        slider (nh.util.SlidingWindow):
            the sliding window over this image

        input_dims (Tuple[int, int]): height / width to resize to after window
            sampling.

        channels (ChannelSpec | str): channel spec needed by the network


    Example:
        >>> full_inputs = {
        >>>     'rgb': kwimage.atleast_3channels(np.arange(0, 16 * 16).reshape(16, 16)),
        >>>     'disparity': np.random.rand(16, 16, 1),
        >>> }
        >>> full_dims = full_inputs['rgb'].shape[0:2]
        >>> window_dims = (8, 8)
        >>> input_dims = (2, 2)
        >>> channels = 'rgb|disparity'
        >>> slider = nh.util.SlidingWindow(full_dims, window=window_dims,
        >>>                                overlap=0,
        >>>                                keepbound=True, allow_overshoot=True)
        >>> self = SingleImageDataset(full_inputs, slider, input_dims, channels)
        >>> index = 0
        >>> item = self[index]
        >>> item = self[1]

        >>> channels = 'rgb,disparity|rgb,disparity'
        >>> slider = nh.util.SlidingWindow(full_dims, window=window_dims,
        >>>                                overlap=0,
        >>>                                keepbound=True, allow_overshoot=True)
        >>> self = SingleImageDataset(full_inputs, slider, input_dims, channels)
        >>> index = 0
        >>> item = self[index]
        >>> item = self[1]
    """

    def __init__(self, full_inputs, slider, input_dims, channels='rgb'):
        self.full_inputs = full_inputs
        self.slider = slider
        self.input_dims = input_dims
        self.window_dims = self.slider.window
        self.channels = ChannelSpec.coerce(channels)

    def __len__(self):
        return self.slider.n_total

    def __getitem__(self, index):
        """
        Ignore:
            self = slider_dataset
            index = 0
        """
        # Lookup the window location
        slider = self.slider
        basis_idx = np.unravel_index(index, slider.basis_shape)
        slice_ = tuple([bdim[i] for bdim, i in zip(slider.basis_slices, basis_idx)])

        # Resize the image patch if necessary
        needs_resize = (self.input_dims is not None and
                        self.input_dims != 'window')
        if needs_resize:
            if isinstance(self.input_dims, str):
                raise TypeError(
                    'input dims is a non-window string but should '
                    'have been resolved before this!')
            letterbox = nh.data.transforms.Resize(None, mode='letterbox')
            letterbox.target_size = self.input_dims[::-1]
            # Record the inverse transformation
            window_size = self.window_dims[::-1]
            input_size = self.input_dims[::-1]
            shift, scale, embed_size = letterbox._letterbox_transform(
                window_size, input_size)
        else:
            letterbox = None
            shift = [0, 0]
            scale = [1, 1]

        # TODO: UNIFY WITH WindowedSamplerDataset.__getitem__

        scale_xy = torch.FloatTensor(scale)
        offset_xy = torch.FloatTensor([slice_[1].start, slice_[0].start])

        # To apply a transform we first scale then shift
        tf_full_to_chip = {
            'scale_xy': torch.FloatTensor(scale_xy),
            'shift_xy': torch.FloatTensor(shift) - (offset_xy * scale_xy),
        }

        # This transform will bring us from chip space back to full img space
        tf_chip_to_full = {
            'scale_xy': 1.0 / tf_full_to_chip['scale_xy'],
            'shift_xy': -tf_full_to_chip['shift_xy'] * (1.0 / tf_full_to_chip['scale_xy']),
        }

        components = {}
        # Sample the image patch
        for chan_key, full_imdata in self.full_inputs.items():
            chip_hwc = full_imdata[slice_]

            # TODO: be careful what we do here based on the channel info
            chip_hwc = kwimage.ensure_float01(chip_hwc, dtype=np.float32)

            if needs_resize:
                # Resize the image
                if letterbox is not None:
                    chip_hwc = kwimage.imresize(
                        chip_hwc, dsize=letterbox.target_size,
                        letterbox=True)
                chip_hwc = kwarray.atleast_nd(chip_hwc, n=3, front=False)

            chip_chw = np.transpose(chip_hwc, (2, 0, 1))
            chip_chw = np.ascontiguousarray(chip_chw)
            tensor_chw = torch.from_numpy(chip_chw)
            components[chan_key] = tensor_chw

        # Do early fusion as specified by channels
        # TODO: we need to ensure `encode` can handle the case where components
        # are pre-fused. It should be able to so separate and refuse where
        # necessary.
        inputs = self.channels.encode(components, axis=0)

        return {
            'inputs': inputs,
            'tf_chip_to_full': tf_chip_to_full,
        }


class WindowedSamplerDataset(torch_data.Dataset, ub.NiceRepr):
    """
    Dataset that breaks up images into windows and optionally resizes those
    windows.

    TODO: Use as a base class for training detectors. This should ideally be
    used as an input to another dataset which handles augmentation.

    Args:
        window_dims: size of a sliding window
        input_dims: size to resize sampled windows to
        window_overlap: amount of overlap between windows
        gids : images to sample from, if None use all of them
    """

    def __init__(self, sampler, window_dims='full', input_dims='window',
                 window_overlap=0.0, gids=None, channels='rgb'):
        self.sampler = sampler
        self.input_dims = input_dims
        self.window_dims = window_dims
        self.window_overlap = window_overlap
        self.channels = ChannelSpec.coerce(channels)
        self.subindex = None
        self.gids = gids
        self._build_sliders()

        self.want_aux = self.channels.unique() - {'rgb'}

    @classmethod
    def demo(WindowedSamplerDataset, key='habcam', **kwargs):
        import ndsampler
        if key == 'habcam':
            dset_fpath = ub.expandpath('~/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_vali.mscoco.json')
            workdir = ub.expandpath('~/work/bioharn')
            dset = kwcoco.CocoDataset(dset_fpath)
            sampler = ndsampler.CocoSampler(dset, workdir=workdir, backend=None)
        else:
            sampler = ndsampler.CocoSampler.demo(key)
        self = WindowedSamplerDataset(sampler, **kwargs)
        return self

    def _build_sliders(self):
        """
        Use the ndsampler.Sampler and sliders to build a flat index that can
        reach every subregion of every image in the training set.

        Ignore:
            window_dims = (512, 512)
            input_dims = 'window'
            window_overlap = 0
        """
        from viame.pytorch import netharn as nh
        window_overlap = self.window_overlap
        window_dims = self.window_dims
        sampler = self.sampler

        gids = self.gids
        if gids is None:
            gids = list(sampler.dset.imgs.keys())

        gid_to_slider = {}
        for gid in gids:
            img = sampler.dset.imgs[gid]
            # if img.get('source', '') == 'habcam_2015_stereo':
            # Hack: todo, cannoncial way to get this effect
            full_dims = [img['height'], img['width']]
            # else:
            #     full_dims = [img['height'], img['width']]

            window_dims_ = full_dims if window_dims == 'full' else window_dims
            slider = nh.util.SlidingWindow(full_dims, window_dims_,
                                           overlap=window_overlap,
                                           keepbound=True,
                                           allow_overshoot=True)
            gid_to_slider[img['id']] = slider

        self.gid_to_slider = gid_to_slider
        self._gids = list(gid_to_slider.keys())
        self._sliders = list(gid_to_slider.values())
        self.subindex = nh.util.FlatIndexer.fromlist(self._sliders)

    def __len__(self):
        return len(self.subindex)

    def __getitem__(self, index):
        """
        Example:
            >>> # xdoctest: +REQUIRES(--data)
            >>> self = WindowedSamplerDataset.demo(window_dims=(512, 512))
            >>> index = 0
            >>> item = self[1]
            >>> # xdoctest: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.autompl()
            >>> kwplot.imshow(item['inputs']['rgb'])
        """
        outer, inner = self.subindex.unravel(index)
        gid = self._gids[outer]
        slider = self._sliders[outer]
        slices = slider[inner]

        tr = {'gid': gid, 'slices': slices}

        # for now load sample only returns rgb
        unique_channels = self.channels.unique()

        assert 'rgb' in unique_channels
        sample = self.sampler.load_sample(tr, with_annots=False)

        if 0:
            # kwcoco should interact with the network's ChannelSpec to know
            # if it needs to coerce grayscale images to 3 channel to pass them
            # to an RGB network.
            #
            # Outline:
            #     - [ ] we should be given `self.channels` which defines the
            #     network input early-fused streams.
            #         - [ ] perhaps this can store input mean / std values for
            #               pre-processing? We can measure speedup from doing
            #               this on CPU.
            #     - [ ] Given the sampler "frames" object specify the window
            #           region and the resize region the early fused inputs and
            #           optionally resize and renormalize.
            #
            if 'rgb' in self.channels.unique():
                chip_hwc = kwimage.atleast_3channels(sample['im'])
        else:
            chip_hwc = kwimage.atleast_3channels(sample['im'])

        chip_dims = tuple(chip_hwc.shape[0:2])

        # Resize the image patch if necessary
        # print('self.input_dims = {!r}'.format(self.input_dims))
        # print('chip_dims = {!r}'.format(chip_dims))

        if isinstance(self.input_dims, str) or self.input_dims is None:
            if self.input_dims is not None and self.input_dims != 'window':
                raise TypeError(
                    'input dims is a non-window string but should '
                    'have been resolved before this!')
            letterbox = None
            shift = [0, 0]
            scale = [1, 1]
        else:
            letterbox = nh.data.transforms.Resize(None, mode='letterbox')
            letterbox.target_size = self.input_dims[::-1]
            # Record the inverse transformation
            chip_size = np.array(chip_dims[::-1])
            input_size = np.array(self.input_dims[::-1])
            shift, scale, embed_size = letterbox._letterbox_transform(chip_size, input_size)
            # Resize the image
            chip_hwc = letterbox.augment_image(chip_hwc)

        scale_xy = torch.FloatTensor(scale)
        offset_xy = torch.FloatTensor([slices[1].start, slices[0].start])

        # TODO: UNIFY WITH SingleImageDataset.__getitem__
        # TODO: kwcoco should contain the correct logic to sample fused
        # streams that includes the rgb sampling.

        # Assume 8-bit image inputs
        chip_chw = np.transpose(chip_hwc, (2, 0, 1))
        chip_chw = np.ascontiguousarray(chip_chw)
        # print('chip_chw.dtype = {!r}'.format(chip_chw.dtype))

        # chip_chw = kwimage.ensure_float01(chip_chw, dtype=np.float32)
        # tensor_rgb = torch.from_numpy(chip_chw)
        tensor_rgb = torch.from_numpy(chip_chw).float() / 255.0
        # print(kwarray.stats_dict(tensor_rgb))

        # To apply a transform we first scale then shift
        tf_full_to_chip = {
            'scale_xy': torch.FloatTensor(scale_xy),
            'shift_xy': torch.FloatTensor(shift) - (offset_xy * scale_xy),
        }

        # This transform will bring us from chip space back to full img space
        tf_chip_to_full = {
            'scale_xy': 1.0 / tf_full_to_chip['scale_xy'],
            'shift_xy': -tf_full_to_chip['shift_xy'] * (1.0 / tf_full_to_chip['scale_xy']),
        }
        components = {
            'rgb': tensor_rgb,
        }
        item = {
            'gid': torch.LongTensor([gid]),
            'tf_chip_to_full': tf_chip_to_full,
        }

        sampler = self.sampler

        if self.want_aux:

            from .detect_dataset import load_sample_auxiliary

            sampler = self.sampler

            want_aux = self.want_aux
            aux_components = load_sample_auxiliary(sampler, tr, want_aux)

            # note: the letterbox augment doesn't handle floats well
            # use the kwimage.imresize instead
            for auxkey, aux_im in aux_components.items():
                if letterbox is not None:
                    aux_components[auxkey] = kwimage.imresize(
                        aux_im, dsize=letterbox.target_size,
                        letterbox=True).clip(0, 1)

            for auxkey, aux_im in aux_components.items():
                aux_im = kwarray.atleast_nd(aux_im, 3)
                components[auxkey] = torch.FloatTensor(
                    aux_im.transpose(2, 0, 1))

        item['inputs'] = self.channels.encode(components, axis=0)
        # print(item['inputs'])
        return item


################################################################################
# CLI


def _coerce_sampler(config):
    from viame.pytorch.netharn import bio_util as util
    from os.path import isdir
    import ndsampler

    # Running prediction is much faster if you can build a sampler.
    sampler_backend = config['sampler_backend']

    if isinstance(config['dataset'], str):
        if config['dataset'].endswith('.json'):
            dataset_fpath = ub.expandpath(config['dataset'])
            coco_dset = kwcoco.CocoDataset(dataset_fpath)
            print('coco hashid = {}'.format(coco_dset._build_hashid()))
        else:
            image_path = ub.expandpath(config['dataset'])
            path_exists = exists(image_path)
            if path_exists and isfile(image_path):
                # Single image case
                coco_dset = kwcoco.CocoDataset()
                coco_dset.add_image(image_path)
            elif path_exists and isdir(image_path):
                # Directory of images case
                img_globs = ['*' + ext for ext in kwimage.im_io.IMAGE_EXTENSIONS]
                fpaths = list(util.find_files(image_path, img_globs))
                if len(fpaths):
                    coco_dset = kwcoco.CocoDataset.from_image_paths(fpaths)
                else:
                    raise Exception('no images found')
            else:
                # Glob pattern case
                import glob
                fpaths = list(glob.glob(image_path))
                if len(fpaths):
                    coco_dset = kwcoco.CocoDataset.from_image_paths(fpaths)
                else:
                    raise Exception('not an image path')

    elif isinstance(config['dataset'], list):
        # Multiple image case
        gpaths = config['dataset']
        gpaths = [ub.expandpath(g) for g in gpaths]
        coco_dset = kwcoco.CocoDataset.from_image_paths(gpaths)
    else:
        raise TypeError(config['dataset'])

    print('Create sampler')
    workdir = ub.expandpath(config.get('workdir'))
    sampler = ndsampler.CocoSampler(coco_dset, workdir=workdir,
                                    backend=sampler_backend)
    return sampler


@profile
def _cached_predict(predictor, sampler, out_dpath='./cached_out', gids=None,
                    draw=False, enable_cache=True, async_buffer=False,
                    verbose=1, draw_truth=True):
    """
    Helper to only do predictions that havent been done yet.

    Note that this currently requires you to ensure that the dest folder is
    unique to this particular dataset.

    Ignore:
        >>> import ndsampler
        >>> config = {}
        >>> config['deployed'] = ub.expandpath('~/work/bioharn/fit/runs/bioharn-det-v13-cascade/ogenzvgt/torch_snapshots/_epoch_00000042.pt')
        >>> predictor = DetectPredictor(config)
        >>> predictor._ensure_model()
        >>> out_dpath = './cached_out'
        >>> gids = None
        >>> coco_dset = kwcoco.CocoDataset(ub.expandpath('~/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_vali.mscoco.json'))
        >>> sampler = ndsampler.CocoSampler(coco_dset, workdir=None,
        >>>                                 backend=None)
    """
    from viame.pytorch.netharn import bio_util as util
    import tempfile
    coco_dset = sampler.dset
    # predictor.config['verbose'] = 1

    det_outdir = ub.ensuredir((out_dpath, 'pred'))
    tmp_fpath = tempfile.mktemp()

    if gids is None:
        gids = list(coco_dset.imgs.keys())

    gid_to_pred_fpath = {
        gid: join(det_outdir, 'dets_gid_{:08d}_v2.mscoco.json'.format(gid))
        for gid in gids
    }

    if enable_cache:
        # Figure out what gids have already been computed
        have_gids = [gid for gid, fpath in gid_to_pred_fpath.items() if exists(fpath)]
    else:
        have_gids = []

    print('enable_cache = {!r}'.format(enable_cache))
    print('Found {} / {} existing predictions'.format(len(have_gids), len(gids)))

    gids = ub.oset(gids) - have_gids
    pred_gen = predictor.predict_sampler(sampler, gids=gids)

    if async_buffer:
        desc = 'buffered detect'
        buffered_gen = util.AsyncBufferedGenerator(pred_gen,
                                                   size=coco_dset.n_images)
        gen = buffered_gen
    else:
        desc = 'unbuffered detect'
        gen = pred_gen

    gid_to_pred = {}
    prog = ub.ProgIter(gen, total=len(gids), desc=desc, verbose=verbose)
    for img_idx, (gid, dets) in enumerate(prog):
        gid_to_pred[gid] = dets

        img = coco_dset.imgs[gid]

        # TODO: need to either add the expected img_root to the coco dataset or
        # reroot the file name to be a full path so the predicted dataset can
        # reference the source images if needed.
        single_img_coco = kwcoco.CocoDataset()
        gid = single_img_coco.add_image(**img)

        for cat in dets.classes.to_coco():
            single_img_coco.add_category(**cat)

        # for cat in coco_dset.cats.values():
        #     single_img_coco.add_category(**cat)
        for ann in dets.to_coco(style='new'):
            ann['image_id'] = gid
            if 'category_name' in ann:
                catname = ann['category_name']
                cid = single_img_coco.ensure_category(catname)
                ann['category_id'] = cid
            single_img_coco.add_annotation(**ann)

        single_pred_fpath = gid_to_pred_fpath[gid]

        # prog.ensure_newline()
        # print('write single_pred_fpath = {!r}'.format(single_pred_fpath))
        # TODO: use safer?
        single_img_coco.dump(tmp_fpath, newlines=True)
        util.atomic_move(tmp_fpath, single_pred_fpath)

        if draw is True or (draw and img_idx < draw):
            draw_outdir = ub.ensuredir((out_dpath, 'draw'))
            img_fpath = coco_dset.get_image_fpath(gid)
            gname = basename(img_fpath)
            viz_fname = ub.augpath(gname, prefix='detect_', ext='.jpg')
            viz_fpath = join(draw_outdir, viz_fname)

            image = kwimage.imread(img_fpath)

            if draw_truth:
                # draw truth if available
                anns = list(ub.take(coco_dset.anns, coco_dset.index.gid_to_aids[gid]))
                true_dets = kwimage.Detections.from_coco_annots(anns,
                                                                dset=coco_dset)
                true_dets.draw_on(image, alpha=None, color='green')

            # flags = dets.scores > .2
            # flags[kwarray.argmaxima(dets.scores, num=10)] = True
            # show_dets = dets.compress(flags)
            show_dets = dets
            toshow = show_dets.draw_on(image, alpha=None)
            # kwplot.imshow(toshow)
            kwimage.imwrite(viz_fpath, toshow, space='rgb')

    if enable_cache:
        pred_fpaths = [gid_to_pred_fpath[gid] for gid in have_gids]
        load_workers = 0
        cached_dets = _load_dets(pred_fpaths, workers=load_workers)
        assert have_gids == [d.meta['gid'] for d in cached_dets]
        gid_to_cached = ub.dzip(have_gids, cached_dets)
        gid_to_pred.update(gid_to_cached)

    return gid_to_pred, gid_to_pred_fpath


def _load_dets(pred_fpaths, workers=0):
    # Process mode is much faster than thread.
    from kwcoco.util import util_futures
    jobs = util_futures.JobPool(mode='process', max_workers=workers)
    for single_pred_fpath in ub.ProgIter(pred_fpaths, desc='submit load dets jobs'):
        job = jobs.submit(_load_dets_worker, single_pred_fpath)
    dets = []
    for job in ub.ProgIter(jobs.jobs, total=len(jobs), desc='loading cached dets'):
        dets.append(job.result())
    return dets


def _load_dets_worker(single_pred_fpath):
    """
    single_pred_fpath = ub.expandpath('$HOME/remote/namek/work/bioharn/fit/runs/bioharn-det-mc-cascade-rgbd-v36/brekugqz/eval/habcam_cfarm_v6_test.mscoc/bioharn-det-mc-cascade-rgbd-v36__epoch_00000018/c=0.2,i=window,n=0.5,window_d=512,512,window_o=0.0/pred/dets_gid_00004070_v2.mscoco.json')
    """
    single_img_coco = kwcoco.CocoDataset(single_pred_fpath, autobuild=False)
    if len(single_img_coco.dataset['images']) != 1:
        raise Exception('Expected predictions for a single image only')
    gid = single_img_coco.dataset['images'][0]['id']
    dets = kwimage.Detections.from_coco_annots(single_img_coco.dataset['annotations'],
                                               dset=single_img_coco)
    dets.meta['gid'] = gid
    return dets


class DetectPredictCLIConfig(scfg.Config):
    default = ub.dict_union(
        {
            'dataset': scfg.Value(None, help='coco dataset, path to images or folder of images'),
            'out_dpath': scfg.Path('./out', help='output directory'),
            'draw': scfg.Value(False),
            'sampler_backend': scfg.Value(None),
            'enable_cache': scfg.Value(False),
            'workdir': scfg.Path('~/work/bioharn', help='work directory for sampler if needed'),

            'async_buffer': scfg.Value(False, help="I've seen this increase prediction rate from 2.0Hz to 2.3Hz, but it increases instability, unsure of the reason"),
            'gids': scfg.Value(None, help='if specified only predict on these image-ids (only applicable to coco input)'),
        },
        DetectPredictConfig.default
    )


def detect_cli(config={}):
    """
    CommandLine:
        python -m bioharn.detect_predict --help

    CommandLine:
        python -m bioharn.detect_predict \
            --dataset=~/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_test.mscoco.json \
            --deployed=/home/joncrall/work/bioharn/fit/runs/bioharn-det-v11-test-cascade/myovdqvi/deploy_MM_CascadeRCNN_myovdqvi_035_MVKVVR.zip \
            --out_dpath=~/work/bioharn/habcam_test_out \
            --draw=100 \
            --input_dims=512,512 \
            --xpu=0 --batch_size=1

    Ignore:
        >>> config = {}
        >>> config['dataset'] = '~/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_vali.mscoco.json'
        >>> config['deployed'] = '/home/joncrall/work/bioharn/fit/runs/bioharn-det-v11-test-cascade/myovdqvi/deploy_MM_CascadeRCNN_myovdqvi_035_MVKVVR.zip'
        >>> config['out_dpath'] = 'out'
    """
    config = DetectPredictCLIConfig(config, cmdline=True)
    print('config = {}'.format(ub.urepr(config.asdict())))

    out_dpath = ub.expandpath(config.get('out_dpath'))
    det_outdir = ub.ensuredir((out_dpath, 'pred'))

    sampler = _coerce_sampler(config)
    print('prepare frames')
    gids = config['gids']
    gids = [gids] if isinstance(gids, int) else gids
    sampler.frames.prepare(workers=config['workers'], gids=gids)

    print('Create predictor')
    pred_config = ub.dict_subset(config, DetectPredictConfig.default)
    if config['verbose'] < 2:
        pred_config['verbose'] = 0
    predictor = DetectPredictor(pred_config)
    print('Ensure model')
    predictor._ensure_model()

    async_buffer = config['async_buffer']

    gid_to_pred, gid_to_pred_fpath = _cached_predict(
        predictor, sampler, out_dpath=out_dpath, gids=gids,
        draw=config['draw'], enable_cache=config['enable_cache'],
        async_buffer=async_buffer)

    if gids is None:
        # Each image produces its own kwcoc files in the "pred" subfolder.
        # Union all of those to make a single coco file that contains all
        # predictions.
        coco_dsets = []
        for gid, pred_fpath in gid_to_pred_fpath.items():
            single_img_coco = kwcoco.CocoDataset(pred_fpath)
            coco_dsets.append(single_img_coco)

        pred_dset = kwcoco.CocoDataset.union(*coco_dsets)
        pred_fpath = join(det_outdir, 'detections.mscoco.json')
        print('Dump detections to pred_fpath = {!r}'.format(pred_fpath))
        pred_dset.dump(pred_fpath, newlines=True)


if __name__ == '__main__':
    detect_cli()
