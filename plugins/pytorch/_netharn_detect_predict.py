"""
Detection wrapper from bioharn
"""
import ubelt as ub
import torch.utils.data as torch_data
import netharn as nh
import numpy as np
import torch
import six
import scriptconfig as scfg
import kwimage


class DetectPredictConfig(scfg.Config):
    default = {

        'deployed': None,
        'batch_size': 4,
        'xpu': 'auto',

        'window_dims': scfg.Value('full', help='size of a sliding window'),  # (512, 512),
        'input_dims': scfg.Value((512, 512), help='The size of the inputs to the network'),

        'workers': 0,

        'overlap': scfg.Value(0.0, help='overlap of the sliding window'),

        # Note: these dont work exactly correct due to mmdetection model
        # differences
        'nms_thresh': 0.4,
        'conf_thresh': 0.1,

        'verbose': 3,
    }


class DetectPredictor(object):
    """
    A detector API for bioharn trained models

    Ignore:
        >>> path_or_image = kwimage.imread('/home/joncrall/data/noaa/2015_Habcam_photos/201503.20150522.131445618.413800.png')[:, :1360]
        >>> full_rgb = path_or_image
        >>> config = dict(
        >>>     deployed='/home/joncrall/work/bioharn/fit/runs/bioharn-det-v11-test-cascade/myovdqvi/deploy_MM_CascadeRCNN_myovdqvi_035_MVKVVR.zip',
        >>>     window_dims=(512, 512),
        >>>     input_dims=(256, 256),
        >>> )
        >>> self = DetectPredictor(config)
        >>> final = self.predict(full_rgb)
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.autompl()
        >>> kwplot.imshow(full_rgb, doclf=True)
        >>> final2 = final.compress(final.scores > .0)
        >>> final2.draw()
    """
    def __init__(self, config):
        self.config = DetectPredictConfig(config)
        self.model = None
        self.xpu = None
        self.coder = None

    def info(self, text):
        if self.config['verbose']:
            print(text)

    def _ensure_model(self):
        if self.model is None:
            xpu = nh.XPU.coerce(self.config['xpu'])
            deployed = nh.export.DeployedModel.coerce(self.config['deployed'])
            model = deployed.load_model()
            if xpu != nh.XPU.from_data(model):
                self.info('Mount {} on {}'.format(deployed, xpu))
                model = xpu.mount(model)
            model.train(False)
            self.model = model
            self.xpu = xpu
            # The model must have a coder
            self.raw_model = self.xpu.raw(self.model)
            self.coder = self.raw_model.coder

    def _rectify_image(self, path_or_image):
        if isinstance(path_or_image, six.string_types):
            self.info('Reading {!r}'.format(path_or_image))
            full_rgb = kwimage.imread(path_or_image, space='rgb')
        else:
            full_rgb = path_or_image
        return full_rgb

    def predict(self, path_or_image):
        """
        Predict on a single large image using a sliding window_dims

        Args:
            path_or_image (PathLike | ndarray): An 8-bit RGB numpy image or a
                path to the image.

        Returns:
            kwimage.Detections: a wrapper around predicted boxes, scores,
                and class indices. See the `.data` attribute for more info.
        """
        self.info('Begin detection prediction')

        # Ensure model is in prediction mode and disable gradients for speed
        self._ensure_model()

        full_rgb = self._rectify_image(path_or_image)
        self.info('Detect objects in image (shape={})'.format(full_rgb.shape))

        full_rgb, pad_offset_rc, window_dims = self._prepare_image(full_rgb)
        pad_offset_xy = torch.FloatTensor(np.ascontiguousarray(pad_offset_rc[::-1]))

        slider_dataset = self._make_dataset(full_rgb, window_dims)

        # Its typically faster to use num_workers=0 here because the full image
        # is already in memory. We only need to slice and cast to float32.
        slider_loader = torch.utils.data.DataLoader(
            slider_dataset, shuffle=False, num_workers=self.config['workers'],
            batch_size=self.config['batch_size'])

        # TODO:
        # mmdetection models need to modify self._raw_model.detector.test_cfg
        prog = ub.ProgIter(slider_loader, total=len(slider_loader),
                           desc='predict', enabled=self.config['verbose'] > 1)
        accum_dets = []
        with torch.set_grad_enabled(False):
            for raw_batch in prog:
                batch = {
                    'im': self.xpu.move(raw_batch['im']),
                    'tf_chip_to_full': raw_batch['tf_chip_to_full'],
                    'pad_offset_xy': pad_offset_xy,
                }
                for dets in self._predict_batch(batch):
                    accum_dets.append(dets)

        # Stitch predicted detections together
        self.info('Accumulate detections')
        all_dets = kwimage.Detections.concatenate(accum_dets)

        # Perform final round of NMS on the stiched boxes
        self.info('Finalize detections')

        if len(all_dets) > 0:
            keep = all_dets.non_max_supression(
                thresh=self.config['nms_thresh'],
                daq={'diameter': all_dets.boxes.width.max()},
            )
            final_dets = all_dets.take(keep)
        else:
            final_dets = all_dets

        self.info('Finished prediction')
        return final_dets

    def _prepare_image(self, full_rgb):
        full_dims = tuple(full_rgb.shape[0:2])
        if self.config['window_dims'] == 'full':
            window_dims = full_dims
        else:
            window_dims = self.config['window_dims']

        # Pad small images to be at least the minimum window_dims size
        dims_delta = np.array(full_dims) - np.array(window_dims)
        if np.any(dims_delta < 0):
            padding = np.maximum(-dims_delta, 0)
            lower_pad = padding // 2
            upper_pad = padding - lower_pad
            pad_width = list(zip(lower_pad, upper_pad))
            ndims_all = len(full_rgb.shape)
            ndims_spti = len(padding)
            if ndims_all > ndims_spti:
                # Handle channels
                extra = [(0, 0)] * (ndims_all - ndims_spti)
                pad_width = pad_width + extra
            full_rgb = np.pad(full_rgb, pad_width, mode='constant',
                              constant_values=127)
            full_dims = tuple(full_rgb.shape[0:2])
            pad_offset_rc = lower_pad[0:2]
        else:
            pad_offset_rc = np.array([0, 0])

        return full_rgb, pad_offset_rc, window_dims

    def _make_dataset(self, full_rgb, window_dims):
        full_dims = tuple(full_rgb.shape[0:2])

        # Break large images into chunks to fit on the GPU
        slider = nh.util.SlidingWindow(full_dims, window=window_dims,
                                       overlap=self.config['overlap'],
                                       keepbound=True, allow_overshoot=True)

        input_dims = self.config['input_dims']
        if input_dims == 'full' or input_dims == window_dims:
            input_dims = None

        slider_dataset = SingleImageDataset(full_rgb, slider, input_dims)
        return slider_dataset

    def _predict_batch(self, batch):
        """
        Runs the torch network on a single batch and postprocesses the outputs

        Yields:
            kwimage.Detections
        """
        # chips = batch['im']
        tf_chip_to_full = batch['tf_chip_to_full']

        scale_xy = tf_chip_to_full['scale_xy']
        shift_xy = tf_chip_to_full['shift_xy']

        if 'pad_offset_xy' in batch:
            pad_offset_xy = batch['pad_offset_xy']
            shift_xy_ = shift_xy - pad_offset_xy[None, :]
        else:
            shift_xy_ = shift_xy

        # All GPU work happens in this line
        if hasattr(self.model.module, 'detector'):
            # HACK FOR MMDET MODELS
            outputs = self.model.forward(batch, return_loss=False,
                                         return_result=True)
            # from bioharn.models.mm_models import _batch_to_mm_inputs
            # mm_inputs = _batch_to_mm_inputs(batch)
            # imgs = mm_inputs.pop('imgs')
            # img_metas = mm_inputs.pop('img_metas')
            # hack_imgs = [g[None, :] for g in imgs]
            # # For whaver reason we cant run more than one test image at the
            # # same time.
            # batch_results = []
            # outputs = {}
            # for one_img, one_meta in zip(hack_imgs, img_metas):
            #     result = self.model.module.detector.forward(
            #         [one_img], [[one_meta]], return_loss=False)
            #     batch_results.append(result)
            # outputs['batch_results'] = batch_results
        else:
            raise NotImplementedError('only hacked mmdet models working')
            # outputs = self.model.forward(chips, return_loss=False)

        # Postprocess GPU outputs
        batch_dets = self.coder.decode_batch(outputs)
        for idx, det in enumerate(batch_dets):
            item_scale_xy = scale_xy[idx].numpy()
            item_shift_xy = shift_xy_[idx].numpy()
            det = det.numpy()
            det = det.scale(item_scale_xy)
            det = det.translate(item_shift_xy)
            # Fix type issue
            det.data['class_idxs'] = det.data['class_idxs'].astype(np.int)
            yield det

    def predict_sampler(self, sampler):
        """
        Predict on all images in a dataset wrapped in a ndsampler.CocoSampler

        Args:
            sampler (ndsampler.CocoDataset): dset wrapped in a sampler
        """
        input_dims = self.config['input_dims']
        window_dims = self.config['window_dims']

        torch_dset = WindowedSamplerDataset(sampler, window_dims=window_dims,
                                            input_dims=input_dims)
        slider_loader = torch.utils.data.DataLoader(
            torch_dset, shuffle=False, num_workers=self.config['workers'],
            batch_size=self.config['batch_size'])

        prog = ub.ProgIter(slider_loader, total=len(slider_loader),
                           chunksize=self.config['batch_size'],
                           desc='predict', enabled=self.config['verbose'] > 1)

        def finalize_dets(ready_dets, ready_gids):
            gid_to_ready_dets = ub.group_items(ready_dets, ready_gids)
            for gid, dets_list in gid_to_ready_dets.items():
                if len(dets_list) == 0:
                    dets = kwimage.Detections.concatenate([])
                elif len(dets_list) == 1:
                    dets = dets_list[0]
                elif len(dets_list) > 1:
                    dets = kwimage.Detections.concatenate(dets_list)
                    keep = dets.non_max_supression(
                        thresh=self.config['nms_thresh'],
                    )
                    dets = dets.take(keep)
                yield (gid, dets)

        xpu = self.xpu
        with torch.set_grad_enabled(False):

            # ----
            buffer_gids = []
            buffer_dets = []
            for raw_batch in prog:
                batch = {
                    'im': xpu.move(raw_batch['im']),
                    'tf_chip_to_full': raw_batch['tf_chip_to_full'],
                }
                batch_gids = raw_batch['gid'].view(-1).numpy()
                batch_dets = list(self._predict_batch(batch))

                # Determine if we have finished an image (assuming images are
                # passed in sequentially in order)
                buffer_gids.extend(batch_gids)
                buffer_dets.extend(batch_dets)

                # Test if we can yield intermediate results for an image
                can_yield = (
                    np.any(np.diff(batch_gids)) or
                    (len(buffer_gids) and buffer_gids[-1] != batch_gids[0])
                )
                if can_yield:
                    ready_idx = max(np.where(np.diff(buffer_gids))[0]) + 1
                    ready_gids = buffer_gids[:ready_idx]
                    ready_dets = buffer_dets[:ready_idx]

                    #
                    buffer_gids = buffer_gids[ready_idx:]
                    buffer_dets = buffer_dets[ready_idx:]
                    for gid, dets in finalize_dets(ready_dets, ready_gids):
                        yield gid, dets
            # ----

            # Finalize anything that remains
            ready_gids = buffer_gids
            ready_dets = buffer_dets
            for gid, dets in finalize_dets(ready_dets, ready_gids):
                yield gid, dets


class SingleImageDataset(torch_data.Dataset):
    """
    Wraps a SlidingWindow in a torch dataset for fast data loading

    This maps image slices into an indexable set for the torch dataloader.

    Calling __getitem__ will result in a dictionary containing a chip for a
    particular window and that chip's offset in the original image.
    """

    def __init__(self, full_image, slider, input_dims):
        self.full_image = full_image
        self.slider = slider
        self.input_dims = input_dims
        self.window_dims = self.slider.window

    def __len__(self):
        return self.slider.n_total

    def __getitem__(self, index):
        # Lookup the window location
        slider = self.slider
        basis_idx = np.unravel_index(index, slider.basis_shape)
        slice_ = tuple([bdim[i] for bdim, i in zip(slider.basis_slices, basis_idx)])

        # Sample the image patch
        chip_hwc = self.full_image[slice_]

        # Resize the image patch if necessary
        if self.input_dims is not None:
            letterbox = nh.data.transforms.Resize(None, mode='letterbox')
            letterbox.target_size = self.input_dims[::-1]
            # Record the inverse transformation
            window_size = self.window_dims[::-1]
            input_size = self.input_dims[::-1]
            print('input_size = {!r}'.format(input_size))
            print('window_size = {!r}'.format(window_size))
            shift, scale, embed_size = letterbox._letterbox_transform(window_size, input_size)
            # Resize the image
            chip_hwc = letterbox.augment_image(chip_hwc)
        else:
            shift = [0, 0]
            scale = [1, 1]
        scale_xy = torch.FloatTensor(scale)

        # Assume 8-bit image inputs
        chip_chw = np.transpose(chip_hwc, (2, 0, 1))
        tensor_chip = torch.FloatTensor(np.ascontiguousarray(chip_chw)) / 255.0
        offset_xy = torch.FloatTensor([slice_[1].start, slice_[0].start])

        # To apply a transform we first scale then shift
        tf_full_to_chip = {
            'scale_xy': torch.FloatTensor(scale_xy),
            'shift_xy': torch.FloatTensor(shift) - (offset_xy * scale_xy),
        }

        if False:
            tf_mat = np.array([
                [tf_full_to_chip['scale_xy'][0], 0, tf_full_to_chip['shift_xy'][0]],
                [0, tf_full_to_chip['scale_xy'][1], tf_full_to_chip['shift_xy'][1]],
                [0, 0, 1],
            ])
            np.linalg.inv(tf_mat)

        # This transform will bring us from chip space back to full img space
        tf_chip_to_full = {
            'scale_xy': 1.0 / tf_full_to_chip['scale_xy'],
            'shift_xy': -tf_full_to_chip['shift_xy'] * (1.0 / tf_full_to_chip['scale_xy']),
        }
        return {
            'im': tensor_chip,
            'tf_chip_to_full': tf_chip_to_full,
        }


class WindowedSamplerDataset(torch_data.Dataset, ub.NiceRepr):
    """
    Dataset that breaks up images into windows and optionally resizes those
    windows.

    TODO: Use as a base class for training detectors. This should ideally be
    used as an input to another dataset which handles augmentation.
    """

    def __init__(self, sampler, window_dims='full', input_dims='native',
                 input_overlap=0.0):
        self.sampler = sampler
        self.input_dims = input_dims
        self.window_dims = window_dims
        self.input_overlap = input_overlap
        self.subindex = None
        self._build_sliders()

    @classmethod
    def demo(WindowedSamplerDataset, key='habcam', **kwargs):
        import ndsampler
        if key == 'habcam':
            dset_fpath = ub.expandpath('~/data/noaa/Habcam_2015_g027250_a00102917_c0001_v2_vali.mscoco.json')
            workdir = ub.expandpath('~/work/bioharn')
            dset = ndsampler.CocoDataset(dset_fpath)
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
            input_dims = 'native'
            input_overlap = 0
        """
        import netharn as nh
        input_overlap = self.input_overlap
        window_dims = self.window_dims
        sampler = self.sampler

        gid_to_slider = {}
        for img in sampler.dset.imgs.values():
            if img.get('source', '') == 'habcam_2015_stereo':
                # Hack: todo, cannoncial way to get this effect
                full_dims = [img['height'], img['width'] // 2]
            else:
                full_dims = [img['height'], img['width']]

            window_dims_ = full_dims if window_dims == 'full' else window_dims
            slider = nh.util.SlidingWindow(full_dims, window_dims_,
                                           overlap=input_overlap,
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
            >>> kwplot.imshow(item['im'])

        Ignore:
            import netharn as nh
            nh.data.collate.padded_collate([self[0], self[1], self[2]])

            self = WindowedSamplerDataset.demo(window_dims='full', input_dims=(512, 512))
            kwplot.imshow(self[19]['im'])
        """
        outer, inner = self.subindex.unravel(index)
        gid = self._gids[outer]
        slider = self._sliders[outer]
        slices = slider[inner]

        tr = {'gid': gid, 'slices': slices}
        sample = self.sampler.load_sample(tr, with_annots=False)
        chip_hwc = sample['im']

        chip_dims = tuple(chip_hwc.shape[0:2])

        # Resize the image patch if necessary
        if self.input_dims != 'native' and self.input_dims != 'window':
            letterbox = nh.data.transforms.Resize(None, mode='letterbox')
            letterbox.target_size = self.input_dims[::-1]
            # Record the inverse transformation
            chip_size = chip_dims[::-1]
            input_size = self.input_dims[::-1]
            shift, scale, embed_size = letterbox._letterbox_transform(chip_size, input_size)
            # Resize the image
            chip_hwc = letterbox.augment_image(chip_hwc)
        else:
            shift = [0, 0]
            scale = [1, 1]
        scale_xy = torch.FloatTensor(scale)

        # Assume 8-bit image inputs
        chip_chw = np.transpose(chip_hwc, (2, 0, 1))
        tensor_chip = torch.FloatTensor(np.ascontiguousarray(chip_chw)) / 255.0
        offset_xy = torch.FloatTensor([slices[1].start, slices[0].start])

        # To apply a transform we first scale then shift
        tf_full_to_chip = {
            'scale_xy': torch.FloatTensor(scale_xy),
            'shift_xy': torch.FloatTensor(shift) - (offset_xy * scale_xy),
        }

        if False:
            tf_mat = np.array([
                [tf_full_to_chip['scale_xy'][0], 0, tf_full_to_chip['shift_xy'][0]],
                [0, tf_full_to_chip['scale_xy'][1], tf_full_to_chip['shift_xy'][1]],
                [0, 0, 1],
            ])
            np.linalg.inv(tf_mat)

        # This transform will bring us from chip space back to full img space
        tf_chip_to_full = {
            'scale_xy': 1.0 / tf_full_to_chip['scale_xy'],
            'shift_xy': -tf_full_to_chip['shift_xy'] * (1.0 / tf_full_to_chip['scale_xy']),
        }
        item = {
            'im': tensor_chip,
            'gid': torch.LongTensor([gid]),
            'tf_chip_to_full': tf_chip_to_full,
        }
        return item


################################################################################
# CLI

import queue  # NOQA
from threading import Thread  # NOQA


class _AsyncConsumerThread(Thread):
    """
    Will fill the queue with content of the source in a separate thread.

    >>> import queue
    >>> q = queue.Queue()
    >>> c = _background_consumer(q, range(3))
    >>> c.start()
    >>> q.get(True, 1)
    0
    >>> q.get(True, 1)
    1
    >>> q.get(True, 1)
    2
    >>> q.get(True, 1) is ub.NoParam
    True
    """
    def __init__(self, queue, source):
        Thread.__init__(self)

        self._queue = queue
        self._source = source

    def run(self):
        for item in self._source:
            self._queue.put(item)
        # Signal the consumer we are done.
        self._queue.put(ub.NoParam)


class AsyncBufferedGenerator(object):
    """Buffers content of an iterator polling the contents of the given
    iterator in a separate thread.
    When the consumer is faster than many producers, this kind of
    concurrency and buffering makes sense.

    The size parameter is the number of elements to buffer.

    The source must be threadsafe.

    References:
        http://code.activestate.com/recipes/576999-concurrent-buffer-for-generators/
    """
    def __init__(self, source, size=100):
        self._queue = queue.Queue(size)

        self._poller = _AsyncConsumerThread(self._queue, source)
        self._poller.daemon = True
        self._poller.start()

    def __iter__(self):
        while True:
            item = self._queue.get(True)
            if item is ub.NoParam:
                return
            yield item


class DetectPredictCLIConfig(scfg.Config):
    default = ub.dict_union(
        {
            'dataset': scfg.Value(None, help='coco dataset, path to images or folder of images'),
            'out_dpath': scfg.Value('./out', help='output directory'),
            'draw': scfg.Value(False),
            'workdir': scfg.Value('~/work/bioharn', help='work directory for sampler if needed'),
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
    import kwarray
    import ndsampler
    from os.path import basename, join, exists, isfile, isdir  # NOQA

    config = DetectPredictCLIConfig(config, cmdline=True)
    print('config = {}'.format(ub.repr2(config.asdict())))

    out_dpath = ub.expandpath(config.get('out_dpath'))

    import six
    if isinstance(config['dataset'], six.string_types):
        if config['dataset'].endswith('.json'):
            dataset_fpath = ub.expandpath(config['dataset'])
            coco_dset = ndsampler.CocoDataset(dataset_fpath)
            # Running prediction is much faster if you can build a sampler.
            sampler_backend = {
                'type': 'cog',
                'config': {
                    'compress': 'JPEG',
                },
                '_hack_old_names': False,  # flip to true to use legacy caches
            }
            sampler_backend = None
            print('coco hashid = {}'.format(coco_dset._build_hashid()))
        else:
            sampler_backend = None
            if exists(config['dataset']) and isfile(config['dataset']):
                # Single image case
                image_fpath = ub.expandpath(config['dataset'])
                coco_dset = ndsampler.CocoDataset()
                coco_dset.add_image(image_fpath)
    elif isinstance(config['dataset'], list):
        # Multiple image case
        gpaths = config['dataset']
        gpaths = [ub.expandpath(g) for g in gpaths]
        coco_dset = ndsampler.CocoDataset()
        for gpath in gpaths:
            coco_dset.add_image(gpath)
    else:
        raise TypeError(config['dataset'])

    draw = config.get('draw')
    workdir = ub.expandpath(config.get('workdir'))

    det_outdir = ub.ensuredir((out_dpath, 'pred'))

    pred_config = ub.dict_subset(config, DetectPredictConfig.default)

    print('Create sampler')
    sampler = ndsampler.CocoSampler(coco_dset, workdir=workdir,
                                    backend=sampler_backend)
    print('prepare frames')
    sampler.frames.prepare(workers=config['workers'])

    print('Create predictor')
    predictor = DetectPredictor(pred_config)
    print('Ensure model')
    predictor._ensure_model()

    pred_dataset = coco_dset.dataset.copy()
    pred_dataset['annotations'] = []
    pred_dset = ndsampler.CocoDataset(pred_dataset)

    # self = predictor
    predictor.config['verbose'] = 1
    pred_gen = predictor.predict_sampler(sampler)
    buffered_gen = AsyncBufferedGenerator(pred_gen, size=coco_dset.n_images)

    gid_to_pred = {}
    prog = ub.ProgIter(buffered_gen, total=coco_dset.n_images,
                       desc='buffered detect')
    for img_idx, (gid, dets) in enumerate(prog):
        gid_to_pred[gid] = dets

        for ann in dets.to_coco():
            ann['image_id'] = gid
            try:
                catname = ann['category_name']
                ann['category_id'] = pred_dset._resolve_to_cid(catname)
            except KeyError:
                if 'category_id' not in ann:
                    cid = pred_dset.add_category(catname)
                    ann['category_id'] = cid
            pred_dset.add_annotation(**ann)

        single_img_coco = pred_dset.subset([gid])
        single_pred_dpath = ub.ensuredir((det_outdir, 'single_image'))
        single_pred_fpath = join(single_pred_dpath, 'detections_gid_{:08d}.mscoco.json'.format(gid))
        single_img_coco.dump(single_pred_fpath, newlines=True)

        if draw is True or (draw and img_idx < draw):
            draw_outdir = ub.ensuredir((out_dpath, 'draw'))
            img_fpath = coco_dset.load_image_fpath(gid)
            gname = basename(img_fpath)
            viz_fname = ub.augpath(gname, prefix='detect_', ext='.jpg')
            viz_fpath = join(draw_outdir, viz_fname)

            image = kwimage.imread(img_fpath)

            flags = dets.scores > .2
            flags[kwarray.argmaxima(dets.scores, num=10)] = True
            top_dets = dets.compress(flags)
            toshow = top_dets.draw_on(image, alpha=None)
            # kwplot.imshow(toshow)
            kwimage.imwrite(viz_fpath, toshow, space='rgb')

    pred_fpath = join(det_outdir, 'detections.mscoco.json')
    print('Dump detections to pred_fpath = {!r}'.format(pred_fpath))
    pred_dset.dump(pred_fpath, newlines=True)


if __name__ == '__main__':
    detect_cli()
