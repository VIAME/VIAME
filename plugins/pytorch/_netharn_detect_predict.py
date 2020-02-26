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

        'window_dims': scfg.Value('native', help='size of a sliding window'),  # (512, 512),
        'input_dims': scfg.Value('window', help='The size of the inputs to the network'),

        'workers': 0,

        'overlap': scfg.Value(0.0, help='overlap of the sliding window'),

        # Note: these dont work exactly correct due to mmdetection model
        # differences
        'nms_thresh': 0.4,
        'conf_thresh': 0.1,

        'verbose': 1,
    }


class DetectPredictor(object):
    """
    A detector API for bioharn trained models

    Example:
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
        }
        @ub.memoize
        def _native_config():
            deployed = nh.export.DeployedModel.coerce(config['deployed'])
            # New models should have relevant params here, which is slightly
            # less hacky than using the eval.
            native_config = deployed.train_info()['other']
            common = set(native_defaults) & set(native_config)
            if len(common) != len(native_defaults):
                # Fallback on the hacky string encoding of the configs
                native_config.update(eval(
                    deployed.train_info()['extra']['config'], {}))
            return native_config
        native = {}
        for key in list(native_defaults.keys()):
            if config[key] == 'native':
                try:
                    native_config = _native_config()
                    native[key] = native_config[key]
                except Exception:
                    import warnings
                    warnings.warn((
                        'WARNING: Unable to determine native {} from model. '
                        'Defaulting to {}! Please ensure this is OK.').format(
                            key, native_defaults[key]
                    ))
                    native[key] = native_defaults[key]
            else:
                native[key] = config[key]
        return native

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
                results = self._predict_batch(batch)
                for dets in results:
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

        if self.config['window_dims'] == 'native':
            native = self._infer_native(self.config)
            window_dims = native['window_dims']

        if window_dims == 'full':
            window_dims = full_dims

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
        tf_chip_to_full = batch['tf_chip_to_full']

        scale_xy = tf_chip_to_full['scale_xy']
        shift_xy = tf_chip_to_full['shift_xy']

        if 'pad_offset_xy' in batch:
            pad_offset_xy = batch['pad_offset_xy']
            shift_xy_ = shift_xy - pad_offset_xy[None, :]
        else:
            shift_xy_ = shift_xy

        if 'disparity' in batch and self.model.module.in_channels > 3:
            batch = batch.copy()
            batch['im'] = torch.cat([batch['im'], batch['disparity']], dim=1)
            pass

        # All GPU work happens in this line
        if hasattr(self.model.module, 'detector'):
            # HACK FOR MMDET MODELS
            outputs = self.model.forward(batch, return_loss=False)
        else:
            outputs = self.model.forward(batch['im'])
            # raise NotImplementedError('only works on mmdet models')

        # Postprocess GPU outputs
        batch_dets = self.coder.decode_batch(outputs)
        for idx, det in enumerate(batch_dets):
            item_scale_xy = scale_xy[idx].numpy()
            item_shift_xy = shift_xy_[idx].numpy()
            det = det.numpy()

            if True and len(det) and np.all(det.boxes.width <= 1):
                # HACK FOR YOLO
                # TODO: decode should return detections in batch input space
                inp_size = np.array(batch['im'].shape[-2:][::-1])
                det = det.scale(inp_size)

            det = det.scale(item_scale_xy)
            det = det.translate(item_shift_xy)
            # Fix type issue
            det.data['class_idxs'] = det.data['class_idxs'].astype(np.int)
            yield det

    def predict_sampler(self, sampler, gids=None):
        """
        Predict on all images in a dataset wrapped in a ndsampler.CocoSampler

        Args:
            sampler (ndsampler.CocoDataset): dset wrapped in a sampler
            gids (List[int], default=None): if specified, then only predict
                on these images.

        Yields:
            Tuple[int, Detections] : image_id, detection pairs
        """
        native = self._infer_native(self.config)
        input_dims = native['input_dims']
        window_dims = native['window_dims']

        torch_dset = WindowedSamplerDataset(sampler, window_dims=window_dims,
                                            input_dims=input_dims, gids=gids)
        if len(torch_dset) == 0:
            return
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

        # raw_batch = ub.peek(prog)
        with torch.set_grad_enabled(False):

            # ----
            buffer_gids = []
            buffer_dets = []

            for raw_batch in prog:
                batch = {
                    'im': xpu.move(raw_batch['im']),
                    'tf_chip_to_full': raw_batch['tf_chip_to_full'],
                }
                if 'disparity' in raw_batch:
                    batch['disparity'] = xpu.move(raw_batch['disparity'])
                batch_gids = raw_batch['gid'].view(-1).numpy()
                batch_dets = list(self._predict_batch(batch))

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
        if self.input_dims is not None and self.input_dims != 'window':
            letterbox = nh.data.transforms.Resize(None, mode='letterbox')
            letterbox.target_size = self.input_dims[::-1]
            # Record the inverse transformation
            window_size = self.window_dims[::-1]
            input_size = self.input_dims[::-1]
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

    Args:
        window_dims: size of a sliding window
        input_dims: size to resize sampled windows to
        window_overlap: amount of overlap between windows
        gids : images to sample from, if None use all of them
    """

    def __init__(self, sampler, window_dims='full', input_dims='native',
                 window_overlap=0.0, gids=None):
        self.sampler = sampler
        self.input_dims = input_dims
        self.window_dims = window_dims
        self.window_overlap = window_overlap
        self.subindex = None
        self.gids = gids
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
            window_overlap = 0
        """
        import netharn as nh
        window_overlap = self.window_overlap
        window_dims = self.window_dims
        sampler = self.sampler

        gids = self.gids
        if gids is None:
            gids = list(sampler.dset.imgs.keys())

        gid_to_slider = {}
        for gid in gids:
            img = sampler.dset.imgs[gid]
            if img.get('source', '') == 'habcam_2015_stereo':
                # Hack: todo, cannoncial way to get this effect
                full_dims = [img['height'], img['width'] // 2]
            else:
                full_dims = [img['height'], img['width']]

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
        chip_hwc = kwimage.atleast_3channels(sample['im'])

        chip_dims = tuple(chip_hwc.shape[0:2])

        # Resize the image patch if necessary
        if self.input_dims != 'native' and self.input_dims != 'window':
            letterbox = nh.data.transforms.Resize(None, mode='letterbox')
            letterbox.target_size = self.input_dims[::-1]
            # Record the inverse transformation
            chip_size = np.array(chip_dims[::-1])
            input_size = np.array(self.input_dims[::-1])
            shift, scale, embed_size = letterbox._letterbox_transform(chip_size, input_size)
            # Resize the image
            chip_hwc = letterbox.augment_image(chip_hwc)
        else:
            letterbox = None
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

        sampler = self.sampler
        img = sampler.dset.imgs[gid]
        if img.get('source', '') in ['habcam_2015_stereo', 'habcam_stereo']:
            from bioharn.detect_dataset import _cached_habcam_disparity_frame
            disp_frame = _cached_habcam_disparity_frame(sampler, gid)
            data_dims = ((img['width'] // 2), img['height'])
            pad = 0
            data_slice, extra_padding, st_dims = sampler._rectify_tr(
                tr, data_dims, window_dims=None, pad=pad)
            # Load the image data
            disp_im = disp_frame[data_slice]
            if extra_padding:
                if disp_im.ndim != len(extra_padding):
                    extra_padding = extra_padding + [(0, 0)]  # Handle channels
                disp_im = np.pad(disp_im, extra_padding, **{'mode': 'constant'})
            if letterbox is not None:
                disp_im = letterbox.augment_image(disp_im)
            if len(disp_im.shape) == 2:
                disp_im = disp_im[None, :, :]
            else:
                disp_im = disp_im.transpose(2, 0, 1)
            item['disparity'] = torch.FloatTensor(disp_im)
        return item
