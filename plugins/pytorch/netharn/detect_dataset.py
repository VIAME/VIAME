from viame.pytorch import netharn as nh
import numpy as np
import torch
import ubelt as ub
import kwarray
import kwimage
import torch.utils.data.sampler
from viame.pytorch.netharn.data.channel_spec import ChannelSpec
from viame.pytorch.netharn.data.data_containers import ItemContainer
from viame.pytorch.netharn.data.data_containers import container_collate
from functools import partial
import numbers

if 0:
    _debug = print
else:
    _debug = ub.identity


class DetectFitDataset(torch.utils.data.Dataset):
    """
    Loads data with ndsampler.CocoSampler and formats it in a way suitable for
    object detection.

    Example:
        >>> # xdoc: +REQUIRES(module:osgeo)
        >>> # xdoctest: +REQUIRES(module:imgaug)
        >>> from .detect_dataset import *  # NOQA
        >>> self = DetectFitDataset.demo(key='shapes', channels='rgb|disparity', augment='heavy', window_dims=(390, 390), segmentation_bootstrap='kpts+ellipse')
        >>> index = 15
        >>> item = self[index]
        >>> # xdoc: +REQUIRES(--show)
        >>> import kwplot
        >>> kwplot.figure(doclf=True, fnum=1)
        >>> kwplot.autompl()  # xdoc: +SKIP
        >>> components = self.channels.decode(item['inputs'], axis=0)
        >>> rgb01 = components['rgb'].data.numpy().transpose(1, 2, 0)
        >>> boxes = kwimage.Boxes(item['label']['cxywh'].data.numpy(), 'cxywh')
        >>> canvas_rgb = np.ascontiguousarray(kwimage.ensure_uint255(rgb01))
        >>> canvas_rgb = boxes.draw_on(canvas_rgb)
        >>> kwplot.imshow(canvas_rgb, pnum=(1, 2, 1), fnum=1)
        >>> if 'disparity' in components:
        >>>     disp = components['disparity'].data.numpy().transpose(1, 2, 0)
        >>>     disp_canvs = np.ascontiguousarray(disp.copy())
        >>>     disp_canvs = disp_canvs - disp_canvs.min()
        >>>     disp_canvs = disp_canvs / disp_canvs.max()
        >>>     disp_canvs = boxes.draw_on(disp_canvs)
        >>>     kwplot.imshow(disp_canvs, pnum=(1, 2, 2), fnum=1)

        # mask = item['label']['class_masks'].data[0]
        # mask.float()

        masks = kwimage.structs.SegmentationList([
            kwimage.Heatmap(
                class_probs=mask.float(),
                img_dims=rgb01.shape[0:2],
                tf_data_to_img=np.eye(3),
            )
            for mask in item['label']['class_masks'].data
        ])

        >>> # Draw masks
        >>> masks = kwimage.structs.MaskList([
        >>>     kwimage.Mask(mask, 'c_mask')
        >>>     for mask in item['label']['class_masks'].data
        >>> ])
        >>> masks.draw()
        >>> kwplot.show_if_requested()

    Ignore:

        masks
    """
    def __init__(self, sampler, augment='simple', window_dims=[512, 512],
                 input_dims='window', window_overlap=0.5, scales=[-3, 6],
                 factor=32, with_mask=True, gravity=0.0,
                 classes_of_interest=None, channels='rgb',
                 blackout_ignore=True, segmentation_bootstrap=None,
                 cat_mapping=None):
        super(DetectFitDataset, self).__init__()

        self.sampler = sampler

        if input_dims == 'window':
            input_dims = window_dims

        if input_dims == 'full':
            widths = set(sampler.dset.images().width)
            heights = set(sampler.dset.images().height)
            if len(widths) != 1 and len(heights) != 1:
                raise AssertionError((
                    'Must specify a consistent input size when using full '
                    'image windows. Images have variable widths={} '
                    'and heights={}'
                ).format(widths, heights))
            else:
                input_dims = (ub.peek(heights), ub.peek(widths))

        self.with_mask = with_mask
        self.channels = ChannelSpec.coerce(channels)

        self.factor = factor  # downsample factor of yolo grid
        self.input_dims = np.array(input_dims, dtype=int)
        self.window_dims = window_dims
        self.window_overlap = window_overlap

        if segmentation_bootstrap is None:
            segmentation_bootstrap = ['given']

        if isinstance(segmentation_bootstrap, str):
            segmentation_bootstrap = segmentation_bootstrap.split('+')

        self.segmentation_bootstrap = segmentation_bootstrap

        if classes_of_interest is None:
            classes_of_interest = []
        self.classes_of_interest = {c.lower() for c in classes_of_interest}

        # Can we do this lazilly?
        self._prebuild_pool()

        window_jitter = 0.5 if augment == 'complex' else 0
        window_jitter = 0.1 if augment == 'medium' else 0
        self.window_jitter = window_jitter

        self.blackout_ignore = blackout_ignore

        self.cat_mapping = cat_mapping

        # assert np.all(self.input_dims % self.factor == 0)
        # FIXME: multiscale training is currently not enabled
        if not scales:
            scales = [1]

        rng = None
        self.rng = kwarray.ensure_rng(rng)

        if not augment:
            self.augmenter = None
        else:
            self.augmenter = DetectionAugmentor(mode=augment, gravity=gravity,
                                                rng=self.rng)

        self.disable_augmenter = False  # flag for forcing augmentor off

        # Used to resize images to the appropriate inp_size without changing
        # the aspect ratio.
        self.letterbox = nh.data.transforms.Resize(None, mode='letterbox')

        self.want_aux = self.channels.unique() - {'rgb'}

        # Storage for input statistics (populated after dataset creation ---
        # because we need the dataset to compute it!), so we can fill with real
        # mean values instead of zeros.
        self.input_stats = None

    @ub.memoize_property
    def input_id(self):
        # Use the sampler to compute an input id
        depends = [
            self.augmenter and self.augmenter.json_id(),
            self.sampler._depends(),
            self.window_dims,
            self.input_dims,
            str(self.channels),
        ]
        input_id = ub.hash_data(depends, hasher='sha512', base='abc')[0:32]
        return input_id

    def _prebuild_pool(self):
        print('Prebuild pool')
        sampler = self.sampler
        window_overlap = self.window_overlap
        window_dims = self.window_dims
        classes_of_interest = self.classes_of_interest
        ratio = 0.1
        verbose = 1

        hashid = self.sampler.dset.hashid

        dpath = ub.ensuredir((self.sampler.regions.workdir, '_cache'))
        depends = ub.odict({
            'window_overlap': window_overlap,
            'window_dims': window_dims,
            'coi': sorted(self.classes_of_interest),
            'hashid': hashid,
            'ratio': ratio,
        })
        cacher = ub.Cacher(
            'preselect_regions_v1', dpath=dpath, depends=depends,
            enabled=hashid is not None,  # need to ensure we have a good hashid to do this
            verbose=100)
        chosen_regions = cacher.tryload()
        if chosen_regions is None:
            positives, negatives = preselect_regions(
                sampler, window_overlap, window_dims, classes_of_interest,
                verbose=verbose)

            positives = kwarray.shuffle(positives, rng=971493943902)
            negatives = kwarray.shuffle(negatives, rng=119714940901)

            num_neg = int(len(positives) * ratio)
            chosen_neg = negatives[0:num_neg]

            chosen_regions = positives + chosen_neg
            cacher.save(chosen_regions)

        self.chosen_regions = chosen_regions

    @classmethod
    def demo(cls, key='shapes8', augment='simple', channels='rgb', window_dims=(512, 512), **kw):
        """

        Example:

            channels = "rgb|disparity,flowx|flowy"

            import sys, ubelt
            sys.path.append(ubelt.expandpath('~/code/bioharn'))
            from .detect_dataset import *  # NOQA
            key = ub.expandpath('$HOME/data/noaa_habcam/combos/habcam_cfarm_v8_vali_dummy_sseg.mscoco.json')
            cls = DetectFitDataset
            self = DetectFitDataset.demo(key, augment='simple', channels='rgb', window_dims=(512, 512))

            import xdev
            globals().update(**xdev.get_func_kwargs(DetectFitDataset.demo))


            loader = self.make_loader(batch_size=4, shuffle=True)
            batch = ub.peek(loader)


        Ignore:
            >>> from .detect_dataset import *  # NOQA
            >>> self = DetectFitDataset.demo('vidshapes8', augment='complex', channels='rgb|disparity,flowx|flowy')
            >>> index = 1
            >>> #
            >>> self.disable_augmenter = False
            >>> item = self[index]
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.figure(doclf=True, fnum=1)
            >>> kwplot.autompl()  # xdoc: +SKIP
            >>> components = self.channels.decode(item['inputs'], axis=0)
            >>> rgb01 = components['rgb'].data.numpy().transpose(1, 2, 0)
            >>> boxes = kwimage.Boxes(item['label']['cxywh'].data.numpy(), 'cxywh')
            >>> canvas_rgb = np.ascontiguousarray(kwimage.ensure_uint255(rgb01))
            >>> canvas_rgb = boxes.draw_on(canvas_rgb)
            >>> pnum_ = kwplot.PlotNums(nSubplots=len(components))
            >>> kwplot.imshow(canvas_rgb, pnum=pnum_(), fnum=1, title='rgb')
            >>> for aux_key, aux_im in ub.dict_diff(components, {'rgb'}).items():
            >>>     aux_canvs = kwimage.normalize(aux_im.data.numpy().transpose(1, 2, 0))
            >>>     aux_canvs = boxes.draw_on(aux_canvs)
            >>>     kwplot.imshow(aux_canvs, pnum=pnum_(), fnum=1, title=aux_key)
            >>> kwplot.show_if_requested()
        """
        import ndsampler
        from os.path import exists

        import inspect
        sig = inspect.signature(cls.__init__)
        cls_kw = ub.dict_isect(kw, list(sig.parameters.keys()))
        sampler_kw = ub.dict_diff(kw, cls_kw)

        channels = ChannelSpec.coerce(channels)
        if exists(key):
            import kwcoco
            dset = kwcoco.CocoDataset(key)
            from .detect_fit import DetectFitConfig
            config = DetectFitConfig()
            sampler = ndsampler.CocoSampler(dset, workdir=config['workdir'])
        else:
            aux = list(channels.difference(ChannelSpec.coerce('rgb')).keys())
            sampler = ndsampler.CocoSampler.demo(key, aux=aux, **sampler_kw)

        self = cls(sampler, augment=augment, window_dims=window_dims,
                   channels=channels, **cls_kw)
        return self

    def __len__(self):
        # TODO: Use sliding windows so detection can be run and trained on
        # larger images
        return len(self.chosen_regions)
        # return len(self.sampler.image_ids)

    def __getitem__(self, spec):
        """
        Example:
            >>> # DISABLE_DOCTSET
            >>> from .detect_dataset import *  # NOQA
            >>> torch_dset = self = DetectFitDataset.demo(
            >>>     key='vidshapes32', channels="rgb|disparity,flowx|flowy",
            >>>     augment='complex', window_dims=(512, 512), gsize=(1920, 1080),
            >>>     segmentation_bootstrap='kpts+given+ellipse')
            >>> index = 10
            >>> spec = {'index': index, 'input_dims': (120, 120)}
            >>> item = self[spec]
            >>> chw01 = self.channels.decode(item['inputs'], axis=0)['rgb'].data.numpy()
            >>> hwc01 = chw01.transpose(1, 2, 0)
            >>> print(hwc01.shape)
            >>> boxes = kwimage.Boxes(item['label']['cxywh'].data.numpy(), 'cxywh')
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> plt = kwplot.autoplt()  # xdoc: +SKIP
            >>> kwplot.figure(doclf=True, fnum=1)
            >>> kwplot.imshow(hwc01)
            >>> labels = ['w={}'.format(w) for w in item['label']['weight'].data]
            >>> boxes.draw(labels=labels)
            >>> for mask in item['label']['class_masks'].data:
            ...     kwimage.Mask(mask.data.cpu().numpy(), 'c_mask').draw()
            >>> fig = plt.gcf()
            >>> for o in fig.findobj():  # http://matplotlib.1069221.n5.nabble.com/How-to-turn-off-all-clipping-td1813.html
            >>>     o.set_clip_on(False)
            >>> kwplot.show_if_requested()

        Ignore:
            >>> from .detect_dataset import *  # NOQA
            >>> fpath = ub.expandpath('$HOME/remote/namek/data/noaa_habcam/Habcam_2015_g027250_a00102917_c0001_v2_vali.mscoco.json')
            >>> fpath = ub.expandpath('$HOME/remote/namek/data/noaa_habcam/combos/may_priority_habcam_cfarm_v7_vali.mscoco.json')
            >>> fpath = ub.expandpath('$HOME/remote/namek/data/noaa_habcam/combos/habcam_cfarm_v8_vali_dummy_sseg.mscoco.json')
            >>> self = DetectFitDataset.demo(key=fpath, augment='complex', channels='rgb|disparity')
            >>> spec = {'index': 954, 'input_dims': (300, 300)}
            >>> item = self[spec]
            >>> hwc01 = item['inputs']['rgb'].data.numpy().transpose(1, 2, 0)
            >>> disparity = item['disparity'].data
            >>> boxes = kwimage.Boxes(item['label']['cxywh'].data.numpy(), 'cxywh')
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> kwplot.figure(doclf=True, fnum=1, pnum=(1, 2, 1))
            >>> kwplot.autompl()  # xdoc: +SKIP
            >>> kwplot.imshow(hwc01, fnum=1, pnum=(1, 2, 1))
            >>> boxes.draw()
            >>> for mask, flag in zip(item['label']['class_masks'].data, item['label']['has_mask'].data):
            >>>      if flag > 0:
            >>>          kwimage.Mask(mask.data.cpu().numpy(), 'c_mask').draw()
            >>> kwplot.imshow(disparity, fnum=1, pnum=(1, 2, 2))
            >>> boxes.draw()
            >>> kwplot.show_if_requested()
        """

        if isinstance(spec, dict):
            index = spec['index']
            input_dims = spec['input_dims']
        elif isinstance(spec, numbers.Integral):
            index = int(spec)
            input_dims = self.input_dims
        else:
            raise TypeError(type(spec))

        inp_size = np.array(input_dims[::-1])

        gid, slices, _ = self.chosen_regions[index]

        if self.augmenter is not None and self.window_jitter and not self.disable_augmenter:
            # jitter the sliding window location a little bit.
            jitter = self.window_jitter
            y1 = slices[0].start
            y2 = slices[0].stop
            x1 = slices[1].start
            x2 = slices[1].stop
            box = kwimage.Boxes([[x1, y1, x2, y2]], 'tlbr')
            rng = self.rng
            offset = (int(box.width[0, 0] * jitter * (0.5 - rng.rand())),
                      int(box.height[0, 0] * jitter * (.5 - rng.rand())))
            box = box.translate(offset)
            x1, y1, x2, y2 = map(int, box.data[0])
            slices = tuple([slice(y1, y2), slice(x1, x2)])

        tr = {'gid': gid, 'slices': slices}
        _debug('tr = {!r}'.format(tr))

        # TODO: instead of forcing ourselfs to compute an iffy pad, we could
        # instead separate out all the non-square geometric augmentations and
        # then augment a bounding polygon representing the original region.
        # Based on that result we sample the appropriate data at the
        # appropriate scale. Then we apply the intensity based augmentors
        # after.
        pad = int((slices[0].stop - slices[0].start) * 0.3)

        aux_components = None
        _debug('self.channels = {!r}'.format(self.channels))

        if self.want_aux:
            sampler = self.sampler
            want_aux = self.want_aux
            aux_components = load_sample_auxiliary(
                sampler, tr, want_aux, pad=pad)
            # if disp_im.max() > 1.0:
            #     raise AssertionError('gid={} {}'.format(gid, ub.repr2(kwarray.stats_dict(disp_im))))
            if not aux_components:
                # disp_im = np.zeros()
                raise Exception('no auxiliary disparity')

        _debug('self.with_mask = {!r}'.format(self.with_mask))
        with_annots = ['boxes']

        if self.with_mask:
            sseg_method = self.rng.choice(self.segmentation_bootstrap)
            with_annots += ['segmentation']
            if sseg_method == 'kpts':
                with_annots += ['keypoints']

        # NOTE: using the gdal backend samples HABCAM images in 16ms, and no
        # backend samples clocks in at 72ms. The disparity speedup is about 2x
        sample = self.sampler.load_sample(tr, visible_thresh=0.05,
                                          with_annots=with_annots, pad=pad)

        _debug('sample = {!r}'.format(sample))
        imdata = kwimage.atleast_3channels(sample['im'])[..., 0:3]

        sample_annots = sample['annots']

        boxes = sample_annots['rel_boxes'].view(-1, 4)
        cids = sample_annots['cids']
        aids = sample_annots['aids']
        anns = list(ub.take(self.sampler.dset.anns, aids))
        weights = [ann.get('weight', 1.0) for ann in anns]

        for idx, cid in enumerate(cids):
            # set weights of uncertain varaibles to zero
            catname = self.sampler.dset._resolve_to_cat(cid)['name']
            if catname.lower() in {'unknown', 'ignore'}:
                weights[idx] = 0

            if self.classes_of_interest:
                if catname.lower() not in self.classes_of_interest:
                    weights[idx] = 0

        if self.cat_mapping is None:
            classes = self.sampler.classes
            class_idxs = np.array([classes.id_to_idx[cid] for cid in cids])
        else:
            # Ensure we are using the class ids / idxs of the "training" or
            # target categories that were passed to the network
            classes = self.cat_mapping['target']
            class_idxs = np.array([
                classes.id_to_idx[self.cat_mapping['id'][cid]] for cid in cids])

        detskw = {
            'boxes': boxes,
            'class_idxs': class_idxs,
            'weights': np.array(weights, dtype=np.float32),
            'classes': classes,
        }

        if 'rel_kpts' in sample_annots:
            detskw['keypoints'] = sample_annots['rel_kpts']
        if 'rel_ssegs' in sample_annots:
            detskw['segmentations'] = sample_annots['rel_ssegs']

        dets = kwimage.Detections(**detskw)
        _debug('dets = {!r}'.format(dets))
        orig_size = np.array(imdata.shape[0:2][::-1])

        if self.augmenter and not self.disable_augmenter:
            _debug('augment')
            imdata, dets, aux_components = self.augmenter.augment_data(
                imdata, dets, aux_components)
            # disp_im.dtype

        _debug('un-pad')
        pad = sample['params']['pad']
        if np.any(pad):
            # if we gave extra padding, crop back to the original shape
            y_sl, x_sl = [slice(d_pad, d - d_pad) for d, d_pad in
                          zip(imdata.shape[0:2], pad)]
            imdata = imdata[y_sl, x_sl]
            if aux_components is not None:
                for auxkey, aux_im in aux_components.items():
                    aux_components[auxkey] = aux_im[y_sl, x_sl]
            dets = dets.translate([-x_sl.start, -y_sl.start])

        # Ignore any box that is cutoff.
        ignore_thresh = 0.4
        h, w = imdata.shape[0:2]
        frame_box = kwimage.Boxes([[0, 0, w, h]], 'xywh')
        isect = dets.boxes.isect_area(frame_box)
        visibility = (isect / dets.boxes.area)[:, 0]
        ignore_flags = (visibility < ignore_thresh).astype(np.float32)
        dets.data['weights'] *= (1.0 - ignore_flags)

        dets = dets.compress(visibility > 0)

        # Apply letterbox resize transform to train and test
        _debug('imresize')
        self.letterbox.target_size = inp_size
        prelb_dims = imdata.shape[0:2]
        imdata = self.letterbox.augment_image(imdata)
        postlb_dims = imdata.shape[0:2]
        if aux_components is not None:
            # note: the letterbox augment doesn't handle floats well
            # use the kwimage.imresize instead
            for auxkey, aux_im in aux_components.items():
                aux_components[auxkey] = kwimage.imresize(
                    aux_im, dsize=self.letterbox.target_size,
                    letterbox=True).clip(0, 1)
        if len(dets):
            _debug('warp')
            dets = dets.warp(self.letterbox,
                             input_dims=prelb_dims,
                             output_dims=postlb_dims)

        # Remove any boxes that are no longer visible or out of bounds
        flags = (dets.boxes.area > 0).ravel()
        dets = dets.compress(flags)

        chw01 = torch.FloatTensor(imdata.transpose(2, 0, 1) / 255.0)
        cxwh = dets.boxes.toformat('cxywh')

        # Return index information in the label as well
        orig_size = torch.LongTensor(orig_size)
        index = torch.LongTensor([index])
        bg_weight = torch.FloatTensor([1.0])

        label = {
            'cxywh': ItemContainer(torch.FloatTensor(cxwh.data), stack=False),
            'class_idxs': ItemContainer(torch.LongTensor(dets.class_idxs), stack=False),
            'weight': ItemContainer(torch.FloatTensor(dets.weights), stack=False),

            'indices': ItemContainer(index, stack=False),
            'orig_sizes': ItemContainer(orig_size, stack=False),
            'bg_weights': ItemContainer(bg_weight, stack=False),
        }
        _debug('label = {!r}'.format(label))

        if self.blackout_ignore:
            # Black out any region marked as ignore
            norm_classes = [c.lower() for c in classes]
            ignore_catname = 'ignore'
            # ignore_catname = 'eff'
            if ignore_catname in norm_classes:
                ignore_cidx = norm_classes.index(ignore_catname)
                ignore_boxes = dets.boxes.compress(dets.class_idxs == ignore_cidx)
                ignore_tlbr = ignore_boxes.to_ltrb()
                for tlbr_row in ignore_tlbr.data:
                    tl_x, tl_y, br_x, br_y  = tlbr_row
                    slx = slice(int(tl_x), int(br_x))
                    sly = slice(int(tl_y), int(br_y))
                    chw01[:, sly, slx] = 0

        if self.with_mask:
            has_mask_list = []
            class_mask_list = []
            if sseg_method == 'given' and 'segmentations' in dets.data:
                # Convert segmentations to masks
                h, w = chw01.shape[1:]
                for sseg in dets.data['segmentations']:
                    if sseg is not None:
                        mask = sseg.to_mask(dims=chw01.shape[1:])
                        c_mask = mask.to_c_mask().data
                        mask_tensor = torch.tensor(c_mask, dtype=torch.uint8)
                        class_mask_list.append(mask_tensor[None, :])
                        has_mask_list.append(1)
                    else:
                        bad_mask = torch.empty((1, h, w), dtype=torch.uint8)
                        bad_mask = torch.full((1, h, w), fill_value=2, dtype=torch.uint8)
                        class_mask_list.append(bad_mask)
                        has_mask_list.append(-1)

                has_mask = torch.tensor(has_mask_list, dtype=torch.int8)
                if len(class_mask_list) == 0:
                    class_masks = torch.empty((0, h, w), dtype=torch.uint8)
                else:
                    class_masks = torch.cat(class_mask_list, dim=0)
                label['class_masks'] = ItemContainer(class_masks, stack=False, cpu_only=True)
                label['has_mask'] = ItemContainer(has_mask, stack=False)

            if sseg_method == 'ellipse':
                for box in dets.data['boxes']:
                    if box is not None:
                        import cv2
                        mask = np.zeros(chw01.shape[1:], dtype=np.float32)

                        center = tuple(map(int, box.center))
                        axes = (int(box.width) // 3, int(box.height) // 3)
                        color_ = 1
                        cv2.ellipse(mask, center, axes, angle=0.0, startAngle=0.0,
                                    endAngle=360.0, color=color_, thickness=-1)

                        mask_tensor = torch.tensor(mask, dtype=torch.float32)
                        class_mask_list.append(mask_tensor[None, :])
                        has_mask_list.append(1)
                    else:
                        bad_mask = torch.empty((1, h, w), dtype=torch.float32)
                        bad_mask = torch.full((1, h, w), fill_value=2, dtype=torch.uint8)
                        class_mask_list.append(bad_mask)
                        has_mask_list.append(-1)

                has_mask = torch.tensor(has_mask_list, dtype=torch.int8)
                if len(class_mask_list) == 0:
                    class_masks = torch.empty((0, h, w), dtype=torch.uint8)
                else:
                    class_masks = torch.cat(class_mask_list, dim=0)
                label['class_masks'] = ItemContainer(class_masks, stack=False, cpu_only=True)
                label['has_mask'] = ItemContainer(has_mask, stack=False)

            if sseg_method == 'kpts' and 'keypoints' in dets.data:
                # Make a pseudo segmentation based on keypoints
                pts_list = dets.data['keypoints']
                for pts in pts_list:
                    if pts is not None:
                        mask = np.zeros(chw01.shape[1:], dtype=np.float32)
                        # mask = pts.data['xy'].fill(mask, value=1.0)
                        pts.data['xy'].soft_fill(mask, coord_axes=[1, 0], radius=5)
                        mask_tensor = torch.tensor(mask, dtype=torch.float32)
                        class_mask_list.append(mask_tensor[None, :])
                        has_mask_list.append(1)
                    else:
                        bad_mask = torch.empty((1, h, w), dtype=torch.float32)
                        bad_mask = torch.full((1, h, w), fill_value=2, dtype=torch.uint8)
                        class_mask_list.append(bad_mask)
                        has_mask_list.append(-1)

                has_mask = torch.tensor(has_mask_list, dtype=torch.int8)
                if len(class_mask_list) == 0:
                    class_masks = torch.empty((0, h, w), dtype=torch.uint8)
                else:
                    class_masks = torch.cat(class_mask_list, dim=0)
                label['class_masks'] = ItemContainer(class_masks, stack=False, cpu_only=True)
                label['has_mask'] = ItemContainer(has_mask, stack=False)

        components = {
            'rgb': chw01,
        }
        if aux_components is not None:
            for auxkey, aux_im in aux_components.items():
                aux_im = kwarray.atleast_nd(aux_im, 3)
                components[auxkey] = torch.FloatTensor(
                    aux_im.transpose(2, 0, 1))
        _debug('components = {!r}'.format(components))
        _debug('aux_components = {!r}'.format(aux_components))

        inputs = {
            k: ItemContainer(v, stack=True)
            for k, v in self.channels.encode(components).items()
        }
        _debug('inputs = {!r}'.format(inputs))

        item = {
            'inputs': inputs,
            'label': label,
            'tr': ItemContainer(sample['tr'], stack=False),
        }
        _debug('item = {!r}'.format(item))
        return item

    def make_loader(self, batch_size=16, num_workers=0, shuffle=False,
                    pin_memory=False, drop_last=False, multiscale=False,
                    balance=False, num_batches='auto', xpu=None):
        """
        CommandLine:
            xdoctest -m /home/joncrall/code/bioharn/bioharn/detect_dataset.py DetectFitDataset.make_loader

        Example:
            >>> # xdoctest: +REQUIRES(module:imgaug)
            >>> from .detect_dataset import *  # NOQA
            >>> self = DetectFitDataset.demo('shapes32')
            >>> self.augmenter = None
            >>> loader = self.make_loader(batch_size=4, shuffle=True, balance='tfidf')
            >>> loader.batch_sampler.index_to_prob
            >>> loader = self.make_loader(batch_size=1, shuffle=True)
            >>> # training batches should have multiple shapes
            >>> shapes = set()
            >>> for raw_batch in ub.ProgIter(iter(loader), total=len(loader)):
            >>>     inputs = raw_batch['inputs']['rgb']
            >>>     # test to see multiscale works
            >>>     shapes.add(inputs.data[0].shape[-1])
            >>>     if len(shapes) > 1:
            >>>         break

        Example:
            >>> # xdoctest: +REQUIRES(module:imgaug)
            >>> from .detect_dataset import *  # NOQA
            >>> import pytest
            >>> self = DetectFitDataset.demo('shapes8')
            >>> with pytest.raises(IndexError):
            >>>     loader = self.make_loader(batch_size=3, num_batches=1000, shuffle=True)
        """
        if len(self) == 0:
            raise ValueError('must have some data')

        if balance != 'tfidf':
            # The case where where replacement is not allowed
            if num_batches == 'auto':
                num_samples = None
            else:
                num_samples = num_batches * batch_size

            if shuffle:
                from viame.pytorch.netharn.data.batch_samplers import PatchedRandomSampler
                item_sampler = PatchedRandomSampler(self, num_samples=num_samples)
            else:
                from viame.pytorch.netharn.data.batch_samplers import SubsetSampler
                import itertools as it
                REBALANCE_DETERMINISTIC_SHUFFLE = True
                if REBALANCE_DETERMINISTIC_SHUFFLE:
                    # Make is such that the first few validation batches
                    # have instances of all true categories
                    dset = self.sampler.dset
                    cid_to_idxs = ub.ddict(list)

                    for idx, region in enumerate(self.chosen_regions):
                        (gid, slices, aids) = region
                        region_cids = dset.annots(aids).lookup('category_id')
                        if len(region_cids) == 0:
                            # Handle case where there are no annots in a region
                            cid_to_idxs[None].append(idx)
                        else:
                            # Mark that this region index contains this category
                            for cid in set(region_cids):
                                cid_to_idxs[cid].append(idx)

                    # Deterministic shuffle
                    rng = kwarray.ensure_rng(58286)
                    idx_groups = []
                    for idxs in cid_to_idxs.values():
                        idxs = sorted(idxs)
                        rng.shuffle(idxs)
                        idx_groups.append(idxs)

                    it.zip_longest
                    flat_idxs = ub.flatten(list(it.zip_longest(*idx_groups)))
                    flat_idxs = [idx for idx in flat_idxs if idx is not None]

                    if num_samples is None:
                        final_idxs = flat_idxs
                    else:
                        final_idxs = flat_idxs[:num_samples]
                    item_sampler = SubsetSampler(final_idxs)
                else:
                    if num_samples is None:
                        item_sampler = torch.utils.data.sampler.SequentialSampler(self)
                    else:
                        stats_idxs = (np.arange(num_samples) % len(self))
                        item_sampler = SubsetSampler(stats_idxs)

            if num_samples is not None:
                # If num_batches is too big, what should the behavior be?
                #   1. Sample with replacement to achieve the requested num_batches
                #   2. Clip num_batches so it can't exceed (num_items // batch_size)
                # ✓ 3. Error
                # We could implement other strategies via configuration options if needed
                if num_samples > len(self):
                    raise IndexError(
                        'num_batches={} and batch_size={} causes '
                        'num_samples={} to be greater than the number '
                        'of data items {}. Try setting num_batches=auto?'.format(
                            num_batches, batch_size, num_samples, len(self)))

        if balance == 'tfidf':
            if not shuffle:
                raise AssertionError('for now you must shuffle when you balance')
            if balance != 'tfidf':
                raise AssertionError('for now balance must be tfidf')

            # label_freq = ub.map_vals(len, self.sampler.dset.index.cid_to_aids)
            anns = self.sampler.dset.anns
            cats = self.sampler.dset.cats

            assert not multiscale, 'multiscale and balance not yet compatible'

            label_to_weight = None
            if self.classes_of_interest:
                # Only give sampling weight to categories we care about
                label_to_weight = {cat['name']: 0 for cat in cats.values()}
                for cname in self.classes_of_interest:
                    label_to_weight[cname] = 1

            index_to_labels = [
                np.array([cats[anns[aid]['category_id']]['name'] for aid in aids], dtype=str)
                for gid, slices, aids in self.chosen_regions
            ]

            batch_sampler = nh.data.batch_samplers.GroupedBalancedBatchSampler(
                index_to_labels, batch_size=batch_size, num_batches=num_batches,
                shuffle=shuffle, label_to_weight=label_to_weight, rng=None
            )
            print('balanced batch_sampler = {!r}'.format(batch_sampler))
            batch_sampler._balance_report()
        elif multiscale:
            batch_sampler = MultiScaleBatchSampler2(
                item_sampler, batch_size=batch_size, drop_last=drop_last,
                factor=32, scales=[-9, 1])
        else:
            # batch_sampler = torch.utils.data.BatchSampler(
            #     item_sampler, batch_size=batch_size, drop_last=drop_last)
            from viame.pytorch.netharn.data.batch_samplers import PatchedBatchSampler
            batch_sampler = PatchedBatchSampler(
                item_sampler, batch_size=batch_size, drop_last=drop_last,
                num_batches=num_batches)

        # torch.utils.data.sampler.WeightedRandomSampler

        if xpu is None:
            num_devices = 1
        else:
            num_devices = len(xpu.devices)

        collate_fn = partial(container_collate, num_devices=num_devices)
        # collate_fn = nh.data.collate.padded_collate

        loader = torch.utils.data.DataLoader(
            self, batch_sampler=batch_sampler,
            collate_fn=collate_fn, num_workers=num_workers,
            pin_memory=pin_memory, worker_init_fn=worker_init_fn)
        return loader


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()  # TODO
    self = worker_info.dataset

    if hasattr(self.sampler.dset, 'connect'):
        # Reconnect to the backend if we are using SQL
        self.sampler.dset.connect(readonly=True)

    # Make loaders more random
    kwarray.seed_global(np.random.get_state()[1][0] + worker_id)
    if self.augmenter:
        rng = kwarray.ensure_rng(None)
        reseed_(self.augmenter, rng)


def load_sample_auxiliary(sampler, tr, want_aux, pad=0):
    """
    Prototype for auxiliary channel loading that should eventually be merged
    into ndsampler itself.

    Args:
        sampler (CocoSampler):
        tr (Dict): spatial localization of the target region
        want_aux (List[str]): list of desired raw channels to load
        pad (int): padding

    TODO:
        - [ ] Handle case when auxiliary data is not aligned with the main data
        - [ ] Don't rely on the auxiliary data being in COG format.
        - [ ] Improve efficiency when all components of a disk-fused stream
              are present.
        - [ ] Improve efficiency of creating the disk_fusion channel spec

    Example:
        >>> # xdoctest: +REQUIRES(module:gdal)
        >>> from .detect_dataset import *  # NOQA
        >>> import ndsampler
        >>> from viame.pytorch.netharn.data.channel_spec import ChannelSpec
        >>> want_aux = ChannelSpec.coerce('disparity,flowx|flowy').unique()
        >>> sampler = ndsampler.CocoSampler.demo('vidshapes8-aux')
        >>> pad = 0
        >>> tr = {'gid': 1, 'cx': 10, 'cy': 10, 'width': 5, 'height': 5}
        >>> aux_components = load_sample_auxiliary(sampler, tr, want_aux)
        >>> print(ub.map_vals(lambda x: x.shape, aux_components))
    """
    gid = tr['gid']
    img = sampler.dset.imgs[gid]

    from ndsampler.utils import util_gdal
    import os
    if 'auxillary' in img:
        img['auxiliary'] = img['auxillary']  # Hack

    if 'auxiliary' not in img:
        raise ValueError('Image does not have auxiliary information')

    disk_fusion = ChannelSpec(','.join([a['channels'] for a in img['auxiliary']]))
    component_indices = disk_fusion.component_indices(axis=0)
    component_indices = ub.dict_isect(component_indices, want_aux)
    aux_components = {}
    for part, (key, subindex) in component_indices.items():
        # TODO: The channels may be fused on disk different than we would
        # like them to be fused in the network. We need a way of rectifying
        # available channels with requested channels.

        # First check if the dataset defines a proper disparity channel
        disp_fpath = sampler.dset.get_auxiliary_fpath(gid, key)

        aux_frame = util_gdal.LazyGDalFrameFile(os.fspath(disp_fpath))
        data_dims = aux_frame.shape[0:2]

        tr = sampler._infer_target_attributes(tr)
        data_slice, extra_padding = kwarray.embed_slice(
            tr['space_slice'], data_dims=data_dims, pad=pad)
        # data_slice, extra_padding, st_dims = sampler._rectify_tr(
        #     tr, data_dims, window_dims=None, pad=pad)

        if part != key:
            # disk input is prefused, need to separate it out
            data_slice = data_slice + tuple(subindex)

        # Load the image data
        aux_im = aux_frame[data_slice]
        if extra_padding:
            if aux_im.ndim != len(extra_padding):
                extra_padding = extra_padding + [(0, 0)]  # Handle channels
            aux_im = np.pad(aux_im, extra_padding, **{'mode': 'constant'})
        aux_components[part] = aux_im

    return aux_components


class RandomSampler(torch.utils.data.sampler.RandomSampler):
    r"""
    Extends torch RandomSampler allow num_samples when replacement=False

    See: https://github.com/pytorch/pytorch/issues/38032
    """

    def __init__(self, data_source, replacement=False, num_samples=None):
        if num_samples == 'auto':
            num_samples = None

        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples

        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        else:
            return iter(torch.randperm(n).tolist()[:self._num_samples])

    def __len__(self):
        return self.num_samples


def reseed_(auger, rng):
    if hasattr(auger, 'seed_'):
        return auger.seed_(rng)
    else:
        return auger.reseed(rng)


class MultiScaleBatchSampler2(torch.utils.data.sampler.BatchSampler):
    """
    Indicies returned in the batch are tuples indicating data index and scale
    index. Requires that dataset has a `multi_scale_inp_size` attribute.

    Args:
        sampler (Sampler): Base sampler. Must have a data_source attribute.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
        resample_freq (int): how often to change scales. if None, then
            only one scale is used.

    Example:
        >>> import torch.utils.data as torch_data
        >>> class DummyDatset(torch_data.Dataset):
        >>>     def __init__(self):
        >>>         super(DummyDatset, self).__init__()
        >>>         self.input_dims = (512, 512)
        >>>     def __len__(self):
        >>>         return 1000
        >>> batch_size = 16
        >>> data_source = DummyDatset()
        >>> sampler = sampler1 = torch.utils.data.sampler.RandomSampler(data_source)
        >>> self = rand = MultiScaleBatchSampler2(sampler1, resample_freq=10)
        >>> sampler2 = torch.utils.data.sampler.SequentialSampler(data_source)
        >>> seq = MultiScaleBatchSampler2(sampler2, resample_freq=None)
        >>> rand_idxs = list(iter(rand))
        >>> seq_idxs = list(iter(seq))
    """

    def __init__(self, sampler, batch_size=16, resample_freq=10, factor=32,
                 scales=[-9, 1], drop_last=False):

        self.sampler = sampler
        self.drop_last = drop_last

        self.resample_interval = resample_freq

        input_dims = np.array(sampler.data_source.input_dims)

        self.base_input_dims = input_dims
        self.base_batch_size = batch_size

        factor_coeff = sorted(range(*scales), key=abs)
        factor = 32
        self.multi_scale_inp_size = [
            input_dims + (factor * i) for i in factor_coeff]

        self._batch_dynamics = [
            {'batch_size': batch_size, 'input_dims': dims}
            for dims in self.multi_scale_inp_size
        ]
        self.batch_size = None
        if ub.allsame(d['batch_size'] for d in self._batch_dynamics):
            self.batch_size = self._batch_dynamics[0]['batch_size']

        self.num_batches = None
        total = len(sampler)
        if self.drop_last:
            self.num_batches = int(np.floor(total / self.base_batch_size))
        else:
            self.num_batches = int(np.ceil(total / self.base_batch_size))

        self._dynamic_schedule = None
        self.rng = kwarray.ensure_rng(None)

    def __nice__(self):
        return str(len(self))

    def __len__(self):
        return self.num_batches

    def _init_dynamic_schedule(self):
        # print("INIT NEW DYNAMIC SCHEDULE")
        self._dynamic_schedule = ub.odict()
        total = len(self.sampler)
        remain = total

        # Always end on the native dynamic
        native_dynamic = {
            'batch_size': self.base_batch_size,
            'input_dims': self.base_input_dims,
        }

        if self.resample_interval is not None:
            final_native = self.resample_interval

            num_final = final_native * native_dynamic['batch_size']

            bx = 0
            while remain > 0:
                if remain <= num_final or bx == 0:
                    # The first and last batches will use the native
                    # input_dims.
                    current = native_dynamic.copy()
                    current['remain'] = remain
                    self._dynamic_schedule[bx] = current
                elif bx % self.resample_interval == 0:
                    dyn_idx = self.rng.randint(len(self._batch_dynamics))
                    current = self._batch_dynamics[dyn_idx]
                    current = current.copy()
                    if remain < 0:
                        current['batch_size'] += remain
                    current['remain'] = remain
                    self._dynamic_schedule[bx] = current

                    if remain < num_final:
                        # Ensure there are enough items for final batches
                        current['remain'] = remain
                        current['batch_size'] -= (num_final - remain)
                        self._dynamic_schedule[bx] = current

                if remain <= current['batch_size']:
                    current['batch_size'] = remain
                    current['remain'] = remain
                    current = current.copy()
                    self._dynamic_schedule[bx] = current

                bx += 1
                remain = remain - current['batch_size']
        else:
            self._dynamic_schedule[0] = {
                'batch_size': self.batch_size,
                'remain': total,
                'input_dims': self.base_input_dims,
            }

        final_bx, final_dynamic = list(self._dynamic_schedule.items())[-1]

        if self.drop_last:
            last = int(np.floor(final_dynamic['remain'] / final_dynamic['batch_size']))
        else:
            last = int(np.ceil(final_dynamic['remain'] / final_dynamic['batch_size']))

        num_batches = final_bx + last
        self.num_batches = num_batches

        # print(ub.repr2(self._dynamic_schedule, nl=1))
        # print('NEW SCHEDULE')

    def __iter__(self):
        # Start first batch
        self._init_dynamic_schedule()

        bx = 0
        batch = []
        if bx in self._dynamic_schedule:
            current_dynamic = self._dynamic_schedule[bx]
        # print('RESAMPLE current_dynamic = {!r}'.format(current_dynamic))

        for idx in self.sampler:
            # Specify dynamic information to the dataset
            index = {
                'index': idx,
                'input_dims': current_dynamic['input_dims'],
            }
            batch.append(index)
            if len(batch) == current_dynamic['batch_size']:
                yield batch

                # Start next batch
                bx += 1
                batch = []
                if bx in self._dynamic_schedule:
                    current_dynamic = self._dynamic_schedule[bx]
                    # print('RESAMPLE current_dynamic = {!r}'.format(current_dynamic))

        if len(batch) > 0 and not self.drop_last:
            yield batch


def preselect_regions(sampler, window_overlap, window_dims,
                      classes_of_interest=None,
                      ignore_coverage_thresh=0.6,
                      negative_classes={'ignore', 'background'}, verbose=1):
    """
    TODO: this might be generalized and added to ndsampler

    Args:
        ignore_coverage_thresh : if specified any window covered by an ignore
            box greater than this percent is not returned.

        negative_classes : classes to consider as negative

    window_overlap = 0.5
    window_dims = (512, 512)

    Ignore:
        sampler = self.sampler
        window_overlap = self.window_overlap
        window_dims = self.window_dims
        classes_of_interest = self.classes_of_interest

        import kwcoco
        fpath = ub.expandpath('$HOME/remote/viame/data/US_ALASKA_MML_SEALION/sealions_all_refined_v8_manual_train.mscoco.json')
        coco_dset = kwcoco.CocoDataset(fpath)

        import ndsampler
        sampler = ndsampler.CocoSampler(coco_dset)
        window_overlap = 0.5
        window_dims = (512, 512)
        ignore_coverage_thresh = 0.8
        negative_classes={'ignore', 'background'}

    """
    from viame.pytorch import netharn as nh

    keepbound = True

    # Create a sliding window object for each specific image (because they may
    # have different sizes, technically we could memoize this)
    dset = sampler.dset
    dset._ensure_imgsize()

    gid_height_width_list = [
        (img['id'], img['height'], img['width'] // 2)
        if img.get('source', '') == 'habcam_2015_stereo' else
        (img['id'], img['height'], img['width'])
        for img in ub.ProgIter(dset.imgs.values(), total=len(dset.imgs),
                               desc='load image sizes', verbose=verbose)]

    if any(h is None or w is None for gid, h, w in gid_height_width_list):
        raise ValueError('All images must contain width and height attrs.')

    @ub.memoize
    def _memo_slider(full_dims, window_dims):
        window_dims_ = full_dims if window_dims == 'full' else window_dims
        slider = nh.util.SlidingWindow(
            full_dims, window_dims_, overlap=window_overlap,
            keepbound=keepbound, allow_overshoot=True)
        slider.regions = list(slider)
        return slider

    gid_to_slider = {
        gid: _memo_slider(full_dims=(height, width), window_dims=window_dims)
        for gid, height, width in ub.ProgIter(
            gid_height_width_list, desc='build sliders', verbose=verbose)
    }

    from ndsampler import isect_indexer
    _isect_index = isect_indexer.FrameIntersectionIndex.from_coco(
        dset, verbose=verbose)

    if hasattr(dset, '_all_rows_column_lookup'):
        # SQL optimization
        aids_cids = dset._all_rows_column_lookup(
            'annotations', ['id', 'category_id'])
        aid_to_cid = dict(aids_cids)
    else:
        aid_to_cid = None

    positives = []
    negatives = []
    for gid, slider in ub.ProgIter(gid_to_slider.items(),
                                   total=len(gid_to_slider),
                                   desc='finding regions with annots',
                                   verbose=verbose):
        # For each image, create a box for each spatial region in the slider
        boxes = []
        regions = list(slider)
        for region in regions:
            y_sl, x_sl = region
            boxes.append([x_sl.start,  y_sl.start, x_sl.stop, y_sl.stop])
        boxes = kwimage.Boxes(np.array(boxes), 'tlbr')

        for region, box in zip(regions, boxes):
            # Check to see what annotations this window-box overlaps with
            aids = _isect_index.overlapping_aids(gid, box)

            # Look at the categories within this region
            # anns = [dset.anns[aid] for aid in aids]
            # cids = [ann['category_id'] for ann in anns]
            if aid_to_cid is None:
                cids = dset.annots(aids).get('category_id')
            else:
                cids = list(ub.take(aid_to_cid, aids))
            cats = [dset.cats[cid] for cid in cids]
            catnames = [cat['name'].lower() for cat in cats]

            if ignore_coverage_thresh:
                ignore_flags = [catname == 'ignore' for catname in catnames]
                if any(ignore_flags):
                    # If the almost the entire window is marked as ignored then
                    # just skip this window.
                    ignore_aids = list(ub.compress(aids, ignore_flags))
                    ignore_boxes = sampler.dset.annots(ignore_aids).boxes

                    # Get an upper bound on coverage to short circuit extra
                    # computation in simple cases.
                    box_area = box.area.sum()
                    coverage_ub = ignore_boxes.area.sum() / box_area
                    if coverage_ub  > ignore_coverage_thresh:
                        max_coverage = ignore_boxes.iooas(box).max()
                        if max_coverage > ignore_coverage_thresh:
                            continue
                        elif len(ignore_boxes) > 1:
                            # We have to test the complex case
                            try:
                                from shapely.ops import cascaded_union
                                ignore_shape = cascaded_union(ignore_boxes.to_shapley())
                                region_shape = box[None, :].to_shapley()[0]
                                coverage_shape = ignore_shape.intersection(region_shape)
                                real_coverage = coverage_shape.area / box_area
                                if real_coverage > ignore_coverage_thresh:
                                    continue
                            except Exception as ex:
                                import warnings
                                warnings.warn(
                                    'ignore region select had non-critical '
                                    'issue ex = {!r}'.format(ex))

            if classes_of_interest:
                # If there are CoIs then only count a region as positive if one
                # of those is in this region
                interest_flags = np.array([
                    catname in classes_of_interest for catname in catnames])
                pos_aids = list(ub.compress(aids, interest_flags))
            elif negative_classes:
                # Don't count negative classes as positives
                nonnegative_flags = np.array([
                    catname not in negative_classes for catname in catnames])
                pos_aids = list(ub.compress(aids, nonnegative_flags))
            else:
                pos_aids = aids

            # aids = sampler.regions.overlapping_aids(gid, box, visible_thresh=0.001)
            if len(pos_aids):
                positives.append((gid, region, aids))
            else:
                negatives.append((gid, region, aids))

    print('Found {} positives'.format(len(positives)))
    print('Found {} negatives'.format(len(negatives)))
    return positives, negatives
    # len([gid for gid, a in sampler.dset.gid_to_aids.items() if len(a) > 0])


class DetectionAugmentor(object):
    """
    CommandLine:
        xdoctest -m bioharn.detect_dataset DetectionAugmentor --show --mode=heavy --num=3
        xdoctest -m bioharn.detect_dataset DetectionAugmentor --show --mode=complex --gravity=1 --num=15
        xdoctest -m bioharn.detect_dataset DetectionAugmentor --show --mode=complex --gravity=0 --num=8

    Example:
        >>> # xdoctest: +REQUIRES(--show)
        >>> # xdoctest: +REQUIRES(module:imgaug)
        >>> from .detect_dataset import *  # NOQA
        >>> import scriptconfig as scfg
        >>> import kwimage
        >>> config = scfg.quick_cli({
        >>>     'fpath': kwimage.grab_test_image_fpath(),
        >>>     'mode': 'simple',
        >>>     'gravity': 0,
        >>>     'num': 8,
        >>>     'rng': None,
        >>>     'ndet': 3,
        >>> })
        >>> import kwplot
        >>> kwplot.autompl()
        >>> orig_imdata = kwimage.imread(config['fpath'])
        >>> augmentor = DetectionAugmentor(config['mode'], gravity=config['gravity'], rng=config['rng'])
        >>> orig_dets = kwimage.Detections.random(num=config['ndet'], segmentations=True)
        >>> orig_dets = orig_dets.scale(tuple(orig_imdata.shape[0:2][::-1]))
        >>> first = kwimage.draw_text_on_image(
        >>>     orig_imdata.copy(), 'orig', (0, 0),
        >>>     valign='top', color='limegreen', border=1)
        >>> first = orig_dets.draw_on(first)
        >>> fnum = 1
        >>> fig = kwplot.figure(fnum=fnum)
        >>> fig.suptitle('Press <spacebar> to re-augment')
        >>> ax = fig.gca()
        >>> def augment_and_draw():
        >>>     print('augment and drawing')
        >>>     augged_images = []
        >>>     for idx in range(config['num']):
        >>>         aug_im, dets2, disp_im = augmentor.augment_data(orig_imdata.copy(), orig_dets)
        >>>         aug_im = dets2.draw_on(aug_im)
        >>>         augged_images.append(aug_im)
        >>>     canvas = kwimage.stack_images_grid([first] + augged_images)
        >>>     kwplot.imshow(canvas, ax=ax)
        >>> augment_and_draw()
        >>> def on_key_press(event):
        >>>     if event and event.key == ' ':
        >>>         augment_and_draw()
        >>>         fig.canvas.draw()
        >>> cid = fig.canvas.mpl_connect('key_press_event', on_key_press)
        >>> kwplot.show_if_requested()
    """
    def __init__(self, mode='simple', gravity=0, rng=None):
        import imgaug as ia
        from imgaug import augmenters as iaa
        self.rng = kwarray.ensure_rng(rng)

        self.mode = mode

        print('gravity = {!r}'.format(gravity))
        self._intensity = iaa.Sequential([])
        self._geometric = iaa.Sequential([])
        self._disp_intensity = iaa.Sequential([])

        # DEFINE NEW MODES HERE
        if mode == 'simple':
            self._geometric = iaa.Sequential([
                iaa.Fliplr(p=.5),
                iaa.Flipud(p=.5 * (1 - gravity)),
                iaa.CropAndPad(px=(0, 4)),
            ])
            self._intensity = iaa.Sequential([])
        elif mode == 'low':
            scale = .25
            rot = 30
            scale_base = ia.parameters.TruncatedNormal(
                1.0, (scale * 2) / 6, low=1 - scale, high=1 + scale)
            rot_base = ia.parameters.TruncatedNormal(
                0.0, (rot * 2) / 6, low=-rot, high=rot)
            scale_rv = ia.parameters.Choice([scale_base, 1], p=[.6, .4])
            rot_rv = ia.parameters.Choice([rot_base, 0], p=[.6, .4])

            self._geometric = iaa.Sequential([
                iaa.Affine(
                    scale=scale_rv,
                    rotate=rot_rv,
                    order=1,
                    cval=(0, 255),
                    backend='cv2',
                ),
                iaa.Fliplr(p=.5),
                iaa.Flipud(p=.5 * (1 - gravity)),
                iaa.Rot90(k=[0, 1, 2, 3]),
                iaa.CropAndPad(px=(-3, 3)),
            ])
            self._intensity = iaa.Sequential([])
        elif mode == 'medium':
            # The weather augmenters are very expensive, so we ditch them
            self._geometric = iaa.Sequential([
                iaa.Sometimes(0.55, iaa.Affine(
                    scale={"x": (1.0, 1.2), "y": (1.0, 1.2)},
                    rotate=(-40, 40),  # rotate by -45 to +45 degrees
                    # use nearest neighbour or bilinear interpolation (fast)
                    # order=[0, 1],
                    order=1,
                    # if mode is constant, use a cval between 0 and 255
                    cval=(0, 255),
                    # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    backend='cv2',
                )),
                iaa.Fliplr(p=.5),
                iaa.Flipud(p=.5 * (1 - gravity)),
                iaa.Rot90(k=[0, 1, 2, 3]),
                iaa.Sometimes(.9, iaa.CropAndPad(px=(-4, 4))),
            ], random_order=False)

            self._intensity = iaa.Sequential([
                iaa.Sequential([
                    iaa.Sometimes(.1, iaa.GammaContrast((0.5, 2.0))),
                    iaa.Sometimes(.1, iaa.LinearContrast((0.5, 1.5))),
                ], random_order=True),
                iaa.Sometimes(.5, iaa.Grayscale(alpha=(0, 1))),
                iaa.Sometimes(.1, iaa.CoarseDropout(p=(.1, .3), size_percent=(0.02, 0.5))),
                iaa.Sometimes(.1, iaa.AddElementwise((-40, 40))),
            ])
        elif mode == 'heavy':
            scale = .25
            rot = 45
            self._geometric = iaa.Sequential([
                # Geometric
                iaa.Sometimes(0.55, iaa.Affine(
                    scale=ia.parameters.TruncatedNormal(1.0, (scale * 2) / 6, low=1 - scale, high=1 + scale),
                    rotate=ia.parameters.TruncatedNormal(0.0, (rot * 2) / 6, low=-rot, high=rot),
                    shear=ia.parameters.TruncatedNormal(0.0, 2.5, low=-16, high=16),
                    order=1,
                    # if mode is constant, use a cval between 0 and 255
                    cval=(0, 255),
                    # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    backend='cv2',
                )),
                iaa.Fliplr(p=.5),
                iaa.Flipud(p=.5 * (1 - gravity)),
                iaa.Rot90(k=[0, 1, 2, 3]),
                iaa.Sometimes(.9, iaa.CropAndPad(px=(-16, 16))),
            ])

            self._intensity = iaa.Sequential([
                # Color, brightness, saturation, and contrast
                iaa.Sometimes(.10, nh.data.transforms.HSVShift(hue=0.1, sat=1.5, val=1.5)),
                iaa.Sometimes(.10, iaa.GammaContrast((0.5, 2.0))),
                iaa.Sometimes(.10, iaa.LinearContrast((0.5, 1.5))),
                iaa.Sometimes(.10, iaa.Multiply((0.5, 1.5), per_channel=0.5)),
                iaa.Sometimes(.10, iaa.Add((-10, 10), per_channel=0.5)),
                iaa.Sometimes(.10, iaa.Grayscale(alpha=(0, 1))),

                # Speckle noise
                iaa.Sometimes(.05, iaa.AddElementwise((-40, 40))),
                iaa.Sometimes(.05, iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                )),
                iaa.Sometimes(.05, iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.CoarseDropout(
                        (.09, .31), size_percent=(0.19, 0.055),
                        per_channel=0.15
                    ),
                ])),
                # Blurring
                iaa.Sometimes(.05, iaa.OneOf([
                    iaa.GaussianBlur((0, 2.5)),
                    iaa.AverageBlur(k=(2, 7)),
                    iaa.MedianBlur(k=(3, 11)),
                ])),
            ], random_order=True)
            self._disp_intensity = iaa.Sequential([
                iaa.Sometimes(.1, iaa.CoarseDropout(p=(.1, .3), size_percent=(0.02, 0.5))),
                iaa.Sometimes(.1, iaa.AddElementwise((-40, 40))),
            ], random_order=True)

        elif mode == 'complex' or mode == 'no_hue':
            """
            notes:
                We have N independent random variables V[n] with
                `P(V[n] == 1) = p[n]`.

                The chance we draw none of them is
                >>> prod = np.prod
                >>> N = 12
                >>> p = [0.1 for n in range(N)]
                >>> prod([1 - p[n] for n in range(N)])

                More generally this is a binomial distribution when all p[n]
                are equal. (Unequal probabilities require poisson binomial,
                which is fairly expensive to compute). See
                https://github.com/scipy/scipy/issues/6000

                >>> from scipy.special import comb
                >>> n = 12
                >>> p = 0.1
                >>> dist = scipy.stats.binom(p=p, n=N)
                >>> # The probability sum(V) <= x is
                >>> # (ie we use at least x augmentors)
                >>> print('P(we use 0 augmentors) = {:.4f}'.format(dist.cdf(x=0)))
                >>> print('P(we use 1 augmentor)  = {:.4f}'.format(dist.cdf(x=1) - dist.cdf(x=0)))
                >>> print('P(we use 2 augmentors) = {:.4f}'.format(dist.cdf(x=2) - dist.cdf(x=1)))
                >>> print('P(we use 6 augmentors) = {:.4f}'.format(dist.cdf(x=6) - dist.cdf(x=5)))
                >>> print('P(we use 8 augmentors) = {:.4f}'.format(dist.cdf(x=8) - dist.cdf(x=7)))
            """
            if mode == 'complex':
                hue_op_per = 0.1
            else:
                hue_op_per = 0.0
            self._geometric = iaa.Sequential([
                iaa.Sometimes(0.55, iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    rotate=(-45, 45),  # rotate by -45 to +45 degrees
                    shear=ia.parameters.TruncatedNormal(0.0, 2.5, low=-16, high=16),
                    order=1,
                    # if mode is constant, use a cval between 0 and 255
                    cval=(0, 255),
                    # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                    backend='cv2',
                )),
                iaa.Fliplr(p=.5),
                iaa.Flipud(p=.5 * (1 - gravity)),
            ] + ([iaa.Rot90(k=[0, 1, 2, 3])] * int(1 - gravity))  +
                [
                    iaa.Sometimes(.9, iaa.CropAndPad(px=(-16, 16))),
                 ],
            )
            self._intensity = iaa.Sequential([
                # Color, brightness, saturation, and contrast
                iaa.Sometimes(hue_op_per, nh.data.transforms.HSVShift(hue=0.1, sat=1.5, val=1.5)),
                iaa.Sometimes(hue_op_per, iaa.GammaContrast((0.5, 2.0))),
                iaa.Sometimes(.1, iaa.LinearContrast((0.5, 1.5))),
                iaa.Sometimes(.1, iaa.Multiply((0.5, 1.5), per_channel=0.5)),
                iaa.Sometimes(.1, iaa.Add((-10, 10), per_channel=0.5)),
                iaa.Sometimes(hue_op_per, iaa.Grayscale(alpha=(0, 1))),

                # Speckle noise
                iaa.Sometimes(hue_op_per, iaa.AddElementwise((-40, 40))),
                iaa.Sometimes(.1, iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                )),
                iaa.Sometimes(hue_op_per, iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.CoarseDropout(
                        (.09, .31), size_percent=(0.19, 0.055),
                        per_channel=0.15
                    ),
                ])),

                # Blurring
                iaa.Sometimes(.05, iaa.OneOf([
                    iaa.GaussianBlur((0, 2.5)),
                    iaa.AverageBlur(k=(2, 7)),
                    iaa.MedianBlur(k=(3, 11)),
                ])),

                # Sharpening
                iaa.Sometimes(.1, iaa.OneOf([
                    iaa.Sometimes(.1, iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))),
                    iaa.Sometimes(.1, iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))),
                ])),

                # Misc
                iaa.Sometimes(hue_op_per, iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0, 0.7)),
                    iaa.DirectedEdgeDetect(
                        alpha=(0, 0.7), direction=(0.0, 1.0)
                    ),
                ])),
            ], random_order=True)
            self._disp_intensity = iaa.Sequential([
                iaa.Sometimes(.1, iaa.CoarseDropout(p=(.1, .3), size_percent=(0.02, 0.5))),
                iaa.Sometimes(.1, iaa.AddElementwise((-40, 40))),
            ], random_order=True)
        else:
            raise KeyError(mode)

        self._augers = ub.odict([
            ('intensity', self._intensity),
            ('geometric', self._geometric),
            ('disp_intensity', self._disp_intensity),
        ])
        self.mode = mode
        self.seed_(self.rng)

    def json_id(self):
        from viame.pytorch import netharn as nh
        params = ub.map_vals(nh.data.transforms.imgaug_json_id, self._augers)
        return params

    def seed_(self, rng):
        for auger in self._augers.values():
            if auger is not None:
                reseed_(auger, rng)

    def augment_data(self, imdata, dets, aux_components=None):
        """
        Ignore:
            self = DetectionAugmentor(mode='heavy')
            s = 128
            rng = kwarray.ensure_rng(0)
            dets = kwimage.Detections.random(segmentations=True, rng=rng).scale(s)

            imdata = (rng.rand(s, s, 3) * 255).astype(np.uint8)
            aux_components = {'disparity': rng.rand(s, s).astype(np.float32).clip(0, 1)}

            import kwplot
            kwplot.imshow(imdata, fnum=1, pnum=(2, 2, 1), doclf=True)
            dets.draw()
            kwplot.imshow(aux_components['disparity'], fnum=1, pnum=(2, 2, 2))
            dets.draw()

            imdata1, dets1, disp_im1 = self.augment_data(imdata, dets, disp_im)

            kwplot.imshow(imdata1, fnum=1, pnum=(2, 2, 3))
            dets1.draw()
            kwplot.imshow(disp_im1, fnum=1, pnum=(2, 2, 4))
            dets1.draw()
        """

        _debug('to det')
        rgb_im_aug_det = self._intensity.to_deterministic()
        geom_aug_det = self._geometric.to_deterministic()
        disp_im_aug_det = self._augers['disp_intensity'].to_deterministic()

        input_dims = imdata.shape[0:2]
        _debug('aug gdo')
        _debug('imdata.dtype = {!r}'.format(imdata.dtype))
        _debug('imdata.shape = {!r}'.format(imdata.shape))
        imdata = geom_aug_det.augment_image(imdata)
        _debug('aug rgb')
        imdata = rgb_im_aug_det.augment_image(imdata)

        if aux_components:
            for key, aux_im in aux_components.items():
                # _debug(kwarray.stats_dict(disp_im))
                aux_im = kwimage.ensure_uint255(aux_im)
                aux_im = disp_im_aug_det.augment_image(aux_im)
                aux_im = geom_aug_det.augment_image(aux_im)
                aux_im = disp_im_aug_det.augment_image(aux_im)
                aux_im = kwimage.ensure_float01(aux_im)
                # _debug(kwarray.stats_dict(aux_im))
                aux_components[key] = aux_im

        output_dims = imdata.shape[0:2]

        if len(dets):
            _debug('aug dets')
            dets = dets.warp(geom_aug_det, input_dims=input_dims,
                             output_dims=output_dims)

        return imdata, dets, aux_components
