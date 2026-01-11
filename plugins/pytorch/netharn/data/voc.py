"""
Simple dataset for loading the VOC 2007 object detection dataset without extra
bells and whistles. Simply loads the images, boxes, and class labels and
resizes images to a standard size.

THIS WILL BE DEPRECATED IN THE FUTURE. WE WILL USE THE COCO FORMAT AS A COMMON
DATA FORMAT FOR DETECTION PROBLEMS.
"""
from os.path import exists
from os.path import join
import re
import torch
import glob
import ubelt as ub
import numpy as np
from viame.pytorch.netharn.data import collate
import torch.utils.data as torch_data


class VOCDataset(torch_data.Dataset, ub.NiceRepr):
    """
    Example:
        >>> # xdoc: +REQUIRES(--voc)
        >>> assert len(VOCDataset(split='train', years=[2007])) == 2501
        >>> assert len(VOCDataset(split='test', years=[2007])) == 4952
        >>> assert len(VOCDataset(split='val', years=[2007])) == 2510
        >>> assert len(VOCDataset(split='trainval', years=[2007])) == 5011

        >>> assert len(VOCDataset(split='train', years=[2007, 2012])) == 8218
        >>> assert len(VOCDataset(split='test', years=[2007, 2012])) == 4952
        >>> assert len(VOCDataset(split='val', years=[2007, 2012])) == 8333

    Example:
        >>> # xdoc: +REQUIRES(--voc)
        >>> years = [2007, 2012]
        >>> self = VOCDataset()
        >>> for i in range(10):
        ...     a, bc = self[i]
        ...     #print(bc[0].shape)
        ...     print(bc[1].shape)
        ...     print(a.shape)

    Example:
        >>> # xdoc: +REQUIRES(--voc)
        >>> self = VOCDataset(split='trainval', years=[2007, 2012])
        >>> test = VOCDataset(split='test', years=[2007])

        >>> self1 = VOCDataset(split='trainval', years=[2012])
        >>> self2 = VOCDataset(split='trainval', years=[2007])


        from .. import util
        util.qtensure()
        self1.show_image(198)
        self2.show_image(300)
    """
    def __init__(self, devkit_dpath=None, split='train', years=[2007, 2012]):
        if devkit_dpath is None:
            # ub.expandpath('~/data/VOC/VOCdevkit')
            devkit_dpath = self.ensure_voc_data(years=years)

        self.devkit_dpath = devkit_dpath
        self.years = years

        # determine train / test splits
        self.gpaths = []
        self.apaths = []
        if split == 'test':
            assert 2007 in years, 'test set is hacked to be only 2007'
            gps, aps = self._read_split_paths('test', 2007)
            self.gpaths += gps
            self.apaths += aps
        else:
            for year in sorted(years):
                gps, aps = self._read_split_paths(split, year)
                self.gpaths += gps
                self.apaths += aps

        self.label_names = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                            'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                            'dog', 'horse', 'motorbike', 'person',
                            'pottedplant', 'sheep', 'sofa', 'train',
                            'tvmonitor')
        self._class_to_ind = ub.invert_dict(dict(enumerate(self.label_names)))
        self.base_wh = [416, 416]

        self.num_classes = len(self.label_names)

        import os
        hashid = ub.hash_data(list(map(os.path.basename, self.gpaths)))
        yearid = '_'.join(map(str, years))
        self.input_id = 'voc_{}_{}_{}'.format(yearid, split, hashid)

    def _read_split_paths(self, split, year):
        """
        split = 'train'
        self = VOCDataset('test')
        year = 2007
        year = 2012
        """
        split_idstrs = []
        data_dpath = join(self.devkit_dpath, 'VOC{}'.format(year))
        split_dpath = join(data_dpath, 'ImageSets', 'Main')
        pattern = join(split_dpath, '*_' + split + '.txt')
        for p in sorted(glob.glob(pattern)):
            rows = [list(re.split(' +', t)) for t in ub.readfrom(p).split('\n') if t]
            # code = -1 if the image does not contain the object
            # code = 1 if the image contains at least one instance
            # code = 0 if the image contains only hard instances of the object
            idstrs = [idstr for idstr, code in rows if int(code) == 1]
            split_idstrs.extend(idstrs)
        split_idstrs = sorted(set(split_idstrs))

        image_dpath = join(data_dpath, 'JPEGImages')
        annot_dpath = join(data_dpath, 'Annotations')
        gpaths = [join(image_dpath, '{}.jpg'.format(idstr))
                  for idstr in split_idstrs]
        apaths = [join(annot_dpath, '{}.xml'.format(idstr))
                  for idstr in split_idstrs]
        return gpaths, apaths
        # for p in gpaths:
        #     assert exists(p)
        # for p in apaths:
        #     assert exists(p)
        # return split_idstrs

    @classmethod
    def ensure_voc_data(VOCDataset, dpath=None, force=False, years=[2007, 2012]):
        """
        Download the Pascal VOC 2007 data if it does not already exist.

        CommandLine:
            python -m netharn.data.voc VOCDataset.ensure_voc_data

        Example:
            >>> # SCRIPT
            >>> # xdoc: +REQUIRES(--voc)
            >>> from .data.voc import *  # NOQA
            >>> VOCDataset.ensure_voc_data()
        """
        if dpath is None:
            dpath = ub.expandpath('~/data/VOC')
        devkit_dpath = join(dpath, 'VOCdevkit')
        # if force or not exists(devkit_dpath):
        ub.ensuredir(dpath)

        fpath1 = ub.grabdata('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar', dpath=dpath)
        if force or not exists(join(dpath, 'VOCdevkit', 'VOCcode')):
            ub.cmd('tar xvf "{}" -C "{}"'.format(fpath1, dpath), verbout=1)

        if 2007 in years:
            # VOC 2007 train+validation data
            fpath2 = ub.grabdata('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar', dpath=dpath)
            if force or not exists(join(dpath, 'VOCdevkit', 'VOC2007', 'ImageSets', 'Main', 'bird_trainval.txt')):
                ub.cmd('tar xvf "{}" -C "{}"'.format(fpath2, dpath), verbout=1)

            # VOC 2007 test data
            fpath3 = ub.grabdata('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar', dpath=dpath)
            if force or not exists(join(dpath, 'VOCdevkit', 'VOC2007', 'ImageSets', 'Main', 'bird_test.txt')):
                ub.cmd('tar xvf "{}" -C "{}"'.format(fpath3, dpath), verbout=1)

        if 2012 in years:
            # VOC 2012 train+validation data
            fpath4 = ub.grabdata('https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar', dpath=dpath)
            if force or not exists(join(dpath, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Main', 'bird_trainval.txt')):
                ub.cmd('tar xvf "{}" -C "{}"'.format(fpath4, dpath), verbout=1)
        return devkit_dpath

    def __nice__(self):
        return '{}'.format(len(self))

    def __len__(self):
        return len(self.gpaths)

    def __getitem__(self, index):
        """
        Returns:
            image, (bbox, class_idxs)

            bbox and class_idxs are variable-length
            bbox is in x1,y1,x2,y2 (i.e. tlbr) format

        CommandLine:
            xdoctest ~/code/netharn/netharn/data/voc.py VOCDataset.__getitem__ --show --voc

        Example:
            >>> # xdoc: +REQUIRES(--voc)
            >>> from .data.voc import *  # NOQA
            >>> self = VOCDataset()
            >>> chw, label = self[1]
            >>> hwc = chw.numpy().transpose(1, 2, 0)
            >>> boxes, class_idxs = label
            >>> # xdoc: +REQUIRES(--show)
            >>> import kwplot
            >>> import kwimage
            >>> kwplot.autompl()  # xdoc: +SKIP
            >>> kwplot.figure(fnum=1, doclf=True)
            >>> kwplot.imshow(hwc, colorspace='rgb')
            >>> kwimage.Boxes(boxes.numpy(), 'tlbr').draw()
            >>> kwplot.show_if_requested()
        """
        if isinstance(index, tuple):
            # Get size index from the batch loader
            index, inp_size = index
        else:
            inp_size = self.base_wh
        hwc, boxes, gt_classes = self._load_item(index, inp_size)

        chw = torch.FloatTensor(hwc.transpose(2, 0, 1))
        gt_classes = torch.LongTensor(gt_classes)
        boxes = torch.FloatTensor(boxes)
        label = (boxes, gt_classes,)
        return chw, label

    def _load_item(self, index, inp_size=None):
        # from .models.yolo2.utils.yolo import _offset_boxes
        import cv2
        image = self._load_image(index)
        annot = self._load_annotation(index)

        boxes = annot['boxes'].astype(np.float32)
        gt_classes = annot['gt_classes']

        # squish the bounding box and image into a standard size
        if inp_size is None:
            return image, boxes, gt_classes
        else:
            w, h = inp_size
            sx = float(w) / image.shape[1]
            sy = float(h) / image.shape[0]
            boxes[:, 0::2] *= sx
            boxes[:, 1::2] *= sy
            interpolation = cv2.INTER_AREA if (sx + sy) <= 2 else cv2.INTER_CUBIC
            hwc = cv2.resize(image, (w, h), interpolation=interpolation)
            return hwc, boxes, gt_classes

    def _load_image(self, index):
        import cv2
        fpath = self.gpaths[index]
        imbgr = cv2.imread(fpath, flags=cv2.IMREAD_COLOR)
        imrgb_255 = cv2.cvtColor(imbgr, cv2.COLOR_BGR2RGB)
        return imrgb_255

    def _load_annotation(self, index):
        import scipy
        import scipy.sparse
        import xml.etree.ElementTree as ET
        fpath = self.apaths[index]
        tree = ET.parse(fpath)
        objs = tree.findall('object')

        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ishards = np.zeros((num_objs), dtype=np.int32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1

            diffc = obj.find('difficult')
            difficult = 0 if diffc is None else int(diffc.text)
            ishards[ix] = difficult

            clsname = obj.find('name').text.lower().strip()
            cls = self._class_to_ind[clsname]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        annots = {'boxes': boxes,
                  'gt_classes': gt_classes,
                  'gt_ishard': ishards,
                  'gt_overlaps': overlaps,
                  'flipped': False,
                  'fpath': fpath,
                  'seg_areas': seg_areas}
        return annots

    def show_image(self, index, fnum=None):
        from .. import util
        hwc, boxes, gt_classes = self._load_item(index, inp_size=None)

        labels = list(ub.take(self.label_names, gt_classes))

        util.figure(doclf=True, fnum=fnum)
        util.imshow(hwc, colorspace='rgb')
        util.draw_boxes(boxes, color='green', box_format='tlbr', labels=labels)

    def make_loader(self, *args, **kwargs):
        """
        We need to do special collation to deal with different numbers of
        bboxes per item.

        Args:
            batch_size (int, optional): how many samples per batch to load
                (default: 1).
            shuffle (bool, optional): set to ``True`` to have the data
                reshuffled at every epoch (default: False).
            sampler (Sampler, optional): defines the strategy to draw samples
                from the dataset. If specified, ``shuffle`` must be False.
            batch_sampler (Sampler, optional): like sampler, but returns a
                batch of indices at a time. Mutually exclusive with batch_size,
                shuffle, sampler, and drop_last.
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means that the data will be loaded in the main
                process.  (default: 0)
            pin_memory (bool, optional): If ``True``, the data loader will copy
                tensors into CUDA pinned memory before returning them.
            drop_last (bool, optional): set to ``True`` to drop the last
                incomplete batch, if the dataset size is not divisible by the
                batch size. If ``False`` and the size of dataset is not
                divisible by the batch size, then the last batch will be
                smaller. (default: False)
            timeout (numeric, optional): if positive, the timeout value for
                collecting a batch from workers. Should always be non-negative.
                (default: 0)
            worker_init_fn (callable, optional): If not None, this will be
                called on each worker subprocess with the worker id (an int in
                ``[0, num_workers - 1]``) as input, after seeding and before
                data loading. (default: None)

        References:
            https://github.com/pytorch/pytorch/issues/1512

        Example:
            >>> # xdoc: +REQUIRES(--voc)
            >>> self = VOCDataset()
            >>> #inbatch = [self[i] for i in range(10)]
            >>> loader = self.make_loader(batch_size=10)
            >>> batch = next(iter(loader))
            >>> images, labels = batch
            >>> assert len(images) == 10
            >>> assert len(labels) == 2
            >>> assert len(labels[0]) == len(images)
        """
        kwargs['collate_fn'] = collate.list_collate
        loader = torch_data.DataLoader(self, *args, **kwargs)
        return loader

    def to_coco(self):
        """ Transform VOC to coco-style dataset """
        from viame.pytorch import netharn as nh
        voc_dset = self
        coco_dset = nh.data.coco_api.CocoDataset()
        coco_dset._build_index()

        for cx, catname in enumerate(self.label_names):
            coco_dset.add_category(catname, cid=int(cx))

        for gx, gpath in enumerate(ub.ProgIter(voc_dset.gpaths,
                                               label='convert coco')):
            coco_dset.add_image(gpath, gid=int(gx))
            voc_anno = self._load_annotation(gx)

            for i in range(len(voc_anno['boxes'])):
                box = nh.util.Boxes(voc_anno['boxes'][i], 'tlbr')
                cx = voc_anno['gt_classes'][i]
                weight = 1 - voc_anno['gt_ishard'][i]
                coco_dset.add_annotation(
                    gid=int(gx), cid=int(cx), bbox=box, weight=weight)
        return coco_dset


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m netharn.data.voc all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
