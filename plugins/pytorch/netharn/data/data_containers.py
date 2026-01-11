"""
Proof-of-concept for porting mmcv DataContainer concept to netharn. Depending
on how well this works these features might be useful as a standalone module or
to contribute to torch proper.

References:
    https://github.com/open-mmlab/mmcv/blob/master/mmcv/parallel/data_container.py
    https://github.com/open-mmlab/mmcv/blob/master/mmcv/parallel/collate.py
    https://github.com/open-mmlab/mmcv/blob/master/mmcv/parallel/scatter_gather.py

FIXME 0 dimension tensors
"""
import torch.utils.data as torch_data
import torch
import ubelt as ub
import numpy as np  # NOQA
import re
import torch.nn.functional as F
# from torch.nn.parallel import DataParallel
from itertools import chain
from viame.pytorch.netharn.device import DataParallel, DataSerial, XPU
from torch.nn.parallel._functions import _get_stream
from torch.nn.parallel._functions import Scatter as OrigScatter
from torch.nn.parallel._functions import Gather as OrigGather

try:
    import collections.abc as container_abcs
    from six import string_types as string_classes
    from six import integer_types as int_classes
except Exception:
    from torch._six import container_abcs
    from torch._six import string_classes, int_classes
default_collate = torch_data.dataloader.default_collate


# numpy_type_map = torch_data.dataloader.numpy_type_map  # moved in torch 1.1.0
numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


class CollateException(Exception):
    pass


_DEBUG = False


class BatchContainer(ub.NiceRepr):
    """
    A container for a set of items in a batch. Usually this is for network
    outputs or a set of items that have already been collated.

    Attributes:
        data (List[Any]): Unlike ItemContainer, data is always a list where
            len(data) is the number of devices this batch will run on.  Each
            item in the list may be either a pre-batched Tensor (in the case
            where the each item in the batch has the same shape) or a list of
            individual item Tensors (in the case where different batch items
            may have different shapes).
    """
    def __init__(self, data, stack=False, padding_value=-1, cpu_only=False,
                 pad_dims=2):
        self.data = data  # type: list
        self.meta = {
            'stack': stack,
            'padding_value': padding_value,
            'cpu_only': cpu_only,
            'pad_dims': pad_dims,
        }

    @property
    def nestshape(self):
        return nestshape(self.data)

    def numel(self):
        """
        The number of scalar elements held by this container
        """
        shapes = self.nestshape
        total = sum([np.prod(s) for s in shapes])
        return total

    @property
    def packshape(self):
        """
        The shape of this data if it was packed
        """
        # shape = np.maximum.reduce(self.nestshape)
        # return shape
        dim = 0
        if self.stack:
            # Should be a straight forward concatenation
            shapes = [d.shape for d in self.data]
            max_shape = np.maximum.reduce(shapes)  # should all be the same here
            stacked_dim = sum([s[dim] for s in shapes])
            max_shape[dim] = stacked_dim
            pack_shape = tuple(max_shape.tolist())
            return pack_shape
        else:
            shapes = nestshape(self.data)
            max_shape = np.maximum.reduce(shapes)
            stacked_dim = sum([s[dim] for s in shapes])
            max_shape[dim] = stacked_dim
            pack_shape = tuple(max_shape.tolist())
            return pack_shape

    def __nice__(self):
        try:
            shape_repr = ub.repr2(self.nestshape, nl=-2)
            return 'nestshape(data)={}'.format(shape_repr)
        except Exception:
            return object.__repr__(self)

    def __getitem__(self, index):
        cls = self.__class__
        return cls([d[index] for d in self.data], **self.meta)

    @property
    def cpu_only(self):
        return self.meta['cpu_only']

    @property
    def stack(self):
        return self.meta['stack']

    @property
    def padding_value(self):
        return self.meta['padding_value']

    @property
    def pad_dims(self):
        return self.meta['pad_dims']

    @classmethod
    def cat(cls, items, dim=0):
        """
        Concatenate data in multiple BatchContainers

        Example:
            >>> d1 = BatchContainer([torch.rand(3, 3, 1, 1), torch.rand(2, 3, 1, 1)])
            >>> d2 = BatchContainer([torch.rand(3, 1, 1, 1), torch.rand(2, 1, 1, 1)])
            >>> items = [d1, d2]
            >>> self = BatchContainer.cat(items, dim=1)
        """
        newdata = []
        num_devices = len(items[0].data)
        for device_idx in range(num_devices):
            parts = [item.data[device_idx] for item in items]
            newpart = torch.cat(parts, dim=dim)
            newdata.append(newpart)
        self = cls(newdata, **items[0].meta)
        return self

    @classmethod
    def demo(cls, key='img', n=5, num_devices=1):
        inbatch = [ItemContainer.demo(key) for _ in range(n)]
        self = ItemContainer._collate(inbatch, num_devices=num_devices)
        return self

    def pack(self):
        """
        Pack all of the data in this container into a single tensor.

        Returns:
            Tensor: packed data, padded with ``self.padding_value`` if
            ``self.stack`` is False.

        Example:
            >>> self = BatchContainer.demo('img')
            >>> print(self.pack())
            >>> self = BatchContainer.demo('box')
            >>> print(self.pack())
            >>> self = BatchContainer.demo('labels')
            >>> print(self.pack())
        """
        if self.stack:
            # Should be a straight forward concatenation
            packed = torch.cat(self.data, dim=0)
        else:
            # Need to account for padding values
            from .data.collate import padded_collate
            inbatch = list(ub.flatten(self.data))
            packed = padded_collate(inbatch, fill_value=self.padding_value)
        return packed

    def to(self, device):
        """ inplace move data onto a device """
        walker = ub.IndexableWalker(self.data)
        for path, val in walker:
            if torch.is_tensor(val):
                walker[path] = val.to(device)
        return self


class ItemContainer(ub.NiceRepr):
    """
    A container for uncollated items that defines a specific collation
    strategy. Based on mmdetections ItemContainer.
    """

    def __init__(
        self,
        data,
        stack=False,
        padding_value=-1,
        cpu_only=False,
        pad_dims=2
    ):
        self._data = data
        assert pad_dims in [None, 1, 2, 3]
        self.meta = {
            'stack': stack,
            'padding_value': padding_value,
            'cpu_only': cpu_only,
            'pad_dims': pad_dims,
        }

    @property
    def nestshape(self):
        return nestshape(self.data)

    def __nice__(self):
        try:
            shape_repr = ub.repr2(self.nestshape, nl=-2)
            return 'nestshape(data)={}'.format(shape_repr)
        except Exception:
            return object.__repr__(self)
            # return super().__repr__()

    @classmethod
    def demo(cls, key='img', rng=None, **kwargs):
        """
        Create data for tests

        Example:
            >>> from .data.data_containers import *  # NOQA
            >>> print(ItemContainer.demo('img'))
            >>> print(ItemContainer.demo('labels'))
            >>> print(ItemContainer.demo('box'))

        """
        import kwarray
        rng = kwarray.ensure_rng(rng)
        if key == 'img':
            shape = kwargs.get('shape', (3, 512, 512))
            data = rng.rand(*shape).astype(np.float32)
            data = torch.from_numpy(data)
            self = cls(data, stack=True)
        elif key == 'labels':
            n = rng.randint(0, 10)
            data = rng.randint(0, 10, n)
            data = torch.from_numpy(data)
            self = cls(data, stack=False)
        elif key == 'box':
            n = rng.randint(0, 10)
            data = rng.rand(n, 4)
            data = torch.from_numpy(data)
            self = cls(data, stack=False)
        else:
            raise KeyError(key)
        return self

    def __getitem__(self, index):
        assert self.stack, 'can only index into stackable items'
        cls = self.__class__
        return cls(self.data[index], **self.meta)

    @property
    def data(self):
        return self._data

    @property
    def datatype(self):
        if isinstance(self.data, torch.Tensor):
            return self.data.type()
        else:
            return type(self.data)

    @property
    def cpu_only(self):
        return self.meta['cpu_only']

    @property
    def stack(self):
        return self.meta['stack']

    @property
    def padding_value(self):
        return self.meta['padding_value']

    @property
    def pad_dims(self):
        return self.meta['pad_dims']

    def size(self, *args, **kwargs):
        return self.data.size(*args, **kwargs)

    @property
    def shape(self):
        return self.data.shape

    def dim(self):
        return self.data.dim()

    @classmethod
    def _collate(cls, inbatch, num_devices=None):
        """
        Collates a sequence of DataContainers

        Args:
            inbatch (Sequence[ItemContainer]): datacontainers with the same
                parameters.

            num_devices (int): number of groups, if None, then uses one group.

        Example:
            >>> print('Collate Image ItemContainer')
            >>> inbatch = [ItemContainer.demo('img') for _ in range(5)]
            >>> print('inbatch = {}'.format(ub.repr2(inbatch)))
            >>> result = ItemContainer._collate(inbatch, num_devices=2)
            >>> print('result1 = {}'.format(ub.repr2(result, nl=1)))
            >>> result = ItemContainer._collate(inbatch, num_devices=1)
            >>> print('result2 = {}'.format(ub.repr2(result, nl=1)))
            >>> result = ItemContainer._collate(inbatch, num_devices=None)
            >>> print('resultN = {}'.format(ub.repr2(result, nl=1)))

            >>> print('Collate Label ItemContainer')
            >>> inbatch = [ItemContainer.demo('labels') for _ in range(5)]
            >>> print('inbatch = {}'.format(ub.repr2(inbatch, nl=1)))
            >>> result = ItemContainer._collate(inbatch, 1)
            >>> print('result1 = {}'.format(ub.repr2(result, nl=1)))
            >>> result = ItemContainer._collate(inbatch, 2)
            >>> print('result2 = {}'.format(ub.repr2(result, nl=1)))
            >>> result = ItemContainer._collate(inbatch, None)
            >>> print('resultN = {}'.format(ub.repr2(result, nl=1)))
        """
        item0 = inbatch[0]
        bsize = len(inbatch)
        if num_devices is None:
            num_devices = 1

        samples_per_device = int(np.ceil(bsize / num_devices))

        # assert bsize % samples_per_device == 0
        stacked = []
        if item0.cpu_only:
            # chunking logic
            stacked = []
            for i in range(0, bsize, samples_per_device):
                stacked.append(
                    [sample.data for sample in inbatch[i:i + samples_per_device]])

        elif item0.stack:
            for i in range(0, bsize, samples_per_device):
                item = inbatch[i]
                pad_dims_ = item.pad_dims
                assert isinstance(item.data, torch.Tensor)

                if pad_dims_ is not None:
                    # Note: can probably reimplement this using padded collate
                    # logic
                    ndim = item.dim()
                    assert ndim > pad_dims_
                    max_shape = [0 for _ in range(pad_dims_)]
                    for dim in range(1, pad_dims_ + 1):
                        max_shape[dim - 1] = item.shape[-dim]
                    for sample in inbatch[i:i + samples_per_device]:
                        for dim in range(0, ndim - pad_dims_):
                            assert item.shape[dim] == sample.shape[dim]
                        for dim in range(1, pad_dims_ + 1):
                            max_shape[dim - 1] = max(max_shape[dim - 1], sample.shape[-dim])
                    padded_samples = []
                    for sample in inbatch[i:i + samples_per_device]:
                        pad = [0 for _ in range(pad_dims_ * 2)]
                        for dim in range(1, pad_dims_ + 1):
                            pad[2 * dim - 1] = max_shape[dim - 1] - sample.shape[-dim]
                        padded_samples.append(
                            F.pad(sample.data, pad, value=sample.padding_value))
                    stacked.append(default_collate(padded_samples))

                elif pad_dims_ is None:
                    stacked.append(
                        default_collate([
                            sample.data
                            for sample in inbatch[i:i + samples_per_device]
                        ]))
                else:
                    raise ValueError(
                        'pad_dims should be either None or integers (1-3)')

        else:
            for i in range(0, bsize, samples_per_device):
                stacked.append(
                    [sample.data for sample in inbatch[i:i + samples_per_device]])
        result = BatchContainer(stacked, **item0.meta)
        return result


def decollate_batch(batch):
    """
    Breakup a collated batch of BatchContainers back into ItemContainers

    Example:
        >>> bsize = 5
        >>> batch_items = [
        >>>     {
        >>>         'im': ItemContainer.demo('img'),
        >>>         'label': ItemContainer.demo('labels'),
        >>>         'box': ItemContainer.demo('box'),
        >>>     }
        >>>     for _ in range(bsize)
        >>> ]
        >>> batch = container_collate(batch_items, num_devices=2)
        >>> decollated = decollate_batch(batch)
        >>> assert len(decollated) == len(batch_items)
        >>> assert (decollated[0]['im'].data == batch_items[0]['im'].data).all()
    """
    import ubelt as ub
    walker = ub.IndexableWalker(batch)
    decollated_dict = ub.AutoDict()
    decollated_walker = ub.IndexableWalker(decollated_dict)
    for path, batch_val in walker:
        if isinstance(batch_val, BatchContainer):
            for bx, item_val in enumerate(ub.flatten(batch_val.data)):
                decollated_walker[[bx] + path] = ItemContainer(item_val)
    decollated = list(decollated_dict.to_dict().values())
    return decollated


def container_collate(inbatch, num_devices=None):
    """Puts each data field into a tensor/DataContainer with outer dimension
    batch size.

    Extend default_collate to add support for
    :type:`~mmcv.parallel.DataContainer`. There are 3 cases.

    1. cpu_only = True, e.g., meta data
    2. cpu_only = False, stack = True, e.g., images tensors
    3. cpu_only = False, stack = False, e.g., gt bboxes

    Ignore:
        >>> # DISABLE_DOCTSET
        >>> dataset = DetectFitDataset.demo(key='shapes8', augment='complex', window_dims=(512, 512), gsize=(1920, 1080))

        >>> inbatch = [dataset[0], dataset[1], dataset[2]]
        >>> raw_batch = container_collate(inbatch)

        >>> target_gpus = [0]
        >>> inputs, kwargs = container_scatter_kwargs(raw_batch, {}, target_gpus)

        >>> loader = torch.utils.data.DataLoader(dataset, collate_fn=container_collate, num_workers=0)


    Example:
        >>> item1 = {
        >>>     'im': torch.rand(3, 512, 512),
        >>>     'label': torch.rand(3),
        >>> }
        >>> item2 = {
        >>>     'im': torch.rand(3, 512, 512),
        >>>     'label': torch.rand(3),
        >>> }
        >>> item3 = {
        >>>     'im': torch.rand(3, 512, 512),
        >>>     'label': torch.rand(3),
        >>> }
        >>> batch = batch_items = [item1, item2, item3]
        >>> raw_batch = container_collate(batch_items)
        >>> print('batch_items = {}'.format(ub.repr2(batch_items, nl=2)))
        >>> print('raw_batch = {}'.format(ub.repr2(raw_batch, nl=2)))

        >>> batch = batch_items = [
        >>>     {'im': ItemContainer.demo('img'), 'label': ItemContainer.demo('labels')},
        >>>     {'im': ItemContainer.demo('img'), 'label': ItemContainer.demo('labels')},
        >>>     {'im': ItemContainer.demo('img'), 'label': ItemContainer.demo('labels')},
        >>> ]
        >>> raw_batch = container_collate(batch, num_devices=2)
        >>> print('batch_items = {}'.format(ub.repr2(batch_items, nl=2)))
        >>> print('raw_batch = {}'.format(ub.repr2(raw_batch, nl=2)))

        >>> raw_batch = container_collate(batch, num_devices=6)
        >>> raw_batch = container_collate(batch, num_devices=3)
        >>> raw_batch = container_collate(batch, num_devices=4)
        >>> raw_batch = container_collate(batch, num_devices=1)
        >>> print('batch = {}'.format(ub.repr2(batch, nl=1)))
    """

    if not isinstance(inbatch, container_abcs.Sequence):
        raise TypeError("{} is not supported.".format(inbatch.dtype))
    item0 = inbatch[0]
    if isinstance(item0, ItemContainer):
        return item0.__class__._collate(inbatch, num_devices=num_devices)
    elif isinstance(item0, container_abcs.Sequence):
        transposed = zip(*inbatch)
        return [container_collate(samples,
                                  num_devices=num_devices)
                for samples in transposed]
    elif isinstance(item0, container_abcs.Mapping):
        return {
            key: container_collate([d[key] for d in inbatch],
                                   num_devices=num_devices)
            for key in item0
        }
    else:
        return default_collate(inbatch)
        # return _collate_else(inbatch, container_collate)


def _collate_else(batch, collate_func):
    """
    Handles recursion in the else case for these special collate functions

    This is duplicates all non-tensor cases from `torch_data.dataloader.default_collate`
    This also contains support for collating slices.
    """
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], slice):
        batch = default_collate([{
            'start': sl.start,
            'stop': sl.stop,
            'step': 1 if sl.step is None else sl.step
        } for sl in batch])
        return batch
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], container_abcs.Mapping):
        # Hack the mapping collation implementation to print error info
        if _DEBUG:
            collated = {}
            try:
                for key in batch[0]:
                    collated[key] = collate_func([d[key] for d in batch])
            except Exception:
                print('\n!!Error collating key = {!r}\n'.format(key))
                raise
            return collated
        else:
            return {key: collate_func([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], tuple) and hasattr(batch[0], '_fields'):  # namedtuple
        return type(batch[0])(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(batch[0], container_abcs.Sequence):
        transposed = zip(*batch)
        return [collate_func(samples) for samples in transposed]
    else:
        raise TypeError((error_msg.format(type(batch[0]))))


# ----


def _fn_scatter(input, devices, streams=None):
    """Scatters tensor across multiple GPUs.

    from mmcv.parallel._functions
    """
    if streams is None:
        streams = [None] * len(devices)

    if isinstance(input, list):
        chunk_size = (len(input) - 1) // len(devices) + 1
        outputs = [
            _fn_scatter(input[i], [devices[i // chunk_size]],
                          [streams[i // chunk_size]]) for i in range(len(input))
        ]
        return outputs
    elif isinstance(input, torch.Tensor):
        output = input.contiguous()
        # TODO: copy to a pinned buffer first (if copying from CPU)
        stream = streams[0] if output.numel() > 0 else None
        with torch.cuda.device(devices[0]), torch.cuda.stream(stream):
            output = output.cuda(devices[0], non_blocking=True)
        return output
    else:
        raise Exception('Unknown type {}.'.format(type(input)))


def synchronize_stream(output, devices, streams):
    if isinstance(output, list):
        chunk_size = len(output) // len(devices)
        for i in range(len(devices)):
            for j in range(chunk_size):
                synchronize_stream(output[i * chunk_size + j], [devices[i]],
                                   [streams[i]])
    elif isinstance(output, torch.Tensor):
        if output.numel() != 0:
            with torch.cuda.device(devices[0]):
                main_stream = torch.cuda.current_stream()
                main_stream.wait_stream(streams[0])
                output.record_stream(main_stream)
    else:
        raise Exception('Unknown type {}.'.format(type(output)))


def get_input_device(input):
    if isinstance(input, list):
        for item in input:
            input_device = get_input_device(item)
            if input_device != -1:
                return input_device
        return -1
    elif isinstance(input, torch.Tensor):
        return input.get_device() if input.is_cuda else -1
    else:
        raise Exception('Unknown type {}.'.format(type(input)))


class ContainerScatter(object):

    @staticmethod
    def forward(target_gpus, input):
        input_device = get_input_device(input)
        streams = None
        if input_device == -1:
            # Perform CPU to GPU copies in a background stream
            # FIXME: the updated version of this function in torch
            # might require a torch device instead of an int.
            streams = [_get_stream(device) for device in target_gpus]

        outputs = _fn_scatter(input, target_gpus, streams)
        # Synchronize with the copy stream
        if streams is not None:
            synchronize_stream(outputs, target_gpus, streams)

        return tuple(outputs)

# ----


class ContainerDataParallel(DataParallel):
    """

    Ignore:
        import torch
        from torch.nn.parallel import DataParallel

        # First lets create a simple model where the forward function accepts
        # kwargs. I don't really care what they do for this example, but imaging
        # they are flags that change the behavior of forward.

        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)

            def forward(self, im, **kwargs):
                return self.conv(im)

        raw_model = MyModel()
        raw_model = raw_model.to(0)

        # Next create some dummy input and verify the model works by itself
        im = torch.zeros(1, 1, 1, 1).to(0)
        raw_model.forward(im)

        # Now create a DataParallel object to map the input across two devices
        par_model = DataParallel(raw_model, device_ids=[0, 1], output_device=0)

        # In the case where kwargs are not specified DataParallel correctly
        # understands that there is only one item in the batch and applies the
        # operation on only one GPU.
        par_model.forward(im)

        # Howver, if you pass kwargs, then data parallel breaks
        par_model.forward(im, flag1=True)

        inputs = (im,)
        kwargs = dict(flag1=True, flag2=False)
        s1, k1 = par_model.scatter(inputs, kwargs, [0, 1])
        replicas = par_model.replicate(par_model.module, par_model.device_ids[:len(s1)])
        outputs = par_model.parallel_apply(replicas, s1, k1)

        container_scatter(inputs, [0, 1])[0]

        inbatch = [ItemContainer.demo('img', shape=(1, 1, 1)) for _ in range(5)]
        im = ItemContainer._collate(inbatch, 5)

        im = torch.zeros(1, 1, 1, 1).to(0)
        inputs = (im,)
        self = ContainerDataParallel(raw_model, device_ids=[0, 1], output_device=0)
        self.forward(*inputs, **kwargs)
    """

    def forward(self, *inputs, **kwargs):
        """
        Unchanged version for torch.nn.DataParallel
        """
        if not self.device_ids:
            return self.module(*inputs, **kwargs)

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError("module must have its parameters and buffers "
                                   "on device {} (device_ids[0]) but found one of "
                                   "them on device: {}".format(self.src_device_obj, t.device))

        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def scatter(self, inputs, kwargs, device_ids):
        return container_scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def gather(self, outputs, output_device):
        # not part of mmcv's original impl
        return container_gather(outputs, output_device, dim=self.dim)

# ----


def container_scatter(inputs, target_gpus, dim=0):
    """Scatter inputs to target gpus.

    from mmcv.parallel.scatter_gather

    The only difference from original :func:`scatter` is to add support for
    :type:`~mmcv.parallel.DataContainer`.
    """

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return OrigScatter.apply(target_gpus, None, dim, obj)
        if isinstance(obj, BatchContainer):
            if obj.cpu_only:
                return obj.data
            else:
                return ContainerScatter.forward(target_gpus, obj.data)
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            out = list(map(list, zip(*map(scatter_map, obj))))
            return out
        if isinstance(obj, dict) and len(obj) > 0:
            out = list(map(type(obj), zip(*map(scatter_map, obj.items()))))
            return out
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


def container_scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    """
    Scatter with support for kwargs dictionary

    Example:
        >>> # xdoctest: +REQUIRES(--multi-gpu)
        >>> inputs = [torch.rand(1, 1, 1, 1)]
        >>> kwargs = dict(a=1, b=2)
        >>> target_gpus = [0, 1]
        >>> a1, k1 = container_scatter_kwargs(inputs, kwargs, target_gpus)

        >>> # xdoctest: +REQUIRES(--multi-gpu)
        >>> inputs = [torch.rand(1, 1, 1, 1)]
        >>> kwargs = dict(a=torch.rand(1, 1, 1, 1), b=2)
        >>> target_gpus = [0, 1]
        >>> a1, k1 = container_scatter_kwargs(inputs, kwargs, target_gpus)
    """
    inputs = container_scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = container_scatter(kwargs, target_gpus, dim) if kwargs else []

    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])

    # patch for cases where #inputs < len(target_gpus) and len(kwargs) > 0
    PATCH = 1
    if PATCH:
        is_empty = [len(p) == 0 for p in inputs]
        num_empty = sum(is_empty)
        num_full = len(inputs) - num_empty
        if num_full > 0 and num_empty > 0:
            kwargs = kwargs[0:num_full]
            inputs = inputs[0:num_full]

    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


def container_gather(outputs, target_device, dim=0):
    r"""
    Gathers tensors from different GPUs on a specified device
      (-1 means the CPU).

    The only difference from original :func:`gather` is to add support for
    :type:`BatchContainer`.

    Ignore:
        >>> import kwarray
        >>> rng = kwarray.ensure_rng(0)
        >>> outputs = [
        >>>     {
        >>>         'batch_results': BatchContainer([
        >>>             torch.rand(rng.randint(0, 10), 5).to(0)
        >>>             for _ in range(4)
        >>>         ], stack=False),
        >>>         'loss_parts': {
        >>>             'part1': torch.rand(2).sum().to(0),
        >>>             'part2': torch.rand(3).sum().to(0),
        >>>         },
        >>>     },
        >>>     {
        >>>         'batch_results': BatchContainer([
        >>>             torch.rand(rng.randint(0, 10), 5).to(1)
        >>>             for _ in range(4)
        >>>         ], stack=False),
        >>>         'loss_parts': {
        >>>             'part1': torch.rand(2).sum().to(1),
        >>>             'part2': torch.rand(3).sum().to(1),
        >>>         }
        >>>     }
        >>> ]
        >>> _report_data_shape(outputs)
        >>> target_device = 0
        >>> dim = 0
        >>> gathered = container_gather(outputs, target_device, dim)
        >>> _report_data_shape(gathered)
    """
    def gather_map(outputs_):
        out = outputs_[0]
        if isinstance(out, torch.Tensor):
            # if all(t.dim() == 0 for t in outputs_) and dim == 0:
            #     # unsqueeze warnings will trigger
            #     import xdev
            #     xdev.embed()
            return OrigGather.apply(target_device, dim, *outputs_)
        if isinstance(out, BatchContainer):
            newdata = [d for dc in outputs_ for d in dc.data]
            if not out.cpu_only:
                from viame.pytorch import netharn as nh
                target_xpu = nh.XPU(target_device)
                newdata = target_xpu.move(newdata)
            return newdata
        if out is None:
            return None
        if isinstance(out, dict):
            out0_keys = set(out.keys())
            output_keys = [set(d.keys()) for d in outputs_]
            if not all(out0_keys == k for k in output_keys):
                problem_keys = (
                    set.union(*output_keys) - set.intersection(*output_keys)
                )
                raise ValueError(
                    'All dicts must have the same keys. '
                    'problem_keys={}'.format(problem_keys))
            return type(out)(((k, gather_map([d[k] for d in outputs_]))
                              for k in out))
        return type(out)(map(gather_map, zip(*outputs_)))

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        res = gather_map(outputs)
    finally:
        gather_map = None
    return res


# ---


class ContainerXPU(XPU):

    def mount(xpu, model):
        """
        Like move, but only for models.
        Note that this works inplace for non-Tensor objects.

        Args:
            model (torch.nn.Module): the model to mount

        Returns:
            DataSerial | DataParallel :
                the model mounted on the XPU (which may be multiple GPUs)

        Example:
            >>> model = torch.nn.Conv2d(1, 1, 1)
            >>> xpu = XPU()
        """
        # Unwrap the core model if necessary
        model = xpu.raw(model)
        model = xpu.move(model)
        if xpu._device_ids and len(xpu._device_ids) > 1:
            model = ContainerDataParallel(
                model, device_ids=xpu._device_ids,
                output_device=xpu._main_device_id)
        else:
            model = DataSerial(model)
        return model

    def move(xpu, data, **kwargs):
        try:
            if xpu.is_gpu():
                return data.to(xpu._main_device_id, **kwargs)
            else:
                return data.to('cpu')
        except AttributeError:
            # Recursive move
            if isinstance(data, container_abcs.Mapping):
                cls = data.__class__
                return cls((k, xpu.move(v)) for k, v in data.items())
            elif isinstance(data, (container_abcs.Sequence, container_abcs.Set)):
                cls = data.__class__
                return cls(xpu.move(v) for v in data)
            elif isinstance(data, BatchContainer):
                return data.to(xpu._main_device_id, **kwargs)
            else:
                raise TypeError('Unknown type {}'.format(type(data)))


def nestshape(data):
    """
    Examine nested shape of the data

    Example:
        >>> data = [np.arange(10), np.arange(13)]
        >>> nestshape(data)
        [(10,), (13,)]

    Ignore:
        >>> # xdoctest: +REQUIRES(module:mmdet)
        >>> from .data.data_containers import *  # NOQA

        >>> from mmdet.core.mask.structures import *  # NOQA
        >>> masks = [
        >>>     [ np.array([0, 0, 10, 0, 10, 10., 0, 10, 0, 0]) ],
        >>>     [ np.array([0, 0, 10, 0, 10, 10., 0, 10, 5., 5., 0, 0]) ]
        >>> ]
        >>> height, width = 16, 16
        >>> polys = PolygonMasks(masks, height, width)
        >>> nestshape(polys)

        >>> dc = BatchContainer([polys], stack=False)
        >>> print('dc = {}'.format(ub.repr2(dc, nl=1)))

        >>> num_masks, H, W = 3, 32, 32
        >>> rng = np.random.RandomState(0)
        >>> masks = (rng.rand(num_masks, H, W) > 0.1).astype(int)
        >>> bitmasks = BitmapMasks(masks, height=H, width=W)
        >>> nestshape(bitmasks)

        >>> dc = BatchContainer([bitmasks], stack=False)
        >>> print('dc = {}'.format(ub.repr2(dc, nl=1)))

    """
    import ubelt as ub

    def _recurse(d):
        import torch
        import numpy as np
        if isinstance(d, dict):
            return ub.odict(sorted([(k, _recurse(v)) for k, v in d.items()]))

        clsname = type(d).__name__
        if 'Container' in clsname:
            meta = ub.odict(sorted([
                ('stack', d.stack),
                # ('padding_value', d.padding_value),
                # ('pad_dims', d.pad_dims),
                # ('datatype', d.datatype),
                ('cpu_only', d.cpu_only),
            ]))
            meta = ub.repr2(meta, nl=0)
            return {type(d).__name__ + meta: _recurse(d.data)}
        elif isinstance(d, list):
            return [_recurse(v) for v in d]
        elif isinstance(d, tuple):
            return tuple([_recurse(v) for v in d])
        elif isinstance(d, torch.Tensor):
            return d.shape
        elif isinstance(d, np.ndarray):
            return d.shape
        elif isinstance(d, (str, bytes)):
            return d
        elif isinstance(d, (int, float)):
            return d
        elif isinstance(d, slice):
            return d
        elif 'PolygonMasks' == clsname:
            # hack for mmdet
            return repr(d)
        elif 'BitmapMasks' == clsname:
            # hack for mmdet
            return repr(d)
        elif hasattr(d, 'shape'):
            return d.shape
        elif hasattr(d, 'items'):
            # hack for dict-like objects
            return ub.odict(sorted([(k, _recurse(v)) for k, v in d.items()]))
        elif d is None:
            return None
        else:
            raise TypeError(type(d))

    # globals()['_recurse'] = _recurse
    d = _recurse(data)
    return d


def _report_data_shape(data):
    d = nestshape(data)
    print('d = {}'.format(ub.repr2(d, nl=-2)))


def _debug_inbatch_shapes(inbatch):
    import ubelt as ub
    print('len(inbatch) = {}'.format(len(inbatch)))
    extensions = ub.util_format.FormatterExtensions()

    @extensions.register((torch.Tensor, np.ndarray))
    def format_shape(data, **kwargs):
        return ub.repr2(dict(type=str(type(data)), shape=data.shape), nl=1, sv=1)

    print('inbatch = ' + ub.repr2(inbatch, extensions=extensions, nl=True))


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest netharn.data.data_containers all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
