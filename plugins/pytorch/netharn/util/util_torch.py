import numpy as np
import torch
import ubelt as ub


class ModuleMixin(object):
    """
    Adds convenience functions to a torch module
    """

    def trainable_layers(self, names=False):
        """
        Get the layers netharn identifies as "trainable"

        Example:
            >>> import torchvision
            >>> model = torchvision.models.AlexNet()
            >>> list(ModuleMixin.trainable_layers(model, names=True))
        """
        return trainable_layers(self, names=names)

    def number_of_parameters(self, trainable=True):
        """
        Tally the number of model paramters.

        Args:
            trainable (bool, default=True): specify if only trainable
                params should be counted.

        Returns:
            int: number of paramaters in this module and all submodules
        """
        return number_of_parameters(self, trainable)

    def _device_dict(self):
        return {key: item.device for key, item in self.state_dict().items()}

    def devices(self):
        """
        Returns all devices this module state is mounted on

        Returns:
            Set[torch.device]: set of devices used by this model

        Example:
            >>> from viame.pytorch import netharn as nh
            >>> self = nh.models.toynet.ToyNet2d()
            >>> ub.peek(self.devices())
            device(type='cpu')
            >>> # xdoctest: +REQUIRES(--multigpu)
            >>> self = nh.XPU([0, 1]).mount(self)
            >>> print(self.devices())
            {device(type='cuda', index=0), device(type='cuda', index=1)}
            >>> print('self.main_device = {!r}'.format(self.main_device))
            self.main_device = device(type='cuda', index=0)
        """
        state_devices = self._device_dict()
        devices = set(state_devices.values())
        if hasattr(self, 'device_ids'):
            # Handle data parallel
            for _id in self.device_ids:
                devices.add(torch.device(_id))
        return devices

    @property
    def main_device(self):
        """
        The main/src torch device used by this model
        """
        if hasattr(self, 'src_device_obj'):
            return self.src_device_obj
        else:
            devices = self.devices()
            if len(devices) > 1:
                raise NotImplementedError('no information maintained on which device is primary')
            else:
                return list(devices)[0]


def number_of_parameters(model, trainable=True):
    """
    Returns number of trainable parameters in a torch module

    Example:
        >>> from viame.pytorch import netharn as nh
        >>> model = nh.models.ToyNet2d()
        >>> number_of_parameters(model)
        824
    """
    if trainable:
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    else:
        model_parameters = model.parameters()
    n_params = sum([np.prod(p.size()) for p in model_parameters])
    return n_params


class grad_context(object):
    """
    Context manager for controlling if autograd is enabled.

    DEPRECATED use ``torch.set_grad_enabled`` instead
    """
    def __init__(self, flag):
        import warnings
        warnings.warn('Deprecated use torch.set_grad_enabled instead',
                      DeprecationWarning)
        if tuple(map(int, torch.__version__.split('.')[0:2])) < (0, 4):
            self.prev = None
            self.flag = flag
        else:
            self.prev = torch.is_grad_enabled()
            self.flag = flag

    def __enter__(self):
        if self.prev is not None:
            torch.set_grad_enabled(self.flag)

    def __exit__(self, *args):
        if self.prev is not None:
            torch.set_grad_enabled(self.prev)
            return False


def _get_method_func(method):
    func = method.__func__
    return func


def _get_method_base_class(method):
    """
    Finds the class in which a particular method function was defined.

    CommandLine:
        xdoctest -m netharn.util.util_torch _get_method_base_class

    Example:
        >>> method = torch.nn.BatchNorm2d(1).forward
        >>> print(_get_method_base_class(method))
        <class 'torch.nn.modules.batchnorm._BatchNorm'>
    """
    import sys
    qualname = method.__qualname__
    module = sys.modules[method.__module__]
    base_name = qualname.split('.')[0]
    base_cls = getattr(module, base_name)
    return base_cls


class IgnoreLayerContext(object):
    """
    Context manager that modifies (monkey-patches) models to temporarily
    remove the forward pass for particular layers.

    Args:
        model (torch.nn.Module): model to modify

        category (type): the module class to be ignored

        enabled (bool, default=True): if True this context manager is enabled
            otherwise it does nothing (i.e. the specified layers will not be
            ignored).

    Example:
        >>> input = torch.rand(1, 1, 10, 10)
        >>> model = torch.nn.BatchNorm2d(1)
        >>> output1 = model(input)
        >>> with IgnoreLayerContext(model, torch.nn.BatchNorm2d):
        ...     output2 = model(input)
        >>> output3 = model(input)
        >>> assert torch.all(output3 == output1)
        >>> assert torch.all(output2 == input)

    Ignore:
        >>> # Test issue with data parallel
        >>> from .util.util_torch import *
        >>> import torch
        >>> from viame.pytorch import netharn as nh
        >>> layer = raw_model = torch.nn.BatchNorm2d(1)
        >>> raw_inputs = torch.rand(8, 1, 10, 10)
        >>> xpu = nh.XPU.coerce([0,1])
        >>> model = xpu.mount(raw_model)
        >>> inputs = xpu.move(raw_inputs)
        >>> output1 = model(inputs)
        >>> with nh.util.IgnoreLayerContext(model, torch.nn.BatchNorm2d):
        ...     print('model.module.forward = {!r}'.format(model.module.forward))
        ...     output2 = model(inputs)
        >>> output3 = model(inputs)
        >>> assert torch.all(output3 == output1)
        >>> assert torch.all(output2 == inputs)
        >>> # ------------
        >>> raw_model = torch.nn.BatchNorm2d(1)
        >>> raw_inputs = torch.rand(8, 1, 10, 10)
        >>> model = xpu.mount(raw_model)
        >>> inputs = xpu.move(raw_inputs)
        >>> output1 = model(inputs)
        >>> self = nh.util.IgnoreLayerContext(model, torch.nn.BatchNorm2d)
        >>> self.__enter__()
        >>> output2 = model(inputs)
        >>> self.__exit__()
        >>> print('CAN WE DO THIS?')
        >>> output3 = model(inputs)
        >>> # ------------
        >>> xpu = nh.XPU.coerce([0,1])
        >>> devices = [torch.device(type='cuda', index=i) for i in [0, 1]]
        >>> replicas = torch.nn.parallel.replicate(xpu.move(raw_model), devices)
        >>> [r.forward for r in replicas]
        >>> print([r.weight for r in replicas])
        >>> r = replicas[1]
        >>> # ------
        >>> assert torch.all(output3 == output1)
        >>> assert torch.all(output2 == inputs)
    """
    def __init__(self, model, category=None, enabled=True):
        self.model = model
        self.category = category
        self.prev_state = None

        self._PATCH_CLASS = True  # are we patching the instance or class?

    def __enter__(self):
        self.prev_state = {}
        def _noop_forward(self, inputs, *args, **kwargs):
            return inputs
        _noop_forward._patched = True

        for name, layer in trainable_layers(self.model, names=True):
            needs_filter = False
            if self.category is not None:
                needs_filter |= isinstance(layer, self.category)

            if needs_filter:
                if self._PATCH_CLASS:
                    func = _get_method_func(layer.forward)
                    already_patched = not getattr(func, '_patched', False)
                    if already_patched:
                        # Patch the entire class if it wasn't already
                        base_cls = _get_method_base_class(layer.forward)
                        assert 'forward' in base_cls.__dict__
                        # print('PATCH FORWARD IN base_cls = {!r}'.format(base_cls))
                        # print('base_cls = {!r}'.format(base_cls))
                        self.prev_state[name] = (base_cls, base_cls.forward)
                        base_cls.forward = _noop_forward
                else:
                    self.prev_state[name] = layer.forward
                    ub.inject_method(layer, _noop_forward, name='forward')
        return self

    def __exit__(self, *args):
        if self.prev_state:
            if self._PATCH_CLASS:
                # Unpatch all patched classes
                for name, state in self.prev_state.items():
                    base_cls, orig = state
                    base_cls.forward = orig
            else:
                for name, layer in trainable_layers(self.model, names=True):
                    if name in self.prev_state:
                        # Unset the instance attribute that overrides the default
                        # class function attribute. Note that we cannot simply
                        # reset the forward attribute to its old value because that
                        # will still leave an entry in the layer.__dict__ that
                        # previously wasn't there. Having the forward method
                        # populated in layer.__dict__ causes issues with data
                        # parallel.
                        del layer.__dict__['forward']


class BatchNormContext(object):
    """
    Sets batch norm training state of `model` to `training` within the context
    manager.

    Args:
        model (torch.nn.Module | Sequnce[Module]): model(s) to modify

        training (bool, default=False):
            if True training of batch norm layers is enabled otherwise it is
            disabled. This is useful for batches of size 1.
    """
    def __init__(self, models, training=True, **kw):
        if not isinstance(models, (tuple, list)):
            models = [models]
        self.models = models
        if kw:
            import warnings
            warnings.warn('the enabled kwarg is deprecated')
            training = kw.pop('enabled', training)
            if len(kw):
                raise ValueError('Unsupported kwargs: {}'.format(list(kw)))
        self.training = training
        self.prev_train_state = None

    def __enter__(self):
        self.prev_train_state = ub.ddict(dict)
        for i, model in enumerate(self.models):
            for name, layer in trainable_layers(model, names=True):
                if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                    self.prev_train_state[i][name] = layer.training
                    layer.training = self.training
        return self

    def __exit__(self, *args):
        if self.prev_train_state:
            for i, model in enumerate(self.models):
                for name, layer in trainable_layers(model, names=True):
                    if name in self.prev_train_state[i]:
                        layer.training = self.prev_train_state[i][name]


DisableBatchNorm = BatchNormContext


def trainable_layers(model, names=False):
    """
    Note:
        This was moved to netharn.initializers.functional.
        Move it back here, or do some other refactoring.

    Example:
        >>> import torchvision
        >>> model = torchvision.models.AlexNet()
        >>> list(trainable_layers(model, names=True))
    """
    if names:
        stack = [('', '', model)]
        while stack:
            prefix, basename, item = stack.pop()
            name = '.'.join([p for p in [prefix, basename] if p])
            if isinstance(item, torch.nn.modules.conv._ConvNd):
                yield name, item
            elif isinstance(item, torch.nn.modules.batchnorm._BatchNorm):
                yield name, item
            elif hasattr(item, 'reset_parameters'):
                yield name, item

            child_prefix = name
            for child_basename, child_item in list(item.named_children())[::-1]:
                stack.append((child_prefix, child_basename, child_item))
    else:
        queue = [model]
        while queue:
            item = queue.pop(0)
            # TODO: need to put all trainable layer types here
            # (I think this is just everything with reset_parameters)
            if isinstance(item, torch.nn.modules.conv._ConvNd):
                yield item
            elif isinstance(item, torch.nn.modules.batchnorm._BatchNorm):
                yield item
            elif hasattr(item, 'reset_parameters'):
                yield item
            # if isinstance(input, torch.nn.modules.Linear):
            #     yield item
            # if isinstance(input, torch.nn.modules.Bilinear):
            #     yield item
            # if isinstance(input, torch.nn.modules.Embedding):
            #     yield item
            # if isinstance(input, torch.nn.modules.EmbeddingBag):
            #     yield item
            for child in item.children():
                queue.append(child)


def one_hot_embedding(labels, num_classes, dtype=None):
    """
    Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N,#classes].

    References:
        https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/4

    CommandLine:
        python -m netharn.loss one_hot_embedding

    Example:
        >>> # each element in target has to have 0 <= value < C
        >>> labels = torch.LongTensor([0, 0, 1, 4, 2, 3])
        >>> num_classes = max(labels) + 1
        >>> t = one_hot_embedding(labels, num_classes)
        >>> assert all(row[y] == 1 for row, y in zip(t.numpy(), labels.numpy()))
        >>> import ubelt as ub
        >>> print(ub.repr2(t.numpy().tolist()))
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ]
        >>> t2 = one_hot_embedding(labels.numpy(), num_classes)
        >>> assert np.all(t2 == t.numpy())
        >>> if torch.cuda.is_available():
        >>>     t3 = one_hot_embedding(labels.to(0), num_classes)
        >>>     assert np.all(t3.cpu().numpy() == t.numpy())
    """
    if isinstance(labels, np.ndarray):
        dtype = dtype or float
        y = np.eye(num_classes, dtype=dtype)
        y_onehot = y[labels]
    else:  # if torch.is_tensor(labels):
        dtype = dtype or torch.float
        y = torch.eye(num_classes, device=labels.device, dtype=dtype)
        y_onehot = y[labels]
    return y_onehot


def one_hot_lookup(probs, labels):
    """
    Return probbility of a particular label (usually true labels) for each item

    Each item in labels corresonds to a row in probs. Returns the index
    specified at each row.

    Example:
        >>> probs = np.array([
        >>>     [0, 1, 2],
        >>>     [3, 4, 5],
        >>>     [6, 7, 8],
        >>>     [9, 10, 11],
        >>> ])
        >>> labels = np.array([0, 1, 2, 1])
        >>> one_hot_lookup(probs, labels)
        array([ 0,  4,  8, 10])
    """
    return probs[np.eye(probs.shape[1], dtype=bool)[labels]]


def torch_ravel_multi_index(multi_index, dims=None, device=None, strides_=None):
    """
    Implementation of `numpy.ravel_multi_index` for torch Tensors

    Args:
        multi_index (List[Tensor] | Tensor) : either a list of indices for
           each dimension, or a tensor containing the same information as
           column vectors.
        dims (Tuple): shape of the array to index into before flattening.
        device (torch.device): default torch device
        strides_ (Tensor, optional): prespecify strides for each dimension
            instead of using dims to compute it.

    Returns:
        torch.LongTensor: flat indices

    Example:
        >>> import torch
        >>> N = 10
        >>> B, A, H, W = 2, 3, 3, 3
        >>> dims = [B, A, 4, H, W]
        >>> device = 'cpu'
        >>> multi_index = [(torch.rand(N) * d).long().to(device) for d in dims]
        >>> multi_index_np = [d.data.cpu().numpy() for d in multi_index]
        >>> numpy_result = np.ravel_multi_index(multi_index_np, dims)
        >>> torch_result = torch_ravel_multi_index(multi_index, dims, device)
        >>> assert np.all(torch_result.data.cpu().numpy() == numpy_result)

    Benchmark:
        >>> import torch
        >>> import ubelt as ub
        >>> N = 10000
        >>> B, A, H, W = 2, 3, 3, 3
        >>> dims = [B, A, 4, H, W]
        >>> device = torch.device('cuda')
        >>> multi_index = [(torch.rand(N) * d).long().to(device) for d in dims]
        >>> #
        >>> ti = ub.Timerit(1000, bestof=10, label='time')
        >>> #
        >>> for timer in ti.reset('cuda'):
        >>>     with timer:
        >>>         torch_ravel_multi_index(multi_index, dims, device)
        >>>         torch.cuda.synchronize()
        >>> #
        >>> for timer in ti.reset('numpy-for-gpu'):
        >>>     with timer:
        >>>         multi_index_np = [d.data.cpu().numpy() for d in multi_index]
        >>>         np.ravel_multi_index(multi_index_np, dims)
        >>>         torch.cuda.synchronize()
        >>> #
        >>> multi_index_np = [d.data.cpu().numpy() for d in multi_index]
        >>> for timer in ti.reset('numpy-native'):
        >>>     with timer:
        >>>         np.ravel_multi_index(multi_index_np, dims)
        >>>         torch.cuda.synchronize()
        >>> #
        >>> strides = np.cumprod(dims[::-1])[::-1][1:].tolist() + [1]
        >>> strides_ = torch.LongTensor(strides).to(device)
        >>> for timer in ti.reset('cuda-precomp-strides'):
        >>>     with timer:
        >>>         torch_ravel_multi_index(multi_index, dims, device, strides_)
        >>>         torch.cuda.synchronize()
    """
    if strides_ is None:
        if 1:
            # this one is the fastest, my guess is because torch
            # doesnt have to worry about managing numpy memory
            nextdim = torch.LongTensor(list(dims[1:]) + [1])
            strides_ = torch.flip(torch.flip(nextdim, (0,)).cumprod(dim=0), (0,))
            strides_ = strides_.to(device)
        elif 0:
            strides = np.cumprod(dims[::-1])[::-1][1:].tolist() + [1]
            strides_ = torch.LongTensor(strides).to(device)
        elif 0:
            strides = np.cumprod([1] + list(dims[::-1][0:-1]))[::-1]
            strides = np.ascontiguousarray(strides)
            strides_ = torch.LongTensor(strides).to(device)
        else:
            strides = np.cumprod([1] + list(dims[::-1][0:-1]))[::-1]
            strides = np.ascontiguousarray(strides)
            strides_ = torch.from_numpy(strides).to(device)
            # strides_ = torch.LongTensor(strides.tolist()).to(device)
    if isinstance(multi_index, (list, tuple)):
        multi_index = torch.cat([x.view(-1, 1) for x in multi_index], dim=1)

    # Could do a mm product if that gets implemented for LongTensors
    result = (multi_index * strides_).sum(dim=1).contiguous()

    # if 1:
    #     if 1:
    #         strides_ = torch.LongTensor(strides).to(device)
    #         multi_index_ = torch.cat([x.view(-1, 1) for x in multi_index], dim=1)
    #         result = (multi_index_ * strides_).sum(dim=1)
    #         # [x * s for s, x in zip(strides_, multi_index)]
    #     else:
    #         strides_ = torch.FloatTensor(strides).view(-1, 1).to(device)
    #         multi_index_ = torch.cat([x.view(-1, 1) for x in multi_index], dim=1).float()
    #         result = torch.mm(multi_index_, strides_).view(-1).long()

    #     # result = (multi_index_ * strides_.view(-1)).sum(dim=1)
    # else:
    #     flat_size = len(multi_index[0])
    #     result = torch.zeros(flat_size, dtype=torch.long, device=device)
    #     for stride, index in zip(strides, multi_index):
    #         if len(index) == 1 and isinstance(index, list):
    #             index = torch.LongTensor(index * flat_size).to(device)
    #         result += stride * index

    return result


class freeze_params(object):
    """
    Context manager for freezing / unfreezing specific layers / params.

    Args:
        layers (Module | List[Module]):
            the modules containing the params you want to freeze

        enabled (bool, default=True): if this layer is enabled

    Example:
        >>> inputs, net = freeze_params.demodata()
        >>> self = freeze_params(net)
        >>> list(self.named_parameters())

    Example:
        >>> from .util.util_torch import *
        >>> inputs, net = freeze_params.demodata()
        >>> with freeze_params(net[1]) as self:
        >>>     x = net(inputs)
        >>> loss = x.sum()
        >>> loss.backward()
        >>> for k, v in net.named_parameters():
        >>>     print('{}.grad = {}'.format(k, v.grad))

        >>> inputs, net = freeze_params.demodata()
        >>> with freeze_params([net[1], net[2]]) as self:
        >>>     x = net(inputs)
        >>> p1 = net(inputs).sum()
        >>> p2 = x.sum()
        >>> loss = p1 + p2
        >>> loss.backward()
        >>> for k, v in net.named_parameters():
        >>>     print('{}.grad = {}'.format(k, v.grad))

        >>> inputs, net = freeze_params.demodata()
        >>> with freeze_params([net[1], net[2]]) as self:
        >>>     x = net(inputs)
        >>> loss = x.sum()
        >>> loss.backward()
        >>> for k, v in net.named_parameters():
        >>>     print('{}.grad = {}'.format(k, v.grad))
    """

    def __init__(self, layers, enabled=True):
        if isinstance(layers, torch.nn.Module):
            self.layers = [layers]
        else:
            if ub.iterable(layers):
                self.layers = list(layers)
            else:
                raise TypeError(layers)
        self.enabled = enabled
        self.state = {}

    @classmethod
    def demodata(cls):
        from viame.pytorch import netharn as nh
        inputs = torch.rand(2, 1, 1, 1)
        net = nh.layers.Sequential(
            nh.layers.ConvNorm2d(1, 1, 1),
            nh.layers.ConvNorm2d(1, 1, 1),
            nh.layers.ConvNorm2d(1, 1, 1),
        )
        return inputs, net

    def named_parameters(self):
        """ Iterate through all the parameters this context manages """
        for i, layer in enumerate(self.layers):
            prefix = str(i)
            for name, param in layer.named_parameters(prefix, recurse=True):
                yield name, param

    def _set_requires_grad(self, flag):
        for name, param in self.named_parameters():
            param.requires_grad = flag

    def _push_state(self):
        assert len(self.state) == 0, 'can only save ones state'
        for name, param in self.named_parameters():
            self.state[name] = param.requires_grad

    def _pop_state(self):
        for name, param in self.named_parameters():
            param.requires_grad = self.state.pop(name)

    def __enter__(self):
        if self.enabled:
            self._push_state()
            self._set_requires_grad(False)
        return self

    def __exit__(self, a, b, c):
        if self.enabled:
            self._pop_state()
