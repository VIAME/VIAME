"""
DataParallel without direct GPU-to-GPU traffic.

Some hosts advertise peer access between GPUs but silently corrupt the copies
(IOMMU / ACS misconfiguration, driver bugs). Nothing raises; replicas simply
train on zeros. ``verify_peer_copies`` detects that at mount time, and the
``Host*`` functions below route every cross-device transfer through host
memory so DataParallel stays correct on such machines.
"""
from collections import OrderedDict
import torch
import torch.cuda.comm as comm
from torch._utils import (
    _flatten_dense_tensors, _get_device_index, _unflatten_dense_tensors)
from torch.autograd import Function

P2P_MODES = ('auto', 'host', 'single', 'require', 'peer')


def verify_peer_copies(device_ids, numel=1 << 20):
    """
    Round-trip a known pattern between every ordered pair of devices along
    each transfer path DataParallel uses.

    Returns:
        List[Tuple[int, int, str]]: (src, dst, path) for every failing copy.
    """
    failures = []
    expected = torch.arange(numel, dtype=torch.float32)
    devices = [torch.device('cuda', _get_device_index(d, True))
               for d in device_ids]
    for src in devices:
        pattern = expected.to(src)
        for dst in devices:
            if dst == src:
                continue
            paths = {
                'copy': lambda: pattern.to(dst),
                'broadcast': lambda: comm.broadcast(pattern, [dst])[0],
                'reduce_add': lambda: comm.reduce_add(
                    [pattern, torch.zeros(numel, device=dst)], dst),
            }
            for name, fn in paths.items():
                try:
                    with torch.no_grad():
                        got = fn().cpu()
                    ok = torch.equal(got, expected)
                except Exception:
                    ok = False
                if not ok:
                    failures.append((src.index, dst.index, name))
    return failures


def _via_host(tensor, device):
    if tensor.device == device:
        return tensor
    return tensor.to('cpu').to(device)


def _coalesced_via_host(tensors, device):
    """
    Move many tensors to ``device`` through host memory with one transfer per
    dtype rather than one per tensor.
    """
    out = [None] * len(tensors)
    groups = OrderedDict()
    for i, t in enumerate(tensors):
        groups.setdefault(t.dtype, []).append(i)
    for idxs in groups.values():
        group = [tensors[i] for i in idxs]
        flat = _via_host(_flatten_dense_tensors(group), device)
        for i, t in zip(idxs, _unflatten_dense_tensors(flat, group)):
            out[i] = t
    return out


class HostBroadcast(Function):
    """Mirror of torch's Broadcast: forward fans out, backward sums."""

    @staticmethod
    def forward(ctx, target_gpus, *inputs):
        ctx.target_gpus = [_get_device_index(x, True) for x in target_gpus]
        ctx.num_inputs = len(inputs)
        if not inputs:
            return ()
        ctx.input_device = inputs[0].get_device()
        outputs = []
        staged = _coalesced_via_host([i.detach() for i in inputs], torch.device('cpu'))
        for dev in ctx.target_gpus:
            if dev == ctx.input_device:
                outputs.extend(inputs)
            else:
                outputs.extend(_coalesced_via_host(staged, torch.device('cuda', dev)))
        non_differentiable = []
        for idx, needs_grad in enumerate(ctx.needs_input_grad[1:]):
            if not needs_grad:
                non_differentiable.extend(
                    outputs[j * ctx.num_inputs + idx]
                    for j in range(len(ctx.target_gpus)))
        ctx.mark_non_differentiable(*non_differentiable)
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        device = torch.device('cuda', ctx.input_device)
        totals = [None] * ctx.num_inputs
        for j in range(len(ctx.target_gpus)):
            replica = grad_outputs[j * ctx.num_inputs:(j + 1) * ctx.num_inputs]
            idxs = [i for i, g in enumerate(replica) if g is not None]
            moved = _coalesced_via_host([replica[i] for i in idxs], device)
            for i, g in zip(idxs, moved):
                if totals[i] is None:
                    totals[i] = g.clone()
                else:
                    totals[i].add_(g)
        return (None,) + tuple(totals)


class HostScatter(Function):

    @staticmethod
    def forward(ctx, target_gpus, chunk_sizes, dim, input):
        target_gpus = [_get_device_index(x, True) for x in target_gpus]
        ctx.dim = dim
        ctx.input_device = input.get_device() if input.is_cuda else -1
        if chunk_sizes is None:
            chunks = input.chunk(len(target_gpus), dim)
        else:
            chunks = input.split(chunk_sizes, dim)
        outputs = tuple(
            _via_host(chunk, torch.device('cuda', dev))
            for chunk, dev in zip(chunks, target_gpus))
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        if ctx.input_device == -1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda', ctx.input_device)
        grads = [_via_host(g, device) for g in grad_outputs]
        return None, None, None, torch.cat(grads, ctx.dim)


class HostGather(Function):

    @staticmethod
    def forward(ctx, target_device, dim, *inputs):
        if target_device == 'cpu' or target_device == -1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda', _get_device_index(target_device, True))
        ctx.dim = dim
        ctx.input_gpus = tuple(i.get_device() for i in inputs)
        ctx.unsqueezed_scalar = all(t.dim() == 0 for t in inputs) and dim == 0
        if ctx.unsqueezed_scalar:
            inputs = tuple(t.view(1) for t in inputs)
        ctx.input_sizes = tuple(i.size(dim) for i in inputs)
        staged = [_via_host(i, device) for i in inputs]
        return torch.cat(staged, dim)

    @staticmethod
    def backward(ctx, grad_output):
        grads = grad_output.split(ctx.input_sizes, ctx.dim)
        grads = [_via_host(g, torch.device('cuda', dev))
                 for g, dev in zip(grads, ctx.input_gpus)]
        if ctx.unsqueezed_scalar:
            grads = [g[0] for g in grads]
        return (None, None) + tuple(grads)


def _host_broadcast_reshape(tensors, devices, detach=False):
    if len(tensors) == 0:
        return []
    if detach:
        with torch.no_grad():
            copies = HostBroadcast.apply(devices, *tensors)
    else:
        copies = HostBroadcast.apply(devices, *tensors)
    return [list(copies[i:i + len(tensors)])
            for i in range(0, len(copies), len(tensors))]


def host_replicate(network, devices, detach=False):
    """
    torch.nn.parallel.replicate with parameters and buffers broadcast via host
    memory. Script modules are not supported.
    """
    if not devices:
        return []
    devices = [_get_device_index(x, True) for x in devices]
    num_replicas = len(devices)

    params = list(network.parameters())
    param_indices = {p: i for i, p in enumerate(params)}
    param_copies = _host_broadcast_reshape(params, devices, detach)

    buffers = list(network.buffers())
    buffers_rg = [b for b in buffers if b.requires_grad and not detach]
    buffers_not_rg = [b for b in buffers if not (b.requires_grad and not detach)]
    buffer_indices_rg = {b: i for i, b in enumerate(buffers_rg)}
    buffer_indices_not_rg = {b: i for i, b in enumerate(buffers_not_rg)}
    buffer_copies_rg = _host_broadcast_reshape(buffers_rg, devices, detach)
    buffer_copies_not_rg = _host_broadcast_reshape(buffers_not_rg, devices, True)

    modules = list(network.modules())
    module_copies = [[] for _ in devices]
    module_indices = {}
    for i, module in enumerate(modules):
        module_indices[module] = i
        for j in range(num_replicas):
            replica = module._replicate_for_data_parallel()
            replica._former_parameters = OrderedDict()
            module_copies[j].append(replica)

    for i, module in enumerate(modules):
        for key, child in module._modules.items():
            for j in range(num_replicas):
                replica = module_copies[j][i]
                if child is None:
                    replica._modules[key] = None
                else:
                    setattr(replica, key, module_copies[j][module_indices[child]])
        for key, param in module._parameters.items():
            for j in range(num_replicas):
                replica = module_copies[j][i]
                if param is None:
                    replica._parameters[key] = None
                else:
                    param_copy = param_copies[j][param_indices[param]]
                    setattr(replica, key, param_copy)
                    replica._former_parameters[key] = param_copy
        for key, buf in module._buffers.items():
            for j in range(num_replicas):
                replica = module_copies[j][i]
                if buf is None:
                    replica._buffers[key] = None
                elif buf.requires_grad and not detach:
                    setattr(replica, key, buffer_copies_rg[j][buffer_indices_rg[buf]])
                else:
                    setattr(replica, key, buffer_copies_not_rg[j][buffer_indices_not_rg[buf]])

    return [module_copies[j][0] for j in range(num_replicas)]


def host_scatter(inputs, target_gpus, dim=0):
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return HostScatter.apply(target_gpus, None, dim, obj)
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return [list(i) for i in zip(*map(scatter_map, obj))]
        if isinstance(obj, dict) and len(obj) > 0:
            return [type(obj)(i) for i in zip(*map(scatter_map, obj.items()))]
        return [obj for _ in target_gpus]

    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None


def host_scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    scattered_inputs = host_scatter(inputs, target_gpus, dim) if inputs else []
    scattered_kwargs = host_scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(scattered_inputs) < len(scattered_kwargs):
        scattered_inputs.extend(
            () for _ in range(len(scattered_kwargs) - len(scattered_inputs)))
    elif len(scattered_kwargs) < len(scattered_inputs):
        scattered_kwargs.extend(
            {} for _ in range(len(scattered_inputs) - len(scattered_kwargs)))
    return tuple(scattered_inputs), tuple(scattered_kwargs)


def host_gather(outputs, target_device, dim=0):
    def gather_map(outputs_):
        out = outputs_[0]
        if isinstance(out, torch.Tensor):
            return HostGather.apply(target_device, dim, *outputs_)
        if out is None:
            return None
        if isinstance(out, dict):
            if not all(len(out) == len(d) for d in outputs_):
                raise ValueError('All dicts must have the same number of keys')
            return type(out)((k, gather_map([d[k] for d in outputs_]))
                             for k in out)
        return type(out)(map(gather_map, zip(*outputs_)))

    try:
        return gather_map(outputs)
    finally:
        gather_map = None


class HostStagedMixin(object):
    """
    Overrides for torch.nn.DataParallel subclasses so replicate / scatter /
    gather never issue a device-to-device copy.
    """

    def replicate(self, module, device_ids):
        return host_replicate(module, device_ids, not torch.is_grad_enabled())

    def scatter(self, inputs, kwargs, device_ids):
        return host_scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def gather(self, outputs, output_device):
        return host_gather(outputs, output_device, dim=self.dim)


if __name__ == '__main__':
    """
    python -m viame.pytorch.netharn.host_parallel [device ids...]
    """
    import sys
    ids = [int(a) for a in sys.argv[1:]] or list(range(torch.cuda.device_count()))
    failures = verify_peer_copies(ids)
    for src, dst, path in failures:
        print('FAIL {}->{} ({})'.format(src, dst, path))
    print('{} of {} peer transfers corrupted across GPUs {}'.format(
        len(failures), 3 * len(ids) * (len(ids) - 1), ids))
    sys.exit(1 if failures else 0)
