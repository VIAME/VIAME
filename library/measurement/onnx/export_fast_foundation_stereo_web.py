# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Export Fast-FoundationStereo as a single ONNX file that onnxruntime-web's
WebGPU provider can run, for DIVE's client-side interactive stereo.

Same I/O contract as the upstream scripts/make_single_onnx.py (ImageNet-
normalised left_image/right_image [1,3,H,W] float32 -> disparity [1,1,H,W],
plus the sidecar yaml) and the same output to ~1e-4 px, so the file is a
drop-in for fast_foundation_stereo_onnx as well. What changes, and why:

1. The group-wise correlation volume is built without the [B,G,C/G,D,H,W]
   tensor (1.5 GB each at 576x960): features are L2-normalised per group once,
   then each disparity shift is a [B,C,H,W] product reduced per group. This is
   also what brings CPU peak memory from ~16 GB down to ~5 GB.
2. Disparity stacks are concatenated in chunks of <= 8, since the native WebGPU
   provider caps a shader at 10 storage buffers (a 48-input Concat fails).
3. Every Conv3d becomes 2-D convolutions over depth slices (Conv3dVia2d):
   the WebGPU 3-D conv kernel is naive (it was ~90% of the runtime) and only
   accepts one pad value, while 2-D conv has the optimised path. Depth
   padding uses zero slices rather than a Pad op so ORT's PadFusion cannot
   fold it into a conv.
4. ConvTranspose3d (unsupported on WebGPU) becomes eight phase Conv3d's on
   the unstrided input, interleaved (the sub-pixel form).

Runtime notes for the browser side (verified probe-by-probe against CPU
onnxruntime): run it on onnxruntime-web's *native* WebGPU provider
(`onnxruntime-web/webgpu`) - the default bundle's JSEP kernels return an
all-zero cost volume for this graph - and with `graphOptimizationLevel`
'basic': one of the extended-level fusions corrupts the GRU gates (~2 px).

Usage:
  python export_fast_foundation_stereo_web.py \
      --model_dir weights/23-36-37/model_best_bp2_serialize.pth \
      --save_path out/ --height 576 --width 960 --valid_iters 8 \
      --onnx_name fast_foundation_stereo_l

  python export_fast_foundation_stereo_web.py --self_test
"""

import argparse
import logging
import os
import sys

os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf

DEFAULT_REPO = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..',
    'packages', 'pytorch-libs', 'fast-foundation-stereo'))


def _add_repo_to_path(repo):
    sys.path.insert(0, repo)
    sys.path.insert(0, os.path.join(repo, 'scripts'))

STACK_CHUNK = 8


def _chunked_stack(tensors, dim):
    chunks = [torch.stack(tensors[i:i + STACK_CHUNK], dim=dim) for i in range(0, len(tensors), STACK_CHUNK)]
    return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=dim)


def build_gwc_volume_web(refimg_fea, targetimg_fea, maxdisp, num_groups, normalize=True):
    B, C, H, W = refimg_fea.shape
    cpg = C // num_groups
    # Everything stays 4-D: normalisation and the per-group reduction run on
    # [B*G, C/G, H, W] views, the same shape of op the model's own feature
    # normalisation uses, which the WebGPU provider is known to get right.
    ref = refimg_fea
    tgt = targetimg_fea
    if normalize:
        ref = F.normalize(ref.reshape(B * num_groups, cpg, H, W), dim=1)
        tgt = F.normalize(tgt.reshape(B * num_groups, cpg, H, W), dim=1)
    else:
        ref = ref.reshape(B * num_groups, cpg, H, W)
        tgt = tgt.reshape(B * num_groups, cpg, H, W)
    slices = []
    for d in range(maxdisp):
        if d == 0:
            shifted = tgt
        elif d >= W:
            shifted = tgt * 0
        else:
            shifted = torch.cat([tgt[:, :, :, :d] * 0, tgt[:, :, :, :W - d]], dim=3)
        slices.append((ref * shifted).sum(dim=1, keepdim=True))   # [B*G, 1, H, W]
    volume = _chunked_stack(slices, dim=1).reshape(B, num_groups, 1, maxdisp, H, W).squeeze(2)
    return volume.contiguous()


GWC_CHUNK = 0


def build_gwc_volume_chunked(refimg_fea, targetimg_fea, maxdisp, num_groups, normalize=True):
    """
    The stock 6-D formulation applied to GWC_CHUNK disparities at a time.
    Fewer, larger kernels than the per-disparity loop (TensorRT: 0.15 s vs
    0.31 s per pair) with 1/(maxdisp/GWC_CHUNK) of the stock export's
    memory. Desktop (onnxruntime-CUDA / TensorRT) only; browsers use
    build_gwc_volume_web.
    """
    B, C, H, W = refimg_fea.shape
    cpg = C // num_groups
    ref = refimg_fea
    tgt = targetimg_fea
    if normalize:
        ref = F.normalize(ref.reshape(B * num_groups, cpg, H, W), dim=1).reshape(B, num_groups, cpg, H, W)
        tgt = F.normalize(tgt.reshape(B * num_groups, cpg, H, W), dim=1).reshape(B, C, H, W)
    else:
        ref = ref.reshape(B, num_groups, cpg, H, W)
    def shifted(d):
        if d == 0:
            return tgt
        if d >= W:
            return tgt * 0
        return torch.cat([tgt[:, :, :, :d] * 0, tgt[:, :, :, :W - d]], dim=3)
    chunks = []
    for start in range(0, maxdisp, GWC_CHUNK):
        ds = list(range(start, min(start + GWC_CHUNK, maxdisp)))
        tv = torch.stack([shifted(d) for d in ds], dim=2).reshape(B, num_groups, cpg, len(ds), H, W)
        chunks.append((ref.unsqueeze(3) * tv).sum(dim=2))          # [B, G, n, H, W]
    return torch.cat(chunks, dim=2).contiguous()


def build_concat_volume_web(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    ref_volume = _chunked_stack([refimg_fea] * maxdisp, dim=2)
    shifted = [targetimg_fea] + [
        targetimg_fea * 0 if d >= W
        else torch.cat([targetimg_fea[:, :, :, :d] * 0, targetimg_fea[:, :, :, :W - d]], dim=3)
        for d in range(1, maxdisp)
    ]
    target_volume = _chunked_stack(shifted, dim=2)
    return torch.cat((ref_volume, target_volume), dim=1).contiguous()


def _zero_pad(x, pads):
    """pads = ((d0, d1), (h0, h1), (w0, w1)) on the last three dims, via zero slices."""
    for axis, (before, after) in zip((2, 3, 4), pads):
        if before == 0 and after == 0:
            continue
        # Zero slices of x rather than Expand: sliced from x so nothing is
        # constant-foldable, and no Expand/Where shape machinery in the graph.
        zero = x.narrow(axis, 0, 1) * 0
        parts = []
        if before:
            parts.append(torch.cat([zero] * before, dim=axis) if before > 1 else zero)
        parts.append(x)
        if after:
            parts.append(torch.cat([zero] * after, dim=axis) if after > 1 else zero)
        x = torch.cat(parts, dim=axis)
    return x


class Conv3dVia2d(nn.Module):
    """
    A Conv3d (groups=1) as 2-D convolutions, which is where onnxruntime-web's
    WebGPU providers have their optimised paths (the 3-D conv kernel is naive
    and took ~90% of the runtime).

    - kernel (kd, 1, 1): one Conv2d with kernel (kd, 1) over [B, C, D, H*W].
    - otherwise: for each depth tap, a Conv2d over [B*D_out, C, H, W] of the
      matching (strided) depth slice, summed. Depth padding is done with zero
      slices so the graph carries no asymmetric-pad conv and no Pad op.
    """

    def __init__(self, conv: nn.Conv3d):
        super().__init__()
        assert conv.groups == 1 and all(d == 1 for d in conv.dilation)
        self.kd, self.kh, self.kw = conv.kernel_size
        self.sd, self.sh, self.sw = conv.stride
        self.pd, self.ph, self.pw = conv.padding
        w = conv.weight.detach()  # [Cout, Cin, kd, kh, kw]
        self.bias = None if conv.bias is None else nn.Parameter(conv.bias.detach(), requires_grad=False)
        if self.kh == 1 and self.kw == 1:
            self.weight = nn.Parameter(w[:, :, :, 0, :].contiguous(), requires_grad=False)  # [Cout, Cin, kd, 1]
        else:
            self.taps = nn.ParameterList([nn.Parameter(w[:, :, i].contiguous(), requires_grad=False) for i in range(self.kd)])

    def forward(self, x):
        B, C, D, H, W = x.shape
        if self.kh == 1 and self.kw == 1:
            assert self.sh == 1 and self.sw == 1
            y = F.conv2d(x.reshape(B, C, D, H * W), self.weight, self.bias, stride=(self.sd, 1), padding=(self.pd, 0))
            return y.reshape(B, y.shape[1], y.shape[2], H, W)
        xp = _zero_pad(x, ((self.pd, self.pd), (0, 0), (0, 0))) if self.pd else x
        Dp = D + 2 * self.pd
        D_out = (Dp - self.kd) // self.sd + 1
        y = None
        for i in range(self.kd):
            sl = xp[:, :, i:i + self.sd * (D_out - 1) + 1:self.sd] if self.sd > 1 else xp[:, :, i:i + D_out]
            sl = sl.permute(0, 2, 1, 3, 4).reshape(B * D_out, C, H, W)
            o = F.conv2d(sl, self.taps[i], None, stride=(self.sh, self.sw), padding=(self.ph, self.pw))
            y = o if y is None else y + o
        Cout, Ho, Wo = y.shape[1], y.shape[2], y.shape[3]
        y = y.reshape(B, D_out, Cout, Ho, Wo).permute(0, 2, 1, 3, 4)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1, 1)
        return y


class PadConv3d(nn.Module):
    """A Conv3d with per-axis padding, expressed as explicit zero padding + unpadded conv."""

    def __init__(self, conv: nn.Conv3d):
        super().__init__()
        self.conv = conv
        self.pads = tuple((p, p) for p in conv.padding)
        conv.padding = (0, 0, 0)

    def forward(self, x):
        return self.conv(_zero_pad(x, self.pads))


class PhaseConvTranspose3d(nn.Module):
    """
    ConvTranspose3d (stride 2) as 2^3 phase convolutions over the *unstrided*
    input, interleaved afterwards (the sub-pixel form). Equivalent to zero
    insertion + full convolution but with 8x fewer FLOPs and no sparse
    intermediate, which is what made the zero-insertion form dominate the
    WebGPU runtime (4.6 of 6.3 s per pair).

    Output position o = 2*i + phi along an axis gathers input positions
    j = i - m for every m with kernel tap t = 2*m + phi + padding inside
    [0, kernel); each phase is therefore a small dense conv with its own taps
    and its own window offset into a once-padded input.
    """

    def __init__(self, deconv: nn.ConvTranspose3d):
        super().__init__()
        assert deconv.groups == 1 and all(d == 1 for d in deconv.dilation)
        assert all(s == 2 for s in deconv.stride), 'phase decomposition written for stride 2'
        w = deconv.weight.detach()  # [Cin, Cout, kd, kh, kw]
        self.bias = None if deconv.bias is None else nn.Parameter(deconv.bias.detach(), requires_grad=False)
        # per axis, per phase: the m offsets (descending, i.e. input positions ascending) and their taps
        self.axes = []
        for k, p, op in zip(deconv.kernel_size, deconv.padding, deconv.output_padding):
            phases = []
            for phi in (0, 1):
                ms = [m for m in range(-k, k + 1) if 0 <= 2 * m + phi + p < k]
                ms.sort(reverse=True)
                phases.append((ms, [2 * m + phi + p for m in ms]))
            self.axes.append(phases)
        self.pad = [max(max(abs(m) for m in ph[0]) for ph in ax) for ax in self.axes]
        weights = {}
        for pd in (0, 1):
            for ph in (0, 1):
                for pw in (0, 1):
                    td = self.axes[0][pd][1]
                    th = self.axes[1][ph][1]
                    tw = self.axes[2][pw][1]
                    sub = w[:, :, td][:, :, :, th][:, :, :, :, tw]  # [Cin, Cout, nd, nh, nw]
                    weights[(pd, ph, pw)] = sub.transpose(0, 1).contiguous()
        self.weights = nn.ParameterDict({
            f'p{pd}{ph}{pw}': nn.Parameter(v, requires_grad=False) for (pd, ph, pw), v in weights.items()
        })

    def forward(self, x):
        B, C, D, H, W = x.shape
        pads = tuple((p, p) for p in self.pad)
        xp = _zero_pad(x, pads)
        outs = {}
        for pd in (0, 1):
            for ph in (0, 1):
                for pw in (0, 1):
                    starts = []
                    sizes = []
                    for axis, (phase, n, pad) in enumerate(zip((pd, ph, pw), (D, H, W), self.pad)):
                        ms = self.axes[axis][phase][0]
                        starts.append(pad - ms[0])   # window begins at input position i - max(m)
                        sizes.append(n + len(ms) - 1)
                    win = xp[:, :, starts[0]:starts[0] + sizes[0], starts[1]:starts[1] + sizes[1], starts[2]:starts[2] + sizes[2]]
                    o = F.conv3d(win, self.weights[f'p{pd}{ph}{pw}'])
                    outs[(pd, ph, pw)] = o[:, :, :D, :H, :W]
        # interleave phases: w, then h, then d
        def merge(get, axis_len, dim):
            a = torch.stack([get(0), get(1)], dim=dim + 1)
            shape = list(a.shape)
            shape[dim] = shape[dim] * 2
            del shape[dim + 1]
            return a.reshape(shape)
        rows = {}
        for pd in (0, 1):
            for ph in (0, 1):
                rows[(pd, ph)] = merge(lambda pw: outs[(pd, ph, pw)], W, 4)
        cols = {pd: merge(lambda ph: rows[(pd, ph)], H, 3) for pd in (0, 1)}
        y = merge(lambda pd: cols[pd], D, 2)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1, 1)
        return y


def strip_noop_casts(model_path):
    """
    Remove Cast nodes whose target type equals their input type (torch's
    .float()/.to(dtype) calls in an fp32 export). onnxruntime-web's WebGPU
    provider (1.27) returns wrong data from such a Cast when its input has
    other consumers, and ORT's own CastElimination does not catch them.
    """
    import onnx
    from onnx import shape_inference
    model = onnx.load(model_path)
    inferred = shape_inference.infer_shapes(model)
    types = {v.name: v.type.tensor_type.elem_type
             for v in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output)}
    types.update({t.name: t.data_type for t in model.graph.initializer})
    graph = model.graph
    outputs = {o.name for o in graph.output}
    rename = {}
    kept = []
    for node in graph.node:
        if node.op_type == 'Cast' and node.output[0] not in outputs:
            to = next(a.i for a in node.attribute if a.name == 'to')
            src = rename.get(node.input[0], node.input[0])
            if types.get(src) == to:
                rename[node.output[0]] = src
                continue
        for i, name in enumerate(node.input):
            if name in rename:
                node.input[i] = rename[name]
        kept.append(node)
    del graph.node[:]
    graph.node.extend(kept)
    onnx.checker.check_model(model)
    onnx.save(model, model_path)
    return len(rename)


def forward_web(self, image1, image2, iters=12, test_mode=True, low_memory=False,
                init_disp=None, profile=False, optimize_build_volume='pytorch1'):
    """
    FastFoundationStereo.forward for export: identical maths, but the two views
    go through the feature extractor one at a time instead of as a batch of two
    that is then sliced. The WebGPU provider mis-reads the second half of that
    batch slice through alias ops (Reshape, Cast), yielding an all-zero cost
    volume.
    """
    fs = sys.modules[type(self).__module__]
    F_ = torch.nn.functional
    features_left = list(self.feature(image1))
    features_right = list(self.feature(image2))
    stem_2x = self.stem_2(image1)

    gwc_volume = fs.build_gwc_volume_optimized_pytorch1(
        features_left[0], features_right[0], self.args.max_disp // 4, self.cv_group,
        normalize=self.args.normalize)
    left_tmp = self.proj_cmb(features_left[0])
    right_tmp = self.proj_cmb(features_right[0])
    concat_volume = fs.build_concat_volume_optimized_pytorch1(left_tmp, right_tmp, maxdisp=self.args.max_disp // 4)
    comb_volume = torch.cat([gwc_volume, concat_volume], dim=1)
    comb_volume = self.corr_stem(comb_volume)
    comb_volume = self.corr_feature_att(comb_volume, features_left[0])
    comb_volume = self.cost_agg(comb_volume, features_left)

    logits = self.classifier(comb_volume).squeeze(1)
    prob = F_.softmax(logits, dim=1)
    if init_disp is None:
        init_disp = fs.disparity_regression(prob, self.args.max_disp // 4)

    cnet_list = list(self.cnet(features_left[0], features_left[1], features_left[2]))
    net_list = [torch.tanh(x[0]) for x in cnet_list]
    inp_list = [torch.relu(x[1]) for x in cnet_list]
    inp_list = [self.cam(x) * x for x in inp_list]
    att = [self.sam(x) for x in inp_list]

    geo_fn = fs.Combined_Geo_Encoding_Volume(
        features_left[0], features_right[0], comb_volume, num_levels=self.args.corr_levels)
    b, c, h, w = features_left[0].shape
    coords = torch.arange(w, dtype=torch.float, device=init_disp.device).reshape(1, 1, w, 1).repeat(b, h, 1, 1)
    disp = init_disp
    disp_up = None
    for itr in range(iters):
        geo_feat = geo_fn(disp, coords, dx=self.dx, low_memory=False)
        net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat, disp, att)
        disp = disp + delta_disp
        if itr == iters - 1:
            disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x)
    return disp_up


def _replace_modules(model, keep_3d_convs=False):
    replaced = {'conv3d_via_2d': 0, 'conv3d_pad': 0, 'deconv3d': 0}
    if keep_3d_convs:
        return replaced
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Conv3d) and child.groups == 1 and all(d == 1 for d in child.dilation):
                setattr(module, child_name, Conv3dVia2d(child))
                replaced['conv3d_via_2d'] += 1
            elif isinstance(child, nn.Conv3d) and len(set(child.padding)) > 1:
                setattr(module, child_name, PadConv3d(child))
                replaced['conv3d_pad'] += 1
            elif isinstance(child, nn.ConvTranspose3d):
                setattr(module, child_name, PhaseConvTranspose3d(child))
                replaced['deconv3d'] += 1
    return replaced


def apply_patches(model, fs_module, keep_3d_convs=False):
    """
    keep_3d_convs=True applies only the memory/graph fixes (lean correlation
    volume, chunked stacks, unbatched features, cast stripping) and leaves the
    3-D convolutions and ConvTranspose3d native. That is the right file for
    VIAME's own onnxruntime-CUDA / TensorRT path, where 3-D conv kernels are
    fast (TensorRT: 0.15 s/pair vs 0.6 s with the 2-D rewrite) and where the
    stock export's 1.5 GB tensors otherwise exhaust a 16 GB GPU. The browser
    needs the full rewrite.
    """
    fs_module.build_gwc_volume_optimized_pytorch1 = build_gwc_volume_chunked if GWC_CHUNK else build_gwc_volume_web
    fs_module.build_concat_volume_optimized_pytorch1 = build_concat_volume_web
    if not keep_3d_convs:
        # Only the browser needs the unbatched feature passes (WebGPU alias
        # bug); the desktop export keeps the batched extractor, which
        # TensorRT runs noticeably faster.
        type(model).forward = forward_web
    return _replace_modules(model, keep_3d_convs)



def self_test(mso):
    """Check the rewrites against torch's own operators."""
    torch.manual_seed(0)
    x = torch.randn(1, 6, 5, 7, 9)
    conv = nn.Conv3d(6, 4, kernel_size=(1, 3, 3), padding=(0, 1, 1))
    ref = conv(x)
    pc = PadConv3d(conv)
    assert torch.allclose(ref, pc(x), atol=1e-6), 'PadConv3d mismatch'
    conv2 = nn.Conv3d(6, 4, kernel_size=(17, 1, 1), padding=(8, 0, 0))
    ref2 = conv2(x)
    assert torch.allclose(ref2, PadConv3d(conv2)(x), atol=1e-6), 'PadConv3d (k,1,1) mismatch'
    for k, st, p in [((3, 3, 3), 1, (1, 1, 1)), ((3, 3, 3), 2, (1, 1, 1)), ((1, 3, 3), 1, (0, 1, 1)), ((17, 1, 1), 1, (8, 0, 0)), ((1, 1, 1), 1, (0, 0, 0))]:
        c3 = nn.Conv3d(6, 4, kernel_size=k, stride=st if k == (3, 3, 3) else 1, padding=p)
        xx = torch.randn(1, 6, 9, 7, 9)
        assert torch.allclose(c3(xx), Conv3dVia2d(c3)(xx), atol=1e-5), f'Conv3dVia2d mismatch {k} s{st}'
    for k, p, op in [((4, 4, 4), (1, 1, 1), (0, 0, 0)), ((3, 3, 3), (1, 1, 1), (1, 1, 1))]:
        de = nn.ConvTranspose3d(6, 4, kernel_size=k, stride=2, padding=p, output_padding=op, bias=False)
        ref3 = de(x)
        out3 = PhaseConvTranspose3d(de)(x)
        assert ref3.shape == out3.shape, (ref3.shape, out3.shape)
        assert torch.allclose(ref3, out3, atol=1e-5), f'PhaseConvTranspose3d mismatch {k}'
    a = torch.randn(1, 16, 6, 10)
    b = torch.randn(1, 16, 6, 10)
    assert torch.allclose(mso._build_gwc_volume_onnx(a, b, 12, 4), build_gwc_volume_web(a, b, 12, 4), atol=1e-5), 'gwc mismatch'
    global GWC_CHUNK
    GWC_CHUNK = 5
    assert torch.allclose(mso._build_gwc_volume_onnx(a, b, 12, 4), build_gwc_volume_chunked(a, b, 12, 4), atol=1e-5), 'chunked gwc mismatch'
    GWC_CHUNK = 0
    assert torch.allclose(mso._build_concat_volume_onnx(a, b, 12), build_concat_volume_web(a, b, 12), atol=1e-6), 'concat mismatch'
    print('web patches match torch reference ops')




def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--repo', default=DEFAULT_REPO, help='Fast-FoundationStereo checkout')
    parser.add_argument('--self_test', action='store_true', help='Check the rewrites against torch and exit')
    parser.add_argument('--model_dir')
    parser.add_argument('--save_path')
    parser.add_argument('--height', type=int, default=576)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--valid_iters', type=int, default=8)
    parser.add_argument('--max_disp', type=int, default=192)
    parser.add_argument('--onnx_name', default='fast_foundation_stereo_l')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gwc_chunk', type=int, default=0,
                        help='Build the correlation volume this many disparities at a time (desktop exports; 0 = per disparity)')
    parser.add_argument('--keep_3d_convs', action='store_true',
                        help='Leave Conv3d/ConvTranspose3d native (for onnxruntime-CUDA/TensorRT, not for browsers)')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    _add_repo_to_path(args.repo)
    import core.foundation_stereo as fs
    import make_single_onnx as mso
    if args.self_test:
        self_test(mso)
        return
    if not args.model_dir or not args.save_path:
        parser.error('--model_dir and --save_path are required')
    assert args.height % 32 == 0 and args.width % 32 == 0, 'height and width must be divisible by 32'

    os.makedirs(args.save_path, exist_ok=True)
    torch.autograd.set_grad_enabled(False)
    model = torch.load(args.model_dir, map_location='cpu', weights_only=False)
    model.args.max_disp = args.max_disp
    model.args.valid_iters = args.valid_iters
    model.args.mixed_precision = False
    model.eval()

    fs.normalize_image = lambda img: img
    global GWC_CHUNK
    GWC_CHUNK = args.gwc_chunk
    replaced = apply_patches(model, fs, keep_3d_convs=args.keep_3d_convs)
    logging.info(f'patched modules: {replaced}')

    wrapper = mso.FastFoundationStereoSingleOnnx(model).to(args.device).eval()
    left = torch.randn(1, 3, args.height, args.width, device=args.device)
    right = torch.randn(1, 3, args.height, args.width, device=args.device)

    onnx_path = os.path.join(args.save_path, args.onnx_name + '.onnx')
    torch.onnx.export(
        wrapper, (left, right), onnx_path, opset_version=17, dynamo=False,
        input_names=['left_image', 'right_image'], output_names=['disparity'],
        do_constant_folding=True,
    )
    removed = 0
    while True:
        n = strip_noop_casts(onnx_path)
        removed += n
        if n == 0:
            break
    logging.info(f'removed {removed} same-type Cast nodes')
    cfg = OmegaConf.to_container(model.args)
    cfg['image_size'] = [args.height, args.width]
    cfg['web_export'] = not args.keep_3d_convs
    cfg['lean_export'] = True
    with open(os.path.join(args.save_path, args.onnx_name + '.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f)
    logging.info(f'wrote {onnx_path}')


if __name__ == '__main__':
    main()
