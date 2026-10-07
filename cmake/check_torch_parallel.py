#!/usr/bin/env python
# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Check that torch's threaded CPU arithmetic gives the right answers.

A Windows source build of torch that links two OpenMP runtimes -- MSVC's
VCOMP140 from `/openmp` and LLVM's libomp140 from `/openmp:llvm` -- loads
both into one process, and each keeps its own thread pool and its own
`omp_get_thread_num`. Parallel regions then slice their work against
inconsistent state and write garbage.

What makes it worth a dedicated check is that it is invisible to ordinary
tests. Everything below ATen's GRAIN_SIZE of 32768 elements runs serially and
is correct, so small fixtures pass; only real images cross the threshold.
The v0.23.5 Windows binaries shipped with it, and RF-DETR training ran to
completion on blank pictures, reporting plausible losses the whole way.

Exits non-zero on failure, so it can gate a build.

  python check_torch_parallel.py
"""

import sys

import torch

# 3 * 104 * 104 = 32448, just under GRAIN_SIZE; 3 * 105 * 105 = 33075, just
# over. The pair brackets the serial/threaded boundary exactly.
SIZES = (64, 104, 105, 128, 224, 256, 512, 1024)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
TOLERANCE = 1e-4


def check_broadcast_arithmetic():
    """In-place broadcasting sub_/div_, which is what the transforms use."""
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    want = (0.5 - MEAN[0]) / STD[0]
    bad = []
    for n in SIZES:
        x = torch.full((3, n, n), 0.5)
        x.sub_(mean).div_(std)
        got = x[0].mean().item()
        if abs(got - want) > TOLERANCE:
            bad.append((n, 3 * n * n, got, want))
    return bad


def check_to_dtype_scale():
    """torchvision's uint8 -> float32 scaling, if torchvision is present."""
    try:
        from torchvision.transforms.v2 import ToDtype, ToImage
    except Exception:
        return None
    convert = ToDtype(torch.float32, scale=True)
    bad = []
    for n in SIZES:
        x = torch.full((3, n, n), 200, dtype=torch.uint8)
        got = convert(ToImage()(x)).max().item()
        want = 200.0 / 255.0
        if abs(got - want) > TOLERANCE:
            bad.append((n, 3 * n * n, got, want))
    return bad


def main():
    print('torch       %s' % torch.__version__)
    print('threads     %d' % torch.get_num_threads())
    try:
        backend = [l for l in torch.__config__.parallel_info().splitlines()
                   if 'parallel backend' in l]
        print('backend     %s' % (backend[0].split(':')[-1].strip()
                                  if backend else 'unknown'))
    except Exception:
        pass

    failures = 0
    for label, bad in (('in-place broadcast sub_/div_', check_broadcast_arithmetic()),
                       ('torchvision ToDtype(scale=True)', check_to_dtype_scale())):
        if bad is None:
            print('SKIP %s (torchvision not importable)' % label)
            continue
        if bad:
            failures += 1
            print('FAIL %s' % label)
            for n, els, got, want in bad:
                print('       %dx%d (%d elements): got %.6f, expected %.6f'
                      % (n, n, els, got, want))
        else:
            print('ok   %s' % label)

    if failures:
        print()
        print('torch is computing the wrong answers on threaded CPU paths.')
        print('A Windows source build linking both VCOMP140 and libomp140 does')
        print('this; check with:')
        print('    dumpbin /dependents torch_cpu.dll | findstr /i omp')
        print('and see USE_OPENMP in cmake/add_project_pytorch.cmake.')
        return 1

    print()
    print('threaded CPU arithmetic is correct above and below GRAIN_SIZE')
    return 0


if __name__ == '__main__':
    sys.exit(main())
