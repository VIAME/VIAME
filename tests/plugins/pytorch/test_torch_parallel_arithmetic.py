# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""torch's threaded CPU arithmetic gives the right answers.

A Windows source build that links two OpenMP runtimes -- MSVC's VCOMP140
from `/openmp` and LLVM's libomp140 from `/openmp:llvm`, which oneDNN needs
-- loads both into one process, and each keeps its own thread pool and its
own `omp_get_thread_num`. Parallel regions then slice their work against
inconsistent state and write garbage. The v0.23.5 Windows binaries shipped
that way.

This is a CRITICAL test rather than a UNIT one because the existing critical
set did not catch it. `viame_examples:train_netharn_cfrnn_from_viame_csv`
passed on the broken build: it checks that training completes, and training
did complete -- on blank images, reporting plausible losses, for 25 epochs,
reaching mAP 0.002 where a good build reaches 0.24.

What makes it invisible is the size threshold. ATen's GRAIN_SIZE is 32768
elements; below it `at::parallel_for` runs serially and is correct, above it
threads spawn and the answers are wrong. Every small fixture is below it, so
the sizes here are chosen to bracket it: 104x104x3 is 32448 elements and
105x105x3 is 33075.
"""

import pytest

torch = pytest.importorskip("torch")

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Either side of GRAIN_SIZE, then well past it.
SIZES = (64, 104, 105, 128, 224, 256, 512, 1024)


def _report(n):
    return "%dx%d (%d elements, GRAIN_SIZE is 32768)" % (n, n, 3 * n * n)


@pytest.mark.parametrize("n", SIZES)
def test_inplace_broadcast_sub_div(n):
    """`tensor.sub_(mean).div_(std)`, which the image transforms use.

    This is the operation that fails: a bare sub_/div_ against a (3, 1, 1)
    tensor returned around -9e8 at 256x256 in the broken build, where
    0.0655 is correct. A division by a scalar, which broadcasts nothing, was
    fine -- so the fault is the vectorised broadcast kernels, not the cast.
    """
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)

    tensor = torch.full((3, n, n), 0.5)
    tensor.sub_(mean).div_(std)

    expected = (0.5 - MEAN[0]) / STD[0]
    assert tensor[0].mean().item() == pytest.approx(expected, abs=1e-4), (
        "in-place broadcast arithmetic is wrong at %s; torch is linking more "
        "than one OpenMP runtime. Check with "
        "`dumpbin /dependents torch_cpu.dll | findstr /i omp` and see "
        "USE_OPENMP in cmake/add_project_pytorch.cmake" % _report(n)
    )


@pytest.mark.parametrize("n", SIZES)
def test_torchvision_to_dtype_scale(n):
    """torchvision's uint8 -> float32 scaling.

    The same defect seen through `transforms.v2.ToDtype(float32, scale=True)`,
    which returned 0.0 for every real image and so handed RF-DETR training a
    blank picture.
    """
    v2 = pytest.importorskip("torchvision.transforms.v2")

    tensor = torch.full((3, n, n), 200, dtype=torch.uint8)
    out = v2.ToDtype(torch.float32, scale=True)(v2.ToImage()(tensor))

    assert out.max().item() == pytest.approx(200.0 / 255.0, abs=1e-4), (
        "uint8 -> float32 scaling is wrong at %s; see "
        "test_inplace_broadcast_sub_div for the cause" % _report(n)
    )


def test_normalize_matches_manual():
    """torchvision's Normalize agrees with the arithmetic done by hand.

    sam2 carries a hand-written per-channel replacement for Normalize
    (packages/patches/sam2/sam2/utils/transforms.py) because of this. When
    this passes on every supported build, that workaround can go.
    """
    transforms = pytest.importorskip("torchvision.transforms")

    tensor = torch.full((3, 256, 256), 0.5)
    through_torchvision = transforms.Normalize(list(MEAN), list(STD))(tensor.clone())

    manual = torch.zeros_like(tensor)
    for channel in range(3):
        manual[channel] = (tensor[channel] - MEAN[channel]) / STD[channel]

    assert torch.allclose(through_torchvision, manual, atol=1e-4), (
        "Normalize disagrees with per-channel arithmetic at 256x256; see "
        "test_inplace_broadcast_sub_div for the cause"
    )
