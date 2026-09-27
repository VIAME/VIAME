"""Run manually on a CUDA host: python benchmark_cuda.py [--repeats 10]."""
import argparse
import statistics
import time
import numpy as np
from viame import image_kernels as cpu
from viame.image_kernels import cuda


def milliseconds(operation, repeats):
    for _ in range(3):
        operation()
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        elapsed.append((time.perf_counter() - start) * 1000)
    return statistics.median(elapsed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not cuda.available():
        parser.exit(1, cuda.availability_error() + "\n")
    rng = np.random.default_rng(1)
    ctx = cuda.Context()
    workspace = cpu.GaussianWorkspace()
    operations = [
        ("Gaussian 1024x1024 size21", rng.random((1024, 1024), dtype=np.float32),
         lambda x: cpu.gaussian_blur(x, 21, workspace=workspace),
         lambda x, out: ctx.gaussian_blur(x, 21, out=out)),
        ("NLM 256x256 patch7 search21", rng.integers(0, 256, (256, 256), dtype=np.uint8),
         lambda x: cpu.denoise(x, 9, 7, 21),
         lambda x, out: ctx.denoise_non_local_means(x, 9, 7, 21, out=out)),
    ]
    for name, source, host_op, device_op in operations:
        device = ctx.upload(source)
        output = device_op(device, None)
        def round_trip():
            ctx.upload(source, out=device)
            device_op(device, output)
            return ctx.download(output)
        expected = host_op(source)
        actual = round_trip()
        if source.dtype == np.uint8:
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-7)
        print(name, flush=True)
        for label, operation in [("CPU", lambda: host_op(source)),
                                 ("CUDA, images on device", lambda: device_op(device, output)),
                                 ("CUDA, upload + compute + download", round_trip)]:
            print(f"  {label}: {milliseconds(operation, args.repeats):.3f} ms", flush=True)


if __name__ == "__main__":
    main()
