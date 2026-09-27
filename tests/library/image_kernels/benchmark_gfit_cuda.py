"""Manual GFIT motion benchmark; run from a CUDA-enabled installed environment."""
import statistics
import time
import numpy as np
from viame.algo import ImageFilter
from viame.modules import modules
from viame.types import Image, ImageContainer
from viame.image_kernels import cuda


def main():
    modules.load_known_modules()
    filters = []
    for name, options in [
        ("vxl_convert_image", dict(format="byte", single_channel="true")),
        ("vxl_average", dict(type="window", window_size="5", round="false", output_variance="true")),
        ("vxl_average", dict(type="window", window_size="30", round="false", output_variance="true")),
        ("vxl_convert_image", dict(format="byte", scale_factor="0.5")),
    ]:
        f = ImageFilter.create(name)
        cfg = f.get_configuration()
        for key, value in options.items():
            cfg.set_value(key, value)
        f.set_configuration(cfg)
        filters.append(f)
    context = cuda.Context()
    rng = np.random.default_rng(78)
    source = rng.integers(0, 256, (960, 1728, 3), dtype=np.uint8)
    device = context.upload(source)
    output = context.gfit_motion(device)
    def cpu():
        grey = filters[0].filter(ImageContainer(Image(source)))
        short = filters[3].filter(filters[1].filter(grey)).asarray().reshape(source.shape[:2])
        long = filters[3].filter(filters[2].filter(grey)).asarray().reshape(source.shape[:2])
        return np.stack([short, grey.asarray().reshape(source.shape[:2]), long], axis=2)
    def resident():
        context.gfit_motion(device, out=output)
    def roundtrip():
        context.upload(source, out=device)
        resident()
        return context.download(output)
    for name, operation in [("CPU filter chain", cpu), ("CUDA device buffers", resident),
                            ("CUDA upload + compute + download", roundtrip)]:
        for _ in range(35):
            operation()
        times = []
        for _ in range(10):
            start = time.perf_counter()
            operation()
            times.append((time.perf_counter() - start) * 1000)
        print(f"{name}: {statistics.median(times):.3f} ms", flush=True)


if __name__ == "__main__":
    main()
