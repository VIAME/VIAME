"""Optional GPU letterbox preprocessing for the GFIT/Netharn classifiers."""
import numpy as np


class CUDAClassifierBatches:
    """Yield the predictor's existing batch format, with tensors already on GPU.

    Iteration stays in the calling process; no CUDA contexts in DataLoader
    workers. Input chips are CPU views. Resize output crosses into PyTorch via
    the CUDA array interface, without downloading it between the two libraries.
    """
    def __init__(self, images, input_dims, batch_size, device):
        import torch
        from viame.image_kernels import cuda
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("CUDA preprocessing requires a CUDA inference device")
        if int(batch_size) < 1:
            raise ValueError("batch_size must be positive")
        self.images = images
        self.input_dims = input_dims
        self.batch_size = int(batch_size)
        self.device = device
        self.context = cuda.Context(device.index if device.index is not None else torch.cuda.current_device())

    def __len__(self):
        return (len(self.images) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        import torch
        import kwimage
        for start in range(0, len(self.images), self.batch_size):
            tensors = []
            for image in self.images[start:start + self.batch_size]:
                image = kwimage.atleast_3channels(image)[:, :, :3]
                if image.dtype != np.uint8:
                    raise TypeError("CUDA classifier preprocessing requires uint8 chips")
                gpu_image = self.context.upload(image)
                if self.input_dims is not None:
                    dsize = tuple(self.input_dims[::-1])
                    if len(dsize) > 3:
                        dsize = (256, 256)
                    gpu_image = self.context.resize_letterbox(gpu_image, *dsize)
                tensor = torch.as_tensor(gpu_image, device=self.device).permute(2, 0, 1)
                # ImageListDataset divides in numpy float64 before FloatTensor.
                tensors.append((tensor.to(torch.float64) / 255.0).to(torch.float32))
            yield {"inputs": {"rgb": torch.stack(tensors)}}


def auto_batches(images, input_dims, batch_size, device):
    """Return CUDA batches if usable, otherwise let the predictor use its CPU loader.

    Probe before processing any chips. Runtime execution failures remain errors.
    """
    import torch
    try:
        from viame.image_kernels import cuda
    except ImportError:
        return None
    if torch.device(device).type != "cuda":
        return None
    if any(image.dtype != np.uint8 for image in images):
        return None
    if not cuda.available():
        return None
    try:
        return CUDAClassifierBatches(images, input_dims, batch_size, device)
    except (ImportError, RuntimeError):
        return None
