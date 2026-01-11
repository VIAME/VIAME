"""
Simple arbitrary-sized datasets for testing / demo purposes
"""
import numpy as np
import itertools as it
import ubelt as ub
import torch
from torch.utils import data as torch_data


class ToyData1d(torch_data.Dataset):
    """
    Spiral xy-data points

    Args:
        n (int, default=2000): dataset size
        rng (RandomCoercable, default=None): seed or random state

    Note:
        this is 1d in the sense that each data point has shape with len(1),
        even though they can be interpreted as 2d vector points.

    CommandLine:
        python -m netharn.data.toydata ToyData1d --show

    Example:
        >>> dset = ToyData1d()
        >>> data, labels = next(iter(dset.make_loader(batch_size=2000)))
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> plt = kwplot.autoplt()
        >>> kwplot.figure(fnum=1, doclf=True)
        >>> cls1 = data[labels == 0]
        >>> cls2 = data[labels == 1]
        >>> a, b = cls1.T.numpy()
        >>> c, d = cls2.T.numpy()
        >>> plt.plot(a, b, 'rx')
        >>> plt.plot(c, d, 'bx')
        >>> kwplot.show_if_requested()
    """

    def __init__(self, n=2000, rng=None):
        import kwarray
        rng = kwarray.ensure_rng(rng)

        # spiral equation in parameteric form:
        # x(t) = r(t) * cos(t)
        # y(t) = r(t) * sin(t)

        # class 1
        n1 = n // 2
        theta1 = rng.rand(n1) * 10
        x1 = theta1 * np.cos(theta1)
        y1 = theta1 * np.sin(theta1)

        n2 = n - n1
        theta2 = rng.rand(n2) * 10
        x2 = -theta2 * np.cos(theta2)
        y2 = -theta2 * np.sin(theta2)

        data = []
        labels = []

        data.extend(list(zip(x1, y1)))
        labels.extend([0] * n1)

        data.extend(list(zip(x2, y2)))
        labels.extend([1] * n2)

        data = np.array(data)
        labels = np.array(labels)

        self.data = data
        self.labels = labels

        suffix = ub.hash_data([
            rng], base='abc', hasher='sha1')[0:16]
        self.input_id = 'TD1D_{}_'.format(n) + suffix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.FloatTensor(self.data[index])
        label = int(self.labels[index])
        return data, label

    def make_loader(self, *args, **kwargs):
        loader = torch_data.DataLoader(self, *args, **kwargs)
        return loader


class ToyData2d(torch_data.Dataset):
    """
    Simple black-on-white and white-on-black images.

    Args:
        n (int, default=100): dataset size
        size (int, default=4): width / height
        border (int, default=1): border mode
        rng (RandomCoercable, default=None): seed or random state

    CommandLine:
        python -m netharn.data.toydata ToyData2d --show

    Example:
        >>> self = ToyData2d()
        >>> data1, label1 = self[0]
        >>> data2, label2 = self[-1]
        >>> # xdoctest: +REQUIRES(--show)
        >>> import kwplot
        >>> plt = kwplot.autoplt()
        >>> kwplot.figure(fnum=1, doclf=True)
        >>> kwplot.imshow(data1.numpy().squeeze(), pnum=(1, 2, 1))
        >>> kwplot.imshow(data2.numpy().squeeze(), pnum=(1, 2, 2))
        >>> kwplot.show_if_requested()
    """
    def __init__(self, size=4, border=1, n=100, rng=None):
        import kwarray
        rng = kwarray.ensure_rng(rng)

        h = w = size

        whiteish = 1 - (np.abs(rng.randn(n, 1, h, w) / 4) % 1)
        blackish = (np.abs(rng.randn(n, 1, h, w) / 4) % 1)

        fw = border
        slices = [slice(None, fw), slice(-fw, None)]

        # class 0 is white block inside a black frame
        data1 = whiteish.copy()
        for sl1, sl2 in it.product(slices, slices):
            data1[..., sl1, :] = blackish[..., sl1, :]
            data1[..., :, sl2] = blackish[..., :, sl2]

        # class 1 is black block inside a white frame
        data2 = blackish.copy()
        for sl1, sl2 in it.product(slices, slices):
            data2[..., sl1, :] = whiteish[..., sl1, :]
            data2[..., :, sl2] = whiteish[..., :, sl2]

        self.data = np.concatenate([data1, data2], axis=0)
        self.labels = np.array(([0] * n) + ([1] * n))

        suffix = ub.hash_data([
            size, border, n, rng
        ], base='abc', hasher='sha1')[0:16]
        self.input_id = 'TD2D_{}_'.format(n) + suffix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.FloatTensor(self.data[index])
        label = int(self.labels[index])
        return data, label

    def make_loader(self, *args, **kwargs):
        loader = torch_data.DataLoader(self, *args, **kwargs)
        return loader


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/netharn/netharn/data/toydata.py all
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
