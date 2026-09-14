"""
Wraps the torchvision mnist dataset
"""
import torchvision


class MNIST(torchvision.datasets.MNIST):
    """
    `MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    This is a loose wrapper around the torchvision object to define category
    names and standardize the attributes between train and test variants.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(MNIST, self).__init__(root, train=train, transform=transform,
                                    target_transform=target_transform,
                                    download=download)
        self.classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six',
                        'seven', 'eight', 'nine']

    @property
    def labels(self):
        if self.train:
            return self.train_labels
        else:
            return self.test_labels

    @property
    def data(self):
        if self.train:
            return self.train_data
        else:
            return self.test_data
