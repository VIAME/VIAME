# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import io
import pickle

from PIL import Image
import torch.utils.data as data

from .storage import DataStorage
from .utilities import DividedDataset

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def make_dataset(detection_file):
    with open(detection_file, 'rb') as f:
        return DividedDataset(pickle.load(f), 1.0, -1.0)


def pil_loader(blob):
    with Image.open(io.BytesIO(blob.read())) as img:
        img = img.resize((224, 224), Image.BILINEAR)
        return img.convert('RGB')


class SiameseDataLoader(data.Dataset):
    """A Siamese data loader where the images are arranged in this way in a file: ::

        img1_path img2_path img3_path

        img1 and img2's label=1: same class
        img1 and img3's label=-1: diff class

    Args:
        train_or_test_file (string): file name, which contains all triplets for training / testing

        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``

     Attributes:
        TODO: add the description about the
    """

    def __init__(self, data_root, train_or_test_file, transform=None):
        self.data_storage = DataStorage(data_root)
        data = make_dataset(train_or_test_file)
        if not data:
            raise RuntimeError("Found 0 images in data")

        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image1, image2, target) where target is 1: same class; -1: diff class.
        """
        (img1_did, img2_did), label = self.data[index]

        img1 = pil_loader(self.data_storage.blob(img1_did, 'img'))
        img2 = pil_loader(self.data_storage.blob(img2_did, 'img'))

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label

    def __len__(self):
        return len(self.data)


class SiameseEXFDataLoader(data.Dataset):
    def __init__(self, image_blobs, transform=None):
        self.img_blobs = image_blobs
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image1, image2, target) where target is 1: same class; -1: diff class.
        """
        img = pil_loader(self.img_blobs[index])
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_blobs)
