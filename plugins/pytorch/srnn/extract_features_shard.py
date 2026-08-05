# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""Compute appearance features for one shard of the detections.

The undivided form of this stage lives in extract_siamese_features and is what
still runs when there is only one device to run it on.  Above that, the parent
starts one of these per device, each pinned to its own card through the
CUDA_VISIBLE_DEVICES it is given and each writing to a store of its own, and
folds the shards into the crop store afterwards.  Separate stores are the point:
the crop store took hours to build, and the way to be sure it survives is for
nothing but the single-threaded merge to write to it.
"""

import argparse
import itertools
import os

import torch
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm
tqdm.monitor_interval = 0

from .models import Siamese

from .siamese_dataset import SiameseEXFDataLoader
from .storage import DataStorage
from .utilities import load_track_feature_file


# As in extract_siamese_features.  Shards are cut on multiples of it, so it has
# to agree with that stage for the two to produce the same descriptors.
BATCH_SIZE = 48


def shard_slice(total, shard_index, shard_count, batch_size=BATCH_SIZE):
    """The half-open range of detections belonging to one shard.

    The cut falls on a batch boundary, so every batch a shard forms holds
    exactly the crops the undivided run would have put in it.  Correctness does
    not need that -- the model is in eval mode, where batch normalization uses
    its running statistics, so a crop's descriptor does not depend on what it
    shares a batch with -- but cuDNN chooses its kernels by shape, and a
    trailing batch of a different size can pick a different one and answer in a
    different last bit.  Aligning the cut keeps the parallel result identical to
    the serial one rather than merely equivalent to it.
    """
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be at least zero and less than"
                         " shard_count")

    n_batches = -(-total // batch_size)
    first = n_batches * shard_index // shard_count
    last = n_batches * (shard_index + 1) // shard_count
    return min(first * batch_size, total), min(last * batch_size, total)


def load_model(model_path):
    """Load the trained Siamese model onto the visible device, in eval mode"""
    model = Siamese()
    model = torch.nn.DataParallel(model).cuda()

    snapshot = torch.load(model_path)
    model.load_state_dict(snapshot['state_dict'])
    print('Model loaded from {}'.format(model_path))
    model.train(False)
    return model


def extract_shard(
        model, read_storage, write_storage, track_feature_file,
        shard_index, shard_count, num_workers=2,
):
    """Extract this shard's appearance features, returning how many.

    Crops are read from read_storage and descriptors written to write_storage,
    which are different stores when this runs under the sharded parent.
    """
    transform = transforms.Compose([
        transforms.Resize(224),  # was transforms.Scale, removed in torchvision 0.15
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    kwargs = {'num_workers': num_workers, 'pin_memory': True}

    dids = load_track_feature_file(track_feature_file)[1]
    total = len(dids)
    start, stop = shard_slice(total, shard_index, shard_count)
    dids = dids[start:stop]
    print('  detections [{}, {}) of {}'.format(start, stop, total))

    # More shards than batches, which only happens on a tiny dataset
    if not dids:
        return 0

    image_blobs = [read_storage.blob(did, 'img') for did in dids]

    data_loader_class = SiameseEXFDataLoader(image_blobs, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        data_loader_class, batch_size=BATCH_SIZE, shuffle=False, **kwargs)

    pbar = tqdm(total=len(test_loader))

    idids = iter(dids)
    # The loader hands the crop store to its worker processes, and a sqlite
    # connection does not survive being inherited by one; closing here lets
    # each reopen its own.
    read_storage.close()

    for img in test_loader:
        input = Variable(img.cuda())

        with torch.no_grad():
            output = model(input)

        np_output = output.data.cpu().numpy()

        f_blobs = [write_storage.blob(did, 'app')
                   for did in itertools.islice(idids, len(np_output))]
        assert np_output.shape[0] == len(f_blobs)

        for npo, fb in zip(np_output, f_blobs):
            fb.write(npo.tobytes())

        pbar.update(1)

    return len(dids)


def main(model_path, data_root, train_feature_file, test_feature_file,
         shard_index, shard_count, shard_root, num_workers=2):
    os.makedirs(shard_root, exist_ok=True)

    model = load_model(model_path)

    with DataStorage(data_root) as read_storage, \
            DataStorage(shard_root) as write_storage:
        # Only the blobs table is used, but the shard is a DataStorage like any
        # other and the merge attaches it as one
        write_storage.create()

        total = 0

        for phase, feature_file in (('train', train_feature_file),
                                    ('test', test_feature_file)):
            print("Extracting {} features for shard {} of {}...".format(
                phase, shard_index, shard_count))
            total += extract_shard(
                model, read_storage, write_storage, feature_file,
                shard_index, shard_count, num_workers,
            )

        print("Shard {} wrote {} appearance feature(s)".format(
            shard_index, total))


def create_parser():
    p = argparse.ArgumentParser(
        description="Compute appearance features for one shard of the"
        " detections using a trained Siamese model")
    p.add_argument('--model-path', required=True,
                   help='Path to a trained Siamese model')
    p.add_argument('--data-root', required=True,
                   help='The path where all feature data is stored')
    p.add_argument('--train-feature-file', required=True,
                   help='Path to a track feature file for training data')
    p.add_argument('--test-feature-file', required=True,
                   help='Path to a track feature file for testing data')
    p.add_argument('--shard-index', type=int, required=True,
                   help='Which shard of the detections to extract, counting'
                   ' from zero')
    p.add_argument('--shard-count', type=int, required=True,
                   help='How many shards the detections are divided into')
    p.add_argument('--shard-root', required=True,
                   help='Directory for this shard\'s own store, which the'
                   ' caller merges into the main one afterwards')
    p.add_argument('--num-workers', dest='num_workers', type=int, default=2,
                   help='Data loading worker processes. On Python 3.14 these '
                   'come from a forkserver rather than a fork, so each is a '
                   'fresh interpreter with its own copy rather than shared '
                   'pages, and the old default of eight was sized for the '
                   'latter.')
    return p


if __name__ == '__main__':
    main(**vars(create_parser().parse_args()))
