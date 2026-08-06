# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import argparse
import os

import torch
from torchvision import transforms

from .models import Siamese

from .siamese_training import train_model
from .g_config import get_config
from .siamese_dataset import SiameseDataLoader
from .utilities import resume_epoch, setupLogger, logging, exp_lr_scheduler
from .ContrastiveLoss import ContrastiveLoss

def main():
    parser = argparse.ArgumentParser(description='Siamese model')
    parser.add_argument('--model-dir', type=str, dest='model_dir',
                        help='path to where models are saved', default='../snapshot/temp')
    parser.add_argument('--load-path', dest='load_path', type=str,
                        help='path to pretrained model', default='')
    parser.add_argument('--data-root', help='Path to root of processed training data')
    parser.add_argument('--train-file', type=str, dest='train_file',
                        help='the file with train tripulet', default='../script/non_itar_siamese_train_set.p')
    parser.add_argument('--test-file', type=str, dest='test_file',
                        help='the file with test tripulet', default='../script/non_itar_siamese_test_set.p')
    parser.add_argument('--num-workers', dest='num_workers', type=int, default=6,
                        help='Data loading worker processes')

    args = parser.parse_args()

    # persistent_workers keeps the workers alive between epochs. Without it
    # every epoch boundary tears the whole set down and spawns a fresh one,
    # and on Python 3.14 each of those is a new interpreter rather than a
    # fork, so the two sets briefly coexist. Two runs of this stage were
    # killed with SIGKILL at exactly that point, one of them having just
    # finished epoch 0 and the other epoch 1.
    # pin_memory is off with persistent workers. Pinned buffers are page
    # locked host memory held by the loader's pinning thread, and a loader
    # kept alive between epochs keeps allocating them: this stage went from
    # 5.4 GB after epoch 0 to 57.7 GB after epoch 1 and was killed early in
    # epoch 2. The copy to GPU is a little slower without it, which is
    # nothing beside not finishing.
    kwargs = {'num_workers': args.num_workers, 'pin_memory': False}

    if args.num_workers > 0:
        kwargs['persistent_workers'] = True
    g_config = get_config()

    trans = transform=transforms.Compose([
                            transforms.Resize(224),  # was transforms.Scale, removed in torchvision 0.15
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                       ])
    train_loader = torch.utils.data.DataLoader(
        SiameseDataLoader(args.data_root, args.train_file, transform=trans),
        batch_size=g_config.train_BatchSize, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        SiameseDataLoader(args.data_root, args.test_file, transform=trans),
        batch_size=g_config.vali_BatchSize, shuffle=False, **kwargs)

    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    setupLogger(os.path.join(model_dir, 'log.txt'))
    g_config.model_dir = model_dir

    criterion = ContrastiveLoss(margin=g_config.margin)

    model = Siamese(pretrained=True)

    # One device by default. Spread over three cards with DataParallel this
    # stage leaked host memory at about 1.8 MB per batch -- linear across
    # epoch boundaries, to 88 GB by the second epoch -- while the identical
    # stage on one card measured a flat 1.7 GB over ten thousand batches of
    # the same data, resumed from the same snapshot. The leak was never
    # reproduced off the three-card path; every other layer (dataset, loader,
    # loss, metrics, resume) was ruled out individually. DataParallel also
    # re-replicates the model on every forward and bought less than 2x from
    # its three cards on this loader-bound stage, so the trade is a modest
    # slowdown for a stage that reliably finishes. VIAME_SRNN_SIAMESE_GPUS
    # restores the old behaviour for hunting the leak properly.
    try:
        siamese_gpus = max(int(os.environ.get('VIAME_SRNN_SIAMESE_GPUS', 1)), 1)
    except ValueError:
        siamese_gpus = 1

    devices = list(range(min(torch.cuda.device_count(), siamese_gpus)))
    model = torch.nn.DataParallel(model, device_ids=devices).to(torch.device("cuda"))

    # load model snapshot
    load_path = args.load_path
    epoch = 0

    if load_path:
        snapshot = torch.load(load_path)
        model.load_state_dict(snapshot['state_dict'])
        epoch = resume_epoch(snapshot, load_path)
        logging('Model loaded from {}'.format(load_path))

    train_model(model, criterion, train_loader, test_loader, g_config, exp_lr_scheduler, epoch)


# Everything above has to stay import-safe: from Python 3.14 on, Linux spawns
# data loading workers through a forkserver, and each one re-imports this file
# as __mp_main__ before it starts.
if __name__ == '__main__':
    main()
