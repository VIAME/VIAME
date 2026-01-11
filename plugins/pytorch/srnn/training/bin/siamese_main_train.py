# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import argparse
import os

import torch
from torchvision import transforms

from ...models import Siamese

from ..siamese_training import train_model
from ..g_config import get_config
from ..siamese_dataset import SiameseDataLoader
from ..utilities import setupLogger, logging, exp_lr_scheduler
from ..ContrastiveLoss import ContrastiveLoss

if __name__ != '__main__':
    raise ImportError

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


args = parser.parse_args()

kwargs = {'num_workers': 6, 'pin_memory': True}
g_config = get_config()

trans = transform=transforms.Compose([
                        transforms.Scale(224),
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
model = torch.nn.DataParallel(model, device_ids=[x for x in range(torch.cuda.device_count())]).to(torch.device("cuda"))

# load model snapshot
load_path = args.load_path
epoch = 0

if load_path:
    snapshot = torch.load(load_path)
    model.load_state_dict(snapshot['state_dict'])
    epoch = snapshot['epoch'] + 1
    logging('Model loaded from {}'.format(load_path))


train_model(model, criterion, train_loader, test_loader, g_config, exp_lr_scheduler, epoch)
