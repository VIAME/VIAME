# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import argparse
from ast import literal_eval
import os

import torch
from torch import nn

from ...models import (
    AppearanceLSTM, MotionLSTM,
    InteractionLSTM, BBoxLSTM,
    RnnType,
)

from ..rnn_training import train_model
from ..g_config import get_config

from ..rnn_dataset import RNNDataLoader
from ..utilities import setupLogger, logging, exp_lr_scheduler

if __name__ != '__main__':
    raise ImportError

g_config = get_config()

parser = argparse.ArgumentParser(description='Train LSTM model')
parser.add_argument('--model-snapshot-dir', type=str, dest='model_snapshot_dir',
                    help='path to save the trained model', default='../snapshot/out_motion_F_snapshot/')
parser.add_argument('--load-path', dest='load_path', type=str,
                    help='path to pretrained model', default='')
parser.add_argument('--data-root', default='../script/vids',
                    help='Path to root of processed training data')
parser.add_argument('--train-file', default='../script/out_F_train_set.p',
                    help='the file with train data')
parser.add_argument('--test-file', default='../script/out_F_test_set.p',
                    help='the file with test data')
parser.add_argument('--RNN-Type', type=str, dest='rnn_type_str',
                    help='Rnn Type (A: Appearance; I: Interaction; M: Motion; B: Bounding Box)', default='M')
parser.add_argument('--model-params', type=literal_eval,
                    help='Python dict literal with parameters for the model constructor')
parser.add_argument('--padding',
                    help='the padding method to use for short sequences')

args = parser.parse_args()

kwargs = {'num_workers': 8, 'pin_memory': True}

device = torch.device("cuda")
if args.rnn_type_str == 'A':
    rnn_type = RnnType.Appearance
    make_model = AppearanceLSTM
elif args.rnn_type_str == 'I':
    rnn_type = RnnType.Interaction
    make_model = InteractionLSTM
elif args.rnn_type_str == 'M':
    rnn_type = RnnType.Motion
    make_model = MotionLSTM
elif args.rnn_type_str == 'B':
    rnn_type = RnnType.BBox
    make_model = BBoxLSTM
else:
    raise ValueError('--RNN-Type has to be one of A, I, M, B!')
model_params = {'normalized': True, **(args.model_params or {})}
model = make_model(**model_params).to(device)

train_loader = torch.utils.data.DataLoader(
    RNNDataLoader(args.data_root, args.train_file, rnn_type, args.padding),
    batch_size=g_config.train_BatchSize, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    RNNDataLoader(args.data_root, args.test_file, rnn_type, args.padding),
    batch_size=g_config.vali_BatchSize, shuffle=False, **kwargs)

model_snapshot_dir = args.model_snapshot_dir
if not os.path.exists(model_snapshot_dir):
    os.makedirs(model_snapshot_dir)

setupLogger(os.path.join(model_snapshot_dir, 'log.txt'))
g_config.model_dir = model_snapshot_dir

criterion = nn.CrossEntropyLoss()

# load model snapshot
load_path = args.load_path
epoch = 0

if load_path:
    snapshot = torch.load(load_path)
    model.load_state_dict(snapshot['state_dict'])
    epoch = snapshot['epoch'] + 1
    logging('Model loaded from {}'.format(load_path))


train_model(model, criterion, train_loader, test_loader, g_config, exp_lr_scheduler, epoch)
