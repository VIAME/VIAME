# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import argparse
from ast import literal_eval
import os

import torch
from torch import nn

from .models import (
    RnnType, TargetLSTM,
)

from .rnn_training import train_model
from .g_config import get_config
from .target_rnn_dataset import TargetRNNDataLoader
from .utilities import setupLogger, logging, exp_lr_scheduler

if __name__ != '__main__':
    raise ImportError

g_config = get_config()

parser = argparse.ArgumentParser(description='Train SRNN model')
parser.add_argument('--model-dir', type=str, dest='model_dir',
                    help='path to where models are saved', default='snapshot/temp/')
parser.add_argument('--app-load-path', dest='app_load_path', type=str,
                    help='path to appearance pretrained model',
                    default='../snapshot/non_itar_app_F_snapshot/App_LSTM_epoch_51.pt')
parser.add_argument('--motion-load-path', dest='motion_load_path', type=str,
                    help='path to motion pretrained model',
                    default='../snapshot/non_itar_motion_F_snapshot/App_LSTM_epoch_51.pt')
parser.add_argument('--interaction-load-path', dest='interaction_load_path', type=str,
                    help='path to interaction pretrained model',
                    default='../snapshot/non_itar_interaction_F_snapshot/App_LSTM_epoch_51.pt')
parser.add_argument('--bbar-load-path', dest='bbar_load_path', type=str,
                    help='path to bbar pretrained model',
                    default='')
parser.add_argument('--load-path', dest='load_path', type=str,
                    help='path to pretrained model', default='')
parser.add_argument('--data-root', default='../script/vids',
                    help='Path to root of processed training data')
parser.add_argument('--train-file', default='../script/out_F_train_set.p',
                    help='the file with train data')
parser.add_argument('--test-file', default='../script/out_F_test_set.p',
                    help='the file with test data')
parser.add_argument('--RNN-component', type=str, dest='rnn_list_str',
                    help='Define the rnn components for the final SRNN (can only be the combination of A I M B)',
                    default='AIM')
parser.add_argument('--model-params', type=literal_eval,
                    help='Python dict literal with parameters for the model constructor')
parser.add_argument('--padding',
                    help='the padding method to use for short sequences')


args = parser.parse_args()

rnn_list = []
for r in args.rnn_list_str:
    if r == 'A':
        rnn_list.append(RnnType.Appearance)
    elif r == 'I':
        rnn_list.append(RnnType.Interaction)
    elif r == 'M':
        rnn_list.append(RnnType.Motion)
    elif r == 'B':
        rnn_list.append(RnnType.BBox)
    else:
        raise ValueError('--RNN-component has to be the combination of A I M B!')

print('rnn_list {}'.format(rnn_list))

kwargs = {'num_workers': 8, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(
    TargetRNNDataLoader(args.data_root, args.train_file, rnn_list, args.padding),
    batch_size=g_config.train_BatchSize, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    TargetRNNDataLoader(args.data_root, args.test_file, rnn_list, args.padding),
    batch_size=g_config.vali_BatchSize, shuffle=False, **kwargs)

model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

setupLogger(os.path.join(model_dir, 'log.txt'))
g_config.model_dir = model_dir

criterion = nn.CrossEntropyLoss()

model_params = {'normalized': True, **(args.model_params or {})}
model = TargetLSTM(app_model=args.app_load_path, motion_model=args.motion_load_path,
                   interaction_model=args.interaction_load_path,
                   bbox_model=args.bbar_load_path,
                   model_list=tuple(rnn_list), **model_params).to(torch.device("cuda"))

# load model snapshot
load_path = args.load_path
epoch = 0

if load_path:
    snapshot = torch.load(load_path)
    model.load_state_dict(snapshot['state_dict'])
    epoch = snapshot['epoch'] + 1
    logging('Model loaded from {}'.format(load_path))


train_model(model, criterion, train_loader, test_loader, g_config, exp_lr_scheduler, epoch, is_target_rnn=True)
