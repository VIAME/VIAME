# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import os
import pickle
from time import localtime, strftime

from PIL import Image
import numpy as np
import torch

from .models import RnnType


def load_track_feature_file(tff):
    """Given a path to a "track feature file" as created by
    .bin.generate_training_files_kw18, return a tuple of:
    - The raw loaded dictionary of tracks
    - All the DetectionIds as one list
    - A list of pairs giving the slice of the big list corresponding
      to each track

    """
    with open(tff, 'rb') as f:
        tracks = pickle.load(f)
    dets = []
    start_indices = [0]
    for track_states in tracks.values():
        dets.extend(d for _, d in track_states)
        start_indices.append(len(dets))
    return tracks, dets, list(zip(start_indices, start_indices[1:]))


def rnn_type_to_feature(rt):
    if rt is RnnType.Appearance:
        return 'app'
    elif rt is RnnType.Motion:
        return 'bc'
    elif rt is RnnType.Interaction:
        return 'grid'
    elif rt is RnnType.BBox:
        return 'bbar'
    raise ValueError


class DividedDataset:
    """View of a sequence of "sequences" combining positive and negative
    examples as a sequence with separate examples and scores

    """
    __slots__ = '_seq', '_pos', '_neg'

    def __init__(self, sequence, positive_score, negative_score):
        """Create a DividedDataset.  Elements of sequence should be tuples x
        such that (*x[:-2], x[-2]) is a positive example and (*x[:-2],
        x[-1]) is a negative one

        """
        self._seq = sequence
        self._pos = positive_score
        self._neg = negative_score

    def __len__(self):
        return len(self._seq) * 2

    def __getitem__(self, x):
        """Get the item at the specified index.  Slices are not supported"""
        seq = self._seq[x // 2]
        if x % 2:
            return seq[:-1], self._pos
        else:
            return seq[:-2] + seq[-1:], self._neg


gLoggerFile = None


def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=2):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr
    if epoch % lr_decay_epoch == 0 and epoch != 0:
        lr *= 0.1

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer, lr


def setupLogger(fpath):
    fileMode = 'w'
    input = None
    while input is None:
        print('Logging file exits, overwrite(o)? append(a)? abort(q)?')
        input = 'o'
        if input == 'o':
            fileMode = 'w'
        elif input == 'a':
            fileMode = 'a'
        elif input == 'q':
            os.exit()
        else:
            break
    global gLoggerFile
    gLoggerFile = open(fpath, fileMode)


def shutdownlogger():
    if gLoggerFile is not None:
        gLoggerFile.close()


def logging(message, mute=False):
    timeStamp = strftime("%Y-%m-%d %H:%M:%S", localtime())
    msgFormatted = '[{}]  {}'.format(timeStamp, message)
    if not mute:
        print(msgFormatted)
    if gLoggerFile is not None:
        gLoggerFile.write(msgFormatted + '\n')
        gLoggerFile.flush()


def modelSize(model):
    params = model.parameters()

    count = 0
    countForEach = []
    for i, a_param in enumerate(params):
        nParam = a_param.numel()
        count = count + nParam
        countForEach.append(nParam)
    return count, torch.LongTensor(countForEach)


def diagnoseGradients(params):
    """
    [[ Diagnose gradients by checking the value range and the ratio of the norms
    ARGS:
      - `params`     : first arg returned by net:parameters()
      - `gradParams` : second arg returned by net:parameters()
    ]]
    """
    pass


def checkpoint(model, epoch=None):
    package = {
        'epoch': epoch if epoch else 'N/A',
        'state_dict': model.state_dict(),
    }
    return package
