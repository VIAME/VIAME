# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import os

from collections import OrderedDict

viame_dir = os.environ['VIAME_INSTALL']
models_dir = os.path.join( 'configs', 'pipelines', 'models' )
model_file = os.path.join( viame_dir, models_dir, 'mdnet_seed.pth' )

opts = OrderedDict()

opts['use_gpu'] = True
opts['model_seed'] = model_file
opts['savefig_dir'] =''
opts['display']=False
opts['visual_log']=True

opts['crop_size'] = 107
opts['padding'] = 1.2
opts['jitter'] = True
opts['result_path']='./result.npy'
opts['adaptive_align']=True
opts['batch_pos'] = 32
opts['batch_neg'] = 96
opts['batch_neg_cand'] = 1024
opts['batch_test'] = 256

opts['n_samples'] = 256
opts['trans_f'] = 0.6
opts['scale_f'] = 1.05
opts['trans_f_expand'] = 1.4

opts['n_bbreg'] = 1000
opts['overlap_bbreg'] = [0.6, 1]
opts['scale_bbreg'] = [1, 2]

opts['lr_init'] = 0.0001 # original = 0.0001
opts['maxiter_init'] = 50 # original = 30
opts['n_pos_init'] = 500
opts['n_neg_init'] = 5000
opts['overlap_pos_init'] = [0.7, 1]
opts['overlap_neg_init'] = [0, 0.5]

opts['lr_update'] = 0.0003 # original = 0.0002
opts['maxiter_update'] = 15 # original = 15
opts['n_pos_update'] = 50
opts['n_neg_update'] = 200
opts['overlap_pos_update'] = [0.7, 1]
opts['overlap_neg_update'] = [0, 0.3]

opts['success_thr'] = 0. # original = 0
opts['n_frames_short'] = 20
opts['n_frames_long'] = 100
opts['long_interval'] = 10

opts['w_decay'] = 0.0005 # original = 0.0005
opts['momentum'] = 0.9
opts['grad_clip'] = 10 # original = 10
opts['lr_mult'] = {'fc6':10}
opts['ft_layers'] = ['fc']
