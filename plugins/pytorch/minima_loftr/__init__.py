# This file is part of VIAME, and is distributed under an OSI-approved
# BSD 3-Clause License. See either the root top-level LICENSE file or
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.
#
# Vendored inference-only subset of MINIMA-LoFTR (Apache-2.0):
#   https://github.com/LSXI7/LoFTR_minima  (src/loftr @ 9ba0871)
#   https://github.com/LSXI7/MINIMA        (CVPR 2025, arXiv 2412.19412)
# which is itself derived from LoFTR (Apache-2.0, https://github.com/zju3dv/LoFTR).
#
# Local modifications:
#   - training-only modules (supervision, geometry) are not vendored
#   - utils/fine_matching.py: kornia dependency replaced with local torch ops
#   - this file: yacs config replaced with the equivalent plain dict

from .loftr import LoFTR

# Inference config equivalent to LoFTR_minima's lower_config(cvpr_ds_config),
# with the post-outdoor_ds.ckpt coordinate fix that MINIMA weights require
# (MINIMA/load_model.py sets temp_bug_fix=True for minima_loftr.ckpt).
default_cfg = {
    'backbone_type': 'ResNetFPN',
    'resolution': (8, 2),
    'fine_window_size': 5,
    'fine_concat_coarse_feat': True,
    'resnetfpn': {
        'initial_dim': 128,
        'block_dims': [128, 196, 256],
    },
    'coarse': {
        'd_model': 256,
        'd_ffn': 256,
        'nhead': 8,
        'layer_names': ['self', 'cross'] * 4,
        'attention': 'linear',
        'temp_bug_fix': True,
    },
    'match_coarse': {
        'thr': 0.2,
        'border_rm': 2,
        'match_type': 'dual_softmax',
        'dsmax_temperature': 0.1,
        'skh_iters': 3,
        'skh_init_bin_score': 1.0,
        'skh_prefilter': True,
        'train_coarse_percent': 0.4,
        'train_pad_num_gt_min': 200,
    },
    'fine': {
        'd_model': 128,
        'd_ffn': 128,
        'nhead': 8,
        'layer_names': ['self', 'cross'] * 1,
        'attention': 'linear',
    },
}
