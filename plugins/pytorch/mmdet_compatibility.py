# ckwg +29
# Copyright 2019 by Kitware, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided
# that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this list of conditions and
#  the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
#  the following disclaimer in the documentation and/or other materials provided with the distribution.
#
#  * Neither name of Kitware, Inc. nor the names of any contributors may be used to endorse or promote
#  products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.

import os
import sys
import torch
import argparse
import re
import tempfile

from collections import OrderedDict
from mmcv import Config

def adjust_for_alt_loss_def_v1( parsed_file ):

    adj_file = []
    skip_counter = 0
    scale_counter = 0

    for line in parsed_file:

        if skip_counter > 0:
            skip_counter = skip_counter - 1
            continue
    
        if "use_sigmoid_cls=True)," in line:
            adj_file.append( "        loss_cls=dict(\n" )
            adj_file.append( "            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),\n" )
            adj_file.append( "        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),\n" )
            continue

        if "reg_class_agnostic=True)," in line:
            adj_file.append( "            reg_class_agnostic=True,\n" )
            adj_file.append( "            loss_cls=dict(\n" )
            adj_file.append( "                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),\n" )
            adj_file.append( "            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),\n" )
            continue

        if "smoothl1_beta=1 / 9.0," in line:
            adj_file.append( "         debug=False),\n" )
            adj_file.append( "    rpn_proposal=dict(\n" )
            adj_file.append( "        nms_across_levels=False,\n" )
            adj_file.append( "        nms_pre=2000,\n" )
            adj_file.append( "        nms_post=2000,\n" )
            adj_file.append( "        max_num=2000,\n" )
            adj_file.append( "        nms_thr=0.7,\n" )
            adj_file.append( "        min_bbox_size=0),\n" )
            skip_counter = 1
            continue

        if "mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)" in line:
            adj_file.append( line )
            adj_file.append( "train_pipeline = [\n" )
            adj_file.append( "    dict(type='LoadImageFromFile'),\n" )
            adj_file.append( "    dict(type='LoadAnnotations', with_bbox=True),\n" )
            adj_file.append( "    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),\n" )
            adj_file.append( "    dict(type='RandomFlip', flip_ratio=0.5),\n" )
            adj_file.append( "    dict(type='Normalize', **img_norm_cfg),\n" )
            adj_file.append( "    dict(type='Pad', size_divisor=32),\n" )
            adj_file.append( "    dict(type='DefaultFormatBundle'),\n" )
            adj_file.append( "    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),\n" )
            adj_file.append( "]\n" )
            adj_file.append( "test_pipeline = [\n" )
            adj_file.append( "    dict(type='LoadImageFromFile'),\n" )
            adj_file.append( "    dict(\n" )
            adj_file.append( "        type='MultiScaleFlipAug',\n" )
            adj_file.append( "        img_scale=(1333, 800),\n" )
            adj_file.append( "        flip=False,\n" )
            adj_file.append( "        transforms=[\n" )
            adj_file.append( "            dict(type='Resize', keep_ratio=True),\n" )
            adj_file.append( "            dict(type='RandomFlip'),\n" )
            adj_file.append( "            dict(type='Normalize', **img_norm_cfg),\n" )
            adj_file.append( "            dict(type='Pad', size_divisor=32),\n" )
            adj_file.append( "            dict(type='ImageToTensor', keys=['img']),\n" )
            adj_file.append( "            dict(type='Collect', keys=['img']),\n" )
            adj_file.append( "        ])\n" )
            adj_file.append( "]\n" )
            continue

        if "        img_scale=(1333, 800)," in line:
            skip_counter = 6
            scale_counter = scale_counter + 1
            if scale_counter < 3:
                adj_file.append( "        pipeline=test_pipeline),\n" )
            else:
                adj_file.append( "        pipeline=test_pipeline))\n" )
            continue

        adj_file.append( line )

    return adj_file

def is_head(key):
    valid_head_list = [
        'bbox_head', 'mask_head', 'semantic_head', 'grid_head', 'mask_iou_head'
    ]

    return any(key.startswith(h) for h in valid_head_list)

def parse_config(config_strings):
    temp_file = tempfile.NamedTemporaryFile()
    config_path = f'{temp_file.name}.py'
    with open(config_path, 'w') as f:
        f.write(config_strings)

    config = Config.fromfile(config_path)
    is_two_stage = True
    is_ssd = False
    is_retina = False
    reg_cls_agnostic = False
    if 'rpn_head' not in config.model:
        is_two_stage = False
        # check whether it is SSD
        if config.model.bbox_head.type == 'SSDHead':
            is_ssd = True
        elif config.model.bbox_head.type == 'RetinaHead':
            is_retina = True
    elif isinstance(config.model['bbox_head'], list):
        reg_cls_agnostic = True
    elif 'reg_class_agnostic' in config.model.bbox_head:
        reg_cls_agnostic = config.model.bbox_head \
            .reg_class_agnostic
    temp_file.close()
    return is_two_stage, is_ssd, is_retina, reg_cls_agnostic

def reorder_cls_channel(val, num_classes=81):
    # bias
    if val.dim() == 1:
        new_val = torch.cat((val[1:], val[:1]), dim=0)
    # weight
    else:
        out_channels, in_channels = val.shape[:2]
        # conv_cls for softmax output
        if out_channels != num_classes and out_channels % num_classes == 0:
            new_val = val.reshape(-1, num_classes, in_channels, *val.shape[2:])
            new_val = torch.cat((new_val[:, 1:], new_val[:, :1]), dim=1)
            new_val = new_val.reshape(val.size())
        # fc_cls
        elif out_channels == num_classes:
            new_val = torch.cat((val[1:], val[:1]), dim=0)
        # agnostic | retina_cls | rpn_cls
        else:
            new_val = val

    return new_val


def truncate_cls_channel(val, num_classes=81):

    # bias
    if val.dim() == 1:
        if val.size(0) % num_classes == 0:
            new_val = val[:num_classes - 1]
        else:
            new_val = val
    # weight
    else:
        out_channels, in_channels = val.shape[:2]
        # conv_logits
        if out_channels % num_classes == 0:
            new_val = val.reshape(num_classes, in_channels, *val.shape[2:])[1:]
            new_val = new_val.reshape(-1, *val.shape[1:])
        # agnostic
        else:
            new_val = val

    return new_val

def truncate_reg_channel(val, num_classes=81):
    # bias
    if val.dim() == 1:
        # fc_reg|rpn_reg
        if val.size(0) % num_classes == 0:
            new_val = val.reshape(num_classes, -1)[:num_classes - 1]
            new_val = new_val.reshape(-1)
        # agnostic
        else:
            new_val = val
    # weight
    else:
        out_channels, in_channels = val.shape[:2]
        # fc_reg|rpn_reg
        if out_channels % num_classes == 0:
            new_val = val.reshape(num_classes, -1, in_channels,
                                  *val.shape[2:])[1:]
            new_val = new_val.reshape(-1, *val.shape[1:])
        # agnostic
        else:
            new_val = val

    return new_val

def convert_v1_to_v2_weights(in_file, out_file, num_classes):
    """Convert keys in checkpoints.

    There can be some breaking changes during the development of mmdetection,
    and this tool is used for upgrading checkpoints trained with old versions
    to the latest one.
    """
    checkpoint = torch.load(in_file)
    in_state_dict = checkpoint.pop('state_dict')
    out_state_dict = OrderedDict()
    meta_info = checkpoint['meta']
    is_two_stage, is_ssd, is_retina, reg_cls_agnostic = parse_config(
        meta_info['config'])
    if meta_info['mmdet_version'] <= '0.5.3' and is_retina:
        upgrade_retina = True
    else:
        upgrade_retina = False

    for key, val in in_state_dict.items():
        new_key = key
        new_val = val
        if is_two_stage and is_head(key):
            new_key = 'roi_head.{}'.format(key)

        # classification
        m = re.search(
            r'(conv_cls|retina_cls|rpn_cls|fc_cls|fcos_cls|'
            r'fovea_cls).(weight|bias)', new_key)
        if m is not None:
            print(f'reorder cls channels of {new_key}')
            new_val = reorder_cls_channel(val, num_classes)

        # regression
        m = re.search(r'(fc_reg|rpn_reg).(weight|bias)', new_key)
        if m is not None and not reg_cls_agnostic:
            print(f'truncate regression channels of {new_key}')
            new_val = truncate_reg_channel(val, num_classes)

        # mask head
        m = re.search(r'(conv_logits).(weight|bias)', new_key)
        if m is not None:
            print(f'truncate mask prediction channels of {new_key}')
            new_val = truncate_cls_channel(val, num_classes)

        m = re.search(r'(cls_convs|reg_convs).\d.(weight|bias)', key)
        # Legacy issues in RetinaNet since V1.x
        # Use ConvModule instead of nn.Conv2d in RetinaNet
        # cls_convs.0.weight -> cls_convs.0.conv.weight
        if m is not None and upgrade_retina:
            param = m.groups()[1]
            new_key = key.replace(param, f'conv.{param}')
            out_state_dict[new_key] = val
            print(f'rename the name of {key} to {new_key}')
            continue

        m = re.search(r'(cls_convs).\d.(weight|bias)', key)
        if m is not None and is_ssd:
            print(f'reorder cls channels of {new_key}')
            new_val = reorder_cls_channel(val, num_classes)

        out_state_dict[new_key] = new_val
    checkpoint['state_dict'] = out_state_dict
    torch.save(checkpoint, out_file)

def check_config_compatibility( input_cfg, input_weights ):

    if not os.path.exists( input_cfg ):
        print( "\nInput model config file: " + input_cfg + " does not exist\n" )
        sys.exit()

    file_lines = []
    auto_edit_performed = False

    with open( input_cfg, 'r' ) as in_file:
        for line in in_file:
            file_lines.append( line )

    for line in file_lines:
        if " use_sigmoid_cls=True)," in line:
            # Do v0.5x to v1 upgrade
            file_lines = adjust_for_alt_loss_def_v1( file_lines )
            auto_edit_performed = True
            break

    if "num_stages" in file_lines[2] or "num_stages" in file_lines[3]:
        # Do v1 to v2 upgrade
        class_count = 1
        for line in file_lines:
            if "num_classes" in line:
                class_count = int(line.rstrip()[line.find("=")+1:line.find(",")])-1
                break
        convert_v1_to_v2_weights( input_weights, input_weights, class_count )

    if auto_edit_performed:
        with open( input_cfg, 'w' ) as out_file:
            for line in file_lines:
                out_file.write( line )
