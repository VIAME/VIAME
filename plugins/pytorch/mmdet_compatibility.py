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

def adjust_for_alt_loss_def( parsed_file ):

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

def check_config_compatibility( input_cfg ):

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
            file_lines = adjust_for_alt_loss_def( file_lines )
            auto_edit_performed = True
            break

    if auto_edit_performed:
        with open( input_cfg, 'w' ) as out_file:
            for line in file_lines:
                out_file.write( line )
