# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import os
import sys
import torch
import argparse
import re
import tempfile

from collections import OrderedDict
from mmcv import Config

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

def check_config_compatibility( input_cfg, input_weights, input_template ):

    if not os.path.exists( input_cfg ):
        print( "\nInput model config file: " + input_cfg + " does not exist\n" )
        sys.exit()

    file_lines = []
    perform_upgrade = False

    with open( input_cfg, 'r' ) as in_file:
        for line in in_file:
            file_lines.append( line )

    for i, line in enumerate( file_lines ):
        if i < 5 and "num_stages" in line:
            perform_upgrade = True
            break
        if " use_sigmoid_cls=True)," in line:
            perform_upgrade = True
            break

    if perform_upgrade:
        if len( input_template ) == 0:
            conf_file = "detector_mmdet.py"
            conf_file = os.path.join( "configs", "pipelines", "templates", conf_file )
            input_template = os.path.join( os.environ[ "VIAME_INSTALL" ], conf_file )
        if not os.path.exists( input_template ):
            print( "\nInput template: " + input_template + " does not exist\n" )
            sys.exit()

        # Do v1 to v2 upgrade
        print( "Upgrading weight file to latest version" )
        class_count = 1
        for line in file_lines:
            if "num_classes" in line:
                class_count = int(line.rstrip()[line.find("=")+1:line.find(",")])
                break
        convert_v1_to_v2_weights( input_weights, input_weights, class_count )

        print( "Upgrading model definition to latest version" )
        for line in file_lines:
            if "img_scale=" in line:
                start_pos = line.find( "img_scale=" ) + 10
                end_pos = line.find( ")," ) + 1
                image_scale = line[ start_pos : end_pos ]
                break

        repl_strs = [ [ "[-CLASS_COUNT_INSERT-]", str( class_count - 1 ) ],
                      [ "[-IMAGE_SCALE_INSERT-]", image_scale ],
                      [ "[-IMAGES_PER_GPU_INSERT-]", "1" ],
                      [ "[-WORKERS_PER_GPU_INSERT-]", "1" ] ]

        fin = open( input_template )
        all_lines = []
        for s in list( fin ):
            all_lines.append( s )

        fout = open( input_cfg, 'w' )
        for repl in repl_strs:
            for i, s in enumerate( all_lines ):
                all_lines[i] = s.replace( repl[0], repl[1] )
        for s in all_lines:
            fout.write( s )

        fout.close()
        fin.close()
