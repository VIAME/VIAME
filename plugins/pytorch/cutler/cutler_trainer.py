# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

from __future__ import print_function
from __future__ import division

import torch
import pickle
import os
import copy
import signal
import sys
import time
import yaml
import mmcv
import mmdet
import random
import gc
import shutil

import numpy as np
import ubelt as ub

from collections import namedtuple
from PIL import Image
from distutils.util import strtobool
from shutil import copyfile
from pathlib import Path
from mmcv.runner import load_checkpoint
from mmdet.utils import collect_env
from mmdet.apis import train_detector
from mmdet.datasets import build_dataset
from kwiver.vital.algo import DetectedObjectSetOutput, TrainDetector
from kwiver.vital.types import (
    BoundingBoxD, CategoryHierarchy, DetectedObject, DetectedObjectSet,
)
import learn.algorithms.MMDET.register_modules
import torch
import torch.nn.functional as F
from mmdet.models.builder import LOSSES
from functools import partial

_Option = namedtuple('_Option', ['attr', 'config', 'default', 'parse', 'help'])

class ConvNextCascadeRCNNTrainer( TrainDetector ):
    """
    Implementation of TrainDetector class
    """

    _options = [
        _Option('_gpu_count', 'gpu_count', -1, int, ''),
        _Option('_launcher', 'launcher', 'pytorch', str, ''), # "none, pytorch, slurm, or mpi" 
        
        _Option('_cutler_config_file', 'cutler_config_file', '', str, ''),

        _Option('_seed_weights', 'seed_weights', '', str, ''),

        _Option('_output_directory', 'output_directory', 'category_models', str, ''),
        _Option('_pipeline_template', 'pipeline_template', '', str, '')
    ]

    def __init__( self ):
        TrainDetector.__init__( self )

        for opt in self._options:
            setattr(self, opt.attr, opt.default)

        self.image_root = ''

    def register_new_losses(self):
        LOSSES._module_dict.pop('EQLv2', None)

        @LOSSES.register_module()
        class EQLv2(torch.nn.Module):
            def __init__(self,
                    use_sigmoid=True,
                    reduction='mean',
                    class_weight=None,
                    loss_weight=1.0,
                    num_classes=len(self.cats),
                    gamma=12,
                    mu=0.8,
                    alpha=4.0,
                    vis_grad=False):
                super().__init__()
                self.use_sigmoid = True
                self.reduction = reduction
                self.loss_weight = loss_weight
                self.class_weight = class_weight
                self.num_classes = num_classes
                self.group = True

                # cfg for eqlv2
                self.vis_grad = vis_grad
                self.gamma = gamma
                self.mu = mu
                self.alpha = alpha

                # initial variables
                self._pos_grad = None
                self._neg_grad = None
                self.pos_neg = None

                def _func(x, gamma, mu):
                    return 1 / (1 + torch.exp(-gamma * (x - mu)))
                self.map_func = partial(_func, gamma=self.gamma, mu=self.mu)
                # logger = get_root_logger()
                # logger.info(f"build EQL v2, gamma: {gamma}, mu: {mu}, alpha: {alpha}")

            def forward(self,
                        cls_score,
                        label,
                        weight=None,
                        avg_factor=None,
                        reduction_override=None,
                        **kwargs):
                self.n_i, self.n_c = cls_score.size()

                self.gt_classes = label
                self.pred_class_logits = cls_score

                def expand_label(pred, gt_classes):
                    target = pred.new_zeros(self.n_i, self.n_c)
                    target[torch.arange(self.n_i), gt_classes] = 1
                    return target

                target = expand_label(cls_score, label)
                pos_w, neg_w = self.get_weight(cls_score)
                weight = pos_w * target + neg_w * (1 - target)
                cls_loss = F.binary_cross_entropy_with_logits(cls_score, target,
                                                            reduction='none')
                cls_loss = torch.sum(cls_loss * weight) / self.n_i
                self.collect_grad(cls_score.detach(), target.detach(), weight.detach())
                return self.loss_weight * cls_loss

            def get_channel_num(self, num_classes):
                num_channel = num_classes + 1
                return num_channel

            def get_activation(self, cls_score):
                cls_score = torch.sigmoid(cls_score)
                n_i, n_c = cls_score.size()
                bg_score = cls_score[:, -1].view(n_i, 1)
                cls_score[:, :-1] *= (1 - bg_score)
                return cls_score

            def collect_grad(self, cls_score, target, weight):
                prob = torch.sigmoid(cls_score)
                grad = target * (prob - 1) + (1 - target) * prob
                grad = torch.abs(grad)

                # do not collect grad for objectiveness branch [:-1]
                pos_grad = torch.sum(grad * target * weight, dim=0)[:-1]
                neg_grad = torch.sum(grad * (1 - target) * weight, dim=0)[:-1]

                # dist.all_reduce(pos_grad)
                # dist.all_reduce(neg_grad)

                self._pos_grad += pos_grad
                self._neg_grad += neg_grad
                self.pos_neg = self._pos_grad / (self._neg_grad + 1e-10)

            def get_weight(self, cls_score):
                # we do not have information about pos grad and neg grad at beginning
                if self._pos_grad is None:
                    self._pos_grad = cls_score.new_zeros(self.num_classes)
                    self._neg_grad = cls_score.new_zeros(self.num_classes)
                    neg_w = cls_score.new_ones((self.n_i, self.n_c))
                    pos_w = cls_score.new_ones((self.n_i, self.n_c))
                else:
                    # the negative weight for objectiveness is always 1
                    neg_w = torch.cat([self.map_func(self.pos_neg), cls_score.new_ones(1)])
                    pos_w = 1 + self.alpha * (1 - neg_w)
                    neg_w = neg_w.view(1, -1).expand(self.n_i, self.n_c)
                    pos_w = pos_w.view(1, -1).expand(self.n_i, self.n_c)
                return pos_w, neg_w
            
            
    def get_configuration( self ):
        # Inherit from the base class
        cfg = super( TrainDetector, self ).get_configuration()

        for opt in self._options:
            cfg.set_value(opt.config, str(getattr(self, opt.attr)))
        return cfg

    def set_configuration( self, cfg_in ):
        cfg = self.get_configuration()
        cfg.merge_config( cfg_in )

        for opt in self._options:
            setattr(self, opt.attr, opt.parse(cfg.get_value(opt.config)))

        config_file = yaml.safe_load(Path(self._cutler_config_file).read_text())
        self.config = config_file['params']
        self.config_dir = os.path.dirname(self._cutler_config_file)
        print('config:', self.config)
        
        self.ckpt = 0 # TODO: not sure about this
        self.stage = 'base' # TODO: also not sure about this 

        device = self.config['device']
        if ub.iterable(device):
            self.device = device
        else:
            if device == -1:
                self.device = list(range(torch.cuda.device_count()))
            else:
                self.device = [device]
        if len(self.device) > torch.cuda.device_count():
            self.device = self.device[:torch.cuda.device_count()]

        if not self._launcher or self._launcher == 'none':
            self._distributed = False
        else:
            self._distributed = True
            from mmcv.runner import init_dist
            if not "RANK" in os.environ or not "WORLD_SIZE" in os.environ:
                os.environ[ "RANK" ] = "0"
                os.environ[ "WORLD_SIZE" ] = "1"
                os.environ[ "MASTER_ADDR" ] = "localhost"
                os.environ[ "MASTER_PORT" ] = "12345"
            init_dist( self._launcher )

        if self._seed_weights:
            self.original_chkpt_file = self._seed_weights
        elif "checkpoint_override" in self.config and self.config["checkpoint_override"]:
            self.original_chkpt_file = self.config["checkpoint_override"]
        else:
            self.original_chkpt_file = self.config["model_checkpoint_file"]

        if not self.config["work_dir"]:
            self.config["work_dir"] = "."

        if config_file["name"] == "CutLER/pretrain_algo.py":
            if os.path.exists(os.path.join(str(self.config["work_dir"]), "pretrain.pth")):
                self.original_chkpt_file = os.path.join(self.config["work_dir"], "pretrain.pth")

        print(f"Found CutLER weights at {self.original_chkpt_file}")


    def set_mmdet_config(self):
        #print('set_mmdet_config\n')
        self.register_new_losses()
        self.mmdet_config_file = os.path.join(self.config_dir,
          os.path.basename(self.config['mmdet_model_config_file']))
        mmdet_config = mmcv.Config.fromfile(self.mmdet_config_file)
        mmdet_config.dataset_type = 'CocoDataset'
        
        mmdet_config.data_root = None
        mmdet_config.data.ann_file = str(os.path.join(self.config["work_dir"], 'train_data_coco.json'))
        
        mmdet_config.classes = tuple(self.cats)

        # print(type(mmdet_config)) # <class 'mmcv.utils.config.Config'>

        if mmdet_config.data.train.type == "RepeatDataset":
            print('using RepeatDataset')
            if self.config["use_class_balanced"] and mmdet_config.data.train.dataset.type != 'ClassBalancedDataset':
                mmdet_config.data.train.dataset.type = 'CocoDataset'
                mmdet_config.data.train.dataset.data_root = None
                mmdet_config.data.train.dataset.ann_file = str(os.path.join(self.config["work_dir"], 'train_data_coco.json'))
                mmdet_config.data.train.dataset.img_prefix = self.image_root
                mmdet_config.data.train.dataset.classes = mmdet_config.classes

                data = copy.deepcopy(mmdet_config.data.train.dataset)
                mmdet_config.data.train.dataset = dict(
                    type='ClassBalancedDataset',
                    oversample_thr=self.config["oversample_thr"],
                    dataset=data)
            elif self.config["use_class_balanced"]:
                mmdet_config.data.train.dataset.dataset.type = 'CocoDataset'
                mmdet_config.data.train.dataset.dataset.data_root = None
                mmdet_config.data.train.dataset.dataset.ann_file = str(os.path.join(self.config["work_dir"], 'train_data_coco.json'))
                mmdet_config.data.train.dataset.dataset.img_prefix = self.image_root
                mmdet_config.data.train.dataset.dataset.classes = mmdet_config.classes
            else:
                mmdet_config.data.train.dataset.type = 'CocoDataset'
                mmdet_config.data.train.dataset.data_root = None
                mmdet_config.data.train.dataset.ann_file = os.path.join(self.config["work_dir"], 'train_data_coco.json')
                mmdet_config.data.train.dataset.img_prefix = self.image_root
                mmdet_config.data.train.dataset.classes = mmdet_config.classes
        elif mmdet_config.data.train.type == 'ClassBalancedDataset' and self.config["use_class_balanced"]:
            print('using ClassBalancedDataset')
            mmdet_config.data.train.oversample_thr = self.config["oversample_thr"]
            mmdet_config.data.train.dataset.type = 'CocoDataset'
            mmdet_config.data.train.dataset.data_root = None
            mmdet_config.data.train.dataset.ann_file = str(os.path.join(self.config["work_dir"], 'train_data_coco.json'))
            mmdet_config.data.train.dataset.img_prefix = self.image_root
            mmdet_config.data.train.dataset.classes = mmdet_config.classes
        else:
            mmdet_config.data.train.type = 'CocoDataset'
            mmdet_config.data.train.data_root = None
            mmdet_config.data.train.ann_file = str(os.path.join(self.config["work_dir"], 'train_data_coco.json'))
            mmdet_config.data.train.img_prefix = self.image_root
            mmdet_config.data.train.classes = mmdet_config.classes # tuple(train_dataset.category_to_category_index.values())

            if self.config["use_class_balanced"]:
                data = copy.deepcopy(mmdet_config.data.train)
                mmdet_config.data.train = dict(
                    type='ClassBalancedDataset',
                    oversample_thr=self.config["oversample_thr"],
                    dataset=data)
                
        mmdet_config.data.val.type = 'CocoDataset'
        mmdet_config.data.val.data_root = None
        mmdet_config.data.val.ann_file = str(os.path.join(self.config["work_dir"], 'train_data_coco.json'))
        mmdet_config.data.val.img_prefix = self.image_root
        mmdet_config.data.val.classes = mmdet_config.classes # tuple(train_dataset.category_to_category_index.values())

        mmdet_config.data.test.type = 'CocoDataset'
        mmdet_config.data.test.data_root = None
        mmdet_config.data.test.ann_file = str(os.path.join(self.config["work_dir"], 'train_data_coco.json'))
        mmdet_config.data.test.img_prefix = self.image_root
        mmdet_config.data.test.classes = mmdet_config.classes

        mmdet_config.log_config.interval = self.config["log_interval"]
        mmdet_config.checkpoint_config.interval = self.config["checkpoint_interval"]
        mmdet_config.data.samples_per_gpu = self.config["batch_size"]  # Batch size
        mmdet_config.gpu_ids = self.device
        mmdet_config.device = 'cuda'
        mmdet_config.work_dir = self.config["work_dir"]

        
        if self.stage == "adapt":
            if "iters_per_ckpt_adapt" in self.config and len(self.config["iters_per_ckpt_adapt"]) > 0 and self.ckpt < len(self.config["iters_per_ckpt_adapt"]):
                num_iter = self.config["iters_per_ckpt_adapt"][self.ckpt]
            else:
                num_iter = self.config["max_iters"]
        else:    
            if "iters_per_ckpt" in self.config and len(self.config["iters_per_ckpt"]) > 0 and self.ckpt < len(self.config["iters_per_ckpt"]):
                num_iter = self.config["iters_per_ckpt"][self.ckpt]
            else:
                num_iter = self.config["max_iters"]

        mmdet_config.lr_config.warmup_iters = 1 if self.config["warmup_iters"] >= num_iter else self.config["warmup_iters"]
        mmdet_config.lr_config.step = [step for step in self.config["lr_steps"] if step < num_iter]
        mmdet_config.runner = {'type': 'IterBasedRunner', 'max_iters': num_iter}
        mmdet_config.optimizer.lr = self.config["lr"]

        # loop over config, and if there are any num_classes, replace it
        # there might be problems with this (i.e. won't work with nested lists)
        # but I think it's fine for mmdet's config structure
        def replace(conf, depth):
            if depth <= 0:
                return
            try:
                for k,v in conf.items():
                    if isinstance(v, dict):
                        replace(v, depth-1)
                    elif isinstance(v, list):
                        for element in v:
                            replace(element, depth-1)
                    else:
                        # print(k,v)
                        if k == 'num_classes':
                            conf[k] = len(self.cats)
                        if k == 'CLASSES':
                            conf[k] = self.toolset['target_dataset'].categories
            except:
                pass

        replace(mmdet_config, 500)
        print(f'mmdet Config:\n{mmdet_config.pretty_text}')

        mmdet_config.dump(str(os.path.join(self.config["work_dir"], 'mmdet_config.py')))
        self.mmdet_config = mmdet_config   


    def check_configuration( self, cfg ):
        return True
        if not cfg.has_value( "config_file" ) or len( cfg.get_value( "config_file") ) == 0:
            print( "A config file must be specified!" )
            return False


    def update_model( self ):
        print('set mmdet config')
        self.set_mmdet_config()

        print('update_model\n')
        # seed = np.random.randint(2**31)
        seed = self.config["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.mmdet_config.seed = seed
        
        meta = dict()
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        print('Environment info:\n' + dash_line + env_info + '\n' +
                    dash_line)
        meta['env_info'] = env_info
        meta['config'] = self.mmdet_config.pretty_text
        meta['seed'] = seed
        
        print("building dataset\n")
        datasets = [build_dataset(self.mmdet_config.data.train)]
        
        print("building detector\n")
        model = mmdet.models.build_detector(
            self.mmdet_config.model,
            train_cfg=self.mmdet_config.get('train_cfg'),
            test_cfg=self.mmdet_config.get('test_cfg')
        )
        
        checkpoint_file = os.path.join(self.config_dir,
          os.path.basename(self.original_chkpt_file))
        chkpt = load_checkpoint(model, checkpoint_file)

        print("training model\n")
        
        model.train()
        train_detector(
            model, datasets,
            self.mmdet_config,
            distributed=False,
            validate=False, meta=meta
        )
        
        self.model = model
        
        if "save_model_every_ckpt" in self.config and self.config["save_model_every_ckpt"]:
            if os.path.exists(os.path.join(self.config["work_dir"], "latest.pth")):
                fname = str(self.stage) + "_" + str(self.ckpt) + "_model.pth"
                shutil.copy(os.path.join(self.config["work_dir"], "latest.pth"), os.path.join(self.config["work_dir"], fname))

        if "eval_train_set" in self.config and self.config["eval_train_set"]:
            if os.path.exists(os.path.join(self.work_dir, "latest.pth")):
                fname = str(self.stage) + "_" + str(self.ckpt) + "_model.pth"
                shutil.copy(os.path.join(self.work_dir, "latest.pth"), os.path.join(self.work_dir, fname))
            
            fname = str(self.stage) + "_" + str(self.ckpt) + "_train_data_coco.json"
            if os.path.exists(os.path.join(self.work_dir, "train_data_coco.json")):
                shutil.copy(os.path.join(self.work_dir, "train_data_coco.json"), os.path.join(self.work_dir, fname))

        self.save_final_model()

        gc.collect()  # Make sure all object have ben deallocated if not used
        torch.cuda.empty_cache()
        return


    def add_data_from_disk( self, categories, train_files, train_dets, test_files, test_dets ):
        #print('add_data_from_disk')

        if len( train_files ) != len( train_dets ):
            print( "Error: train file and groundtruth count mismatch" )
            return
        
        cats = []
        self.cats = categories.all_class_names()
        for cat in self.cats:
            cat_id = categories.get_class_id(cat)
            cats.append({'name': cat, 'id': int(cat_id)})

        for split in [train_files, test_files]:
            is_train = ( split == train_files )
            num_images = len(split)
            
            images = []
            annotations = []
            annotation_id = 0

            for index in range(num_images):
                filename = split[index]
                img = mmcv.image.imread(filename)
                height, width = img.shape[:2]
                targets = train_dets[index] if is_train else test_dets[index]

                image_dct = {'file_name': filename, # dataset.root + '/' + dataset.image_fnames[index],
                    'height': height,
                    'width': width,
                    'id': int(index)
                }

                image_anno_ctr = 0

                for target in targets:
                    bbox = [  target.bounding_box.min_x(),
                              target.bounding_box.min_y(),
                              target.bounding_box.max_x(),
                              target.bounding_box.max_y() ] # tlbr
                    
                    # skip bboxes with 0 width or height
                    if (bbox[2] - bbox[0]) <= 0 or (bbox[3] - bbox[1]) <= 0:
                        continue

                    class_lbl = target.type.get_most_likely_class()
                    if categories is not None:
                        class_id = categories.get_class_id( class_lbl )
                    else:
                        if class_lbl not in self._categories:
                            self._categories.append( class_lbl )
                        class_id = self._categories.index( class_lbl )

                    annotation_dct = {'bbox': [bbox[0], bbox[1], (bbox[2]-bbox[0]), (bbox[3]-bbox[1])],  # coco annotations in file need x, y, width, height
                        'image_id': int(index),
                        'area': (bbox[2]-bbox[0]) * (bbox[3]-bbox[1]),
                        'category_id': class_id,
                        'id': annotation_id,
                        'iscrowd': 0
                    }
                    annotations.append(annotation_dct)
                    annotation_id += 1
                    image_anno_ctr += 1

                images.append(image_dct)

            coco_format_json = dict(
                images=images,
                annotations=annotations,
                categories=cats)

            fn = 'train_data_coco.json' if is_train else 'test_data_coco.json'
            output_file = os.path.join(self.config["work_dir"], fn)
            mmcv.dump(coco_format_json, output_file)
            
            print(f"Transformed the dataset into COCO style: {output_file} "
                  f"Num Images {len(images)} and Num Annotations: {len(annotations)}")
           

    def interupt_handler( self ):
        self.proc.send_signal( signal.SIGINT )
        timeout = 0
        while self.proc.poll() is None:
            time.sleep( 0.1 )
            timeout += 0.1
            if timeout > 5:
                self.proc.kill()
                break
        self.save_final_model()
        sys.exit( 0 )

    def save_final_model( self ):
        if not self._output_directory:
            return

        final_model = os.path.join( self.config["work_dir"], "latest.pth" )

        net_fn = "convnext_xl_cascade_rcnn.py"
        weight_fn = "convnext_xl_cascade_rcnn.pth"
        label_fn = "convnext_xl_cascade_rcnn.txt"
        pipe_fn = "detector.pipe"
  
        if not os.path.exists( self._output_directory ):
            os.mkdir( self._output_directory )

        output_net = os.path.join( self._output_directory, net_fn )
        output_model = os.path.join( self._output_directory, weight_fn )
        output_label = os.path.join( self._output_directory, label_fn )
        output_pipe = os.path.join( self._output_directory, pipe_fn )

        if not os.path.exists( final_model ):
            print( "\nModel failed to finsh training\n" )
            sys.exit( 0 )

        # Copy final model weights
        copyfile( final_model, output_model )

        # Copy network py file
        self.insert_training_params( self.mmdet_config_file, output_net )

        # Write out labels file
        with open( output_label, "w" ) as fout:
            for category in self.cats:
                fout.write( category + "\n" )

        with open( os.path.join( self._output_directory, "tmp.zip" ), "w" ) as fout:
            fout.write( "placeholder\n" )

        # Write out pipeline template
        if len( self._pipeline_template ) > 0:
            self.insert_model_files( self._pipeline_template,
                                     output_pipe,
                                     net_fn,
                                     weight_fn,
                                     label_fn )

        # Output additional completion text
        print( "\nWrote finalized model to " + output_model )

    def insert_training_params( self, input_cfg, output_cfg ):
        repl_strs = [ [ "[-CLASS_COUNT_INSERT-]", str(len(self.cats)+1) ] ]
        self.replace_strs_in_file( input_cfg, output_cfg, repl_strs )

    def insert_model_files( self, input_cfg, output_cfg, net, wgt, cls ):
        repl_strs = [ [ "[-NETWORK-CONFIG-]", net ],
                      [ "[-NETWORK-WEIGHTS-]", wgt ],
                      [ "[-NETWORK-CLASSES-]", cls ],
                      [ "[-LEARN-FLAG-]", "true" ] ]
        self.replace_strs_in_file( input_cfg, output_cfg, repl_strs )

    def replace_strs_in_file( self, input_cfg, output_cfg, repl_strs ):

        fin = open( input_cfg )
        fout = open( output_cfg, 'w' )

        all_lines = []
        for s in list( fin ):
            all_lines.append( s )

        for repl in repl_strs:
            for i, s in enumerate( all_lines ):
                all_lines[i] = s.replace( repl[0], repl[1] )
        for s in all_lines:
            fout.write( s )

        fout.close()
        fin.close()

def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    # Register Algorithm
    implementation_name = "mmdet_convnext"

    if algorithm_factory.has_algorithm_impl_name(
      ConvNextCascadeRCNNTrainer.static_type_name(), implementation_name ):
        return

    algorithm_factory.add_algorithm(
      implementation_name,
      "PyTorch ConvNext CascadeRcnn supervised mmdet training routine",
      ConvNextCascadeRCNNTrainer
    )

    algorithm_factory.mark_algorithm_as_loaded( implementation_name )
