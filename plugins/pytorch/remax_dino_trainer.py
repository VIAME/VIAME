# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import time
import os
from pathlib import Path
import pickle
import mmcv
import numpy as np
import torch
import scipy
import sys
import ubelt as ub
import yaml

from collections import namedtuple

from kwiver.vital.algo import TrainDetector

from viame.pytorch.remax.util.slconfig import SLConfig
from viame.pytorch.remax.model.dino import build_dino
from viame.pytorch.remax.util.coco import build as build_dataset

from viame.pytorch.remax.util.box_ops import box_xyxy_to_cxcywh

from viame.pytorch.remax.ReMax import ReMax

_Option = namedtuple('_Option', ['attr', 'config', 'default', 'parse', 'help'])


class ReMaxDINOTrainer( TrainDetector ):
    '''
    Implementation of TrainDetector class for ReMax Training
    '''
    _options = [
        _Option('_gpu_count', 'gpu_count', -1, int, ''),
        _Option('_norm_degree', 'norm_degree', 1, int, ''),
        _Option('_launcher', 'launcher', 'pytorch', str, ''), # "none, pytorch, slurm, or mpi" 
        
        _Option('_dino_config', 'dino_config', '', str, ''),
        _Option('_device', 'device', '', str, ''),
        _Option('_threshold', 'threshold', 0.0, float, ''),
        _Option('_model_checkpoint_file', 'model_checkpoint_file', '', str, ''),
        _Option('_work_dir', 'work_dir', '', str, ''),
        _Option('_output_directory', 'output_directory', '', str, ''),
        _Option('_feature_directory', 'feature_dir', '', str, ''),
        _Option('_debug_mode', 'debug_mode', False, bool, ''),
        _Option('_feature_cache', 'feature_cache', '', str, '')
    ]


    def __init__( self ):
        TrainDetector.__init__(self)
        for opt in self._options:
            setattr(self, opt.attr, opt.default)

        self.image_root = ''
        self._config_file = ""
        self._seed_weights = ""
        self._train_directory = "deep_training"
        self._output_directory = "category_models"
        self._output_prefix = "custom_cfrnn"
        self._pipeline_template = ""
        self._gpu_count = -1
        self._random_seed = "none"
        self._tmp_training_file = "training_truth.pickle"
        self._tmp_validation_file = "validation_truth.pickle"
        self._validate = True
        self._gt_frames_only = False
        self._launcher = "pytorch"  # "none, pytorch, slurm, or mpi" 
        self._train_in_new_process = True
        self._categories = []
    
    def get_configuration( self ):
        # Inherit from the base class
        cfg = super( TrainDetector, self ).get_configuration()

        for opt in self._options:
            cfg.set_value(opt.config, str(getattr(self, opt.attr)))
        return cfg
    
    def set_configuration( self, cfg_in):
        cfg = self.get_configuration()
        cfg.merge_config( cfg_in )
        for opt in self._options:
            setattr(self, opt.attr, opt.parse(cfg.get_value(opt.config)))

        self.ckpt = 0 # TODO: not sure about this
        self.stage = 'base' # TODO: also not sure about this 

        device = self._device
        if ub.iterable(device):
            self.device = device
        else:
            if device == -1:
                self.device = list(range(torch.cuda.device_count()))
            else:
                self.device = [device]
        if len(self.device) > torch.cuda.device_count():
            self.device = self.device[:torch.cuda.device_count()]
            
        self.original_chkpt_file = self._model_checkpoint_file
        self.load_model()
        
    def check_configuration( self, cfg ):
        return True
        if not cfg.has_value( "config_file" ) or len( cfg.get_value( "config_file") ) == 0:
            print( "A config file must be specified!" )
            return False
        return True
    
    def __getstate__( self ):
        return self.__dict__

    def __setstate__( self, dict ):
        self.__dict__ = dict

    def bbox_iou(self, boxA, boxB):
        # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        # ^^ corrected.

        # Determine the (x, y)-coordinates of the intersection rectangle
        t0 = time.time()
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interW = xB - xA + 1
        interH = yB - yA + 1

        # Correction: reject non-overlapping boxes
        if interW <=0 or interH <=0 :
            return -1.0

        interArea = interW * interH
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou


    def match_bboxes(self, bbox_gt, bbox_pred, IOU_THRESH=0.5):
        '''
        Given sets of true and predicted bounding-boxes,
        determine the best possible match.
        Parameters
        ----------
        bbox_gt, bbox_pred : N1x4 and N2x4 np array of bboxes [x1,y1,x2,y2]. 
        The number of bboxes, N1 and N2, need not be the same.
        
        Returns
        -------
        (idxs_true, idxs_pred, ious, labels)
            idxs_true, idxs_pred : indices into gt and pred for matches
            ious : corresponding IOU value of each match
            labels: vector of 0/1 values for the list of detections
        '''
        n_true = bbox_gt.shape[0]
        n_pred = bbox_pred.shape[0]
        MAX_DIST = 1.0
        MIN_IOU = 0.0
        # NUM_GT x NUM_PRED
        iou_matrix = np.zeros((n_true, n_pred))
        for i in range(n_true):
            for j in range(n_pred):
                iou_matrix[i, j] = self.bbox_iou(bbox_gt[i,:], bbox_pred[j,:])
        if n_pred > n_true:
            # there are more predictions than ground-truth - add dummy rows
            diff = n_pred - n_true
            iou_matrix = np.concatenate( (iou_matrix,
                                            np.full((diff, n_pred), MIN_IOU)),
                                        axis=0)
        if n_true > n_pred:
            # more ground-truth than predictions - add dummy columns
            diff = n_true - n_pred
            iou_matrix = np.concatenate( (iou_matrix,
                                            np.full((n_true, diff), MIN_IOU)),
                                        axis=1)

        # call the Hungarian matching
        idxs_true, idxs_pred = scipy.optimize.linear_sum_assignment(1 - iou_matrix)
        if (not idxs_true.size) or (not idxs_pred.size):
            ious = np.array([])
        else:
            ious = iou_matrix[idxs_true, idxs_pred]

        # remove dummy assignments
        sel_pred = idxs_pred<n_pred
        idx_pred_actual = idxs_pred[sel_pred]
        idx_gt_actual = idxs_true[sel_pred]
        ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
        sel_valid = (ious_actual > IOU_THRESH)
        label = sel_valid.astype(int)

        return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid], label

    def load_model( self ):
        self.dino_config = SLConfig.fromfile(self._dino_config)
        self.dino_config.device = 'cuda'
        self.dino_config.checkpoint_path = self.original_chkpt_file
        self.dino_config
        self.model, self.criterion, self.postprocessors = build_dino(self.dino_config)
        checkpoint = torch.load(self.original_chkpt_file)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

    def update_model( self ):
        self.dino_config.coco_path = str(os.path.join(self._work_dir, 'train_data_coco.json')) # the path of coco
        self.dino_config.fix_size = False
        self.dino_config.image_root = self.image_root
        dataset_train = build_dataset(image_set='train', args=self.dino_config)
        if self._debug_mode and os.path.exists(self._feature_cache):
            with open(self._feature_cache, 'rb') as f:
                train_data = torch.load(f)
        else:
            train_feats = {}
            count = 0
            for image, targets in dataset_train:
                count += 1
                if count % 50 == 0:
                    print("processed ", count, "images")
                # build gt_dict for vis
                gt_label = [str(int(item)) for item in targets['labels']]
                iid = targets['image_id']
                
                # build pred_dict for vis
                output = self.model.cuda()(image[None].cuda())
                output = self.postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]
                scores = output['scores']
                labels = output['labels']
                feats = output['feats']
                boxes = box_xyxy_to_cxcywh(output['boxes'])
                select_mask = scores > self._threshold

                pred_label = [str(int(item)) for item in labels[select_mask]]
                if len(pred_label) == 0 and len(gt_label) == 0:
                    continue
                pred_feats = feats[select_mask]
                pred_boxes = boxes[select_mask]
                idxs_true, idxs_pred, _, _ = self.match_bboxes(targets['boxes'], pred_boxes)
                for i in range(idxs_pred.size):
                    label = gt_label[idxs_true[i]]
                    feats = pred_feats[idxs_pred[i]]
                    boxes = pred_boxes[idxs_pred[i]]

                    if label not in train_feats.keys():
                        train_feats[label] = feats.unsqueeze(0)
                    else:
                        train_feats[label] = torch.cat((train_feats[label], feats.unsqueeze(0)), 0)

                for i in range(len(pred_label)):
                    if i not in idxs_pred:
                        if '-1' not in train_feats.keys():
                            train_feats['-1'] = pred_feats[i].unsqueeze(0)
                        else:
                            train_feats['-1'] = torch.cat((train_feats['-1'], pred_feats[i].unsqueeze(0)), 0)



            train_data = []
            for cls in train_feats.keys():
                train_data.append(train_feats[cls])
            train_data = torch.cat(train_data, dim=0)
            if self._debug_mode and not os.path.exists(self._feature_cache):
                torch.save(train_data, self._feature_cache)

                
        # Normalization step        
        train_data = torch.linalg.norm(train_data, dim=1, ord=self._norm_degree)

        self.remax_model = ReMax(train_data)

        self.save_final_model()

        return {"type": "dino_remax"}

    def save_final_model( self ):
        output_model_name = "remax.pkl"
        output_model = os.path.join( self._output_directory,
            output_model_name )
        with open(output_model, 'wb') as f:
            pickle.dump(self.remax_model, f)


        if not os.path.exists( output_model ):
            print( "\nModel failed to finsh training\n" )
            sys.exit( 0 )


        # Output additional completion text
        print( "\nWrote finalized model to " + output_model )

    def add_data_from_disk( self, categories, train_files, train_dets, test_files, test_dets ):
        print('add_data_from_disk')

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
                if not self.image_root:
                    self.image_root = os.path.dirname(filename) # TODO: there might be a better way to get this?
                    print("self.image_root", self.image_root)

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
            output_file = os.path.join(self._work_dir, fn)
            mmcv.dump(coco_format_json, output_file)
            
            print(f"Transformed the dataset into COCO style: {output_file} "
                  f"Num Images {len(images)} and Num Annotations: {len(annotations)}")
        
           

def __vital_algorithm_register__():
    from kwiver.vital.algo import algorithm_factory

    # Register Algorithm
    implementation_name = "dino_remax"

    if algorithm_factory.has_algorithm_impl_name(
      ReMaxDINOTrainer.static_type_name(), implementation_name):
        return

    algorithm_factory.add_algorithm(implementation_name,
      "PyTorch MMDetection inference routine", ReMaxDINOTrainer)

    algorithm_factory.mark_algorithm_as_loaded(implementation_name)